"""Non-text2query approaches to CQL Agents"""

from typing import Iterable, Optional, Any, Dict


from pydantic import Field

from langchain.tools import BaseTool
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain.agents.agent import AgentExecutor
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.base_language import BaseLanguageModel
from langchain.callbacks.base import BaseCallbackManager
from langchain.chains.llm import LLMChain

from cassandra.cluster import Session


DESCRIBE_TABLE_TEMPLATE = """Table name: {keyspace_name}.{table_name}
Column list:
{column_list}
Partition key columns: {partition_key_columns}

"""

class CQLDatabaseInspectorForAgent:

    def __init__(self, session, keyspace, excludedTables=set()):
        self.session = session
        self.keyspace = keyspace
        self.excludedTables = excludedTables

    def getTableNames(self) -> Iterable[str]:
        return self.session.cluster.metadata.keyspaces[self.keyspace].tables.keys()

    def describeKeyspace(self):
        return '\n'.join(
            self.describeTable(tab)
            for tab in self.getTableNames()
            if tab not in self.excludedTables
        )

    def getColumnNames(self, tableName):
        tab = self.session.cluster.metadata.keyspaces[self.keyspace].tables[tableName]
        return (col.name for _, col in tab.columns.items())

    def getPartitionKey(self, tableName):
        tab = self.session.cluster.metadata.keyspaces[self.keyspace].tables[tableName]
        return (col.name for col in tab.partition_key)

    def describeTable(self, tableName):
        tab = self.session.cluster.metadata.keyspaces[self.keyspace].tables[tableName]
        return DESCRIBE_TABLE_TEMPLATE.format(
            keyspace_name=self.keyspace,
            table_name=tab.name,
            column_list='\n'.join(
                '    %s (%s)' % (col.name, col.cql_type)
                for _, col in tab.columns.items()
            ),
            partition_key_columns=', '.join(self.getPartitionKey(tableName)),
        )


class CQLDescribeKeyspace(BaseTool):
    name = "describe_keyspace"
    description = """The tool to get a full description of the available data tables.
Each table is described through its name, its columns with their type, and which columns make up the partition key.
The input to this tool is discarded.
"""
    session: Session = Field(exclude=True)
    keyspace: str = Field(exclude=True)
    inspector: CQLDatabaseInspectorForAgent = Field(exclude=True)

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Use the tool."""
        return self.inspector.describeKeyspace()
    
    async def _arun(self, query: str,  run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("no async")

class CQLGetRowsFromTable(BaseTool):
    name = "get_table_rows"
    description = """The tool to retrieve rows from a table.
The input is in the form "table_name partition_key". No other input is allowed.
All rows matching the partition key given in the input will be returned.
There is no other way to get data from the table:
    - if data from several partitions are required, this tool is used multiple times
    - if a subset of data is needed, only a portion of the return value of this tool will be used.
    - if you are asked to find the maximum or minimum value, just inspect the full results.
The resulting records are in CSV format with a header line, as in:
    title,year,rating
    Gone with the wind,2013,8
    The reaper,1998,6
    Not too fast,2019,9
"""
    session: Session = Field(exclude=True)
    keyspace: str = Field(exclude=True)
    inspector: CQLDatabaseInspectorForAgent = Field(exclude=True)

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Use the tool."""
        full_table, pkey_value = [p.strip() for p in query.split(' ') if p.strip()!='']
        table = list(full_table.split('.'))[-1]
        all_cols = list(self.inspector.getColumnNames(table))
        pkeys = list(self.inspector.getPartitionKey(table))
        assert(len(pkeys) == 1)
        pkey = pkeys[0]
        stmt = f'SELECT {", ".join(all_cols)} FROM {full_table} WHERE {pkey}=%s;'
        results = list(self.session.execute(stmt, (pkey_value, )))
        data_string = '\n'.join(
            ','.join(str(getattr(rs, f)) for f in all_cols)
            for rs in results
        )
        res_string = f'{",".join(all_cols)}\n{data_string}'
        return res_string
    
    async def _arun(self, query: str,  run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("no async")

PREFIX = """
    You are a helpful data-expert assistant able to read records, or rows, from tables of a database
    in order to answer questions.

    The tools to inspect the table schema and the values stored in the database are made available to you.
    You can read table data multiple times if this is needed.

    Your action will start by getting the schema of the tables.
"""

def create_cql_agent(
    llm: BaseLanguageModel,
    session: Session,
    keyspace: str,
    callback_manager: Optional[BaseCallbackManager] = None,
    verbose: bool = False,
    prefix: str = PREFIX,
    agent_executor_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs: Dict[str, Any],
) -> AgentExecutor:
    """Construct a box-shuffler agent from an LLM (and other stuff)"""
    session.execute('USE %s;' % keyspace)
    inspector = CQLDatabaseInspectorForAgent(session, keyspace, excludedTables={'pqdata'})
    tools = [
        CQLGetRowsFromTable(session=session, keyspace=keyspace, inspector=inspector),
        CQLDescribeKeyspace(session=session, keyspace=keyspace, inspector=inspector),
    ]
    ### 
    prompt = ZeroShotAgent.create_prompt(tools, prefix=prefix)
    llm_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        callback_manager=callback_manager,
    )
    tool_names = [tool.name for tool in tools]
    agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names, **kwargs)
    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        callback_manager=callback_manager,
        verbose=verbose,
        **(agent_executor_kwargs or {}),
    )


if __name__ == '__main__':
    # Cassandra session, move along
    from cqlsession import getCqlSession
    astraSession = getCqlSession()

    from langchain.chat_models import ChatOpenAI
    llm = ChatOpenAI(temperature=0)
    cqlagent = create_cql_agent(
        llm=llm,
        session=astraSession,
        keyspace='pqdemo',
        verbose=True,
    )

    cqlagent.run("Are there people in milan below age 10? City names are lowercase.")

    # inspe = CQLDatabaseInspectorForAgent(astraSession, 'pqdemo')
    # rowgetter = CQLGetRowsFromTable(session=astraSession, keyspace='pqdemo', inspector=inspe)
    # print(rowgetter._run('pqdemo.people milan'))
    # schemagetter = CQLDescribeKeyspace(session=astraSession, keyspace='pqdemo', inspector=inspe)
    # print(schemagetter._run('whatever'))

https://python.langchain.com/en/latest/modules/agents/toolkits/examples/csv.html