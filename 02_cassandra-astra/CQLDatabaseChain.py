# DEMO-LEVEL STUFF #

# we uncomfortably make Cassandra class a subclass of - ahem - SQLDatabase
# to speed up a somewhat-working demo

from typing import Any, Iterable, List, Optional, Dict

from langchain import SQLDatabase, SQLDatabaseChain
from langchain.base_language import BaseLanguageModel
from langchain.prompts.base import BasePromptTemplate
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.prompts.prompt import PromptTemplate
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.sql_database.prompt import DECIDER_PROMPT
from langchain.chains import SQLDatabaseSequentialChain

CREATE_TABLE_TEMPLATE = """CREATE TABLE {keyspace_name}.{table_name} (
{columns}
{primary_key}
);"""


_cql_prompt = """You are a Cassandra CQL expert. Given an input question,
first create a syntactically correct CQL query to run,
then look at the results of the query and return the answer to the input question.
Unless the user specifies in the question a specific number of examples to obtain,
query for at most {top_k} results using the LIMIT clause as per CQL.
Never query for all columns from a table.
CQL queries must always specify equalities for the partition key values in the WHERE clause.
If this is impossible, refuse to execute the query.
You must query only the columns that are needed to answer the question.
Wrap each column name in double quotes (") to denote them as delimited identifiers.
Pay attention to use only the column names you can see in the tables below.
Be careful to not query for columns that do not exist.
Also, pay attention to which column is in which table.
Remember that CQL statement must end with a semicolon (;).

The query cannot contain the ORDER BY clause.

Use the following format:

Question: "Question here"
SQLQuery: "SQL Query to run"
SQLResult: "Result of the SQLQuery"
Answer: "Final answer here"

Only use the following tables:
{table_info}

The last line in each CREATE TABLE statement is of the form "PRIMARY KEY ( (partition keys), clustering columns)"

Question: {input}"""

CQL_PROMPT = PromptTemplate(
    input_variables=["input", "table_info", "top_k"],
    template=_cql_prompt,
)


class CQLDatabaseInspector:

    def __init__(self, session, keyspace):
        self.session = session
        self.keyspace = keyspace

    def get_table_names(self) -> Iterable[str]:
        return self.session.cluster.metadata.keyspaces[self.keyspace].tables.keys()

    # Keyspace description
    @staticmethod
    def _desc_col(col):
        return f'{col.name} {col.cql_type}{" static" if col.is_static else ""}'

    @staticmethod
    def _desc_cols(tab):
        return '\n'.join('    %s,' % CQLDatabaseInspector._desc_col(tb) for _, tb in tab.columns.items())

    @staticmethod
    def _desc_pk(tab):
        partk_spec = ' , '.join([col.name for col in tab.partition_key])
        clustering_cols = [col.name for col in tab.clustering_key]
        if clustering_cols != []:
            clustering_spec = f" , {' , '.join([col.name for col in tab.clustering_key])}"
        else:
            clustering_spec = ''
        return f'    PRIMARY KEY ( ( {partk_spec} ){clustering_spec} )'

    def describeTable(self, tableName):
        tab = self.session.cluster.metadata.keyspaces[self.keyspace].tables[tableName]
        return CREATE_TABLE_TEMPLATE.format(
            keyspace_name=self.keyspace,
            table_name=tab.name,
            columns=CQLDatabaseInspector._desc_cols(tab),
            primary_key=CQLDatabaseInspector._desc_pk(tab),
        )


class CQLDatabase(SQLDatabase):

    def __init__(
        self,
        session,
        keyspace,
        ignore_tables: Optional[List[str]] = None,
        include_tables: Optional[List[str]] = None,
        custom_table_info: Optional[dict] = None,
    ):
        self.session = session
        self.keyspace = keyspace
        if include_tables and ignore_tables:
            raise ValueError("Cannot specify both include_tables and ignore_tables")

        self._inspector = CQLDatabaseInspector(self.session, self.keyspace)

        self._all_tables = set(
            self._inspector.get_table_names()
        )

        self._include_tables = set(include_tables) if include_tables else set()
        if self._include_tables:
            missing_tables = self._include_tables - self._all_tables
            if missing_tables:
                raise ValueError(
                    f"include_tables {missing_tables} not found in database"
                )
        self._ignore_tables = set(ignore_tables) if ignore_tables else set()
        if self._ignore_tables:
            missing_tables = self._ignore_tables - self._all_tables
            if missing_tables:
                raise ValueError(
                    f"ignore_tables {missing_tables} not found in database"
                )
        usable_tables = self.get_usable_table_names()
        self._usable_tables = set(usable_tables) if usable_tables else self._all_tables

        self._custom_table_info = custom_table_info
        if self._custom_table_info:
            if not isinstance(self._custom_table_info, dict):
                raise TypeError(
                    "table_info must be a dictionary with table names as keys and the "
                    "desired table info as values"
                )
            # only keep the tables that are also present in the database
            intersection = set(self._custom_table_info).intersection(self._all_tables)
            self._custom_table_info = dict(
                (table, self._custom_table_info[table])
                for table in self._custom_table_info
                if table in intersection
            )

    def get_usable_table_names(self) -> Iterable[str]:
        """Get names of tables available."""
        if self._include_tables:
            return self._include_tables
        return self._all_tables - self._ignore_tables

    def get_table_names(self) -> Iterable[str]:
        """Get names of tables available."""
        warnings.warn(
            "This method is deprecated - please use `get_usable_table_names`."
        )
        return self.get_usable_table_names()

    @property
    def table_info(self) -> str:
        """Information about all tables in the database."""
        return self.get_table_info()

    def get_table_info(self, table_names: Optional[List[str]] = None) -> str:
        """Get information about specified tables.
        Sample rows not implemented at this point
        """
        all_table_names = self.get_usable_table_names()
        if table_names is not None:
            missing_tables = set(table_names).difference(all_table_names)
            if missing_tables:
                raise ValueError(f"table_names {missing_tables} not found in database")
            all_table_names = table_names

        tables = []
        for table in self._usable_tables:
            if self._custom_table_info and table.name in self._custom_table_info:
                tables.append(self._custom_table_info[table.name])
            else:
                tables.append(self._inspector.describeTable(table))
        final_str = "\n\n".join(tables)
        return final_str

    def run(self, command: str, fetch: str = "all") -> str:
        """Execute a CQL command and return a string representing the results.

        If the statement returns rows, a string of the results is returned, such as:
                [('stringValue', 123), ('stringV2', 456)]
            or
                [('stringVal', )]
        If the statement returns no rows, an empty string is returned.
        """

        def _toTuple(res):
            return tuple(getattr(res, f) for f in res._fields)

        self.session.execute(f'USE {self.keyspace};')
        _q = self.session.execute(command)
        if fetch == 'all':
            results = list(_q)
            # if results:
            resTuples = [_toTuple(res) for res in results]
            return str(resTuples)
            # else:
            #     return ""
        elif fetch == 'one':
            result = _q.one()
            if result:
                resTuple = _toTuple(result)
                return str(resTuple)
            else:
                return ""
        else:
            raise ValueError("Fetch parameter must be either 'one' or 'all'")

    @property
    def dialect(self) -> str:
        """Return string representation of dialect to use."""
        return 'cql'


class CQLDatabaseChain(SQLDatabaseChain):

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        db: SQLDatabase,
        prompt: Optional[BasePromptTemplate] = None,
        **kwargs: Any,
    ):
        prompt = prompt or CQL_PROMPT
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        return cls(llm_chain=llm_chain, database=db, **kwargs)


class CQLDatabaseSequentialChain(SQLDatabaseSequentialChain):
    """CQL variant."""

    decider_chain: LLMChain
    sql_chain: CQLDatabaseChain
    input_key: str = "query"  #: :meta private:
    output_key: str = "result"  #: :meta private:
    return_intermediate_steps: bool = False

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        database: CQLDatabase,
        query_prompt: BasePromptTemplate = CQL_PROMPT,
        decider_prompt: BasePromptTemplate = DECIDER_PROMPT,
        **kwargs: Any,
    ):
        """Load the necessary chains."""
        sql_chain = CQLDatabaseChain(
            llm=llm, database=database, prompt=query_prompt, **kwargs
        )
        decider_chain = LLMChain(
            llm=llm, prompt=decider_prompt, output_key="table_names"
        )
        return cls(sql_chain=sql_chain, decider_chain=decider_chain, **kwargs)