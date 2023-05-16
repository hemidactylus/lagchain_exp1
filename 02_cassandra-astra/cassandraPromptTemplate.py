"""
A prompt template that automates retrieving rows from Cassandra and making their
content into variables in a prompt.
"""

from functools import reduce
from typing import List, Any, Dict, Callable

from DependencyfulPromptTemplate import DependencyfulPromptTemplate


def _table_primary_key_columns(session, keyspace, tableName) -> List[str]:
    table = session.cluster.metadata.keyspaces[keyspace].tables[tableName]
    return [
        col.name for col in table.partition_key
    ] + [
        col.name for col in table.clustering_key
    ]


# Since subclassing for thins one is a mess, with pydantic and so many changed parameters,
# we opt for a factory function

def createCassandraPromptTemplate(session, keyspace, template, input_variables, field_mapper, literal_nones=False):
    
    # what tables do we need?
    tablesNeeded = {fmv[0] for fmv in field_mapper.values()}
    # what's the primary key for each of these tables
    primaryKeyMap = {
        tableName: _table_primary_key_columns(session, keyspace, tableName)
        for tableName in tablesNeeded
    }
    # for the 'getter', first build the list of all inputs for the getter
    allPrimaryKeyColumns = list(reduce(lambda accum, nw: accum | set(nw), primaryKeyMap.values(), set()))
    # for the 'getter', build the function itself
    # TODO not optimized yet! (should group fields to return by table, etc)
    # TODO inspection to see if really must retrieve whole row or if it can do "select field-list ..."
    def _getter(deps, **kwargs):
        _session = deps['session']
        _keyspace = deps['keyspace']
        def _retrieve_field(_session2, _keyspace2, _tableName2, _keyColumns, _columnOrExtractor, _keyValueMap):
            selector = 'SELECT * FROM {keyspace}.{tableName} WHERE {whereClause} LIMIT 1;'.format(
                keyspace=_keyspace2,
                tableName=_tableName2,
                whereClause=' AND '.join(
                    f'{kc} = %s'
                    for kc in _keyColumns
                ),
            )
            values = tuple([
                _keyValueMap[kc]
                for kc in _keyColumns
            ])
            row = _session2.execute(selector, values).one()
            if row:
                if callable(_columnOrExtractor):
                    return _columnOrExtractor(row)
                else:
                    return getattr(row, _columnOrExtractor)
            else:
                if literal_nones:
                    return None
                else:
                    raise ValueError('No data found for %s from %s.%s' % (
                        str(_columnOrExtractor),
                        _keyspace2,
                        _tableName2,
                    ))
        
        return {
            field: _retrieve_field(_session, _keyspace, tableName, primaryKeyMap[tableName], columnOrExtractor, kwargs)
            for field, (tableName, columnOrExtractor) in field_mapper.items()
        }
    
    cassandraPromptTemplate = DependencyfulPromptTemplate(
        template=template,
        dependencies={'session': session, 'keyspace': keyspace},
        getter=_getter,
        input_variables=input_variables,
        forceGetterArguments=allPrimaryKeyColumns,
    )
    
    return cassandraPromptTemplate