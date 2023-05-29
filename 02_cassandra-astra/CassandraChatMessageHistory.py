# import json
# from typing import List


# from langchain.schema import (
#     AIMessage,
#     BaseChatMessageHistory,
#     BaseMessage,
#     HumanMessage,
#     _message_to_dict,
#     messages_from_dict,
# )


# DEFAULT_TTL_SECONDS = 86400

# _createTableCQLTemplate = """
# CREATE TABLE IF NOT EXISTS {keyspace}.{tableName} (
#     session_id TEXT,
#     message_id TIMEUUID,
#     message TEXT,
#     PRIMARY KEY (( session_id ) , message_id )
# ) WITH CLUSTERING ORDER BY (message_id ASC);
# """
# _getSessionMessagesCQLTemplate = """
# SELECT message
#     FROM {keyspace}.{tableName}
# WHERE session_id=%s;
# """
# _storeSessionMessageCQLTemplate = """
# INSERT INTO {keyspace}.{tableName} (
#     session_id,
#     message_id,
#     message
# ) VALUES (
#     %s,
#     now(),
#     %s
# ){ttlSpec};
# """
# _clearSessionCQLTemplate = """
# DELETE FROM {keyspace}.{tableName} WHERE session_id = %s;
# """


# class CassandraChatMessageHistory(BaseChatMessageHistory):
    
#     def __init__(self, session_id, session, keyspace, ttl_seconds=DEFAULT_TTL_SECONDS):
#         self.session_id = session_id
#         self.session = session
#         self.keyspace = keyspace
#         self.ttlSeconds = ttl_seconds
#         self.tableName = 'langchain_chat_history'
#         # Schema creation, if needed
#         if self.ttlSeconds:
#             self.ttlSpec = f' USING TTL {self.ttlSeconds}'
#         else:
#             self.ttlSpec = ''
#         createTableCQL = _createTableCQLTemplate.format(
#             keyspace=self.keyspace,
#             tableName=self.tableName,
#         )
#         session.execute(createTableCQL)

#     @property
#     def messages(self) -> List[BaseMessage]:  # type: ignore
#         """Retrieve all session messages from DB"""
#         getSessionMessagesCQL = _getSessionMessagesCQLTemplate.format(
#             keyspace=self.keyspace,
#             tableName=self.tableName,
#         )
#         messageRows = self.session.execute(
#             getSessionMessagesCQL,
#             (self.session_id, )
#         )
#         items = [json.loads(row.message) for row in messageRows]
#         messages = messages_from_dict(items)
#         return messages

#     def add_user_message(self, message: str) -> None:
#         self.append(HumanMessage(content=message))

#     def add_ai_message(self, message: str) -> None:
#         self.append(AIMessage(content=message))

#     def append(self, message: BaseMessage) -> None:
#         """Write a message to the table"""
#         storeSessionMessageCQL = _storeSessionMessageCQLTemplate.format(
#             keyspace=self.keyspace,
#             tableName=self.tableName,
#             ttlSpec=self.ttlSpec,
#         )
#         self.session.execute(
#             storeSessionMessageCQL,
#             (
#                 self.session_id,
#                 json.dumps(_message_to_dict(message)),
#             )
#         )

#     def clear(self) -> None:
#         """Clear session memory from DB"""
#         clearSessionCQL = _clearSessionCQLTemplate.format(
#             keyspace=self.keyspace,
#             tableName=self.tableName,
#         )
#         self.session.execute(clearSessionCQL, (self.session_id, ))
