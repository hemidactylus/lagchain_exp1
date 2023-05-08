from langchain import OpenAI, SQLDatabase, SQLDatabaseChain

db = SQLDatabase.from_uri("sqlite:///Chinook.db")
llm = OpenAI(temperature=0)

db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)

# db_chain.run("Are there albums by Queen?")
db_chain.run("What is the average temperature on Saturn ?")

