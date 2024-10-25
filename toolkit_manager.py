from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
# from langchain_community.agent_toolkits import SQLDatabaseToolkit
from my_toolkit import My_SQLDatabaseToolkit
from langchain_tool import SQLDatabaseToolkit

def get_all_toolkits(db: SQLDatabase, model: ChatOpenAI):
    """Returns all tools including the ones from LangChain and custom ones."""
    
    # Input prebuilt toolkit (LangChain's SQL Database Toolkit)
    langchain_toolkit = SQLDatabaseToolkit(db=db, llm=model)
    langchain_tools = langchain_toolkit.get_tools()

    # Input custom toolkit
    custom_toolkit = My_SQLDatabaseToolkit(db=db, llm=model)
    custom_tools = custom_toolkit.get_tools()

    # Combine LangChain tools with custom tools
    all_tools = custom_tools + langchain_tools

    return all_tools
