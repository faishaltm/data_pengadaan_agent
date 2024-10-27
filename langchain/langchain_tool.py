from typing import Any, Dict, Optional, Sequence, Type, Union, List
from sqlalchemy.engine import Result

from pydantic import BaseModel, Field, field_validator

from langchain_core.language_models import BaseLanguageModel
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.prompts import PromptTemplate
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.tools import BaseTool
from langchain_community.tools.sql_database.prompt import QUERY_CHECKER
from langchain_core.tools.base import BaseToolkit



class BaseSQLDatabaseTool(BaseModel):
    """Base tool for interacting with a SQL database."""

    db: SQLDatabase = Field(exclude=True)

    class Config:
        arbitrary_types_allowed = True  

    # class Config(BaseTool.Config):
    #     pass



# class _QuerySQLDataBaseToolInput(BaseModel):
#     query: str = Field(..., description="A detailed and correct SQL query.")

# class QuerySQLDataBaseTool(BaseSQLDatabaseTool, BaseTool):
#     """Tool for querying a SQL database."""

#     name: str = "sql_db_query"
#     description: str = """
#     Execute a SQL query against the database and get back the result..
#     If the query is not correct, an error message will be returned.
#     If an error is returned, rewrite the query, check the query, and try again.
#     """
#     args_schema: Type[BaseModel] = _QuerySQLDataBaseToolInput

#     def _run(
#         self,
#         query: str,
#         run_manager: Optional[CallbackManagerForToolRun] = None,
#     ) -> Union[str, Sequence[Dict[str, Any]], Result]:
#         """Execute the query, return the results or an error message."""
#         return self.db.run_no_throw(query)



class _InfoSQLDatabaseToolInput(BaseModel):
    table_names: str = Field(
        ...,
        description=(
            "A comma-separated list of the table names for which to return the schema. "
            "Example input: 'table1, table2, table3'"
        ),
    )


class InfoSQLDatabaseTool(BaseSQLDatabaseTool, BaseTool):
    """Tool for getting metadata about a SQL database."""

    name: str = "sql_db_schema"
    description: str = "Get the schema and sample rows for the specified SQL tables."
    args_schema: Type[BaseModel] = _InfoSQLDatabaseToolInput

    def _run(
        self,
        table_names: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Get the schema for tables in a comma-separated list."""
        return self.db.get_table_info_no_throw(
            [t.strip() for t in table_names.split(",")]
        )



class _ListSQLDataBaseToolInput(BaseModel):
    tool_input: str = Field("", description="An empty string")


class ListSQLDatabaseTool(BaseSQLDatabaseTool, BaseTool):
    """Tool for getting tables names."""

    name: str = "sql_db_list_tables"
    description: str = "Input is an empty string, output is a comma-separated list of tables in the database."
    args_schema: Type[BaseModel] = _ListSQLDataBaseToolInput

    def _run(
        self,
        tool_input: str = "",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Get a comma-separated list of table names."""
        return ", ".join(self.db.get_usable_table_names())



class _QuerySQLCheckerToolInput(BaseModel):
    query: str = Field(..., description="A detailed and SQL query to be checked.")

class QuerySQLCheckerTool(BaseSQLDatabaseTool, BaseTool):
    """Use an LLM to check if a query is correct."""

    template: str = QUERY_CHECKER
    llm: BaseLanguageModel
    llm_chain: Any = None  # Set default to None
    name: str = "sql_db_query_checker"
    description: str = """
    Use this tool to double check if your query is correct before executing it.
    Always use this tool before executing a query with sql_db_query!
    """
    args_schema: Type[BaseModel] = _QuerySQLCheckerToolInput

    def model_post_init(self, __context: Optional[Dict[str, Any]] = None) -> None:
        """Initialize llm_chain after the model is instantiated."""
        if self.llm_chain is None:
            from langchain.chains.llm import LLMChain

            self.llm_chain = LLMChain(
                llm=self.llm,
                prompt=PromptTemplate(
                    template=self.template, input_variables=["dialect", "query"]
                ),
            )

        if self.llm_chain.prompt.input_variables != ["dialect", "query"]:
            raise ValueError(
                "LLM chain for QueryCheckerTool must have input variables ['query', 'dialect']"
            )

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the LLM to check the query."""
        return self.llm_chain.predict(
            query=query,
            dialect=self.db.dialect,
            callbacks=run_manager.get_child() if run_manager else None,
        )

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        return await self.llm_chain.apredict(
            query=query,
            dialect=self.db.dialect,
            callbacks=run_manager.get_child() if run_manager else None,
        )
    
class SQLDatabaseToolkit(BaseToolkit):
    """SQLDatabaseToolkit for interacting with SQL databases.

    Setup:
        Install ``langchain-community``.

        .. code-block:: bash

            pip install -U langchain-community

    Key init args:
        db: SQLDatabase
            The SQL database.
        llm: BaseLanguageModel
            The language model (for use with QuerySQLCheckerTool)

    Instantiate:
        .. code-block:: python

            from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
            from langchain_community.utilities.sql_database import SQLDatabase
            from langchain_openai import ChatOpenAI

            db = SQLDatabase.from_uri("sqlite:///Chinook.db")
            llm = ChatOpenAI(temperature=0)

            toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    Tools:
        .. code-block:: python

            toolkit.get_tools()

    Use within an agent:
        .. code-block:: python

            from langchain import hub
            from langgraph.prebuilt import create_react_agent

            # Pull prompt (or define your own)
            prompt_template = hub.pull("langchain-ai/sql-agent-system-prompt")
            system_message = prompt_template.format(dialect="SQLite", top_k=5)

            # Create agent
            agent_executor = create_react_agent(
                llm, toolkit.get_tools(), state_modifier=system_message
            )

            # Query agent
            example_query = "Which country's customers spent the most?"

            events = agent_executor.stream(
                {"messages": [("user", example_query)]},
                stream_mode="values",
            )
            for event in events:
                event["messages"][-1].pretty_print()
    """  # noqa: E501

    db: SQLDatabase = Field(exclude=True)
    llm: BaseLanguageModel = Field(exclude=True)

    @property
    def dialect(self) -> str:
        """Return string representation of SQL dialect to use."""
        return self.db.dialect

    class Config:
        arbitrary_types_allowed = True

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        list_sql_database_tool = ListSQLDatabaseTool(db=self.db)
        info_sql_database_tool_description = (
            "Input to this tool is a comma-separated list of tables, output is the "
            "schema and sample rows for those tables. "
            "Be sure that the tables actually exist by calling "
            f"{list_sql_database_tool.name} first! "
            "Example Input: table1, table2, table3"
        )
        info_sql_database_tool = InfoSQLDatabaseTool(
            db=self.db, description=info_sql_database_tool_description
        )
        # query_sql_database_tool_description = (
        #     "Input to this tool is a detailed and correct SQL query, output is a "
        #     "result from the database. If the query is not correct, an error message "
        #     "will be returned. If an error is returned, rewrite the query, check the "
        #     "query, and try again. If you encounter an issue with Unknown column "
        #     f"'xxxx' in 'field list', use {info_sql_database_tool.name} "
        #     "to query the correct table fields."
        # )
        # query_sql_database_tool = QuerySQLDataBaseTool(
        #     db=self.db, description=query_sql_database_tool_description
        # )
        # query_sql_checker_tool_description = (
        #     "Use this tool to double check if your query is correct before executing "
        #     "it. Always use this tool before executing a query with "
        #     f"{query_sql_database_tool.name}!"
        # )
        # query_sql_checker_tool = QuerySQLCheckerTool(
        #     db=self.db, llm=self.llm, description=query_sql_checker_tool_description
        # )
        return [
            # query_sql_database_tool,
            info_sql_database_tool,
            list_sql_database_tool,
            # query_sql_checker_tool,
        ]


    def get_context(self) -> dict:
        """Return db context that you may want in agent prompt."""
        return self.db.get_context()