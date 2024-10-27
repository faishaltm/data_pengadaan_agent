import logging
import pandas as pd
from typing import Dict, Optional, Type, List
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from pydantic import BaseModel, Field
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain_community.tools import BaseSQLDatabaseTool
from langchain_community.utilities import SQLDatabase
from langchain.schema.language_model import BaseLanguageModel
from pydantic.v1.config import ConfigDict
from langchain.agents.agent_toolkits.base import BaseToolkit
from langchain_core.callbacks.base import AsyncCallbackHandler
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import create_engine, inspect, text
from openai import OpenAI 
import api_keys
import os
import ast
from sqlalchemy import create_engine
import json

os.environ['OPENAI_API_KEY'] = api_keys.openai_key
client = OpenAI()

def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-large" 
    )
    return response.data[0].embedding

columns = ['id', 'item_name', 'city', 'department', 'description', 'detailed_description', 'amount', 'timestamp', 'keywords']

# Set up logging
logger = logging.getLogger(__name__)

# New input classes for the added tools
class _IntermediaryDataFrameToolInput(BaseModel):
    query: str = Field(..., description="A SQL query to retrieve data.")

class IntermediaryDataFrameTool(BaseSQLDatabaseTool, BaseTool):
    name: str = "itermediary_dataframe_retrieval"
    description: str = """
        This tool is used to create an intermediary dataframe that later can be used to visualize.
        don't limit the query. It will be stored to dataframe. You can read only the head.
        input: sql query
        output: db schema for later use and first five data.
        Use the data from this tool to visualize dataframe.
        """
    args_schema: Type[BaseModel] = _IntermediaryDataFrameToolInput

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        result = self.db.run_no_throw(query)
        data = ast.literal_eval(result)
        df = pd.DataFrame(data, columns=columns)

        current_directory = os.getcwd()
        db_directory = os.path.join(current_directory, 'data')

        if not os.path.exists(db_directory):
            os.makedirs(db_directory)
        
        db_path = os.path.join(db_directory, 'intermediary_data.db')

        engine = create_engine(f'sqlite:///{db_path}')
        
        df.to_sql('intermediary_table', engine, if_exists='replace', index=False)

        db_info = {}

        inspector = inspect(engine)
        table_names = inspector.get_table_names()
        
        schema_info = {}
        for table_name in table_names:
            schema_info[table_name] = []
            columns_info = inspector.get_columns(table_name)
            
            for column in columns_info:
                schema_info[table_name].append({
                    'name': column['name'],
                    'type': str(column['type'])
                })
        
        # Add the schema to the db_info
        db_info['schema'] = schema_info
        
        # Get the first 5 rows from the intermediary_table
        with engine.connect() as connection:
            result = connection.execute(text('SELECT * FROM intermediary_table LIMIT 5'))
            
            # Fetch the top 5 rows
            rows = result.fetchmany(5)
            
            # Get column names from the result's metadata
            columns_result = result.keys()
            
            # Convert the result into a list of dictionaries
            results = [dict(zip(columns_result, row)) for row in rows]
        
        # Add the first rows to the db_info
        db_info['first_rows'] = results
        
        # Return the JSON formatted result
        return json.dumps(db_info, indent=4)


# Third Tool: Visualization

class _VisualizationValidatorToolInput(BaseModel):
    user_request: str = Field(..., description="The user visualization request in graphing the data")
    sql_query: str = Field(..., description="The SQL query to validate.")
    graph: str = Field(..., description="graph that inteded to build")

class VisualizationValidatorTool(BaseSQLDatabaseTool, BaseTool):
    """
    Validates the SQL query and its results against the user's request.
    Executes the SQL query, checks if the results match the user's intent,
    and provides guidance on whether to proceed or adjust the query.
    adjusting the query using this tool before moving forward to visualization tool.
    """
    name: str = "visualizationValidatorTool"
    description: str = """
    Use this tool to validate the SQL query before visualizing the data.
    the data is in the intermediary_table.
    Input: The user's visualization request, SQL query to validate, and the graph you mant to make.
    Output: Instruction to give the query to visualization tool if the results table is meaningful,
    or an adjusted SQL query if the results do not match.
    """
    args_schema: Type[BaseModel] = _VisualizationValidatorToolInput
    llm: BaseLanguageModel = Field(exclude=True)

    def _run(
        self,
        user_request: str,
        sql_query: str,
        graph: str, 
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        # Append LIMIT to the SQL query to limit results for validation
        limited_sql_query = sql_query.strip().rstrip(';')

        engine = create_engine("sqlite:///data/intermediary_data.db")
        conn = engine.connect()
        result = pd.read_sql_query(limited_sql_query, conn)
        
        sample_data = result.head(5).to_string()

        # Prepare the prompt for the LLM
        prompt = f"""
        The user request is: "{user_request}"
        The SQL query is: "{sql_query}"
        The sample data is:
        {sample_data}
        Does this data is good to visualize for this type of {graph}? Answer 'Yes' or 'No' and provide a brief explanation.
        """

        # Get the LLM's response
        try:
            response = self.llm.predict(prompt)
        except Exception as e:
            return f"Error during LLM prediction: {e}"

        if "Yes" in response:
            return "The SQL query are suitable for the graph. Proceed giving the query to the visualization tool"
        else:
            return "The SQL query are not suitable for the graph. Please adjust the query and reuse this visualization validator tool"

    async def _arun(
        self,
        user_request: str,
        sql_query: str,
        run_manager: Optional[AsyncCallbackHandler] = None,
    ) -> str:
        """Asynchronous execution is not implemented."""
        raise NotImplementedError("Asynchronous execution is not implemented.")


class _BarChartToolInput(BaseModel):
    sql_query: str = Field(..., description="SQL query to retrieve data from for the bar chart.")
    x_column: str = Field(..., description="Column name for x-axis.")
    y_column: str = Field(..., description="Column name for y-axis.")
    chart_title: str = Field(..., description="Title suitable for the chart")
    image_filename: str = Field(..., description="Filename to save the image (e.g., 'bar_chart.png').")
    image_directory: Optional[str] = Field(default="./images", description="Directory where the image will be saved.")

# BarChartTool Definition
class BarChartTool(BaseTool):
    name: str = "bar_chart_tool"
    description: str = """
    Creates a bar chart from data queried from the intermediary database.
    this tool is ONLY used to retrieve data from 'intermediary_table', not data_pengadaan.
    the schema for intermediary_table is the result from the itermediary_dataframe_retrieval tool.    
    Input is SQL query, x-axis column, y-axis column, image filename, and optional image directory.
    Output is the image path.
    """
    args_schema: Type[BaseModel] = _BarChartToolInput

    def _run(
        self,
        sql_query: str,
        x_column: str,
        y_column: str,
        chart_title: str, 
        image_filename: str,
        image_directory: Optional[str] = "./images",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        import os
        from sqlalchemy import create_engine

        # Connect to the intermediary database
        db_path = os.path.join(os.getcwd(), 'data', 'intermediary_data.db')
        engine = create_engine(f'sqlite:///{db_path}')

        # Execute the SQL query
        try:
            df = pd.read_sql_query(sql_query, engine)
        except Exception as e:
            return f"Error executing SQL query: {e}"

        # Ensure columns exist
        if x_column not in df.columns:
            return f"Error: Column '{x_column}' not found in DataFrame."
        if y_column not in df.columns:
            return f"Error: Column '{y_column}' not found in DataFrame."

        # Create the bar plot
        plt.figure()
        try:
            sns.barplot(data=df, x=x_column, y=y_column)
            plt.xticks(rotation=45)
            plt.title(chart_title)
        except Exception as e:
            return f"Error creating bar chart: {e}"

        # Ensure the image directory exists
        try:
            os.makedirs(image_directory, exist_ok=True)
        except OSError as e:
            return f"Error creating directory '{image_directory}': {e}"

        # Save the plot to a file
        image_path = os.path.join(image_directory, image_filename)
        try:
            plt.savefig(image_path, format='png', bbox_inches='tight')
            plt.close()
        except Exception as e:
            return f"Error saving image: {e}"

        # Return the image path
        return f"Image saved at {image_path}"

# PieChartTool Input Schema
class _PieChartToolInput(BaseModel):
    sql_query: str = Field(..., description="SQL query to retrieve data for the pie chart.")
    labels_column: str = Field(..., description="Column name for labels.")
    values_column: str = Field(..., description="Column name for values.")
    chart_title: str = Field(..., description="Title suitable for the chart")
    image_filename: str = Field(..., description="Filename to save the image (e.g., 'pie_chart.png').")
    image_directory: Optional[str] = Field(default="./images", description="Directory where the image will be saved.")

# PieChartTool Definition
class PieChartTool(BaseTool):
    name: str = "pie_chart_tool"
    description: str = """
    Creates a pie chart from data queried from the intermediary database.
    this tool is ONLY used to retrieve data from 'intermediary_table', not data_pengadaan.    
    Input is SQL query, labels column, values column, image filename, and optional image directory.
    Output is the image path.
    """
    args_schema: Type[BaseModel] = _PieChartToolInput

    def _run(
        self,
        sql_query: str,
        labels_column: str,
        values_column: str,
        chart_title: str, 
        image_filename: str,
        image_directory: Optional[str] = "./images",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        import pandas as pd
        import matplotlib.pyplot as plt
        import os
        from sqlalchemy import create_engine

        # Connect to the intermediary database
        db_path = os.path.join(os.getcwd(), 'data', 'intermediary_data.db')
        engine = create_engine(f'sqlite:///{db_path}')

        # Execute the SQL query
        try:
            df = pd.read_sql_query(sql_query, engine)
        except Exception as e:
            return f"Error executing SQL query: {e}"

        # Ensure columns exist
        if labels_column not in df.columns:
            return f"Error: Column '{labels_column}' not found in DataFrame."
        if values_column not in df.columns:
            return f"Error: Column '{values_column}' not found in DataFrame."

        # Create the pie chart
        plt.figure()
        try:
            plt.pie(df[values_column], labels=df[labels_column], autopct='%1.1f%%')
            plt.title(chart_title)
        except Exception as e:
            return f"Error creating pie chart: {e}"

        # Ensure the image directory exists
        try:
            os.makedirs(image_directory, exist_ok=True)
        except OSError as e:
            return f"Error creating directory '{image_directory}': {e}"

        # Save the plot to a file
        image_path = os.path.join(image_directory, image_filename)
        try:
            plt.savefig(image_path, format='png', bbox_inches='tight')
            plt.close()
        except Exception as e:
            return f"Error saving image: {e}"

        # Return the image path
        return f"Image saved at {image_path}"

# HistogramTool Input Schema
class _HistogramToolInput(BaseModel):
    sql_query: str = Field(..., description="SQL query to retrieve data for the histogram.")
    column: str = Field(..., description="Column name to plot the histogram.")
    bins: Optional[int] = Field(default=10, description="Number of bins for the histogram.")
    chart_title: str = Field(..., description="Title suitable for the chart")
    image_filename: str = Field(..., description="Filename to save the image (e.g., 'histogram.png').")
    image_directory: Optional[str] = Field(default="./images", description="Directory where the image will be saved.")

# HistogramTool Definition
class HistogramTool(BaseTool):
    name: str = "histogram_tool"
    description: str = """
    Creates a histogram from data queried from the intermediary database.
    this tool is ONLY used to retrieve data from 'intermediary_table', not data_pengadaan.
    Input is SQL query, column to plot, number of bins (optional), image filename, and optional image directory.
    Output is the image path.
    """
    args_schema: Type[BaseModel] = _HistogramToolInput

    def _run(
        self,
        sql_query: str,
        column: str,
        chart_title: str, 
        image_filename: str,
        bins: int = 10,
        image_directory: Optional[str] = "./images",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        import os
        from sqlalchemy import create_engine

        # Connect to the intermediary database
        db_path = os.path.join(os.getcwd(), 'data', 'intermediary_data.db')
        engine = create_engine(f'sqlite:///{db_path}')

        # Execute the SQL query
        try:
            df = pd.read_sql_query(sql_query, engine)
        except Exception as e:
            return f"Error executing SQL query: {e}"

        # Ensure column exists
        if column not in df.columns:
            return f"Error: Column '{column}' not found in DataFrame."

        # Create the histogram
        plt.figure()
        try:
            sns.histplot(data=df, x=column, bins=bins)
            plt.title(chart_title)
        except Exception as e:
            return f"Error creating histogram: {e}"

        # Ensure the image directory exists
        try:
            os.makedirs(image_directory, exist_ok=True)
        except OSError as e:
            return f"Error creating directory '{image_directory}': {e}"

        # Save the plot to a file
        image_path = os.path.join(image_directory, image_filename)
        try:
            plt.savefig(image_path, format='png', bbox_inches='tight')
            plt.close()
        except Exception as e:
            return f"Error saving image: {e}"

        # Return the image path
        return f"Image saved at {image_path}"

# MiniRetrieveSimilarKeywordsTool Input
class _MiniRetrieveSimilarKeywordsToolInput(BaseModel):
    query: str = Field(..., description="The search query keyword to find similar ones.")
    top_k: int = Field(10, description="Number of top similar results to return. Default is 10.")

# MiniRetrieveSimilarKeywordsTool Definition
class MiniRetrieveSimilarKeywordsTool(BaseTool):
    """
    Use this tool to find available keywords that you can use to search in the keyword list.
    The tool performs a search to retrieve the most similar keywords based on cosine similarity.
    Input: Query and top_k.
    Output: A DataFrame with the most similar keywords and their similarity scores.
    """
    name: str = "mini_retrieve_similar_keywords"
    args_schema: Type[BaseModel] = _MiniRetrieveSimilarKeywordsToolInput

    def __init__(self, description: str, **kwargs):
        super().__init__(description=description, **kwargs)

    def _run(
        self,
        query: str,
        top_k: int = 10,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        import pandas as pd
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity

        # Load DataFrame from CSV
        try:
            df = pd.read_csv('v2_key.csv')
        except Exception as e:
            return f"Error loading DataFrame: {e}"

        # Ensure 'embedding' column is converted from string representation to numeric vectors
        try:
            df['embedding'] = df['embedding'].apply(lambda x: np.fromstring(x.strip('[]'), sep=','))
        except Exception as e:
            return f"Error processing embeddings: {e}"

        # Get embedding for the query
        try:
            query_embedding = get_embedding(query)
        except Exception as e:
            return f"Error generating embedding for query: {e}"

        # Calculate cosine similarity
        try:
            similarities = df['embedding'].apply(lambda x: cosine_similarity([query_embedding], [x])[0][0])
        except Exception as e:
            return f"Error calculating similarities: {e}"

        # Add similarity scores to the DataFrame
        df['similarity'] = similarities

        # Sort DataFrame based on similarity scores
        results = df.sort_values(by='similarity', ascending=False)

        # Retrieve the top_k results
        results = results.head(top_k)

        # Return the DataFrame with keyword and similarity, removing embedding for clarity
        return results[['keyword', 'similarity']].to_json(orient='records')

    async def _arun(
        self,
        query: str,
        top_k: int = 10,
        run_manager: Optional[AsyncCallbackHandler] = None,
    ) -> str:
        """Asynchronous execution is not implemented."""
        raise NotImplementedError("Asynchronous execution is not implemented.")

# Input schema for the SQLQueryValidatorTool
class _SQLQueryValidatorToolInput(BaseModel):
    user_request: str = Field(..., description="The original user request.")
    sql_query: str = Field(..., description="The SQL query to validate.")

class SQLQueryValidatorTool(BaseSQLDatabaseTool, BaseTool):
    """
    Validates the SQL query and its results against the user's request.
    Executes the SQL query, checks if the results match the user's intent,
    and provides guidance on whether to proceed or adjust the query.
    """
    name: str = "sql_query_validator"
    description: str = """
    Use this tool to validate the SQL query and its results against the user's request.
    Input: The user's request and the SQL query to validate.
    Output: Instruction to proceed with the full query if the results match,
    or an adjusted SQL query if the results do not match.
    """
    args_schema: Type[BaseModel] = _SQLQueryValidatorToolInput
    llm: BaseLanguageModel = Field(exclude=True)

    def _run(
        self,
        user_request: str,
        sql_query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        # Append LIMIT to the SQL query to limit results for validation
        limited_sql_query = sql_query.strip().rstrip(';')

        engine = create_engine("sqlite:///data_pengadaan.db")
        conn = engine.connect()
        result = pd.read_sql_query(limited_sql_query, conn)
        
        sample_data = result.head(5).to_string()

        # Prepare the prompt for the LLM
        prompt = f"""
        The user request is: "{user_request}"
        The SQL query is: "{sql_query}"
        The sample data is:
        {sample_data}
        Does the data satisfy the user's request? Answer 'Yes' or 'No' and provide a brief explanation.
        """

        # Get the LLM's response
        try:
            response = self.llm.predict(prompt)
        except Exception as e:
            return f"Error during LLM prediction: {e}"

        if "Yes" in response:
            return "The SQL query returns results matching the user's request. Proceed with the full query without limit."
        else:
            return "The SQL query does not return results matching the user's request. Please adjust the query."

    async def _arun(
        self,
        user_request: str,
        sql_query: str,
        run_manager: Optional[AsyncCallbackHandler] = None,
    ) -> str:
        """Asynchronous execution is not implemented."""
        raise NotImplementedError("Asynchronous execution is not implemented.")
    




# Updated SQLDatabaseToolkit to include the new tools
class My_SQLDatabaseToolkit(BaseToolkit):
    db: SQLDatabase = Field(exclude=True)
    llm: BaseLanguageModel = Field(exclude=True)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        tools = []

        # RetrieveDataFrameTool
        intermediary_dataframe_tool_description = ("""
            This tool is used to create an intermediary dataframe that later can be used to visualize.
            this tool can only be used to retrieve the data from 'data_pengadaan' table to make a new table named 'intermediary_table' 
            don't limit the query. It will be stored to dataframe. You can read only the head.
            input: sql query
            output: db schema for later use and first five data.
            Use the data from this tool to visualize dataframe."""
        )
        intermediary_dataframe_tool = IntermediaryDataFrameTool(
            db=self.db, description=intermediary_dataframe_tool_description
        )
        tools.append(intermediary_dataframe_tool)

        bar_chart_tool_description = (
            "Creates a bar chart from data queried from the intermediary database. "
            "if the user don't give you what to put at X and Y label, ASK USER. DON'T ASSUME"
            "this tool is ONLY used to retrieve data from 'intermediary_table', not 'data_pengadaan'"
            "Input is SQL query, x-axis column, y-axis column, chart title, image filename, and optional image directory. "
            "Output is the image path."
        )
        bar_chart_tool = BarChartTool(
            description=bar_chart_tool_description
        )
        tools.append(bar_chart_tool)

        # PieChartTool
        pie_chart_tool_description = (
            "Creates a pie chart from data queried from the intermediary database. "
            "if the user don't give you what to put at X and Y label, ASK USER. DON'T ASSUME"
            "this tool is ONLY used to retrieve data from 'intermediary_table', not 'data_pengadaan'"
            "Input is SQL query, labels column, values column, chart title, image filename, and optional image directory. "
            "Output is the image path."
        )
        pie_chart_tool = PieChartTool(
            description=pie_chart_tool_description
        )
        tools.append(pie_chart_tool)

        # HistogramTool
        histogram_tool_description = (
            "Creates a histogram from data queried from the intermediary database. "
            "if the user don't give you what to put at X and Y label, ASK USER. DON'T ASSUME"
            "this tool is ONLY used to retrieve data from 'intermediary_table', not 'data_pengadaan'"
            "Input is SQL query, column to plot, number of bins (optional), chart title, image filename, and optional image directory. "
            "Output is the image path."
        )
        histogram_tool = HistogramTool(
            description=histogram_tool_description
        )
        tools.append(histogram_tool)


        # MiniRetrieveSimilarKeywordsTool
        mini_retrieve_similar_keywords_tool_description = (
            "Retrieve the most similar keywords based on cosine similarity between the query and "
            "the embeddings in the DataFrame. Input is a search query and top_k. "
            "Output is a DataFrame containing similar keywords and their similarity scores."
        )
        mini_retrieve_similar_keywords_tool = MiniRetrieveSimilarKeywordsTool(
            description=mini_retrieve_similar_keywords_tool_description
        )
        tools.append(mini_retrieve_similar_keywords_tool)

        sql_query_validator_tool_description = (
            "Validates the SQL query and its results against the user's request. "
            "Input: The user's request and the SQL query to validate. "
            "Output: Instruction to proceed with the full query if the results match, "
            "or adjust the query and reuse this tool until the result match"
        )
        sql_query_validator_tool = SQLQueryValidatorTool(
            db=self.db, llm=self.llm, description=sql_query_validator_tool_description
        )
        tools.append(sql_query_validator_tool)

        visualization_query_validator_tool_description = (
            "Use this tool when you want to make a graph." 
            "make sure the SQL query is align with the intended graph"
            "the data is located in the intermediary_table"
            "Input: The user's request, the SQL query to validate and the graph you want to make"
            "Output: Instruction to proceed with the query if the query is suitable for the graph, "
            "or an adjusted SQL query if the results do not match."
        )
        visualization_query_validator_tool = VisualizationValidatorTool(
            db=self.db, llm=self.llm, description=visualization_query_validator_tool_description
        )
        tools.append(visualization_query_validator_tool)

        return tools