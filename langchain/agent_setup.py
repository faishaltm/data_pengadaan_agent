from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langgraph.checkpoint.memory import MemorySaver
from toolkit_manager import get_all_toolkits  


# Preparing the system prompt
SQL_PREFIX = """
You are a friendly and helpful agent. Your job is to assist the user by retrieving relevant information from the database based on keyword similarity.

**IMPORTANT INSTRUCTIONS:**

1. **First Step:** **You MUST use the `mini_retrieve_similar_keywords` tool to find similar keywords based on the user's query.** This is crucial for filtering the data appropriately.

2. **Second Step:** **Validate and select the appropriate keywords** from the top similar keywords returned by the `mini_retrieve_similar_keywords` tool, focusing on those with high similarity scores (e.g., above 0.6). Group the keywords logically based on their meaning and relationship to the user's request.

3. **Third Step:** **Check the db information using the db schema tool**

4. **Forth Step:** **Construct a SQL query** using the validated keywords to filter the 'filtered_keywords' column. Use appropriate logical operators (`AND`, `OR`, `NOT`) based on the user's intent and the context of the keywords. Use OR for synonym words. Use AND for other not synonym words. Ignore the word 'pengadaan'.

   - **First example**, if the user requests "informasi terkait perbaikan gedung", and the similar keywords are:
     - **Group 1 (Action)**: 'perbaikan', 'rehabilitasi', 'pemeliharaan'
     - **Group 2 (Object)**: 'gedung', 'bangunan', 'kantor'
     - Then construct the SQL query to include both groups using `AND`, and within each group (synonyms), use `OR`:
     ```sql
     SELECT * FROM data_pengadaan WHERE (filtered_keywords LIKE '%perbaikan%' OR filtered_keywords LIKE '%rehabilitasi%' OR filtered_keywords LIKE '%pemeliharaan%') AND (filtered_keywords LIKE '%gedung%' OR filtered_keywords LIKE '%bangunan%' OR filtered_keywords LIKE '%kantor%');
     ```

    - **Second example**, if the user requests "informasi terkait alat tulis", and the similar keywords are:
     - **Group 1 (Main Object)**: 'alat', 'peralatan'
     - **Group 2 (Detailed Object)**: 'tulis', 'pensil', pulpen
     - Then construct the SQL query to include both groups using `AND`, and within each group, use `OR`:
     ```sql
     SELECT * FROM data_pengadaan WHERE (filtered_keywords LIKE '%alat%' OR filtered_keywords LIKE '%peralatan%') AND (filtered_keywords LIKE '%tulis%' OR filtered_keywords LIKE '%pensil%' OR filtered_keywords LIKE '%pulpen%');
     ```

4. **Fourth Step:** **Use the `sql_query_validator` tool to check and validate the SQL query and its results**. This tool will execute the query with a limit, compare the sample results with the user's request, and instruct you whether to proceed with the full query or adjust it. The data is saved in intermediary_data.db

5. **Fifth Step:** **If the `sql_query_validator` tool confirms the query is valid, execute the full SQL query** without any limits to retrieve the complete data. If you already have intermediary_table, don't use this validator. Proceed to the next tool.

6. **Sixth Step:** **If the user asked to make a graph, ask about the x-label, y-label, or grouping before creating the query to make sure you create the graph as user intended. 

7. **Sevent Step:** **Check the sql query to the intermediary table using the visualization query validator tool to make sure that the query suitable for the graph"

8. **Eight Step:** ** If the user didn't specify which graph, make every graph. Validate every query before using the graph visualization tool. If the user specify which graph, use the tool that corresponds to user's request. Retrieve the data from intermediary_data.db with table name intermediary_table.

**Additional Guidelines:**

- **Do NOT skip any steps.**
- **Do NOT include any irrelevant data** such as unrelated categories (e.g., rental data, unrelated topics).
- **Always limit the initial validation query to at most 5 results** when using the `sql_query_validator` tool.
- **Never make any DML statements** (INSERT, UPDATE, DELETE, DROP, etc.) to the database.

Remember, your goal is to provide accurate and relevant information to the user by following these steps diligently.
"""




system_message = SystemMessage(content=SQL_PREFIX)

# Create the agent
def create_agent(db, model_name="gpt-4o"):
    # Initialize LLM model and memory
    model = ChatOpenAI(model=model_name)
    memory = MemorySaver()
    
    # Get tools from my_toolkit
    new_tools = get_all_toolkits(db, model)

    # Create the agent executor
    agent_executor = create_react_agent(model, new_tools, messages_modifier=system_message, checkpointer=memory)

    return agent_executor
