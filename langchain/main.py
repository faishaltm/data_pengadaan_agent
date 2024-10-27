import os
import api_keys
import warnings
import logging
from db_connection import create_db_connection  
from agent_setup import create_agent   
from langchain_core.messages import HumanMessage         

# set tracking environment
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = api_keys.langchain_api_key

# Remove warnings and set logging level to ERROR
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

# Create the database connection
print("..creating the connection..")

db = create_db_connection(db_name='data_pengadaan.db')

# Create the agent (with history and tools)
agent_executor = create_agent(db)

config = {"recursion_limit": 50, "configurable": {"thread_id": "abc123"}}
print("..preparing the agent..")
# Start the interaction loop
while True:
    # Take user input
    user_input = input("Input query (type 'quit' to exit): ")
    
    # Check if the user wants to quit
    if user_input.lower() == "quit":
        print("Exiting the stream.")
        break
    
    # Prepare the input message for the agent
    stream = agent_executor.stream(
        {"messages": [HumanMessage(content=user_input)]}, config
    )

    # Stream the responses
    try:
        while True:
            # Get the next chunk from the stream
            chunk = next(stream)
            if 'agent' in chunk and 'messages' in chunk['agent'] and len(chunk['agent']['messages']) > 0:
                # Print the agent's message content
                print(chunk['agent']['messages'][0].content)
            else:
                # Skip if no valid 'agent' or 'messages' in the chunk
                continue

    except StopIteration:
        # Handle stream completion for the current input
        print("----")