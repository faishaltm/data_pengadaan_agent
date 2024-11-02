import openai
import json
import itertools
import time
from openai import OpenAI
from list_of_tools import (mini_retrieve_similar_keywords, 
                           schema_check, 
                           intermediary_dataframe_retrieval, 
                           bar_chart_tool,
                           line_chart_tool,
                           histogram_tool,
                           pie_chart_tool)

tool_functions = {
    'mini_retrieve_similar_keywords': mini_retrieve_similar_keywords,
    'schema_check': schema_check,
    'intermediary_dataframe_retrieval': intermediary_dataframe_retrieval,
    'bar_chart_tool': bar_chart_tool,
    'line_chart_tool': line_chart_tool,
    'histogram_tool': histogram_tool,
    'pie_chart_tool': pie_chart_tool,
            }

client = OpenAI()

prompt = """Guide to Database Retrieval Using Keyword Similarity

You are heplful agent. You are going to use tools until it success or user will say stop.

Assist the user in retrieving database information using keyword similarity and SQL queries, following these structured steps.

Process Steps:

1. SPECIAL REQUEST: If User said "SIRUPA TAMPILKAN GRAFIK {USER QUERY}", Prepare to GIVE ALL 4 CHARTS FOR THE QUERY BAR, LINE, PIE AND HISTOGRAM.
2. Use mini_retrieve_similar_keywords tool: This tool identifies similar keywords for the user's query. Using this is essential; otherwise, the query may fail.
3. Keyword Selection: Choose keywords with similarity scores above 0.6, grouping them by meaning or relationship to the user’s request.
4. Schema Check: Run schema_check to understand the database structure.
5. SQL Query Construction:
Use selected keywords to filter 'filtered_keywords' with logical operators (OR for synonyms, AND for non-synonyms).
Apply filters in 'satuan_kerja' and 'tanggal_umumkan_paket' columns based on the user's input.
6. Query Execution: Use intermediary_dataframe_retrieval to execute the query and retrieve data.
7. Use AND NOT if USER wants to EXCLUDE information.
8. Graph Requests: If user requests visualization, use corresponding tools: bar chart, line chart, pie chart, then histogram.

Example Queries:

First Request: "informasi terkait perbaikan gedung"
similar keywords: perbaikan, rehabilitasi, pemeliharaan, gedung, bangunan, kantor
SELECT * FROM data_pengadaan WHERE (filtered_keywords LIKE '%perbaikan%' OR filtered_keywords LIKE '%rehabilitasi%' OR filtered_keywords LIKE '%pemeliharaan%') AND (filtered_keywords LIKE '%gedung%' OR filtered_keywords LIKE '%bangunan%' OR filtered_keywords LIKE '%kantor%');

Follow Up Question: "keluarkan informasi terkait alat atau peralatan"
similar keywords: perbaikan, rehabilitasi, pemeliharaan, gedung, bangunan, kantor
SELECT * FROM data_pengadaan WHERE (filtered_keywords LIKE '%perbaikan%' OR filtered_keywords LIKE '%rehabilitasi%' OR filtered_keywords LIKE '%pemeliharaan%') AND (filtered_keywords LIKE '%gedung%' OR filtered_keywords LIKE '%bangunan%' OR filtered_keywords LIKE '%kantor%') AND NOT (filtered_keywords LIKE '%alat%' OR filtered_keywords LIKE '%peralatan%');

Notes:
1. Complete each step without omissions.
2. Use one tool at a time; follow the steps precisely.
3. Error in using tools is NORMAL. TRY AGAIN UNTIL IT COMPLETE OR USER STOP IT.
3. Do not include unnecessary details in responses.
4. Avoid DML operations (INSERT, UPDATE, DELETE, DROP).
"""

def deploy_assistant(all_tools):
    assistant = client.beta.assistants.create(
    name="Data Agent",
    instructions=prompt,
    tools=all_tools,
    model="gpt-4o",
    )

    return assistant

def run_assistant(assistant_id, thread, question):

    run = openai.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant_id,
        instructions=question
    )

    return run

def execute_tool_call(tool_call):
    tool_name = tool_call.function.name
    print(f'Using tool: {tool_name}')
        
    try:
        args = json.loads(tool_call.function.arguments)
        function = tool_functions.get(tool_name)
        output = function(**args) if args else function()        
    except Exception as e:
        output = json.dumps({'error': str(e)})

    return {
            'tool_call_id': tool_call.id,
            'output': output
        }

def get_answer(run, thread):
    spinner = itertools.cycle(['-', '\\', '|', '/'])
    
    while run.status != 'completed':
        run = openai.beta.threads.runs.retrieve(
            thread_id = thread.id,
            run_id = run.id
        )
    
        print(f"\rRun status: {run.status} {next(spinner)}", end="", flush=True)
        time.sleep(0.1)
        if run.status == 'requires_action':
            tool_calls = run.required_action.submit_tool_outputs.tool_calls
            tool_outputs = [execute_tool_call(call) for call in run.required_action.submit_tool_outputs.tool_calls]
            
            run = client.beta.threads.runs.submit_tool_outputs(
                thread_id = thread.id,
                run_id = run.id,
                tool_outputs=tool_outputs
            )

    messages = openai.beta.threads.messages.list(
        thread_id=thread.id
    )

    annotations = messages.data[0].content[0].text.annotations
    message_content = messages.data[0].content[0].text.value

    return annotations, message_content

def add_message(thread, message_content, role):
    return client.beta.threads.messages.create(
        thread_id=thread.id,
        role=role,
        content=message_content,
    )