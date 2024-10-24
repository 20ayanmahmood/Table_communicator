import streamlit as st
import oracledb
import pandas as pd
from langchain_core.prompts import PromptTemplate
from langchain.llms import Cohere
import google.generativeai as genai
import os
import matplotlib.pyplot as plt
import json
os.environ['API_KEY']=""
genai.configure(api_key=os.environ["API_KEY"])


llm = Cohere(cohere_api_key='',temperature=0.7)
model = genai.GenerativeModel("gemini-1.5-flash")


def running():
    sql_query = 'SELECT * FROM EBIZAI."Purchase Data"'
    connection = oracledb.connect(user="ebizai", password="ebizai", dsn="localhost:1521/FREEPDB1")
    # Fetch data from the database and load into a pandas DataFrame
    df = pd.read_sql(sql_query, con=connection)
    return df

def relevant_table(question,df):
    list1=[]
    for i in df.columns:
        list1.append(df[i].dtype)

    prompt = """
    Write Python pandas code based on the user's question: {question}.
    Print the most relevant table
    You have access to the following DataFrame column names:
    {df}
    datatype of columns list: {datatype}
    Sample of Data: {sample}
    **Dataframe is df**.
    Make sure to include comments in the code.
    Output **only** the Python code, no explanation, no text, just code.
    Just show the relevant columns.
    The final output should be  an dataframe not an numeric or anything 
    **final output should be an pandas dataframe**
    
    # Your code starts below:
    """

    prompt = PromptTemplate(template=prompt
    , input_variables=["question","df","datatype","sample"])
    prompt_formatted_str = prompt.format(
        question=question,df=df.columns,datatype=list1,sample=df.sample()
    )
    prediction = model.generate_content(prompt_formatted_str)
    prediction =prediction.text
    prediction=prediction.replace("```python","")
    prediction=prediction.replace("```","")
    return prediction
def multiple_graphs(df):
    dtype = [str(df[col].dtype) for col in df.columns]
    columns=df.columns
    prompt = PromptTemplate(template="""    
    Assume df is the dataframe
    Generate One Python Graphs atleast Graph using matplotlib or seaborn for the graph based on the column: {columns}
    Graph can be a line graph, bar , scatter,pie etc.
    Output **only** the Python code, no explanation, no text, just code.
    Make the rotation of x-axis.
    Use comments
    Datatype of columns: {dtype}
    Sample Data: {sample}
    
    Make the graph colorful and visually appealing by:
    - Adding appropriate titles, axis labels, and legends.
    - Using single colors for bar graphs.
    - Use plt.xticks(rotation=45)
    - Adding gridlines for better readability.
    - Including trend lines if applicable.
    - Rotating the x-axis labels for better visibility if needed.
    - Using different styles for markers and lines.
    - Adjusting figure size to make it clear and legible.
    - No Need to generate Sample data 
    - Use plt.tightlayout()
    Dont be biased towards bar graph also you can use multiple columns 
    for y-axis dont provide values range in 1e10 show the full value

    Ensure the graph is informative and well-presented.
    use plt.show() at the end
    
    Your code:
    """, input_variables=["columns","dtype","sample"])
    prompt_formatted_str= prompt.format(
    dtype=dtype,columns=columns,sample=df.sample())
    prediction1 = model.generate_content(prompt_formatted_str)
    prediction1=prediction1.text
    if "```python" in prediction1:
        prediction1=prediction1.replace("```python","")
        prediction1=prediction1.replace("```","")
    plt.clf()
    exec(prediction1)
    path = "output_plot.png"
    plt.savefig(path)
    return path

def insights(value):
    prompt = """
    Based on the provided dataset: {data}, analyze and extract **key insights** related to **supplier performance**, focusing on the **patterns and trends** in **delivery time** and **cost**. 
    Ensure the insights cover areas such as:
    - Any correlation between **delivery time** and **cost** for different suppliers.
    - Trends or outliers in **delivery**.
    - ANy average , location pattern etc.
    - Patterns in **cost variations** across suppliers.
    - Any notable changes in **supplier efficiency** over time.

    The final answer should be concise and presented as bullet points. 
    Do **not** include any summaries or introductions, just the key insights.

    Final Answer:
    """
    prompt = PromptTemplate(template=prompt
    , input_variables=["data"])
    prompt_formatted_str = prompt.format( data=value
    )
    prediction = model.generate_content(prompt_formatted_str)
    return prediction.text
def handle_question_click(question):
    st.write(f"You clicked on: **{question}**")  # Display or handle the clicked question

def suggestions(question,value):
    prompt = """
    Given the data: {data}, and based on the question: {question},
    Provide Answer for only user input.
    generate 3 additional related questions.
    Ensure the questions are insightful and cover different aspects related to the original query.
    Format your response as:
    {{ "Answer_to_user_question": "",
    "Auto_prompt": {{
        "Question_1": "Generated Question 1 based on the data and input question",
        "Question_2": "Generated Question 2 that explores a different angle",
        "Question_3": "Generated Question 3 for deeper insight or clarification",
        }}
    }}
    """
    prompt = PromptTemplate(template=prompt
    , input_variables=["question","data"])
    prompt_formatted_str = prompt.format( data=value, question=question
    )
    prediction = model.generate_content(prompt_formatted_str)
    return prediction.text


def connection(user,password,dsn):
    try:
        connection = oracledb.connect(user=user, password=password, dsn=dsn)
        cursor = connection.cursor()
        cursor.execute("SELECT 1 FROM dual")
        result = cursor.fetchone()
        # If the result is as expected, return success message
        if result[0] == 1:
            return "Connection established successfully!"
    except Exception as e:
        return "Encountered An Error"

st.sidebar.title("Connecting with Database")
st.sidebar.write("Enter Connection Details:")
user=st.sidebar.text_input("user:")
pwd=st.sidebar.text_input("password:")
dsn=st.sidebar.text_input("Dsn:",value="localhost:1521/FREEPDB1")
button=st.sidebar.button("Connect")

if button:
    connect = connection(user,pwd,dsn)
    st.sidebar.write(connect)

st.title("Table Coomunicator")
df=running()
button_dataframe=st.button("Dataframe")
Graph=st.button("Generate Graph")
if Graph:
    graphs=multiple_graphs(df)
    st.image(graphs)
if button_dataframe:
    st.dataframe(df)
question=st.text_input("Please Write Your Question:")
if question:
    try:
        reponse=relevant_table(question,df)
        st.write(reponse)
        local_vars={}
        exec(reponse,globals(),local_vars)
        last_key = list(local_vars.keys())[-1]
                # Store the DataFrame result in session state
        st.session_state.df_result = local_vars[last_key]

        # Display the DataFrame
        st.dataframe(st.session_state.df_result)
        b1 = st.button("Insights")
        if b1:
            repo = insights(str(local_vars))
            st.write(repo)
        b2=st.button("Suggestions")
        if b2:
            reo=suggestions(question,local_vars)
            reo=reo.replace("```json","")
            reo=reo.replace("```","")
            qa_data=json.loads(reo)
            st.write("Answer To User Question:")
            st.write(qa_data['Answer_to_user_question'])
            auto_prompt=qa_data['Auto_prompt']

            for question_key, question_value in auto_prompt.items():
                    # Create a clickable link for each question
                    if st.button(question_value):
                        handle_question_click(question_value)


    except Exception as e:
            st.error(e)
# Handle Insights button
