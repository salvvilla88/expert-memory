import streamlit as st

from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain.agents import load_tools, initialize_agent, AgentType

st.title('🧙 Question Wiz')

openai_api_key = st.secrets["OPENAI"]
tool_names = ['serpapi']

def generate_response(input_text):
  llm = OpenAI(temperature=0, model_name='gpt-3.5-turbo',openai_api_key=openai_api_key)
  tools = load_tools(tool_names,llm=llm)
  agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
  st.info(agent.run(input_text))
  

with st.form('my_form'):
  text = st.text_area('Prompt:', 'Write your question here')
  submitted = st.form_submit_button('Submit')
  if not openai_api_key.startswith('sk-'):
    st.warning('Please enter your OpenAI API key!', icon='⚠')
  if submitted and openai_api_key.startswith('sk-'):
    generate_response(text)