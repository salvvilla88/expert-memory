import streamlit as st

from langchain import LLMChain
from langchain.agents import ZeroShotAgent, AgentExecutor, initialize_agent, Tool 
from langchain.llms import OpenAIChat
from langchain.memory import ConversationBufferMemory
from langchain.utilities import SerpAPIWrapper

st.title('🧙 Question Wiz')

openai_api_key = st.secrets['OPENAI']
search = SerpAPIWrapper()
tools = [
    Tool(
        name = "Search",
        func=search.run,
        description="useful for when you need to answer questions about current events or the current state of the world"
    ),
]
llm = OpenAIChat(temperature=0, model_name='gpt-3.5-turbo',openai_api_key=openai_api_key)
memory = ConversationBufferMemory(memory_key='chat_history')


def generate_response(input_text):
  search = SerpAPIWrapper()
  tools = [
      Tool(
          name = "Search",
          func=search.run,
          description="useful for when you need to answer questions about current events or the current state of the world"
      ),
  ]
  llm = OpenAIChat(temperature=0, model_name='gpt-3.5-turbo',openai_api_key=openai_api_key)
  memory = ConversationBufferMemory(memory_key='chat_history')

  prefix = """Have a conversation with a human, answering the following questions as best you can. You have access to the following tools:"""
  suffix = """Begin!"

  {chat_history}
  Question: {input}
  {agent_scratchpad}"""

  prompt = ZeroShotAgent.create_prompt(
      tools,
      prefix=prefix,
      suffix=suffix,
      input_variables=["input", "chat_history", "agent_scratchpad"]
  )

  llm_chain = LLMChain(llm=llm, prompt=prompt)
  tool_names = [tool.name for tool in tools]
  agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
  agent_chain  = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True, memory=memory
  )
  st.info(agent_chain .run(input_text))
  

with st.form('my_form'):
  text = st.text_area('Prompt:', 'Write your question here')
  submitted = st.form_submit_button('Submit')
  if not openai_api_key.startswith('sk-'):
    st.warning('Please enter your OpenAI API key!', icon='⚠')
  if submitted and openai_api_key.startswith('sk-'):
    generate_response(text)