import streamlit as st  # Importing Streamlit library
import os

# Importing required modules from langchain package
from langchain.agents import ConversationalChatAgent, AgentExecutor, load_tools
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.utilities import SerpAPIWrapper
from template.template import CustomPromptTemplate

# Creating an instance of SerpAPIWrapper
search = SerpAPIWrapper()

# Setting the title for the Streamlit app
st.title('🔮 Eight Wizs')

# Retrieving API keys from Streamlit secrets
openai_api_key = st.secrets['OPENAI']
serpapi_api_key = st.secrets['SERPAPI_API_KEY']

# Initializing chat message history and memory
msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(
    chat_memory=msgs, return_messages=True, memory_key="chat_history", output_key="output"
)

# Handling chat history reset if button is clicked
if len(msgs.messages) == 0 or st.sidebar.button("Reset chat history"):
    msgs.clear()
    msgs.add_ai_message("How can I help you?")
    st.session_state.steps = {}

# Avatar mapping for messages
avatars = {"human": "user", "ai": "assistant"}

# Loop through the chat messages and display them with relevant formatting
for idx, msg in enumerate(msgs.messages):
    with st.chat_message(avatars[msg.type]):
        # Render intermediate steps if any were saved
        for step in st.session_state.steps.get(str(idx), []):
            if step[0].tool == "_Exception":
                continue
            with st.status(f"**{step[0].tool}**: {step[0].tool_input}", state="complete"):
                st.write(step[0].log)
                st.write(step[1])
        st.write(msg.content)

# Accept user input through a chat input box
if prompt := st.chat_input(placeholder="What would you like to know?"):
    st.chat_message("user").write(prompt)

    # Check if OpenAI API key is provided
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    # Initialize the ChatOpenAI model and tools for the chat agent
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, streaming=True)
    tools = load_tools(['serpapi'])
    chat_agent = ConversationalChatAgent.from_llm_and_tools(llm=llm, tools=tools, verbose=True)

    # Initialize an executor to process the user's input and generate a response
    executor = AgentExecutor.from_agent_and_tools(
        agent=chat_agent,
        tools=tools,
        memory=memory,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
        verbose=True,
    )
    

    # Display the assistant's response
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = executor(prompt, callbacks=[st_cb])
        print(f"**************************{response}****************************\n\n")
        st.write(response["output"])
        st.session_state.steps[str(len(msgs.messages) - 1)] = response["intermediate_steps"]
