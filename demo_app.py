import streamlit as st

# Import Langchain modules
from langchain.tools import DuckDuckGoSearchRun
from langchain.agents.tools import Tool
from langchain import OpenAI
from langchain.agents import initialize_agent
# Streamlit UI Callback
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chains import LLMMathChain
from langchain.memory import ConversationBufferMemory

# Import modules related to streaming response
import os
import time
# key 1- Using Streamlit call backhandler
# key 2- Streaming the response

OPENAI_API_KEY = ""
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

st.title("Streamlit Callback Handler and Streaming response tutorial")

# Initialize the memory within the session state
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history")

# Initialize chat history if it doesn't already exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize the OpenAI language model and search tool
llm = OpenAI(temperature=0)
# Tool 1 / Agents
search = DuckDuckGoSearchRun()
# Tool 2
llm_math_chain = LLMMathChain(llm=llm, verbose=True)

# Set up the tool for responding to general questions
tools = [
    Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description='useful for when you need to answer questions about math.'
    )
]

# Set up the tool for performing internet searches
search_tool = Tool(
    name="DuckDuckGo Search",
    func=search.run,
    description="Useful for when you need to do a search on the internet to find information that another tool can't find"
)

# We append both the tools

tools.append(search_tool)

# Initialize the Zero-shot agent with the tools and language model
conversational_agent = initialize_agent(
    agent="conversational-react-description",
    tools=tools,
    llm=llm,
    verbose= True,
    max_iterations=10,
    memory = st.session_state.memory
)

question = st.chat_input("Ask me any question")

# st.session_state.messages
# Display previous chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if question:
    with st.chat_message("user"):
        st.markdown(question)

    # Add the user's question to the chat history
    st.session_state.messages.append({"role": "user", "content": question})
    
    with st.chat_message("assistance"):
        # Setup the callback handler
        st_callback = StreamlitCallbackHandler(st.container())
        message_placeholder = st.empty()
        full_response = ""
        
        assistance_response = conversational_agent.run(question, callbacks=[st_callback])
        # st.markdown(assistance_response)

        # Simulate a streaming response with a slight delay
        for chunk in assistance_response.split():
            full_response += chunk + " "
            time.sleep(0.05)

            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(full_response + "▌")

            # Display the full response
            message_placeholder.info(full_response)

            # Add the assistant's response to the chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})
