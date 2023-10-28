import streamlit as st

# Import Langchain modules
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import DuckDuckGoSearchRun
# Streamlit UI Callback
from langchain.callbacks import StreamlitCallbackHandler
from langchain.memory import ConversationBufferMemory

# Import modules related to streaming response
import os
# key 1- Using Streamlit call backhandler
# key 2- Streaming the response

st.set_page_config(page_title="Langchain Agents + MRKL", page_icon=":parrot:", layout="wide", initial_sidebar_state="collapsed")
st.title(":parrot: Langchain Agents + MRKL")

# Initialize the memory within the session state
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history")

# Initialize chat history if it doesn't already exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar contents for logIN, choose plugin, and export chat
with st.sidebar:
    st.title('👋😁💬 Langchain Agent MRKL')
    if OPENAI_API_KEY := st.text_input('Enter OpenAI API token:', type="password"):
        os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

        # Initialize the OpenAI language model and search tool
        llm = ChatOpenAI(model="gpt-3.5-turbo",temperature=0, streaming=True, openai_api_key=os.getenv("OPENAI_API_KEY"))
        # Set up the tool for responding to general questions
        tools = [DuckDuckGoSearchRun(name="Search")]

        # Initialize the Zero-shot agent with the tools and language model
        conversational_agent = initialize_agent(
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            tools=tools,
            llm=llm,
            verbose= True,
            memory = st.session_state.memory,
            handle_parsing_errors=True,
        )

# Display previous chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if question:= st.chat_input("Ask me any question", disabled=not os.getenv('OPENAI_API_KEY')):
    with st.chat_message("user"):
        st.markdown(question)

    # Add the user's question to the chat history
    st.session_state.messages.append({"role": "user", "content": question})
    
    with st.chat_message("assistance"):
        # Setup the callback handler
        st_callback = StreamlitCallbackHandler(st.container())
        
        assistance_response = conversational_agent.run(question, callbacks=[st_callback])
        st.markdown(assistance_response)

        # Add the assistant's response to the chat history
        st.session_state.messages.append({"role": "assistant", "content": assistance_response})
