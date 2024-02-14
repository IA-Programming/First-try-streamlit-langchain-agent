import streamlit as st
import os
from pathlib import Path
# # Import Langchain modules
from langchain_openai import ChatOpenAI
from langchain.agents import load_tools, initialize_agent, AgentType, Tool
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchRun
from langchain_community.utilities.sql_database import SQLDatabase
from langchain.chains import create_sql_query_chain as SQLDatabaseChain
# Streamlit UI Callback
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.memory import ConversationBufferMemory

# Import modules related to streaming response
from utils import add_text
from callbacks.capturing_callback_handler import playback_callbacks, CapturingCallbackHandler
# key 1- Using Streamlit call backhandler
# key 2- Streaming the response

DB_PATH = (Path(__file__).parent / "Chinook.db").absolute()

SAVED_SESSIONS = [
    "Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?",
    "What is the full name of the artist who recently released an album called "
    "'The Storm Before the Calm' and are they in the FooBar database? If so, what albums of theirs "
    "are in the FooBar database?",
    "Give me a brief overview of the major events leading to the Renaissance.",
    "Solve the equation 3x + 7 = 22 for x."
]

st.set_page_config(page_title="Langchain Agents + MRKL", page_icon="🦜", layout="wide", initial_sidebar_state="collapsed")
st.markdown("<h1 style='text-align: center;'>🦜🔗 Langchain Agents + MRKL</h1>", unsafe_allow_html=True)

# Add custom CSS styles to center the button and ensure consistent size
st.markdown(
    """
    <style>
        div[data-testid="column"] button
        {
            width: 100%;
            margin: 0 auto;
            height: 120px;
            display: flex;
        }
    </style>
    """,unsafe_allow_html=True
)

# Initialize chat history if it doesn't already exist and Initialize the memory within the session state
msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(chat_memory=msgs, return_messages=True, memory_key="chat_history", output_key="output")

def validacion(text: str)->bool:
    for msg in msgs.messages:
        if msg.content == text:
            return True
    return False

col1, col2 = st.columns(2)
with col1:
    st.button(SAVED_SESSIONS[0], key= 0, disabled= validacion(SAVED_SESSIONS[0]))
    st.button(SAVED_SESSIONS[1], key= 1, disabled= validacion(SAVED_SESSIONS[1]))
with col2:
    st.button(SAVED_SESSIONS[2], key= 2, disabled= validacion(SAVED_SESSIONS[2]))
    st.button(SAVED_SESSIONS[3], key= 3, disabled= validacion(SAVED_SESSIONS[3]))

# Sidebar contents for logIN, choose plugin, and export chat
with st.sidebar:
    st.title('👋😁💬 Langchain Agent MRKL')
    if OPENAI_API_KEY := st.text_input('Enter OpenAI API token:', type="password"):
        os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

        # Initialize the OpenAI language model and search tool
        llm = ChatOpenAI(model="gpt-3.5-turbo-0613",temperature=0, streaming=True, openai_api_key=os.getenv("OPENAI_API_KEY"))
        # Set up the tool for responding to general questions
        db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")
        db_chain = SQLDatabaseChain.from_llm(llm, db)
        tools = load_tools(["llm-math"], llm=llm)
        tools.append(DuckDuckGoSearchRun(name="Search"))
        tools.append(Tool(
            name="FooBar DB",
            func=db_chain.run,
            description="useful for when you need to answer questions about FooBar. Input should be in the form of a question containing full context",
            ))

        # Initialize the Zero-shot agent with the tools and language model
        conversational_agent = initialize_agent(
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            tools=tools,
            llm=llm,
            verbose= True,
            memory = memory,
            # return_intermediate_steps=True,
            handle_parsing_errors=True,
            kwargs={'return_intermediate_steps': True}
        )

# Display previous chat messages from history
avatars = {"human": "user", "ai": "assistant"}
for msg in msgs.messages:
        with st.chat_message(avatars[msg.type], avatar='🦜' if avatars[msg.type]=='assistant' else None):
            st.markdown(msg.content)

def mensaje():
    for i in range(4):
        if st.session_state[i]:
            question = SAVED_SESSIONS[i]
            return question
    return None

question = st.chat_input("Ask me any question", disabled=not os.getenv('OPENAI_API_KEY')) or mensaje()

if question:
    with st.chat_message("user"):
        st.markdown(question)
    
    # Create an instance of CapturingCallbackHandler
    capturing_callback_handler = CapturingCallbackHandler()

    with st.chat_message("assistant", avatar="🦜"):
        # Setup the callback handler
        st_callback = StreamlitCallbackHandler(st.container())
        
        if question in SAVED_SESSIONS:
            session_path = (Path(__file__).parent /  f"runs/{add_text(question)}.pickle").absolute()
            answer = playback_callbacks(handlers=[st_callback], records_or_filename=session_path, max_pause_time=1)
            st.markdown(answer)
            msgs.add_user_message(question)
            msgs.add_ai_message(answer)

        else:

            assistance_response = conversational_agent(question, callbacks=[st_callback, capturing_callback_handler])
            # assistance_response = conversational_agent.run(question, callbacks=[st_callback])
            st.markdown(assistance_response)

    # nombre = add_text(question)
    # Salvar = (Path(__file__).parent / f"runs/{nombre}.pickle").absolute()
    # # After the conversation, you can access the captured records from capturing_callback_handler
    # capturing_callback_handler.dump_records_to_file(Salvar)