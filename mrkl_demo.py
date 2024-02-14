from pathlib import Path

import streamlit as st

from langchain_community.utilities.sql_database import SQLDatabase
from langchain.agents import AgentType
from langchain.agents import initialize_agent, Tool
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain.chains import LLMMathChain
from langchain_openai import OpenAI
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
# from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.utilities.sql_database import SQLDatabase
from langchain.chains import create_sql_query_chain as SQLDatabaseChain

from utils import add_text
from callbacks.capturing_callback_handler import playback_callbacks
from clear_results import with_clear_container

DB_PATH = (Path(__file__).parent / "Chinook.db").absolute()

SAVED_SESSIONS = {
    "Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?": "a56933289196024e74b80729e6a77c08.pickle",
    "What is the full name of the artist who recently released an album called "
    "'The Storm Before the Calm' and are they in the FooBar database? If so, what albums of theirs "
    "are in the FooBar database?": "f994fd83f7a84a833f6b83931e947355.pickle",
    "hello, my name is joseph, How are you?": "44428a2576e8b46bb2a899869e5ed393.pickle"
}

st.set_page_config(page_title="MRKL", page_icon="🦜", layout="wide", initial_sidebar_state="collapsed")
st.markdown("<h1 style='text-align: center;'>🦜🔗 MRKL</h1>", unsafe_allow_html=True)

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

col1, col2 = st.columns(2)
with col1:
    st.button(SAVED_SESSIONS[0], key= 0, disabled= validacion(SAVED_SESSIONS[0]))
    st.button(SAVED_SESSIONS[1], key= 1, disabled= validacion(SAVED_SESSIONS[1]))
with col2:
    st.button(SAVED_SESSIONS[2], key= 2, disabled= validacion(SAVED_SESSIONS[2]))
    st.button(SAVED_SESSIONS[3], key= 3, disabled= validacion(SAVED_SESSIONS[3]))

# Setup credentials in Streamlit
if user_openai_api_key:= st.sidebar.text_input("OpenAI API Key", type="password", help="Set this to run your own custom questions."):
    openai_api_key = user_openai_api_key
    enable_custom = True
else:
    openai_api_key = "not_supplied"
    enable_custom = False

# Tools setup
llm = OpenAI(temperature=0, openai_api_key=openai_api_key, streaming=True)
search = DuckDuckGoSearchAPIWrapper()
llm_math_chain = LLMMathChain.from_llm(llm)
db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")
db_chain = SQLDatabaseChain.from_llm(llm, db)
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events. You should ask targeted questions",
    ),
    Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="useful for when you need to answer questions about math",
    ),
    Tool(
        name="FooBar DB",
        func=db_chain.run,
        description="useful for when you need to answer questions about FooBar. Input should be in the form of a question containing full context",
    ),
]

# Initialize agent
mrkl = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

with st.form(key="form"):
    user_input = ""
    if enable_custom:
        user_input = st.text_input("Ask your own question")    
    elif not enable_custom:
        "Ask one of the sample questions, or enter your API Key in the sidebar to ask your own custom questions."
        user_input = st.selectbox("Sample questions", sorted(SAVED_SESSIONS.keys())) or ""

    submit_clicked = st.form_submit_button("Submit Question")

output_container = st.empty()
if with_clear_container(submit_clicked):
    output_container = output_container.container()
    output_container.chat_message("user").write(user_input)

    answer_container = output_container.chat_message("assistant", avatar="🦜")
    st_callback = StreamlitCallbackHandler(answer_container)

    # If we've saved this question, play it back instead of actually running LangChain (so that we don't exhaust our API calls unnecessarily)
    if user_input in SAVED_SESSIONS:
        session_name = add_text(user_input)
        session_path = Path(__file__).parent / "runs" / session_name
        print(f"Playing saved session: {session_path}")
        answer = playback_callbacks([st_callback], str(session_path), max_pause_time=1)
        from pprint import pprint
        print('\033[1;31m' + '#'*200)
        pprint(type(st_callback), sort_dicts=False)
        print('#'*100 + '\033[1;32m' + '#'*100)
        pprint(vars(st_callback), sort_dicts=False)
        completed_thoughts_list = st_callback._completed_thoughts
        for thought in completed_thoughts_list:
            print('#'*100 + '\033[1;33m' + '#'*100)
            pprint(vars(thought), sort_dicts=False)
        print('#'*200 + '\033[0m')

    else:
        answer = mrkl.run(user_input, callbacks=[st_callback])
        from pprint import pprint
        print('\033[1;35m' + '#'*200)
        pprint(type(st_callback), sort_dicts=False)
        print('#'*100 + '\033[1;36m' + '#'*100)
        pprint(vars(st_callback), sort_dicts=False)
        print('#'*200 + '\033[0m')

    answer_container.write(answer)