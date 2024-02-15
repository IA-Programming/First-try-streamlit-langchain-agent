import streamlit as st
import os
import time
from pathlib import Path
# # Import Langchain modules
from langchain_openai import ChatOpenAI
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchRun
# Streamlit UI Callback
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.memory import ConversationBufferMemory

# Import modules related to streaming response
from helpers.utils import add_text
from callbacks.capturing_callback_handler import playback_callbacks
# key 1- Using Streamlit call backhandler
# key 2- Streaming the response

SAVED_SESSIONS = [
    "If a bus travels at an average speed of 60 kilometers per hour, how long will it take the bus to travel 180 kilometers?",
    "Give me a brief overview of the major events leading to the Renaissance.",
    "what day is today in argentina?",
    "Solve the equation 3x + 7 = 22 for x."
]

st.set_page_config(page_title="Langchain Agents + MRKL", page_icon="🦜", initial_sidebar_state="collapsed")
st.markdown("<h1 style='text-align: center;'>🦜🔗 Langchain Agents + MRKL</h1>", unsafe_allow_html=True)

with st.expander(":rainbow[ℹ️ ABOUT-THIS-PROJECT]", expanded=False):
    "AI-driven tool or POC. Features: include Search and math abilities from langchain, a Virtual agent that has pre-recorded runs to see his capabilities. This project is useful experience a taste of langchain agents."

if 'steps' not in st.session_state:
    st.session_state['steps']={}
    st.session_state["Count"]=True

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

def validacion(Number: int)->bool:
    if len(msgs.messages)!=0:
        for msg in msgs.messages:
            if msg.content == SAVED_SESSIONS[Number]:
                return True
    return False

col1, col2 = st.columns(2)
with col1:
    st.button(SAVED_SESSIONS[0], key= 0, disabled= True if '0' in st.session_state and st.session_state.get('0') or validacion(0) else False)
    st.button(SAVED_SESSIONS[1], key= 1, disabled= True if '1' in st.session_state and st.session_state.get('1') or validacion(1) else False)
with col2:
    st.button(SAVED_SESSIONS[2], key= 2, disabled= True if '2' in st.session_state and st.session_state.get('2') or validacion(2) else False)
    st.button(SAVED_SESSIONS[3], key= 3, disabled= True if '3' in st.session_state and st.session_state.get('3') or validacion(3) else False)

# Sidebar contents for logIN, choose plugin, and export chat
with st.sidebar:
    st.title('👋😁💬 Langchain Agent MRKL')
    if OPENAI_API_KEY := st.text_input('Enter OpenAI API token:', type="password"):
        try:
            ChatOpenAI(model="gpt-3.5-turbo-0613",temperature=0, openai_api_key=OPENAI_API_KEY).invoke('hello!')
            os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
            st.success(body='Your Api key provided is good to go', icon='🎉')
        except:
            st.warning(body='Check out your api key has the model gpt-3.5-turbo-0613 available to use this product', icon='⚠️')
            st.stop()
        
        # Initialize the OpenAI language model and search tool
        llm = ChatOpenAI(model="gpt-3.5-turbo-0613",temperature=0, streaming=True, openai_api_key=os.getenv("OPENAI_API_KEY"))
        # Set up the tool for responding to general questions
        tools = load_tools(["llm-math"], llm=llm)
        tools.append(DuckDuckGoSearchRun(name="Search"))

        # Initialize the Zero-shot agent with the tools and language model
        conversational_agent = initialize_agent(
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            tools=tools,
            llm=llm,
            verbose= True,
            memory = memory,
            return_intermediate_steps=True,
            handle_parsing_errors=True,
        )

# Display previous chat messages from history
avatars = {"human": "user", "ai": "assistant"}
for idx, msg in enumerate(msgs.messages):
        with st.chat_message(avatars[msg.type], avatar='🦜' if avatars[msg.type]=='assistant' else None):
            # Render intermediate steps if any were saved
            for step in st.session_state.steps.get(str(idx), []):
                if step[0].tool == "_Exception":
                    continue
                with st.expander(f"✔️ **{step[0].tool}**: {step[0].tool_input}"):
                    st.markdown(f'```\n{step[0].log}\n```')
                    st.write(f"**{step[1]}**")
            st.markdown(msg.content)

def mensaje():
    for i in range(4):
        if st.session_state[i]:
            question = SAVED_SESSIONS[i]
            return question
    return None

question = st.chat_input("🧑‍💻 Write your message here 👇", key="input", disabled=True if ('input' in st.session_state and st.session_state.get('input') and st.session_state.get('Count')) or not os.getenv('OPENAI_API_KEY') else False) or mensaje()

if question:
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🦜"):
        # Setup the callback handler
        st_callback = StreamlitCallbackHandler(st.container())
        
        if question in SAVED_SESSIONS:
            session_path = (Path(__file__).parent /  f"runs/{add_text(question)}.pickle").absolute()
            if os.path.exists(session_path):
                assistance_response = playback_callbacks(handlers=[st_callback], records_or_filename=session_path, max_pause_time=1)
                st.markdown(assistance_response["output"])
                msgs.add_user_message(question)
                msgs.add_ai_message(assistance_response["output"])

        else:

            assistance_response = conversational_agent(question, callbacks=[st_callback])
            st.write(assistance_response["output"])
        st.session_state.steps[str(len(msgs.messages) - 1)] = assistance_response["intermediate_steps"]
        st.session_state.Count=True
        time.sleep(float(10))
        st.rerun()