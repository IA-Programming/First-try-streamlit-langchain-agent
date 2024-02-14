import streamlit as st
import os
from pathlib import Path
# # Import Langchain modules
from langchain import hub
from langchain.tools.render import render_text_description_and_args
from langchain_community.llms.cohere import Cohere
from langchain.agents import AgentExecutor
from langchain.agents.output_parsers import JSONAgentOutputParser
from langchain.agents.format_scratchpad import format_log_to_messages
from langchain.tools.ddg_search.tool import DuckDuckGoSearchResults
# Streamlit UI Callback
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.memory import ConversationBufferMemory

# Import modules related to streaming response
from utils import add_text
from callbacks.capturing_callback_handler import playback_callbacks

# key 1- Using Streamlit call backhandler
# key 2- Streaming the response

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

# st.write(st.session_state["langchain_messages"])

# Add custom CSS styles to center the button and ensure consistent size
st.markdown(
    """
    <style>
        div[data-testid="column"]:nth-of-type(1)
        {
            text-align: end;
        } 

        div[data-testid="column"]:nth-of-type(2)
        {
            text-align: start;
        }

        div[data-testid="column"] button
        {
            width: 480px;
            height: 160px;
        }
    </style>
    """,unsafe_allow_html=True
)

# Initialize chat history if it doesn't already exist
msgs = StreamlitChatMessageHistory()
# Initialize the memory within the session state
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
    if COHERE_API_KEY := st.text_input('Enter Cohere API token:', type="password"):
        os.environ['COHERE_API_KEY'] = COHERE_API_KEY

        # Initialize the OpenAI language model and search tool
        llm = Cohere(model="command-nightly", temperature=0, streaming=True, cohere_api_key=os.getenv("COHERE_API_KEY"))

        prompt = hub.pull("hwchase17/react-multi-input-json")


        llm_with_stop = llm.bind(stop=["Observation"])
        # Set up the tool for responding to general questions
        tools=[DuckDuckGoSearchResults(name="duck_duck_go")]

        prompt = prompt.partial(
            tools=render_text_description_and_args(tools),
            tool_names=", ".join([t.name for t in tools]),
        )

        # Initialize the Zero-shot agent with the tools and language model
        agent = (
            {
                "input": lambda x: x["input"],
                "agent_scratchpad": lambda x: format_log_to_messages(x["intermediate_steps"]),
            }
            | prompt
            | llm_with_stop
            | JSONAgentOutputParser()
        )

        conversational_agent = AgentExecutor(tools=tools, llm=llm, agent=agent, memory=memory, verbose=True, max_iterations=8, handle_parsing_errors=True, return_intermediate_steps=True,
                               prefix="If you think you have enough information to answer the user's initial question, use always '''Final Answer''' action which 'action_input' must be written in a language understandable by the user.")

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

question= st.chat_input("Ask me any question", disabled=not os.getenv('COHERE_API_KEY')) or mensaje()

if question:
    session_path = (Path(__file__).parent /  f"runs/{add_text(question)}.pickle").absolute()

    if os.path.exists(session_path):

            st.chat_message(name="user").markdown(question)
            answer_container = st.empty().container().chat_message("assistant", avatar="🦜")
            st_callback = StreamlitCallbackHandler(answer_container)
            answer = playback_callbacks(handlers=[st_callback], records_or_filename=session_path, max_pause_time=1)
            answer_container.markdown(answer)
            memory.save_context({"question": question}, {"output": answer})        

    else:

        with st.chat_message("user"):
            st.markdown(question)    

        with st.chat_message("assistant", avatar="🦜"):
            # Setup the callback handler
            st_callback = StreamlitCallbackHandler(st.container())
            
            assistance_response = conversational_agent.invoke({"input": question}, config={"callbacks": [st_callback]})
            # assistance_response = conversational_agent.run(question, callbacks=[st_callback])
            st.markdown(assistance_response['output'])