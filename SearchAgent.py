import streamlit as st
from langchain.agents import ConversationalChatAgent, AgentExecutor
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchRun
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper

st.set_page_config(page_title="LangChain: Chat with search", page_icon="🦜")
st.title("🦜 LangChain: Chat with search")

API_KEY = st.sidebar.text_input("Put your API Key", type="password")

msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(chat_memory=msgs, return_messages=True, memory_key="chat_history", output_key="output")
if len(msgs.messages) == 0 or st.sidebar.button("Reset chat history"):
    msgs.clear()
    msgs.add_user_message('Of What are you capable of?')
    msgs.add_ai_message("I'm capable to search any information that you wanna know")
    st.session_state.steps = {}

st.write(st.session_state.get('langchain_messages', 'Not found'))

avatars = {"human": "user", "ai": "assistant"}
for idx, msg in enumerate(msgs.messages):
    with st.chat_message(avatars[msg.type], avatar='🦜' if avatars[msg.type]=='assistant' else None):
        # Render intermediate steps if any were saved
        for step in st.session_state.steps.get(str(idx), []):
            if step[0].tool == "_Exception":
                continue
            with st.expander(f"✔️ **{step[0].tool}**: {step[0].tool_input}"):
                st.write(step[0].log)
                st.write(f"**{step[1]}**")
        st.write(msg.content)

if prompt := st.chat_input(placeholder="Who won the Women's U.S. Open in 2018?", disabled=not API_KEY):
    st.chat_message("user").write(prompt)

    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=API_KEY, streaming=True, convert_system_message_to_human=True)
    tools = [DuckDuckGoSearchRun(name="Search", api_wrapper=DuckDuckGoSearchAPIWrapper(max_results=6))]
    chat_agent = ConversationalChatAgent.from_llm_and_tools(llm=llm, tools=tools)
    executor = AgentExecutor.from_agent_and_tools(
        agent=chat_agent,
        tools=tools,
        memory=memory,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
    )
    with st.chat_message("assistant", avatar='🦜'):
        st_cb = StreamlitCallbackHandler(st.container())
        response = executor(prompt, callbacks=[st_cb])
        st.write(response["output"])
        st.session_state.steps[str(len(msgs.messages) - 1)] = response["intermediate_steps"]