from langchain_community.llms import Ollama
from langchain.schema import SystemMessage, HumanMessage, AIMessage

import streamlit as st
from streamlit_chat import message

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import FileChatMessageHistory

from langchain.globals import set_verbose

set_verbose(True)

history = FileChatMessageHistory(".chat_history.json")

memory : ConversationBufferMemory = ConversationBufferMemory(
    memory_key = "chat_history",
    chat_memory = history,
    return_messages=True
)

st.set_page_config(
    page_title="Your Custom Assistant",
    page_icon="👽"
)

st.subheader('Your Custom ChatGPT')

chat = Ollama(model="llama3.2",temperature=0)


if 'messages' not in st.session_state:
    st.session_state.messages = memory.chat_memory.messages

system_message = next((msg for msg in st.session_state.messages if isinstance(msg, SystemMessage)), None)

with st.sidebar:
    system_message_input = st.text_input(label="System Role",value=system_message.content if system_message else " ")
    user_prompt = st.text_input(label="Send a Message")
    if not system_message and system_message_input:
        system_message = SystemMessage(content=system_message_input)
        st.session_state.messages.append(system_message)
        memory.chat_memory.add_message(system_message)
    if user_prompt:
        st.session_state.messages.append(HumanMessage(content=user_prompt))
        memory.chat_memory.add_message(HumanMessage(content=user_prompt))

        with st.spinner("The Model is thinking..."):
            response = chat.invoke(st.session_state.messages)
            st.session_state.messages.append(AIMessage(content=response))
            memory.chat_memory.add_message(AIMessage(content=response))

if len(st.session_state.messages) > 0:
    for i, msg in enumerate(st.session_state.messages[1:]):
        if isinstance(msg, HumanMessage):
            message(msg.content, key=f'{i}+ HM', is_user=True)
        if isinstance(msg, AIMessage):
            message(msg.content, key=f'{i}+ AIM', is_user=False)
