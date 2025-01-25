# https://medium.com/@petarjloncheski/web-chatgpt-chatbot-using-langchain-and-streamlit-5c0bb1740814
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from langchain_core.messages import SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import FileChatMessageHistory
import streamlit as st

# Initialize the model
llm = Ollama(model="llama3.2")

# file for chat history
history = FileChatMessageHistory('.chat_history.json')

# initializing memory
memory: ConversationBufferMemory = ConversationBufferMemory(
    memory_key='chat_history',
    chat_memory=history,
    return_messages=True
)
# Define the prompt template
prompt = ChatPromptTemplate(
    input_variables=['content'],
    messages=[SystemMessage(content='You are a chatbot having a conversation with a human.'),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessagePromptTemplate.from_template('{content}')],

)

# Set up the chain
chain = prompt | llm | StrOutputParser()

while True:
    content = input('Your prompt ')
    if content.lower() in ['quit', 'exit', 'bye']:
        print('Goodbye!')
        break
    chat_history = memory.load_memory_variables({})['chat_history']
    response = chain.invoke({
        "content" : content,
        "chat_history" : chat_history
    })
    memory.save_context({"content":content},{"response":response})
    print(response)
    print('-' * 50)
