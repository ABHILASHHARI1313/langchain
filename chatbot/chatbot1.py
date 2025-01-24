# https://medium.com/@petarjoncheski/web-chatgpt-chatbot-using-langchain-and-streamlit-5c0bb1740814
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from langchain_community.chat_message_histories import FileChatMessageHistory
import streamlit as st

# Initialize the model
llm = Ollama(model="llama3.2")

# Define the prompt template
prompt = ChatPromptTemplate.from_template("You are a chatbot having a conversation with a human about the following {content}. Give the response in Spanish.")

# Set up the chain
chain = prompt | llm | StrOutputParser()

while True:
    content = input('Your prompt ')
    if content.lower() in ['quit', 'exit', 'bye']:
        print('Goodbye!')
        break
    response = chain.invoke({"content": content})
    print(response)
    print('-' * 50)
