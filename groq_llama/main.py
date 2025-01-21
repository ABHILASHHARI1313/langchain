from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os 
import streamlit as st
import time
load_dotenv()

groq_api_key=os.getenv("GROQ_API_KEY")

st.title("ChatGroq with LLAMA3 Demo")

llm = ChatGroq(groq_api_key=groq_api_key,model_name="Llama3-8b-8192")


Prompt = ChatPromptTemplate.from_template(
    ''' Answer the questions based on the provided context only.
        please provide the most accurate response based on the question
    <context>
    {context}
    </context>
    Question:{input}
    '''
)
def vector_embeddings():
    if "vector" not in st.session_state:
        st.session_state.embeddings = OllamaEmbeddings(model="llama3.2")
        st.session_state.loader = PyPDFDirectoryLoader("./us_census")
        st.session_state.documents = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        st.session_state.final_docs = st.session_state.text_splitter.split_documents(st.session_state.documents[:1])
        st.session_state.vector = FAISS.from_documents(st.session_state.final_docs,st.session_state.embeddings)

prompt1 = st.text_input("Enter your question from the document.")
if st.button("Documents Embedding"):
    st.write("Making embedding started")
    vector_embeddings()
    st.write("Vector store db is ready")

if prompt1:
    document_chain = create_stuff_documents_chain(llm,Prompt)
    retriever = st.session_state.vector.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever,document_chain)
    start = time.process_time()
    response = retrieval_chain.invoke({"input":prompt1})
    print("The response time is :",time.process_time()-start)
    st.write(response['answer'])

    with st.expander("Document Similarity Search"):
        for i,doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("-------------------------------------------------------")