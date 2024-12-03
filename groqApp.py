## pdf docs loader
from langchain_community.document_loaders import PyPDFDirectoryLoader
## text splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
## huggingface embeddings
from langchain_huggingface import HuggingFaceEmbeddings
## vector store
from langchain_community.vectorstores import FAISS
## chatgroq
from langchain_groq import ChatGroq
## retriever
from langchain.chains import create_retrieval_chain
## combine documnets
from langchain.chains.combine_documents import create_stuff_documents_chain
## chat prompt
from langchain_core.prompts import ChatPromptTemplate
## streamlit 
import streamlit as st
# operating syestem control by os
import os
from dotenv import load_dotenv
load_dotenv()

## assign groq api key
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
## huggingface api key
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
## groq api key
groq_api_key = os.getenv('GROQ_API_KEY')

## app title
st.title("ChatGroq Q&A Chatbaot")

## llm
model_name = "Llama3-8b-8192"
llm = ChatGroq(groq_api_key=groq_api_key, model_name = model_name)

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions base on the provided context only.
    Pease provide the most accurate response based on the questions
    <context>
    {context}
    <context>
    Question:{input}
    """
)

def create_vector_embeddings():
    if "vector" not in st.session_state:
        ## huggingface embeddings
        st.session_state.embeddings = HuggingFaceEmbeddings()
        ## document loader
        st.session_state.loader = PyPDFDirectoryLoader("pdf_dir") ## data ingetion step
        st.session_state.documents = st.session_state.loader.load() ## documnet loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.documents[:50])
        st.session_state.vectors = FAISS().from_documents(st.session_state.final_documents, st.session_state.embeddings)

user_prompt = st.text_input("Enter your query form the research paper")

if st.button("Documet Embedding"):  
    create_vector_embeddings()
    st.write("vector Database is ready")

import time

if user_prompt:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retriever_chain = create_retrieval_chain(retriever, document_chain)
    start = time.process_time()
    response = retriever_chain.invoke({"input": user_prompt})
    print(f"Response time :{time.process_time()-start}")
    st.write(response['answer'])

    ## with streamlit expander
    with st.expander("Documnet similarity search"):
        for i, doc in enumerate(response['context']):
            st.write(doc.page_context)
            st.write("----------------------------------------")






