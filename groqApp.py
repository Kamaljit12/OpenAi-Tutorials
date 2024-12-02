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
from langchain.chains import create_retrievel_chain
## combine documnets
from langchain.chains import create_stuff_documents_chain
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


## app title
st.title("Chat Groq Q&A Chatbaot")




