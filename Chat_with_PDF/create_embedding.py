from llm_call import get_embedding_model,get_llm
import streamlit as st
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
from langchain.vectorstores import Chroma 
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, PDFMinerLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.chains import RetrievalQA 
import os


embedding_model=get_embedding_model()
chat_llm=get_llm()


def get_file_size(file):
    """Get file size in MB"""
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    return f"{round(file_size/1000000, 2)} MB"

def get_pdf_text(pdf_file):
    """Extract text from PDF"""
    loader = PDFMinerLoader(pdf_file)
    return loader

def get_text_chunks(loader):
    """Split text into chunks"""
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=500)
    texts = text_splitter.split_documents(documents)
    return texts

def create_embeddings(pdf_file):
    """Create embeddings using HuggingFace"""
    file_data=get_pdf_text(pdf_file)
    text_chunks=get_text_chunks(file_data)
    vectorstore = Chroma.from_texts(texts=text_chunks, embedding=embedding_model)
    return vectorstore

def get_conversation_chain(vectorstore):
    """Create conversation chain"""
    llm = chat_llm
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

@st.cache_resource
def qa_llm():
    db = Chroma(embedding_function = embedding_model)
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm = chat_llm,
        chain_type = "stuff",
        retriever = retriever,
        return_source_documents=True
    )
    return qa