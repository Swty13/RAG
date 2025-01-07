
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, PDFMinerLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter 
import os


def get_file_size(file):
    """Get file size in MB"""
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    return f"{round(file_size/1000000, 2)} MB"

def get_pdf_text(pdf_file):
    """Extract text from PDF"""
    text = ""
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text):
    """Split text into chunks"""
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = splitter.split_text(text)
    return chunks
