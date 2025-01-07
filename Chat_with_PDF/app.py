import streamlit as st
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import os
from PIL import Image
import io
from streamlit_chat import message
from util import *


def process_answer(instruction):
    response = ''
    instruction = instruction
    chat_llm = qa_llm()
    generated_text = chat_llm(instruction)
    answer = generated_text['result']
    return answer

# Display conversation history using Streamlit messages
def display_conversation(history):
    for i in range(len(history["generated"])):
        message(history["past"][i], is_user=True, key=str(i) + "_user")
        message(history["generated"][i],key=str(i))


def main():
    st.set_page_config(layout="wide", page_title="Interactive PDF Assistant")

    # Custom CSS for styling and background
    st.markdown("""
        <style>
            /* Main container background */
            .stApp {
               
                background-size: cover;
                background-repeat: no-repeat;
            }
            
            /* Sidebar styling */
            .css-1d391kg {  /* Sidebar class */
                background-color: rgba(255, 255, 255, 0.95);
                padding: 1rem;
                border-right: 1px solid #ddd;
            }
            
            /* Main content styling */
            .main-content {
                background-color: rgba(255, 255, 255, 0.95);
                padding: 2rem;
                border-radius: 10px;
                margin: 1rem;
            }
            
            /* Chat container styling */
            .chat-container {
                background-color: white;
                border-radius: 10px;
                padding: 1rem;
                margin-top: 1rem;
            }
            
            /* Message styling */
            .chat-message {
                padding: 1rem;
                margin: 0.5rem 0;
                border-radius: 5px;
            }
            .user-message {
                background-color: #e6f3ff;
            }
            .assistant-message {
                background-color: #f0f0f0;
            }
            
            /* File uploader styling */
            .stFileUploader {
                padding: 1rem;
                background-color: white;
                border-radius: 10px;
                margin: 1rem 0;
            }
        </style>
    """, unsafe_allow_html=True)

    # Sidebar with instructions
    with st.sidebar:
        st.markdown("### 📖 How to Use This App")
        st.markdown("""
        #### Steps to use this application:
        1. **Upload PDF** 📤
           - Click the upload button to select your PDF file
           - File will be processed automatically
        
        2. **View PDF** 👀
           - Preview your uploaded document
           - Use scroll bar to view pages
        
        3. **Chat Interface** 💬
           - Ask questions about your document
           - Get AI-powered responses
        
        4. **Navigation** 🔍
           - Use page buttons to move through document
           - Scroll through chat history
        
        #### Tips for better results:
        - 💡 Ask specific questions
        - ⏳ Wait for processing to complete
        - 📝 Keep questions focused
        
        #### Note:
        Processing time varies based on PDF size and complexity.
        """)

    st.markdown("<h1 style='text-align: center; color: #2E86C1;'>Interactive PDF Assistant 📚</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #566573;'>Your Smart Document Companion</h3>", unsafe_allow_html=True)

    st.markdown("<h2 style='text-align: center; color:red;'>Upload your PDF 👇</h2>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("", type=["pdf"])

    if uploaded_file is not None:
        file_details = {
            "Filename": uploaded_file.name,
            "File size": get_file_size(uploaded_file)
        }
        filepath = "docs/"+uploaded_file.name
        with open(filepath, "wb") as temp_file:
                temp_file.write(uploaded_file.read())

        col1, col2= st.columns([1,2])
        with col1:
            # st.markdown("<h4 style color:black;'>File details</h4>", unsafe_allow_html=True)
            # # st.json(file_details)
            st.markdown("<h4 style color:black;'>File preview</h4>", unsafe_allow_html=True)
            pdf_view = displayPDF(filepath)

        with col2:
            with st.spinner('Embeddings are in process...'):
                ingested_data = create_embedding(filepath)
            st.success('Embeddings are created successfully!')
            st.markdown("<h4 style color:black;'>Chat Here</h4>", unsafe_allow_html=True)


            user_input = st.text_input("", key="input")

            # Initialize session state for generated responses and past messages
            if "generated" not in st.session_state:
                st.session_state["generated"] = ["I am ready to help you"]
            if "past" not in st.session_state:
                st.session_state["past"] = ["Hey there!"]
                
            # Search the database for a response based on user input and update session state
            if user_input:
                answer = process_answer({'query': user_input})
                st.session_state["past"].append(user_input)
                response = answer
                st.session_state["generated"].append(response)

            # Display conversation history using Streamlit messages
            if st.session_state["generated"]:
                display_conversation(st.session_state)


if __name__ == "__main__":
    main()
