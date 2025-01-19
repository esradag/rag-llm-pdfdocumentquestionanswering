import streamlit as st
import base64
from dotenv import load_dotenv
import os
from functions import *

# Load API key from .env file
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# If the API key is not found, display an error and stop execution
if not api_key:
    st.error("OpenAI API key not found. Please ensure it is defined in the .env file.")
    st.stop()

# Function to display the uploaded PDF file in the Streamlit app
def display_pdf(uploaded_file):
    bytes_data = uploaded_file.getvalue()
    base64_pdf = base64.b64encode(bytes_data).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# Function to load Streamlit page with file uploader and greeting text
def load_streamlit_page():
    st.set_page_config(layout="wide", page_title="LLM Tool")
    col1, col2 = st.columns([0.5, 0.5], gap="large")
    with col1:
        st.header("Upload file")
        uploaded_file = st.file_uploader("Please upload your PDF document:", type="pdf")
    return col1, col2, uploaded_file

# Load page layout and get uploaded file
col1, col2, uploaded_file = load_streamlit_page()

# Check if the user uploaded a file
if uploaded_file is not None:
    # Display the uploaded PDF
    with col2:
        display_pdf(uploaded_file)

    # Process the PDF and create vector store
    documents = get_pdf_text(uploaded_file)
    st.session_state.vector_store = create_vectorstore_from_texts(documents, api_key=api_key, file_name=uploaded_file.name)
    st.write("Input Processed")

# Input field for user's query
with col1:
    query = st.text_area("Ask a question about the research paper")

    if st.button("Generate Answer"):
        with st.spinner("Generating answer..."):
            # Query the document with the user input
            answer = query_document(vectorstore=st.session_state.vector_store, 
                                    query=query,  # User-defined query
                                    api_key=api_key)
            
            # Display the answer in the Streamlit app
            st.write(answer)
