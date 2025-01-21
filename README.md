# rag-llm-pdfdocumentquestionanswering
This repository contains a Streamlit web application that allows users to upload PDF documents and ask questions based on the content of the document. The application uses the LangChain library, OpenAI GPT-4, and Chroma for document retrieval and question answering.
<img width="1472" alt="Ekran Resmi 2025-01-21 23 32 11" src="https://github.com/user-attachments/assets/53b8916b-f82b-47b1-a3bc-1dd2122a7344" />
<img width="1489" alt="Ekran Resmi 2025-01-21 23 32 26" src="https://github.com/user-attachments/assets/1af5ec6f-bb45-4ec5-bd52-40d53b98e3bd" />





## Features

- **PDF Upload**: Users can upload PDF documents to the system.
- **Text Extraction**: Extracts and splits text from the uploaded PDF for processing.
- **Question Answering**: Users can ask questions about the document and receive answers based on its content.
- **Vector Store**: The system processes and stores document content in a vector store for efficient retrieval.
- **Streamlit Interface**: A user-friendly interface that allows users to interact with the system via a web app.

## How It Works

1. **PDF Upload**: The user uploads a PDF file through the Streamlit app.
2. **Text Processing**: The uploaded PDF is processed to extract the text.
3. **Text Splitting**: The document text is split into smaller chunks to make it easier to search through.
4. **Vector Store Creation**: Chunks of the document are converted into vectors using OpenAI embeddings, and a vector store is created using Chroma.
5. **Question Answering**: The user can ask a question, and the system retrieves relevant context from the document to generate an answer using GPT-4.

## Prerequisites

- Python 3.11+
- OpenAI API key

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/rag-llm-pdfdocumentquestionanswering.git
cd rag-llm-pdfdocumentquestionanswering


### 2. Install dependencies

Create a virtual environment (recommended) and install dependencies from `requirements.txt`:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install --no-cache-dir -r requirements.txt

### 3. Set up OpenAI API key

Make sure you have an OpenAI API key. Add it to a `.env` file in the root directory of the project:

```makefile
OPENAI_API_KEY=your-api-key-here

### 4. Run the Streamlit app

You can start the Streamlit app by running:

```bash
streamlit run streamlit.py

## Docker Support

To run the application using Docker, follow these steps:

### 1. Build the Docker image

```bash
docker build -t rag-llm-pdfdocumentquestionanswering .
### 2. Run the Docker container

```bash
docker run -p 8501:8501 rag-llm-pdfdocumentquestionanswering
## Acknowledgments

- **LangChain** - The framework used for document processing and question answering.
- **OpenAI GPT-4** - Used for generating answers based on document content.
- **Chroma** - Vector store for efficient document retrieval.


