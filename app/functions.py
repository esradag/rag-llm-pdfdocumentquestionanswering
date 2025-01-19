from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

import os
import tempfile
import uuid
import pandas as pd
import re

# Define the prompt template for question-answering
PROMPT_TEMPLATE = """
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer
the question. If you don't know the answer, say that you
don't know. DON'T MAKE UP ANYTHING.

{context}

---

Answer the question based on the above context: {question}
"""

# Dosya adındaki "(sayı)" desenini kaldırma fonksiyonu
def clean_filename(filename):
    """
    Remove "(number)" pattern from a filename 
    (because this could cause error when used as collection name when creating Chroma database).

    Parameters:
        filename (str): The filename to clean

    Returns:
        str: The cleaned filename
    """
    new_filename = re.sub(r'\s\(\d+\)', '', filename)  
    return new_filename  # Temizlenmiş dosya adını döndürüyoruz

# Yüklenen PDF dosyasını işleyip, belgeyi döndüren fonksiyon
def get_pdf_text(uploaded_file): 
    """
    Load a PDF document from an uploaded file and return it as a list of documents

    Parameters:
        uploaded_file (file-like object): The uploaded PDF file to load

    Returns:
        list: A list of documents created from the uploaded PDF file
    """
    try:
        input_file = uploaded_file.read()

        # Geçici bir dosya oluşturuyoruz
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(input_file)
        temp_file.close()

        loader = PyPDFLoader(temp_file.name)
        documents = loader.load()  # PDF içeriğini belgelere dönüştürüyoruz

        return documents
    
    finally:
        os.unlink(temp_file.name)

# Belgeyi daha küçük parçalara ayıran fonksiyon
def split_document(documents, chunk_size, chunk_overlap):    
    """
    Function to split generic text into smaller chunks.
    chunk_size: The desired maximum size of each chunk (default: 400)
    chunk_overlap: The number of characters to overlap between consecutive chunks (default: 20).

    Returns:
        list: A list of smaller text chunks created from the generic text
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                          chunk_overlap=chunk_overlap,
                                          length_function=len,
                                          separators=["\n\n", "\n", " "])  # Metni ayırmak için ayırıcılar belirliyoruz
    
    return text_splitter.split_documents(documents)  # Belgeleri parçalara ayırıp döndürüyoruz

# OpenAI Embeddings fonksiyonunu döndüren fonksiyon
def get_embedding_function(api_key):
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002", openai_api_key=api_key
    )
    return embeddings

# Metin parçalarından vektör mağazası oluşturma fonksiyonu
def create_vectorstore(chunks, embedding_function, file_name, vector_store_path="db"):
    ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.page_content)) for doc in chunks]
    
    unique_ids = set()
    unique_chunks = []

    for chunk, id in zip(chunks, ids):     
        if id not in unique_ids:
            unique_ids.add(id)
            unique_chunks.append(chunk)

    vectorstore = Chroma.from_documents(documents=unique_chunks, 
                                        collection_name=clean_filename(file_name),  
                                        embedding=embedding_function, 
                                        ids=list(unique_ids), 
                                        persist_directory=vector_store_path)

    vectorstore.persist() 
    
    return vectorstore

# Metinlerden vektör mağazası oluşturma fonksiyonu
def create_vectorstore_from_texts(documents, api_key, file_name):
    docs = split_document(documents, chunk_size=1000, chunk_overlap=200)
    
    embedding_function = get_embedding_function(api_key)

    vectorstore = create_vectorstore(docs, embedding_function, file_name)
    
    return vectorstore

# Daha önce oluşturulmuş bir vektör mağazasını diskten yükleyen fonksiyon
def load_vectorstore(file_name, api_key, vectorstore_path="db"):
    embedding_function = get_embedding_function(api_key)
    
    return Chroma(persist_directory=vectorstore_path, 
                  embedding_function=embedding_function, 
                  collection_name=clean_filename(file_name))

# Belgeleri tek bir string'e dönüştüren fonksiyon
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Cevapları içeren sınıf
class AnswerWithSources(BaseModel):
    answer: str = Field(description="Answer to question")

def query_document(vectorstore, query, api_key):
    """
    Query a vector store with a question and return a structured response.

    :param vectorstore: A Chroma vector store object
    :param query: The question to ask the vector store
    :param api_key: The OpenAI API key to use when calling the OpenAI Embeddings API

    :return: The answer as a string
    """
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)

    retriever = vectorstore.as_retriever(search_type="similarity")

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt_template
            | llm.with_structured_output(AnswerWithSources, strict=True)
        )

    structured_response = rag_chain.invoke(query)

    # Düz metin olarak yanıtı çıkarıyoruz
    if isinstance(structured_response, dict):
        answer = structured_response['answer']  # Get only the answer text
    else:
        answer = structured_response  # If it's not a dictionary, treat it as a string

    return answer

