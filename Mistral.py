import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
import tempfile
import os
import time
import requests

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "retrieval_chain" not in st.session_state:
    st.session_state.retrieval_chain = None

# Function to handle API rate limit error
def handle_api_rate_limit_error():
    st.warning("Rate limit exceeded. Retrying in 60 seconds...")
    time.sleep(60)  # Sleep for 60 seconds before retrying

def process_pdf_with_faiss(uploaded_file, api_key):
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    # Load PDF
    loader = PyPDFLoader(tmp_path)
    docs = loader.load()
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    documents = text_splitter.split_documents(docs)
    
    # Create embeddings
    embeddings = MistralAIEmbeddings(
        model="mistral-embed",
        mistral_api_key=api_key
    )
    
    # Create FAISS vector store
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    # Create retriever
    retriever = vectorstore.as_retriever()
    
    # Initialize LLM
    model = ChatMistralAI(mistral_api_key=api_key)
    
    # Define prompt template
    prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:
    <context>
    {context}
    </context>
    Question: {input}
    
    If the answer cannot be found in the context, please say "I cannot find information about this in the document."
    """)
    
    # Create retrieval chain
    document_chain = create_stuff_documents_chain(model, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    # Clean up temporary file
    os.unlink(tmp_path)
    
    return retrieval_chain

def invoke_with_retry(retrieval_chain, prompt):
    try:
        response = retrieval_chain.invoke({"input": prompt})
        return response["answer"]
    except requests.exceptions.HTTPError as err:
        if err.response.status_code == 429:  # Check for rate limit error
            handle_api_rate_limit_error()
            # Retry the request after handling rate limit
            return invoke_with_retry(retrieval_chain, prompt)
        else:
            raise  # Reraise other errors

        # Streamlit UI
st.title("📚 PDF Chat Assistant")

# API key input
api_key = st.sidebar.text_input("Enter Mistral API Key", type="password")

# File upload
uploaded_file = st.sidebar.file_uploader("Upload PDF", type=['pdf'])

if uploaded_file and api_key:
    with st.spinner("Processing PDF..."):
        st.session_state.retrieval_chain = process_pdf_with_faiss(uploaded_file, api_key)
    st.sidebar.success("PDF processed successfully!")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your PDF"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display response
    if st.session_state.retrieval_chain is not None:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = invoke_with_retry(st.session_state.retrieval_chain, prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        with st.chat_message("assistant"):
            st.markdown("Please upload a PDF file and provide an API key first.")
            st.session_state.messages.append({"role": "assistant", "content": "Please upload a PDF file and provide an API key first."})