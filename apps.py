import streamlit as st
from PyPDF2 import PdfReader
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download all required NLTK data at startup
def download_nltk_data():
    try:
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('stopwords')
        return True
    except Exception as e:
        st.error(f"Error downloading NLTK data: {str(e)}")
        return False

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file"""
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def split_into_sentences(text):
    """Split text into sentences using simpler method"""
    sentences = []
    current = ""
    for char in text:
        current += char
        if char in '.!?' and len(current.strip()) > 0:
            sentences.append(current.strip())
            current = ""
    if current.strip():
        sentences.append(current.strip())
    return sentences

def process_text(text):
    """Process text and create TF-IDF matrix"""
    if not text.strip():
        return [], None, None
        
    sentences = split_into_sentences(text)
    
    if not sentences:
        return [], None, None
    
    # Create TF-IDF matrix
    vectorizer = TfidfVectorizer(stop_words='english')  # Added stop_words removal
    tfidf_matrix = vectorizer.fit_transform(sentences)
    
    return sentences, vectorizer, tfidf_matrix

def get_response(question, sentences, vectorizer, tfidf_matrix):
    """Get response based on question with enhanced context"""
    if not sentences or vectorizer is None or tfidf_matrix is None:
        return "Sorry, there was an error processing the document."
    
    # Vectorize the question
    question_vector = vectorizer.transform([question])
    
    # Calculate similarity
    similarities = cosine_similarity(question_vector, tfidf_matrix).flatten()
    
    # Get more similar sentences (increased from 3 to 5)
    most_similar_idx = similarities.argsort()[-7:][::-1]  # Get top 7 sentences
    
    # Build response from relevant sentences with context
    response_sentences = []
    prev_idx = None
    
    for idx in most_similar_idx:
        if similarities[idx] > 0.05:  # Lowered threshold for more matches
            # Add the previous sentence for context if available
            if idx > 0 and prev_idx != idx - 1:
                response_sentences.append(sentences[idx - 1])
            
            response_sentences.append(sentences[idx])
            
            # Add the next sentence for context if available
            if idx < len(sentences) - 1 and idx + 1 not in most_similar_idx:
                response_sentences.append(sentences[idx + 1])
            
            prev_idx = idx
    
    # Remove duplicates while preserving order
    response_sentences = list(dict.fromkeys(response_sentences))
    
    # Combine sentences into a coherent response
    response = " ".join(response_sentences)
    
    return response.strip() if response else "I couldn't find a relevant answer. Please try rephrasing your question."

def main():
    st.title("PDF Question Answering")
    st.write("Upload a PDF and ask questions about its content")
    
    # Download NLTK data at startup
    if 'nltk_downloaded' not in st.session_state:
        st.session_state.nltk_downloaded = download_nltk_data()
    
    # Initialize session state for storing processed data
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    
    # File upload
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        # Process PDF button
        if st.button("Process PDF"):
            try:
                with st.spinner("Processing PDF..."):
                    text = extract_text_from_pdf(uploaded_file)
                    
                    if not text.strip():
                        st.error("Could not extract text from the PDF. Please ensure it's a text-based PDF.")
                        return
                    
                    sentences, vectorizer, tfidf_matrix = process_text(text)
                    
                    if not sentences:
                        st.error("Could not process the text properly. Please try a different PDF.")
                        return
                    
                    st.session_state.processed_data = {
                        'sentences': sentences,
                        'vectorizer': vectorizer,
                        'tfidf_matrix': tfidf_matrix
                    }
                    st.success("PDF processed! You can now ask questions.")
            except Exception as e:
                st.error(f"An error occurred while processing the PDF: {str(e)}")
        
        # Question input
        question = st.text_input("Ask a question about the PDF:")
        if question:
            if st.session_state.processed_data:
                with st.spinner("Finding answer..."):
                    response = get_response(
                        question,
                        st.session_state.processed_data['sentences'],
                        st.session_state.processed_data['vectorizer'],
                        st.session_state.processed_data['tfidf_matrix']
                    )
                    st.write("Answer:", response)
            else:
                st.warning("Please process the PDF first by clicking the 'Process PDF' button.")

if __name__ == "__main__":
    main()
