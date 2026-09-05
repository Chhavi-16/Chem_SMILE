import streamlit as st
import os
from pathlib import Path

st.set_page_config(page_title="ChemBot", layout="wide")

# Check file exists
data_path = r"C:\Users\Dell\OneDrive\Desktop\CHEMRAG\data\esol.csv.xls"
if not os.path.exists(data_path):
    st.error(f"Data file not found: {data_path}")
    st.stop()

# Load data and create retriever
@st.cache_resource
def load_retriever():
    try:
        from langchain_community.document_loaders import CSVLoader
        from langchain_text_splitters import CharacterTextSplitter
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS
        
        st.info("Loading dataset...")
        loader = CSVLoader(file_path=data_path)
        documents = loader.load()
        st.info(f"✓ Loaded {len(documents)} documents")
        
        st.info("Splitting documents...")
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = splitter.split_documents(documents)
        st.info(f"✓ Split into {len(docs)} chunks")
        
        st.info("Creating embeddings...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        st.info("✓ Embeddings created")
        
        st.info("Building FAISS index...")
        db = FAISS.from_documents(docs, embeddings)
        retriever = db.as_retriever()
        st.info("✓ FAISS index ready")
        
        return retriever
    except Exception as e:
        st.error(f"Error loading retriever: {str(e)}")
        st.error("Make sure all dependencies are installed: pip install -r requirements.txt")
        st.stop()

# Load the text generation pipeline
@st.cache_resource
def load_pipeline():
    try:
        from transformers import pipeline
        st.info("Loading language model...")
        pipe = pipeline("text-generation", model="google/flan-t5-base")
        st.info("✓ Language model loaded")
        return pipe
    except Exception as e:
        st.error(f"Error loading pipeline: {str(e)}")
        st.stop()

def ask_question(query, retriever, pipe):
    try:
        docs = retriever.invoke(query)
        context = "\n".join([doc.page_content for doc in docs])
        
        prompt = f"""Answer the question using the context below:

Context:
{context}

Question: {query}"""
        
        result = pipe(prompt, max_length=512)[0]["generated_text"]
        return result
    except Exception as e:
        return f"Error generating answer: {str(e)}"

# Streamlit UI
st.title("🧪 ChemBot - Chemistry RAG Chatbot")

st.write("Ask questions about chemical compounds and their properties based on the ESOL dataset.")

st.info("Initializing ChemBot... This may take a moment on first run.")

try:
    retriever = load_retriever()
    pipe = load_pipeline()
    
    st.success("✓ ChemBot initialized successfully!")
    
    query = st.text_input("Enter your question:", placeholder="e.g., What is the solubility of Amigdalin?")
    
    if st.button("Ask", type="primary"):
        if query:
            with st.spinner("Generating answer..."):
                answer = ask_question(query, retriever, pipe)
            st.write("**Answer:**")
            st.write(answer)
        else:
            st.warning("Please enter a question.")

except Exception as e:
    st.error(f"Failed to initialize ChemBot: {str(e)}")
    st.error("Troubleshooting steps:")
    st.error("1. Make sure all dependencies are installed: `pip install -r requirements.txt`")
    st.error("2. Check that the data file exists at: " + data_path)
    st.error("3. Ensure you have enough disk space for model downloads")