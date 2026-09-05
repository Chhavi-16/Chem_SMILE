# ChemBot - Chemistry RAG Chatbot

This is a Streamlit-based chatbot that uses Retrieval-Augmented Generation (RAG) to answer questions about chemical compounds based on the ESOL dataset.

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

3. Open the provided URL in your browser and start asking questions about chemical properties!

## Features

- Loads and processes the ESOL dataset (CSV format)
- Splits documents into chunks
- Uses HuggingFace embeddings for vectorization
- Stores vectors in FAISS for efficient retrieval
- Generates answers using Google's FLAN-T5 model

## Usage

Enter a question in the text box, such as:
- "What is the solubility of Amigdalin?"
- "Which compound has the highest molecular weight?"
- "Describe the properties of molecules with low solubility."

The bot will retrieve relevant information from the dataset and generate a coherent answer.