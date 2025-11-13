import os
import streamlit as st
import pypdf
import chromadb
from sentence_transformers import SentenceTransformer  # <-- We are using this again
import google.generativeai as genai
import time

# --- Helper Function to Read and Chunk PDF ---
def get_pdf_chunks(pdf_file):
    try:
        pdf_reader = pypdf.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        chunk_size = 1000
        overlap = 200
        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            if len(chunk) > 0:
                chunks.append(chunk)
        return chunks
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return []

# --- Main Application Logic ---

st.set_page_config(page_title="PDF Chat Assistant", layout="wide")
st.title("📄 AI Task Assistant")
st.markdown("Upload a PDF and ask questions about it.")

# --- SIDEBAR for API Key and Setup ---
with st.sidebar:
    st.header("1. Setup")
    
    # --- THIS IS THE NEW WAY TO GET THE KEY ---
    # It reads from Streamlit's native "st.secrets"
    # We will set this up in Step 4
    try:
        if st.secrets["GOOGLE_API_KEY"]:
            st.session_state.api_key = st.secrets["GOOGLE_API_KEY"]
            st.success("API Key loaded from secrets.")
        else:
            st.session_state.api_key = None
            st.error("API Key not found in Streamlit Secrets.")
    except:
        st.error("Could not load secrets. Please add your GOOGLE_API_KEY to st.secrets.")
    # --- END NEW KEY PART ---

    st.header("2. Upload PDF")
    uploaded_file = st.file_uploader("Upload your PDF document", type="pdf")
    process_button = st.button("Process Document")

# Initialize session state
if "memory" not in st.session_state:
    st.session_state.memory = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- PART 1: Processing the PDF (Ingestion & Indexing) ---
if process_button:
    if not st.session_state.get("api_key"):
        st.error("Please add your Google API Key to Streamlit Secrets to run this app.")
    elif not uploaded_file:
        st.error("Please upload a PDF document in the sidebar.")
    else:
        with st.spinner("Processing document... This may take a moment."):
            try:
                genai.configure(api_key=st.session_state.api_key)
                llm = genai.GenerativeModel('gemini-flash-latest')
            except Exception as e:
                st.error(f"Error configuring Google AI: {e}")
                st.stop()

            chunks = get_pdf_chunks(uploaded_file)
            if not chunks:
                st.error("Could not read any text from the PDF.")
                st.stop()

            st.write(f"PDF processed! Found {len(chunks)} text chunks.")

            # 3. Embeddings (BACK TO THE LOCAL MODEL)
            st.write("Loading local embedding model (all-MiniLM-L6-v2)...")
            embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            st.write("Creating vector database (in memory)...")
            client = chromadb.Client()
            
            if "pdf_collection" in [c.name for c in client.list_collections()]:
                client.delete_collection(name="pdf_collection")
            collection = client.create_collection(name="pdf_collection")

            st.write("Embedding chunks and loading into vector store...")
            embeddings = embedding_model.encode(chunks)
            
            collection.add(
                documents=chunks,
                embeddings=list(embeddings),
                ids=[str(i) for i in range(len(chunks))]
            )
            
            # Save "Retriever" to session memory
            st.session_state.memory = {
                "collection": collection,
                "embedding_model": embedding_model, # We add the model back
                "llm": llm
            }
            
            st.session_state.chat_history = []
            st.success("Document processed! You can now ask questions.")
            time.sleep(1)

# --- PART 2: The Chat Interface ---
st.subheader("Chat with your Document")

for role, text in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(text)

user_query = st.chat_input("Ask a question about your document...")

if user_query:
    if st.session_state.memory is None:
        st.error("Please upload and process a document first.")
    elif not st.session_state.get("api_key"):
        st.error("API Key not configured in st.secrets.")
    else:
        st.session_state.chat_history.append(("user", user_query))
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.spinner("Thinking..."):
            
            # 1. Retrieve: Get relevant context
            collection = st.session_state.memory["collection"]
            embedding_model = st.session_state.memory["embedding_model"] # Get model from memory
            llm = st.session_state.memory["llm"]

            # Embed the query LOCALLY
            query_embedding = embedding_model.encode([user_query])
            
            try:
                results = collection.query(
                    query_embeddings=list(query_embedding), # Use local embedding
                    n_results=3
                )
                context = "\n".join(results['documents'][0])
            except Exception as e:
                st.error(f"Error querying vector store: {e}")
                st.stop()

            # 2. Augment: Create the prompt
            prompt = f"""
            You are a helpful assistant. Answer the following question based *only*
            on the provided context. If the answer is not in the context,
            say 'I cannot find the answer in the document.'

            Context:
            {context}

            Question:
            {user_query}
            """
            
            # 3. Generate: Get the answer from the LLM
            try:
                response = llm.generate_content(prompt)
                answer = response.text
                st.session_state.chat_history.append(("assistant", answer))
                with st.chat_message("assistant"):
                    st.markdown(answer)
            except Exception as e:
                st.error(f"Error generating response from LLM: {e}")