import os
import streamlit as st
import pypdf
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import time

# --- Helper Function to Read and Chunk PDF ---
def get_pdf_chunks(pdf_file):
    """
    Reads a PDF, extracts text, and splits it into manageable chunks.
    """
    try:
        pdf_reader = pypdf.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # --- Simple "Manual" Chunking Strategy ---
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
st.markdown("Upload a PDF and ask questions about it. Built 100% free with Streamlit, ChromaDB, and Gemini.")

# --- SIDEBAR for API Key and Setup ---
with st.sidebar:
    st.header("1. Setup")

    # --- MODIFIED PART ---
    # Read the API key from environment variables (for deployment)
    # or from Streamlit's secrets (for Streamlit Community Cloud)
    api_key_env = os.environ.get("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
    
    if api_key_env:
        # If the key is found in the environment, set it in the session state
        st.session_state.api_key = api_key_env
        st.success("API Key loaded from environment.")
    else:
        # Fallback to text input if no key is in the environment (for local testing)
        api_key_input = st.text_input("Enter your Google Gemini API Key:", type="password")
        if api_key_input:
            st.session_state.api_key = api_key_input
    # --- END MODIFIED PART ---

    st.header("2. Upload PDF")
    # File uploader
    uploaded_file = st.file_uploader("Upload your PDF document", type="pdf")

    # The "Process" button
    process_button = st.button("Process Document")

# We use Streamlit's session_state to store variables.
if "memory" not in st.session_state:
    st.session_state.memory = None # This will hold our "Retriever"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- PART 1: Processing the PDF (Ingestion & Indexing) ---
# This block runs ONLY when the user clicks "Process Document"
if process_button:
    # --- MODIFIED PART 2: Check for the key from session_state ---
    if not st.session_state.get("api_key"):
        st.error("Please enter your Gemini API Key in the sidebar.")
    elif not uploaded_file:
        st.error("Please upload a PDF document in the sidebar.")
    else:
        with st.spinner("Processing document... This may take a moment."):
            # 1. Configure the LLM
            try:
                # Use the key from session state
                genai.configure(api_key=st.session_state.api_key)
                llm = genai.GenerativeModel('gemini-flash-latest')
            except Exception as e:
                st.error(f"Error configuring Google AI: {e}")
                st.stop() # Stop the app if the key is invalid

            # 2. Read and Chunk the PDF
            chunks = get_pdf_chunks(uploaded_file)
            if not chunks:
                st.error("Could not read any text from the PDF.")
                st.stop()

            st.write(f"PDF processed! Found {len(chunks)} text chunks.")

            # 3. Embeddings
            st.write("Loading embedding model (runs locally)...")
            embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # 4. Vector Store
            st.write("Creating vector database (in memory)...")
            client = chromadb.Client() # This is an in-memory client
            
            if "pdf_collection" in [c.name for c in client.list_collections()]:
                client.delete_collection(name="pdf_collection")
            collection = client.create_collection(name="pdf_collection")

            # 5. Add Chunks to Vector Store
            st.write("Embedding chunks and loading into vector store...")
            embeddings = embedding_model.encode(chunks)
            
            collection.add(
                documents=chunks,
                embeddings=list(embeddings),
                ids=[str(i) for i in range(len(chunks))] # Simple IDs
            )
            
            # 6. Save "Retriever" to session memory
            st.session_state.memory = {
                "collection": collection,
                "embedding_model": embedding_model,
                "llm": llm
            }
            
            # Clear chat history for the new document
            st.session_state.chat_history = []
            
            st.success("Document processed! You can now ask questions.")
            time.sleep(1) # Give a small pause

# --- PART 2: The Chat Interface ---
st.subheader("Chat with your Document")

# Display past chat history
for role, text in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(text)

# The user enters a new prompt
user_query = st.chat_input("Ask a question about your document...")

if user_query:
    # --- MODIFIED PART 3: Check for key and memory ---
    if st.session_state.memory is None:
        st.error("Please upload and process a document first.")
    elif not st.session_state.get("api_key"): # Also check for the key
        st.error("API Key not configured. Please enter it in the sidebar.")
    else:
        # Add user's message to chat history
        st.session_state.chat_history.append(("user", user_query))
        with st.chat_message("user"):
            st.markdown(user_query)

        # Show a "thinking" spinner
        with st.spinner("Thinking..."):
            
            # 1. Retrieve: Get relevant context
            collection = st.session_state.memory["collection"]
            embedding_model = st.session_state.memory["embedding_model"]
            llm = st.session_state.memory["llm"]

            query_embedding = embedding_model.encode([user_query])
            
            try:
                results = collection.query(
                    query_embeddings=list(query_embedding),
                    n_results=3 # Get top 3 most relevant chunks
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

                # Add assistant's response to chat history
                st.session_state.chat_history.append(("assistant", answer))
                with st.chat_message("assistant"):
                    st.markdown(answer)
            except Exception as e:
                st.error(f"Error generating response from LLM: {e}")