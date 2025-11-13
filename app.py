import os
import streamlit as st
import pypdf
import chromadb
# We no longer import SentenceTransformer
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

    api_key_env = os.environ.get("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
    
    if api_key_env:
        st.session_state.api_key = api_key_env
        st.success("API Key loaded from environment.")
    else:
        api_key_input = st.text_input("Enter your Google Gemini API Key:", type="password")
        if api_key_input:
            st.session_state.api_key = api_key_input

    st.header("2. Upload PDF")
    uploaded_file = st.file_uploader("Upload your PDF document", type="pdf")
    process_button = st.button("Process Document")

# We use Streamlit's session_state to store variables.
if "memory" not in st.session_state:
    st.session_state.memory = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- PART 1: Processing the PDF (Ingestion & Indexing) ---
if process_button:
    if not st.session_state.get("api_key"):
        st.error("Please enter your Gemini API Key in the sidebar.")
    elif not uploaded_file:
        st.error("Please upload a PDF document in the sidebar.")
    else:
        with st.spinner("Processing document... This may take a moment."):
            # 1. Configure the LLM
            try:
                genai.configure(api_key=st.session_state.api_key)
                llm = genai.GenerativeModel('gemini-flash-latest')
            except Exception as e:
                st.error(f"Error configuring Google AI: {e}")
                st.stop()

            # 2. Read and Chunk the PDF
            chunks = get_pdf_chunks(uploaded_file)
            if not chunks:
                st.error("Could not read any text from the PDF.")
                st.stop()

            st.write(f"PDF processed! Found {len(chunks)} text chunks.")

            # 3. Embeddings (NEW, FREE, LIGHTWEIGHT METHOD)
            # We no longer load a model. We use the genai API.
            st.write("Embedding chunks via Google's API (this is free)...")
            try:
                # Use Google's embedding model
                result = genai.embed_content(
                    model="models/embedding-001",  # The correct model for this
                    content=chunks,
                    task_type="retrieval_document" # Specify the task
                )
                embeddings = result['embedding']
            except Exception as e:
                st.error(f"Error creating embeddings: {e}")
                st.stop()
            
            st.write(f"Successfully embedded {len(embeddings)} chunks.")

            # 4. Vector Store
            st.write("Creating vector database (in memory)...")
            client = chromadb.Client()
            
            if "pdf_collection" in [c.name for c in client.list_collections()]:
                client.delete_collection(name="pdf_collection")
            collection = client.create_collection(name="pdf_collection")

            # 5. Add Chunks to Vector Store
            st.write("Embedding chunks and loading into vector store...")
            
            collection.add(
                documents=chunks,
                embeddings=embeddings, # Use the embeddings from the API
                ids=[str(i) for i in range(len(chunks))]
            )
            
            # 6. Save "Retriever" to session memory
            # We NO LONGER save the embedding_model
            st.session_state.memory = {
                "collection": collection,
                "llm": llm
            }
            
            st.session_state.chat_history = []
            st.success("Document processed! You can now ask questions.")

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
        st.error("API Key not configured. Please enter it in the sidebar.")
    else:
        st.session_state.chat_history.append(("user", user_query))
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.spinner("Thinking..."):
            
            # 1. Retrieve: Get relevant context
            collection = st.session_state.memory["collection"]
            llm = st.session_state.memory["llm"]

            # --- MODIFIED PART: Embed the user's query ---
            # We use the same API to embed the query
            try:
                query_result = genai.embed_content(
                    model="models/embedding-001",
                    content=user_query,
                    task_type="retrieval_query" # Specify the task
                )
                query_embedding = query_result['embedding']
            except Exception as e:
                st.error(f"Error embedding query: {e}")
                st.stop()

            # Query the vector store
            try:
                results = collection.query(
                    # Use the new query_embedding
                    query_embeddings=[query_embedding], 
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