import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set USER_AGENT if not set
if not os.getenv("USER_AGENT"):
    os.environ["USER_AGENT"] = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                "AppleWebKit/537.36 (KHTML, like Gecko) "
                                "Chrome/119.0.0.0 Safari/537.36")

# Initialize session state for persistence
if "vectorDB" not in st.session_state:
    st.session_state.vectorDB = None
if "website_docs" not in st.session_state:
    st.session_state.website_docs = None
if "pdf_docs" not in st.session_state:
    st.session_state.pdf_docs = None
if "all_chunks" not in st.session_state:
    st.session_state.all_chunks = None

def extract_data(website_url, uploaded_file):
    website_docs = []
    pdf_docs = []
    
    # --- Scrape Website ---
    if website_url:
        try:
            st.write(f"Scraping website: {website_url}")
            web_loader = WebBaseLoader(website_url)
            website_docs = web_loader.load()
            st.success("Website data loaded!")
            st.write(f"Total characters extracted: {len(website_docs[0].page_content)}")
        except Exception as e:
            st.error(f"Error scraping website: {e}")
    
    # --- Load PDF ---
    if uploaded_file:
        with open("temp_uploaded.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        try:
            pdf_loader = PyPDFLoader("temp_uploaded.pdf")
            pdf_docs = pdf_loader.load()
            st.success(f"PDF loaded with {len(pdf_docs)} page(s).")
        except Exception as e:
            st.error(f"Error loading PDF: {e}")
    
    return website_docs, pdf_docs

def process_data(website_docs, pdf_docs):
    all_chunks = []
    # --- Splitting Documents into Chunks ---
    if website_docs:
        web_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        web_chunks = web_splitter.split_documents(website_docs)
        st.write(f"Total website chunks: {len(web_chunks)}")
        all_chunks.extend(web_chunks)
    if pdf_docs:
        pdf_splitter = RecursiveCharacterTextSplitter(chunk_size=120, chunk_overlap=45)
        pdf_chunks = pdf_splitter.split_documents(pdf_docs)
        st.write(f"Total PDF chunks: {len(pdf_chunks)}")
        all_chunks.extend(pdf_chunks)
    st.write(f"Total combined chunks: {len(all_chunks)}")
    return all_chunks

def build_vector_store(chunks):
    hugging_face_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorDB = FAISS.from_documents(chunks, hugging_face_embeddings)
    st.success("Vector store built successfully!")
    return vectorDB

def main():
    st.title("Web & PDF Data Extraction with LangChain")
    
    # Show extraction interface only if data hasn't been extracted
    if st.session_state.vectorDB is None:
        st.header("Data Extraction")
        
        # --- User Input for Website URL & PDF Upload inside a form ---
        with st.form("extraction_form"):
            website_url = st.text_input("Enter the URL of the website to scrape:")
            uploaded_file = st.file_uploader("Upload a PDF (e.g., resume, document)", type="pdf")
            submitted = st.form_submit_button("Extract Data")
        
        if submitted:
            if not website_url and not uploaded_file:
                st.error("Please provide a website URL or upload a PDF.")
                return
            # Extract data from website and PDF
            website_docs, pdf_docs = extract_data(website_url, uploaded_file)
            st.session_state.website_docs = website_docs
            st.session_state.pdf_docs = pdf_docs
            
            if website_docs or pdf_docs:
                all_chunks = process_data(website_docs, pdf_docs)
                st.session_state.all_chunks = all_chunks
                st.session_state.vectorDB = build_vector_store(all_chunks)
            else:
                st.error("No data was extracted. Please check your inputs.")
    
    else:
        st.info("Data already extracted. You can query the stored vectors below.")
    
    # --- Query Interface ---
    if st.session_state.vectorDB is not None:
        st.header("Query the Data")
        query = st.text_input("Enter your query:")
        if st.button("Run Query") and query:
            vectorDB = st.session_state.vectorDB
            result = vectorDB.similarity_search(query=query)
            st.subheader("Most Relevant Document Snippet:")
            if result:
                st.write(result[0].page_content)
            else:
                st.write("No relevant data found.")
            
            # --- LLM Response ---
            llm = ChatGroq(model_name="llama-3.3-70b-versatile")
            template = ChatPromptTemplate.from_template(
                """Answer the following based on the provided context:
                
<context>
{context}
</context>

Answer the following question:
{input}"""
            )
            document_chain = create_stuff_documents_chain(llm=llm, prompt=template)
            retriever = vectorDB.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            response = retrieval_chain.invoke({"input": query})
            st.subheader("LLM Generated Answer:")
            st.write(response["answer"])
    else:
        st.info("Please extract data first to enable querying.")

if __name__ == "__main__":
    main()
