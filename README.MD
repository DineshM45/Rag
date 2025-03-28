Smart Document Analyzer

Smart Document Analyzer is an AI-powered Streamlit application that enables users to extract and analyze text from both websites and PDF documents. By leveraging LangChain, FAISS, and HuggingFace embeddings, this tool processes documents into manageable chunks, builds a vector store for efficient retrieval, and uses a language model (via ChatGroq) to generate insightful responses to user queries.



Features
Data Extraction:

Website Scraping: Enter a website URL and extract text content.

PDF Upload: Upload a PDF document to extract text.

Text Processing:

Splits the extracted text into chunks for better analysis.

Supports customizable chunk sizes and overlaps.

Vector Store & Retrieval:

Uses FAISS to build a vector store from text chunks.

Retrieves the most relevant text snippets based on user queries.

AI Analysis:

Integrates ChatGroq for generating AI-powered answers.

Provides clear, structured answers in Markdown format.

Session Persistence:

Retains extracted data and vector store using Streamlit session state.

Reset functionality available for a fresh start.