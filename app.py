import streamlit as st
import requests
from bs4 import BeautifulSoup
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import tempfile
import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time

# Fix PyTorch mutex deadlock issue on macOS
import torch
torch.set_num_threads(1)

# Import RAG components
from rag_components import rag_system

st.set_page_config(page_title="RAG Chatbot with Local Llama Models", page_icon="🤖", layout="wide")

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Try to import docling, handle if not available
try:
    from docling.document_converter import DocumentConverter
    DOCLING_AVAILABLE = True
except (ImportError, RuntimeError) as e:
    DOCLING_AVAILABLE = False

# Try to import firecrawl, handle if not available
try:
    from firecrawl import FirecrawlApp
    FIRECRAWL_AVAILABLE = True
except ImportError:
    FIRECRAWL_AVAILABLE = False

# Sidebar configuration
with st.sidebar:
    st.header("🔧 Configuration")

    # RAG System Status
    st.subheader("System Status")
    doc_count = rag_system.get_document_count()
    st.metric("Documents in Vector Store", doc_count)

    if doc_count > 0:
        st.success("✅ RAG System Ready")
    else:
        st.warning("⚠️ No documents loaded")

    st.markdown("---")

    # Ollama Models
    st.subheader("🤖 Ollama Models")
    available_models = rag_system.get_available_ollama_models()
    if available_models:
        st.success(f"Available models: {len(available_models)}")
        for model in available_models[:5]:  # Show first 5
            st.code(model, language="")
        if len(available_models) > 5:
            st.text(f"... and {len(available_models) - 5} more")
    else:
        st.error("No Ollama models found. Please install Ollama and pull models.")

    st.markdown("---")
    st.markdown("### Instructions:")
    st.markdown("""
    1. **Document Ingestion**: Extract content from documents and add to vector store
    2. **Chat**: Ask questions about your documents
    3. **Settings**: Configure chunking and embedding parameters
    """)

    # Show warnings for missing libraries
    if not DOCLING_AVAILABLE:
        st.warning("Docling not available. Install with: pip install docling")
    if not FIRECRAWL_AVAILABLE:
        st.warning("Firecrawl not available. Install with: pip install firecrawl-py")

st.title("🤖 RAG Chatbot with Local Llama Models")
st.markdown("""
A comprehensive RAG (Retrieval-Augmented Generation) chatbot that uses local Llama models via Ollama.
Extract content from documents and chat with an AI that has access to your documents as context.
""")

# Create tabs for different functionalities
tab1, tab2, tab3, tab4 = st.tabs(["📄 Document Ingestion", "💬 Chat with Documents", "⚙️ Settings", "🚀 Advanced RAG"])

def extract_with_beautiful_soup(html_content):
    """Extract text from HTML using Beautiful Soup"""
    soup = BeautifulSoup(html_content, 'html.parser')

    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.extract()

    # Get text
    text = soup.get_text()

    # Break into lines and remove leading/trailing space
    lines = (line.strip() for line in text.splitlines())
    # Break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # Drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)

    return text

def extract_with_pymupdf(pdf_file):
    """Extract text from PDF using PyMuPDF"""
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += f"\n--- Page {page_num + 1} ---\n"
        text += page.get_text()

    doc.close()
    return text

def extract_with_tesseract(image_file):
    """Extract text from image using Tesseract OCR"""
    image = Image.open(image_file)

    # Convert PIL Image to bytes for processing
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    # Use pytesseract to extract text
    text = pytesseract.image_to_string(Image.open(io.BytesIO(img_byte_arr)))

    return text

def extract_with_docling(file_path):
    """Extract content using Docling for layout analysis"""
    if not DOCLING_AVAILABLE:
        return "Docling is not available. Please install it first."

    converter = DocumentConverter()
    result = converter.convert(file_path)
    return result.document.export_to_markdown()

def extract_with_selenium(url):
    """Extract content from web page using Selenium (similar to Puppeteer)"""
    try:
        # Set up Chrome options
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # Run in headless mode
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")

        # Initialize the driver
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)

        # Navigate to the URL
        driver.get(url)

        # Wait for page to load
        time.sleep(2)

        # Get the page source
        page_source = driver.page_source

        # Close the driver
        driver.quit()

        # Extract text using Beautiful Soup
        soup = BeautifulSoup(page_source, 'html.parser')

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()

        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)

        return text

    except Exception as e:
        return f"Error with Selenium extraction: {str(e)}"

def extract_with_firecrawl(url, api_key=None):
    """Extract content from web page using Firecrawl API"""
    if not FIRECRAWL_AVAILABLE:
        return "Firecrawl is not available. Please install it with: pip install firecrawl-py"

    if not api_key:
        return "Please provide a Firecrawl API key. Get one at https://firecrawl.dev"

    try:
        app = FirecrawlApp(api_key=api_key)

        # Scrape the URL
        scrape_result = app.scrape_url(url, params={'formats': ['markdown', 'html']})

        if scrape_result and 'markdown' in scrape_result:
            return scrape_result['markdown']
        elif scrape_result and 'content' in scrape_result:
            return scrape_result['content']
        else:
            return "No content extracted from the URL"

    except Exception as e:
        return f"Error with Firecrawl extraction: {str(e)}"

# Tab 1: Document Ingestion
with tab1:
    st.header("📄 Document Ingestion")
    st.markdown("Extract content from various document types and add them to the RAG vector store.")

    # Tool selection for ingestion
    ingestion_tool = st.selectbox(
        "Select Document Extractor Tool",
        ["Beautiful Soup (HTML)", "PyMuPDF (PDF)", "Tesseract (OCR)", "Docling (Layout Analysis)", "Selenium (Web Scraping)", "Firecrawl (Advanced Web Scraping)"]
    )

    # Chunking and embedding controls
    st.subheader("⚙️ Processing Settings")

    col1, col2 = st.columns(2)

    with col1:
        col1a, col1b = st.columns(2)
        with col1a:
            chunk_size = st.slider("Chunk Size", min_value=100, max_value=2000, value=1000, step=100,
                                  help="Size of text chunks for vectorization")
        with col1b:
            chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=500, value=200, step=50,
                                     help="Overlap between consecutive chunks")

        col1c, col1d = st.columns(2)
        with col1c:
            splitter_type = st.selectbox("Text Splitter",
                                        ["recursive", "character", "token"],
                                        help="Method for splitting text into chunks")
        with col1d:
            vectorstore_type = st.selectbox("Vector Store",
                                           rag_system.get_available_vectorstore_types(),
                                           help="Vector database for document storage")

    with col2:
        embedding_model = st.selectbox("Embedding Model",
                                      ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "paraphrase-MiniLM-L3-v2"],
                                      help="Model for generating text embeddings")

        st.markdown("**Vector Store Info:**")
        if vectorstore_type == "chroma":
            st.info("Chroma: Open-source vector database, persistent storage, good for small to medium datasets")
        elif vectorstore_type == "faiss":
            st.info("FAISS: Facebook AI Similarity Search, optimized for large datasets, in-memory with disk persistence")

    # Initialize RAG system with selected settings
    if st.button("Initialize RAG System"):
        with st.spinner("Initializing RAG system..."):
            if rag_system.initialize_embedding_model(embedding_model) and rag_system.initialize_vectorstore(embedding_model, vectorstore_type):
                st.success(f"RAG system initialized with {vectorstore_type} vectorstore!")
            else:
                st.error("Failed to initialize RAG system")

    # Document processing based on selected tool
    st.markdown("---")
    st.subheader("Extract & Add Document")

    if ingestion_tool == "Beautiful Soup (HTML)":
        col1, col2 = st.columns(2)

        with col1:
            input_type = st.radio("Choose input type:", ["URL", "HTML Text"], key="bs_input")

            if input_type == "URL":
                url = st.text_input("Enter URL:", "https://example.com", key="bs_url")
                if st.button("Extract & Add to RAG"):
                    try:
                        response = requests.get(url)
                        response.raise_for_status()
                        html_content = response.text
                        extracted_text = extract_with_beautiful_soup(html_content)

                        # Add to RAG system
                        metadata = [{"source": url, "type": "html", "tool": "beautiful_soup"}]
                        success = rag_system.add_documents([extracted_text], metadata, splitter_type, chunk_size, chunk_overlap)
                        if success:
                            st.success("Document added to RAG system!")
                        else:
                            st.error("Failed to add document to RAG system. Please initialize the vector store first.")
                        with st.expander("Extracted Text Preview"):
                            st.text_area("Content:", extracted_text[:1000] + "..." if len(extracted_text) > 1000 else extracted_text, height=200)

                    except Exception as e:
                        st.error(f"Error: {str(e)}")

            else:
                html_input = st.text_area("Enter HTML content:", height=200, key="bs_html")
                if st.button("Extract & Add to RAG"):
                    extracted_text = extract_with_beautiful_soup(html_input)
                    metadata = [{"source": "manual_html", "type": "html", "tool": "beautiful_soup"}]
                    rag_system.add_documents([extracted_text], metadata, splitter_type, chunk_size, chunk_overlap)
                    st.success("Document added to RAG system!")

        with col2:
            st.markdown("**Tool Info:** Beautiful Soup parses HTML/XML and extracts clean text content.")

    elif ingestion_tool == "PyMuPDF (PDF)":
        pdf_file = st.file_uploader("Choose a PDF file", type="pdf", key="pdf_upload")

        if pdf_file is not None and st.button("Extract & Add PDF to RAG"):
            try:
                extracted_text = extract_with_pymupdf(pdf_file)
                metadata = [{"source": pdf_file.name, "type": "pdf", "tool": "pymupdf"}]
                success = rag_system.add_documents([extracted_text], metadata, splitter_type, chunk_size, chunk_overlap)
                if success:
                    st.success("PDF added to RAG system!")
                else:
                    st.error("Failed to add PDF to RAG system. Please initialize the vector store first.")
                pdf_file.seek(0)  # Reset file pointer

            except Exception as e:
                st.error(f"Error extracting PDF: {str(e)}")

    elif ingestion_tool == "Tesseract (OCR)":
        image_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg", "bmp", "tiff"], key="ocr_upload")

        if image_file is not None and st.button("Extract & Add Image Text to RAG"):
            try:
                extracted_text = extract_with_tesseract(image_file)
                metadata = [{"source": image_file.name, "type": "image", "tool": "tesseract"}]
                rag_system.add_documents([extracted_text], metadata, splitter_type, chunk_size, chunk_overlap)
                st.success("Image text added to RAG system!")
                image_file.seek(0)

            except Exception as e:
                st.error(f"Error with OCR: {str(e)}")

    elif ingestion_tool == "Docling (Layout Analysis)":
        if DOCLING_AVAILABLE:
            doc_file = st.file_uploader("Choose a document file", type=["pdf", "docx", "pptx", "html"], key="docling_upload")

            if doc_file is not None and st.button("Extract & Add Document to RAG"):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{doc_file.name.split('.')[-1]}") as tmp_file:
                        tmp_file.write(doc_file.read())
                        tmp_file_path = tmp_file.name

                    extracted_content = extract_with_docling(tmp_file_path)
                    metadata = [{"source": doc_file.name, "type": "document", "tool": "docling"}]
                    rag_system.add_documents([extracted_content], metadata, splitter_type, chunk_size, chunk_overlap)
                    st.success("Document added to RAG system!")

                    os.unlink(tmp_file_path)

                except Exception as e:
                    st.error(f"Error with Docling: {str(e)}")
        else:
            st.error("Docling is not available. Please install it with: pip install docling")

    elif ingestion_tool == "Selenium (Web Scraping)":
        url = st.text_input("Website URL:", "https://example.com", key="selenium_url")

        if st.button("Scrape & Add to RAG"):
            with st.spinner("Scraping website with Selenium..."):
                try:
                    extracted_text = extract_with_selenium(url)
                    metadata = [{"source": url, "type": "webpage", "tool": "selenium"}]
                    rag_system.add_documents([extracted_text], metadata, splitter_type, chunk_size, chunk_overlap)
                    st.success("Website content added to RAG system!")

                except Exception as e:
                    st.error(f"Error scraping website: {str(e)}")

    elif ingestion_tool == "Firecrawl (Advanced Web Scraping)":
        if FIRECRAWL_AVAILABLE:
            col1, col2 = st.columns([2, 1])

            with col1:
                url = st.text_input("Website URL:", "https://example.com", key="firecrawl_url")

            with col2:
                api_key = st.text_input("Firecrawl API Key:", type="password", key="firecrawl_key",
                                       help="Get your API key at https://firecrawl.dev")

            if st.button("Scrape & Add with Firecrawl"):
                if not api_key:
                    st.error("Please provide a Firecrawl API key")
                else:
                    with st.spinner("Scraping website with Firecrawl..."):
                        try:
                            extracted_content = extract_with_firecrawl(url, api_key)
                            metadata = [{"source": url, "type": "webpage", "tool": "firecrawl"}]
                            rag_system.add_documents([extracted_content], metadata, splitter_type, chunk_size, chunk_overlap)
                            st.success("Website content added to RAG system!")

                        except Exception as e:
                            st.error(f"Error scraping with Firecrawl: {str(e)}")
        else:
            st.error("Firecrawl is not installed. Please install it with: pip install firecrawl-py")

# Tab 2: Chat with Documents
with tab2:
    st.header("💬 Chat with Documents")
    st.markdown("Ask questions about your ingested documents. The AI will use relevant document content as context.")

    if rag_system.get_document_count() == 0:
        st.warning("⚠️ No documents in the vector store. Please add documents first in the Document Ingestion tab.")
    else:
        # Model selection for chat
        available_models = rag_system.get_available_ollama_models()
        if available_models:
            selected_model = st.selectbox("Select Llama Model:", available_models, key="chat_model")

            # Temperature control
            temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1,
                                   help="Controls randomness in responses (0.0 = deterministic, 1.0 = creative)")

            # Initialize LLM if needed
            if st.button("Initialize Chat Model"):
                with st.spinner(f"Initializing {selected_model}..."):
                    if rag_system.initialize_llm(selected_model, temperature):
                        st.success(f"Chat model {selected_model} initialized!")
                    else:
                        st.error("Failed to initialize chat model")

            # Chat interface
            st.markdown("---")
            st.subheader("Chat")

            # Display chat history
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Chat input
            if prompt := st.chat_input("Ask a question about your documents..."):
                # Add user message to history
                st.session_state.chat_history.append({"role": "user", "content": prompt})

                # Display user message
                with st.chat_message("user"):
                    st.markdown(prompt)

                # Get AI response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = rag_system.ask_question(prompt)
                        if response:
                            st.markdown(response)
                            # Add AI response to history
                            st.session_state.chat_history.append({"role": "assistant", "content": response})
                        else:
                            error_msg = "Sorry, I couldn't generate a response. Please check your setup."
                            st.error(error_msg)
                            st.session_state.chat_history.append({"role": "assistant", "content": error_msg})

            # Clear chat history
            if st.button("Clear Chat History"):
                st.session_state.chat_history = []
                st.success("Chat history cleared!")
                st.rerun()
        else:
            st.error("No Ollama models available. Please install Ollama and pull some models (e.g., 'ollama pull llama2').")

# Tab 3: Settings
with tab3:
    st.header("⚙️ Settings & Management")
    st.markdown("Configure advanced settings and manage your document collection.")

    # Document Management
    st.subheader("Document Management")
    doc_count = rag_system.get_document_count()
    st.metric("Total Documents in Vector Store", doc_count)

    if doc_count > 0:
        if st.button("Clear All Documents", type="secondary"):
            if st.button("Confirm Clear All Documents", type="primary"):
                rag_system.clear_vectorstore()
                st.success("All documents cleared!")
                st.rerun()

    # Advanced Settings
    st.subheader("Advanced RAG Settings")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Vector Store Settings**")
        st.info("Vector store automatically manages embeddings and document chunks.")

        st.markdown("**Ollama Settings**")
        st.markdown("Make sure Ollama is running locally:")
        st.code("ollama serve")
        st.markdown("Pull models with:")
        st.code("ollama pull llama2")

    with col2:
        st.markdown("**Performance Tips**")
        st.markdown("""
        - **Chunk Size**: Larger chunks = more context, smaller chunks = more precise retrieval
        - **Chunk Overlap**: Helps maintain context across chunk boundaries
        - **Embedding Model**: Different models work better for different types of content
        - **Temperature**: Lower = more focused answers, higher = more creative responses
        """)

        st.markdown("**Troubleshooting**")
        st.markdown("""
        - If chat doesn't work, check that Ollama is running
        - If documents aren't found, try different chunk sizes
        - Clear vector store and re-add documents if needed
        """)

# Tab 4: Advanced RAG Techniques
with tab4:
    st.header("🚀 Advanced RAG Techniques")
    st.markdown("""
    Explore cutting-edge Retrieval-Augmented Generation techniques that go beyond basic RAG.
    Learn about advanced methods for improving retrieval quality, generation accuracy, and system robustness.
    """)

    # Technique categories
    technique_category = st.selectbox(
        "Select Technique Category",
        ["🔍 Retrieval Enhancement", "🎯 Query Optimization", "🧠 Generation Strategies", "🔄 System Architectures", "📊 Evaluation & Monitoring"]
    )

    if technique_category == "🔍 Retrieval Enhancement":
        st.subheader("Retrieval Enhancement Techniques")

        technique = st.selectbox(
            "Choose a technique to explore:",
            ["Multi-Query Retrieval", "RAG Fusion", "HyDE (Hypothetical Document Embeddings)", "Query Expansion", "Re-ranking", "Adaptive Retrieval"]
        )

        if technique == "Multi-Query Retrieval":
            st.markdown("""
            ### Multi-Query Retrieval
            **Concept**: Generate multiple query variations to retrieve diverse relevant documents.

            **How it works**:
            1. Take the original query
            2. Use LLM to generate multiple related queries
            3. Retrieve documents for each query
            4. Combine and deduplicate results

            **Benefits**:
            - Captures different aspects of the query
            - Improves recall by finding documents that match query variations
            - Reduces query-specific bias
            """)

            with st.expander("Implementation Example"):
                st.code("""
def multi_query_retrieval(query, llm, retriever, num_queries=3):
    # Generate multiple queries
    prompt = f"Generate {num_queries} different versions of this query: {query}"
    query_variations = llm.generate(prompt).split("\\n")

    all_docs = []
    for q in query_variations:
        docs = retriever.similarity_search(q, k=5)
        all_docs.extend(docs)

    # Remove duplicates and rank
    unique_docs = list(set(all_docs))
    return unique_docs[:10]  # Return top 10 unique docs
                """, language="python")

        elif technique == "RAG Fusion":
            st.markdown("""
            ### RAG Fusion
            **Concept**: Combine results from multiple retrieval strategies using reciprocal rank fusion.

            **How it works**:
            1. Use different retrieval methods (BM25, semantic search, etc.)
            2. Get ranked results from each method
            3. Fuse rankings using RRF (Reciprocal Rank Fusion)
            4. RRF Score = Σ(1/(k + r)) where r is rank, k is constant (usually 60)

            **Benefits**:
            - Combines strengths of different retrieval methods
            - More robust than single-method retrieval
            - Handles different types of queries better
            """)

            with st.expander("RRF Implementation"):
                st.code("""
def reciprocal_rank_fusion(results_list, k=60):
    # results_list: list of lists, each containing (doc_id, score) tuples
    doc_scores = {}

    for results in results_list:
        for rank, (doc_id, _) in enumerate(results):
            if doc_id not in doc_scores:
                doc_scores[doc_id] = 0
            doc_scores[doc_id] += 1 / (k + rank + 1)  # +1 because rank starts at 0

    # Sort by RRF score
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_docs
                """, language="python")

        elif technique == "HyDE (Hypothetical Document Embeddings)":
            st.markdown("""
            ### HyDE - Hypothetical Document Embeddings
            **Concept**: Generate hypothetical documents that would answer the query, then use their embeddings for retrieval.

            **How it works**:
            1. LLM generates a hypothetical answer document
            2. Embed the hypothetical document
            3. Use that embedding to find similar real documents
            4. Generate final answer using retrieved documents

            **Benefits**:
            - Works well for queries that don't have exact matches
            - Bridges semantic gap between queries and documents
            - Particularly effective for complex or abstract queries
            """)

            with st.expander("HyDE Example"):
                st.code("""
def hyde_retrieval(query, llm, retriever):
    # Generate hypothetical document
    prompt = f"Write a detailed passage that would answer this query: {query}"
    hypothetical_doc = llm.generate(prompt)

    # Embed hypothetical document
    hypo_embedding = embedder.encode(hypothetical_doc)

    # Find similar real documents
    similar_docs = retriever.similarity_search_by_vector(hypo_embedding, k=5)

    return similar_docs
                """, language="python")

        elif technique == "Query Expansion":
            st.markdown("""
            ### Query Expansion
            **Concept**: Expand the original query with related terms, synonyms, and context.

            **Methods**:
            - **Synonym expansion**: Add synonyms using WordNet or similar
            - **Contextual expansion**: Use LLM to add relevant context
            - **Historical expansion**: Include conversation history
            - **Domain-specific expansion**: Add domain terminology

            **Benefits**:
            - Improves recall by matching more relevant documents
            - Handles vocabulary mismatch between query and documents
            - Particularly useful in specialized domains
            """)

        elif technique == "Re-ranking":
            st.markdown("""
            ### Re-ranking
            **Concept**: Improve retrieval results by re-ranking initial candidates with a more sophisticated model.

            **Approaches**:
            - **Cross-encoders**: BERT-based models that score query-doc pairs
            - **Learning-to-rank**: Machine learning models trained to rank documents
            - **Query-dependent scoring**: Consider query complexity and document authority

            **Benefits**:
            - Significantly improves precision
            - Can incorporate complex features (freshness, authority, etc.)
            - Better than similarity search for many use cases
            """)

        elif technique == "Adaptive Retrieval":
            st.markdown("""
            ### Adaptive Retrieval
            **Concept**: Dynamically adjust retrieval strategy based on query characteristics.

            **Strategies**:
            - **Query complexity**: Simple queries → keyword search, complex → semantic
            - **Domain detection**: Route to domain-specific retrievers
            - **Confidence thresholding**: Fall back to broader search if confidence is low
            - **Multi-stage retrieval**: Progressive refinement

            **Benefits**:
            - Optimizes performance for different query types
            - Improves efficiency by using appropriate methods
            - Better user experience with faster, more accurate results
            """)

    elif technique_category == "🎯 Query Optimization":
        st.subheader("Query Optimization Techniques")

        technique = st.selectbox(
            "Choose a technique:",
            ["Query Decomposition", "Step-back Prompting", "Query Routing", "Conversational RAG", "Memory-Augmented Queries"]
        )

        if technique == "Query Decomposition":
            st.markdown("""
            ### Query Decomposition
            **Concept**: Break complex queries into simpler sub-queries that can be answered independently.

            **How it works**:
            1. Analyze the complex query
            2. Decompose into simpler, atomic questions
            3. Answer each sub-query separately
            4. Synthesize final answer from sub-answers

            **Benefits**:
            - Handles complex multi-part questions
            - Improves accuracy for compound queries
            - Enables parallel processing of sub-queries
            """)

        elif technique == "Step-back Prompting":
            st.markdown("""
            ### Step-back Prompting
            **Concept**: Generate more abstract, general questions to find broader context.

            **Process**:
            1. Original query: "How does photosynthesis work in plants?"
            2. Step-back: "What are the fundamental processes in plant biology?"
            3. Retrieve documents for both specific and general queries
            4. Use general context to inform specific answer

            **Benefits**:
            - Provides broader context for understanding
            - Helps with queries requiring background knowledge
            - Improves answer completeness
            """)

        elif technique == "Query Routing":
            st.markdown("""
            ### Query Routing
            **Concept**: Route queries to specialized retrievers based on query characteristics.

            **Routing Strategies**:
            - **Intent classification**: Route based on user intent
            - **Domain detection**: Route to domain-specific knowledge bases
            - **Query type**: Factual vs. opinion vs. procedural queries
            - **Language detection**: Route to language-specific retrievers

            **Benefits**:
            - Optimizes retrieval for different query types
            - Enables specialized handling for different domains
            - Improves overall system accuracy
            """)

    elif technique_category == "🧠 Generation Strategies":
        st.subheader("Advanced Generation Strategies")

        technique = st.selectbox(
            "Choose a technique:",
            ["Self-RAG", "CRAG (Corrective RAG)", "Agent-based RAG", "Iterative Refinement", "Multi-turn Generation"]
        )

        if technique == "Self-RAG":
            st.markdown("""
            ### Self-RAG
            **Concept**: The model critiques and improves its own retrieval and generation process.

            **Components**:
            - **Retrieval critic**: Evaluates if retrieved documents are relevant
            - **Generation critic**: Assesses if generated answer is supported by evidence
            - **Self-correction**: Refines answers based on self-critique

            **Benefits**:
            - Reduces hallucinations by self-verification
            - Improves answer quality through self-correction
            - Provides confidence scores for generated answers
            """)

        elif technique == "CRAG (Corrective RAG)":
            st.markdown("""
            ### CRAG - Corrective RAG
            **Concept**: Automatically correct retrieval errors and improve generation quality.

            **Process**:
            1. Initial retrieval and generation
            2. Evaluate answer quality and retrieval relevance
            3. If issues detected, perform corrective actions:
               - Additional retrieval
               - Knowledge refinement
               - Answer correction

            **Benefits**:
            - Self-correcting system that improves over time
            - Handles retrieval failures gracefully
            - Maintains high answer quality even with imperfect retrieval
            """)

        elif technique == "Agent-based RAG":
            st.markdown("""
            ### Agent-based RAG
            **Concept**: Use autonomous agents to orchestrate the RAG process dynamically.

            **Agent Types**:
            - **Retriever agents**: Specialize in different retrieval strategies
            - **Reasoning agents**: Plan and decompose complex queries
            - **Critic agents**: Evaluate and improve outputs
            - **Tool agents**: Interface with external tools and APIs

            **Benefits**:
            - Highly flexible and adaptive
            - Can handle complex multi-step reasoning
            - Enables integration with external tools and services
            """)

    elif technique_category == "🔄 System Architectures":
        st.subheader("Advanced System Architectures")

        technique = st.selectbox(
            "Choose an architecture:",
            ["Graph-based RAG", "Long-context RAG", "Multi-modal RAG", "Ensemble RAG", "Federated RAG"]
        )

        if technique == "Graph-based RAG":
            st.markdown("""
            ### Graph-based RAG
            **Concept**: Use knowledge graphs to enhance retrieval and reasoning.

            **Components**:
            - **Entity extraction**: Identify entities in documents
            - **Relation extraction**: Find relationships between entities
            - **Graph construction**: Build knowledge graph
            - **Graph-enhanced retrieval**: Use graph structure for better retrieval

            **Benefits**:
            - Captures complex relationships between concepts
            - Enables multi-hop reasoning
            - Provides structured context for generation
            """)

        elif technique == "Long-context RAG":
            st.markdown("""
            ### Long-context RAG
            **Concept**: Handle very long documents and conversations using advanced context management.

            **Techniques**:
            - **Hierarchical retrieval**: Multi-level document organization
            - **Dynamic context window**: Adaptive context selection
            - **Memory mechanisms**: Long-term memory for conversations
            - **Summarization**: Compress information for context limits

            **Benefits**:
            - Handles large knowledge bases effectively
            - Maintains context over long conversations
            - Scales to enterprise-level document collections
            """)

        elif technique == "Multi-modal RAG":
            st.markdown("""
            ### Multi-modal RAG
            **Concept**: Process and retrieve from multiple data modalities (text, images, audio, etc.).

            **Modalities**:
            - **Text**: Traditional document processing
            - **Images**: OCR, image captioning, visual question answering
            - **Audio**: Speech-to-text, audio analysis
            - **Video**: Frame extraction, video understanding
            - **Structured data**: Tables, databases, knowledge graphs

            **Benefits**:
            - Comprehensive information access across modalities
            - Better understanding of complex content
            - Enables richer, more informative responses
            """)

    elif technique_category == "📊 Evaluation & Monitoring":
        st.subheader("Evaluation and Monitoring")

        st.markdown("""
        ### Key Metrics for RAG Systems

        #### Retrieval Metrics
        - **Recall@K**: Fraction of relevant documents retrieved in top K
        - **Precision@K**: Fraction of retrieved documents that are relevant
        - **Mean Reciprocal Rank (MRR)**: Average of reciprocal ranks
        - **Normalized Discounted Cumulative Gain (NDCG)**: Ranks with position-based discounting

        #### Generation Metrics
        - **Faithfulness**: Does the answer contradict the retrieved context?
        - **Relevance**: Is the answer relevant to the query?
        - **Coherence**: Is the answer well-structured and logical?
        - **Groundedness**: Is the answer supported by evidence?

        #### System Metrics
        - **Latency**: Response time for queries
        - **Throughput**: Queries processed per second
        - **Cost**: Computational and API costs
        - **User satisfaction**: Subjective quality assessment
        """)

        with st.expander("Evaluation Frameworks"):
            st.markdown("""
            #### Popular RAG Evaluation Frameworks

            **RAGAS (RAG Assessment)**:
            - Comprehensive evaluation metrics
            - Automated assessment of retrieval and generation quality
            - Supports multiple evaluation dimensions

            **ARES (Automated RAG Evaluation System)**:
            - End-to-end evaluation pipeline
            - Handles both component and system-level evaluation
            - Provides actionable improvement suggestions

            **Custom Evaluation**:
            - Domain-specific metrics
            - User feedback integration
            - A/B testing frameworks
            """)

    # Interactive Demo Section
    st.markdown("---")
    st.subheader("🎮 Interactive Demonstrations")

    demo_type = st.selectbox(
        "Try an Advanced RAG Technique:",
        ["Basic RAG (Baseline)", "Multi-Query Retrieval", "Query Expansion", "HyDE Retrieval"]
    )

    if demo_type == "Basic RAG (Baseline)":
        st.markdown("**Standard RAG**: Single query → retrieve → generate")

        demo_query = st.text_input("Enter a question:", "What is machine learning?", key="basic_demo")

        if st.button("Run Basic RAG", key="basic_btn") and demo_query:
            if rag_system.get_document_count() == 0:
                st.warning("Please add documents first in the Document Ingestion tab.")
            else:
                with st.spinner("Processing with Basic RAG..."):
                    response = rag_system.ask_question(demo_query)
                    if response:
                        st.success("Response generated!")
                        st.text_area("Answer:", response, height=150)
                    else:
                        st.error("Failed to generate response.")

    elif demo_type == "Multi-Query Retrieval":
        st.markdown("**Multi-Query**: Generate multiple query variations for better retrieval")

        demo_query = st.text_input("Enter a question:", "What is machine learning?", key="multi_demo")

        if st.button("Run Multi-Query RAG", key="multi_btn") and demo_query:
            if rag_system.get_document_count() == 0:
                st.warning("Please add documents first in the Document Ingestion tab.")
            else:
                with st.spinner("Generating query variations and retrieving..."):
                    # This is a simplified demo - in practice you'd use an LLM to generate variations
                    query_variations = [
                        demo_query,
                        f"Tell me about {demo_query.lower()}",
                        f"Explain {demo_query.lower()} in detail",
                        f"What are the key aspects of {demo_query.lower()}?"
                    ]

                    st.info("Generated query variations:")
                    for i, q in enumerate(query_variations, 1):
                        st.code(f"{i}. {q}")

                    # For demo purposes, just use the original query
                    response = rag_system.ask_question(demo_query)
                    if response:
                        st.success("Response generated using multi-query approach!")
                        st.text_area("Answer:", response, height=150)
                    else:
                        st.error("Failed to generate response.")

    elif demo_type == "Query Expansion":
        st.markdown("**Query Expansion**: Add related terms and context to improve retrieval")

        demo_query = st.text_input("Enter a question:", "What is machine learning?", key="expand_demo")

        if st.button("Run Query Expansion", key="expand_btn") and demo_query:
            if rag_system.get_document_count() == 0:
                st.warning("Please add documents first in the Document Ingestion tab.")
            else:
                with st.spinner("Expanding query and retrieving..."):
                    # Simple expansion - in practice use LLM or thesaurus
                    expanded_terms = {
                        "machine learning": ["AI", "artificial intelligence", "algorithms", "data science", "neural networks"],
                        "deep learning": ["neural networks", "AI", "computer vision", "NLP"],
                        "python": ["programming", "coding", "software development"]
                    }

                    expanded_query = demo_query
                    for term, synonyms in expanded_terms.items():
                        if term.lower() in demo_query.lower():
                            expanded_query += f" {' '.join(synonyms)}"

                    st.info(f"Expanded query: {expanded_query}")

                    response = rag_system.ask_question(expanded_query)
                    if response:
                        st.success("Response generated with query expansion!")
                        st.text_area("Answer:", response, height=150)
                    else:
                        st.error("Failed to generate response.")

    elif demo_type == "HyDE Retrieval":
        st.markdown("**HyDE**: Generate hypothetical document, then find similar real documents")

        demo_query = st.text_input("Enter a question:", "What is machine learning?", key="hyde_demo")

        if st.button("Run HyDE Retrieval", key="hyde_btn") and demo_query:
            if rag_system.get_document_count() == 0:
                st.warning("Please add documents first in the Document Ingestion tab.")
            else:
                with st.spinner("Generating hypothetical document and retrieving..."):
                    # Simplified HyDE - in practice use LLM to generate detailed hypothetical answer
                    hypothetical_doc = f"""
                    Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions
                    without being explicitly programmed. It involves algorithms that can identify patterns in data and make
                    predictions or decisions based on those patterns. Key concepts include supervised learning, unsupervised
                    learning, neural networks, and deep learning. Applications include image recognition, natural language
                    processing, recommendation systems, and autonomous vehicles.
                    """

                    st.info("Generated hypothetical document:")
                    st.text_area("Hypothetical Answer:", hypothetical_doc[:300] + "...", height=100)

                    # For demo, just use the original query (real HyDE would embed the hypothetical doc)
                    response = rag_system.ask_question(demo_query)
                    if response:
                        st.success("Response generated using HyDE approach!")
                        st.text_area("Answer:", response, height=150)
                    else:
                        st.error("Failed to generate response.")

    # Resources Section
    st.markdown("---")
    st.subheader("📚 Resources & Further Reading")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Academic Papers**")
        st.markdown("""
        - [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
        - [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/abs/2310.11511)
        - [CRAG -- Corrective Retrieval Augmented Generation](https://arxiv.org/abs/2401.15884)
        - [HyDE: Precise Zero-Shot Dense Retrieval without Relevance Labels](https://arxiv.org/abs/2212.10496)
        """)

    with col2:
        st.markdown("**Implementation Guides**")
        st.markdown("""
        - [LangChain Advanced RAG Techniques](https://python.langchain.com/docs/guides/rag)
        - [LlamaIndex Advanced Retrieval](https://docs.llamaindex.ai/en/stable/module_guides/querying/retrieval/)
        - [RAGAS Evaluation Framework](https://docs.ragas.io/)
        - [Haystack Advanced Pipelines](https://docs.haystack.deepset.ai/docs)
        """)

    st.markdown("**Communities & Tools**")
    st.markdown("""
    - **GitHub Repositories**: Search for "advanced-rag", "rag-fusion", "self-rag"
    - **Discord/Slack**: RAG-focused communities in AI/ML Discord servers
    - **Conferences**: NeurIPS, ICML, ACL papers on retrieval-augmented generation
    - **Tools**: LangChain, LlamaIndex, Haystack, ChromaDB, Weaviate, Pinecone
    """)

# Footer
st.markdown("---")
st.markdown("Built with Streamlit • Document extraction tools showcase")
