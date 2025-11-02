# 🚀 Advanced RAG Chatbot with Local Llama Models

<div align="center">

![RAG Architecture](https://img.shields.io/badge/RAG-Retrieval--Augmented--Generation-blue?style=for-the-badge)
![Ollama](https://img.shields.io/badge/Ollama-Local--LLMs-green?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-Web--App-red?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

*A comprehensive implementation of Retrieval-Augmented Generation (RAG) with advanced techniques, featuring multiple document extraction tools and local LLM integration via Ollama.*

[📖 Table of Contents](#-table-of-contents) • [🚀 Quick Start](#-quick-start) • [🎯 Features](#-features) • [🧠 Advanced RAG Techniques](#-advanced-rag-techniques)

</div>

---

## 📖 Table of Contents

- [🎯 What is RAG?](#-what-is-rag)
- [🧠 Advanced RAG Techniques](#-advanced-rag-techniques)
- [✨ Key Features](#-key-features)
- [🛠️ Installation & Setup](#️-installation--setup)
- [📚 Usage Guide](#-usage-guide)
- [🔧 Architecture & Implementation](#-architecture--implementation)
- [🎮 Interactive Demos](#-interactive-demos)
- [📊 Evaluation & Metrics](#-evaluation--metrics)
- [🔍 Troubleshooting](#-troubleshooting)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

---

## 🎯 What is RAG?

**Retrieval-Augmented Generation (RAG)** is a cutting-edge AI technique that combines the power of large language models with external knowledge retrieval to provide more accurate, up-to-date, and contextually relevant responses.

### How RAG Works

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Query    │───▶│  Retrieval       │───▶│  Generation     │
│                 │    │  (Find relevant  │    │  (LLM creates   │
│  "What is X?"   │    │   documents)     │    │   response)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              ▲                        │
                              │                        ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │  Knowledge Base  │    │   Enhanced      │
                       │  (Vector Store)  │    │   Response      │
                       └──────────────────┘    └─────────────────┘
```

### Traditional LLM vs RAG

| Aspect | Traditional LLM | RAG System |
|--------|----------------|------------|
| **Knowledge** | Trained cutoff date | Always up-to-date |
| **Accuracy** | May hallucinate | Grounded in facts |
| **Domain Knowledge** | General training | Domain-specific |
| **Cost** | High (large models) | Efficient (smaller models + retrieval) |
| **Customization** | Limited | Highly customizable |

### Why RAG Matters

- **Reduces Hallucinations**: Responses are grounded in retrieved documents
- **Domain Adaptation**: Easy to add domain-specific knowledge
- **Cost Efficiency**: Use smaller, faster models with external knowledge
- **Explainability**: Can show sources for generated answers
- **Privacy**: Keep sensitive data local and secure

---

## 🧠 Advanced RAG Techniques

This codebase implements several cutting-edge RAG techniques that go beyond basic retrieval and generation.

### 1. 🔍 Multi-Query Retrieval

**Concept**: Generate multiple query variations to retrieve diverse relevant documents.

**Benefits**:
- Captures different aspects of the query
- Improves recall by finding documents that match query variations
- Reduces query-specific bias

**Implementation**: Uses LLM to generate 3-5 different query formulations, retrieves documents for each, then combines and deduplicates results.

### 2. 🔄 RAG Fusion

**Concept**: Combine results from multiple retrieval strategies using reciprocal rank fusion.

**How it works**:
1. Use different retrieval methods (semantic, keyword, hybrid)
2. Get ranked results from each method
3. Fuse rankings using RRF (Reciprocal Rank Fusion)
4. RRF Score = Σ(1/(k + r)) where r is rank, k=60

**Benefits**:
- Combines strengths of different retrieval methods
- More robust than single-method retrieval
- Handles different types of queries better

### 3. 🧠 HyDE (Hypothetical Document Embeddings)

**Concept**: Generate hypothetical documents that would answer the query, then use their embeddings for retrieval.

**Process**:
1. LLM generates a hypothetical answer document
2. Embed the hypothetical document
3. Find similar real documents using the hypothetical embedding
4. Generate final answer using retrieved documents

**Benefits**:
- Works well for queries that don't have exact matches
- Bridges semantic gap between queries and documents
- Particularly effective for complex or abstract queries

### 4. 🎯 Query Decomposition

**Concept**: Break complex queries into simpler sub-queries that can be answered independently.

**Benefits**:
- Handles complex multi-part questions
- Improves accuracy for compound queries
- Enables parallel processing of sub-queries

### 5. 📝 Self-RAG

**Concept**: The model critiques and improves its own retrieval and generation process.

**Components**:
- **Retrieval critic**: Evaluates if retrieved documents are relevant
- **Generation critic**: Assesses if generated answer is supported by evidence
- **Self-correction**: Refines answers based on self-critique

**Benefits**:
- Reduces hallucinations by self-verification
- Improves answer quality through self-correction
- Provides confidence scores for generated answers

### 6. 🔧 Adaptive Retrieval

**Concept**: Dynamically adjust retrieval strategy based on query characteristics.

**Strategies**:
- **Query complexity**: Simple queries → keyword search, complex → semantic
- **Domain detection**: Route to domain-specific retrievers
- **Confidence thresholding**: Fall back to broader search if confidence is low

---

## ✨ Key Features

### 🤖 **Local LLM Integration**
- **Ollama Integration**: Run Llama models locally without API costs
- **Multiple Models**: Support for llama2, llama3, codellama, mistral, phi3, deepseek-r1, qwen, and more
- **Configurable Parameters**: Temperature control for response creativity
- **No API Keys**: Completely local and private

### 📄 **Advanced Document Ingestion**
The application supports six different document extraction tools:

#### 1. **Beautiful Soup (HTML Parser)**
- Parse HTML/XML with powerful searching and filtering
- Web scraping and HTML document processing
- Handles malformed HTML gracefully

#### 2. **PyMuPDF (PDF Processor)**
- Extract text, images, and layout from PDFs
- Fast processing with encrypted PDF support
- Maintains document structure and formatting

#### 3. **Tesseract (OCR Engine)**
- Extract text from images using advanced OCR
- Supports 100+ languages and various image formats
- Handles scanned documents and handwritten notes

#### 4. **Docling (Layout Analysis)**
- Advanced document layout and structure analysis
- Extracts tables, figures, and text with markdown output
- Handles complex document formats (PDF, DOCX, PPTX, HTML)

#### 5. **Selenium (Web Scraping)**
- Automate web browsers for dynamic content extraction
- Handles JavaScript-heavy websites and SPAs
- Waits for page loading and user interactions

#### 6. **Firecrawl (Advanced Web Scraping)**
- Enterprise-grade web scraping with API
- Removes ads and navigation elements automatically
- Structured data extraction and content cleaning

### 🎛️ **Configurable Processing Pipeline**

#### **Text Chunking Strategies**
- **Chunk Size**: 100-2000 characters (configurable)
- **Chunk Overlap**: 0-500 characters (configurable)
- **Splitters**: Recursive, character-based, or token-based splitting

#### **Embedding Models**
- `all-MiniLM-L6-v2`: Fast, balanced performance
- `all-mpnet-base-v2`: Higher quality, slower processing
- `paraphrase-MiniLM-L3-v2`: Optimized for paraphrase detection

#### **Vector Stores**
- **ChromaDB**: Open-source, persistent, good for small-medium datasets
- **FAISS**: Facebook AI Similarity Search, optimized for large datasets

### 💬 **Interactive Chat Interface**
- Real-time conversation with document context
- Conversation history preservation
- Loading indicators and error handling
- Clear chat functionality

### 📊 **System Monitoring**
- Document count tracking
- Vector store status monitoring
- Performance metrics and system health
- Automatic initialization and error recovery

---

## 🛠️ Installation & Setup

### Prerequisites

- **Python 3.8+**
- **Tesseract OCR** (for OCR functionality)
  - **macOS**: `brew install tesseract`
  - **Ubuntu/Debian**: `sudo apt-get install tesseract-ocr`
  - **Windows**: Download from [GitHub releases](https://github.com/UB-Mannheim/tesseract/wiki)

### Quick Installation

```bash
# Clone or download the project
cd rag-doc

# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Ollama Setup

1. **Install Ollama**: Visit [ollama.ai](https://ollama.ai) for installation instructions

2. **Start Ollama service**:
   ```bash
   ollama serve
   ```

3. **Pull models** (in another terminal):
   ```bash
   # Popular models
   ollama pull llama2          # General purpose, fast
   ollama pull llama3.1:8b      # Latest Llama, high quality
   ollama pull codellama        # Code generation specialist
   ollama pull mistral          # Fast and capable
   ollama pull deepseek-r1:7b   # Reasoning specialist
   ```

---

## 📚 Usage Guide

### Getting Started

```bash
# Run the application
streamlit run app.py
```

Navigate to `http://localhost:8501` in your browser.

### Three Main Workflows

#### 📄 **Document Ingestion Tab**

1. **Configure Processing Settings**:
   ```python
   # Recommended settings for most use cases
   Chunk Size: 1000
   Chunk Overlap: 200
   Text Splitter: recursive
   Embedding Model: all-MiniLM-L6-v2
   Vector Store: chroma
   ```

2. **Initialize RAG System**: Click "Initialize RAG System"

3. **Extract & Add Documents**:
   - **Web Content**: Use Beautiful Soup or Selenium for URLs
   - **PDFs**: Upload with PyMuPDF
   - **Images**: OCR with Tesseract
   - **Complex Docs**: Layout analysis with Docling
   - **Advanced Web**: Firecrawl for enterprise scraping

#### 💬 **Chat with Documents Tab**

1. **Select Model**: Choose from available Ollama models
2. **Configure Temperature**: 0.0 (deterministic) to 1.0 (creative)
3. **Initialize Chat Model**: Set up the selected LLM
4. **Start Chatting**: Ask questions about your documents

**Example Queries**:
- "What are the main topics covered in these documents?"
- "Summarize the key findings from the research papers"
- "Explain the methodology used in the experiments"
- "What are the limitations mentioned in the studies?"

#### ⚙️ **Settings & Management Tab**

- Monitor document count and system status
- Clear vector store when needed
- Access performance tips and troubleshooting guides

### Example Workflow

```bash
# 1. Start Ollama (in one terminal)
ollama serve

# 2. Pull a model (in another terminal)
ollama pull llama2

# 3. Run the app
streamlit run app.py

# 4. In the app:
#    - Configure chunking (1000 chars, 200 overlap)
#    - Initialize RAG system
#    - Add documents using various tools
#    - Chat with your documents
```

---

## 🔧 Architecture & Implementation

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Web App                        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐    │
│  │ Document    │ │   Chat      │ │   Settings &        │    │
│  │ Ingestion   │ │ Interface   │ │   Management        │    │
│  └─────────────┘ └─────────────┘ └─────────────────────┘    │
├─────────────────────────────────────────────────────────────┤
│                    RAG Components Layer                     │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │ Text        │ │ Embeddings  │ │ Vector      │           │
│  │ Extraction  │ │ Models      │ │ Stores      │           │
│  └─────────────┘ └─────────────┘ └─────────────┘           │
├─────────────────────────────────────────────────────────────┤
│                    Ollama LLM Layer                         │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │ llama2      │ │ llama3      │ │ codellama   │           │
│  │ mistral     │ │ deepseek    │ │ qwen        │           │
│  └─────────────┘ └─────────────┘ └─────────────┘           │
└─────────────────────────────────────────────────────────────┘
```

### Core Components

#### **RAGSystem Class** (`rag_components.py`)

```python
class RAGSystem:
    def __init__(self):
        self.vectorstore = None
        self.embedding_model = None
        self.llm = None

    def initialize_embedding_model(self, model_name):
        # SentenceTransformer embeddings

    def initialize_vectorstore(self, embedding_model, vectorstore_type):
        # ChromaDB or FAISS initialization

    def add_documents(self, texts, metadata, splitter_config):
        # Document processing pipeline

    def ask_question(self, question):
        # Retrieval + Generation pipeline
```

#### **Document Processing Pipeline**

1. **Input Processing**: Files uploaded or URLs provided
2. **Content Extraction**: Tool-specific extraction logic
3. **Text Cleaning**: Remove unwanted elements (scripts, styles)
4. **Text Chunking**: Split into manageable chunks with overlap
5. **Embedding Generation**: Convert text to vector representations
6. **Vector Storage**: Store in ChromaDB or FAISS
7. **Retrieval**: Find relevant documents for queries
8. **Generation**: LLM creates responses using retrieved context

### Advanced Features Implementation

#### **Auto-Initialization**
- Vectorstore initializes automatically when adding documents
- LLM initializes automatically when asking questions
- Reduces user friction and prevents errors

#### **Error Handling & Recovery**
- Graceful handling of missing dependencies
- Automatic fallback for failed operations
- User-friendly error messages

#### **Performance Optimizations**
- CPU-only operations to avoid macOS threading issues
- Efficient text chunking and embedding generation
- Persistent vector stores for reuse

---

## 🎮 Interactive Demos

The application includes interactive demonstrations of advanced RAG techniques:

### Multi-Query Retrieval Demo
- Generates multiple query variations
- Shows how diverse queries improve retrieval
- Demonstrates result combination and deduplication

### Query Expansion Demo
- Adds related terms and synonyms
- Shows expanded query construction
- Demonstrates improved retrieval with broader search terms

### HyDE Retrieval Demo
- Generates hypothetical documents
- Shows semantic matching without exact terms
- Demonstrates abstract query handling

### RAG Fusion Demo
- Combines multiple retrieval strategies
- Shows reciprocal rank fusion in action
- Demonstrates improved ranking and relevance

---

## 📊 Evaluation & Metrics

### Retrieval Metrics

- **Recall@K**: Fraction of relevant documents retrieved in top K
- **Precision@K**: Fraction of retrieved documents that are relevant
- **Mean Reciprocal Rank (MRR)**: Average of reciprocal ranks
- **Normalized Discounted Cumulative Gain (NDCG)**: Position-based ranking quality

### Generation Metrics

- **Faithfulness**: Answer consistency with retrieved context
- **Relevance**: Answer relevance to the query
- **Coherence**: Answer logical flow and structure
- **Groundedness**: Answer support by evidence

### System Metrics

- **Latency**: Query response time
- **Throughput**: Queries processed per second
- **Cost**: Computational resource usage
- **User Satisfaction**: Subjective quality assessment

### Benchmarking Your RAG System

```python
# Example evaluation workflow
from rag_components import rag_system

# Add test documents
test_docs = ["Document content here..."]
rag_system.add_documents(test_docs)

# Test queries
test_queries = [
    "What is machine learning?",
    "Explain neural networks",
    "What are the advantages of RAG?"
]

# Evaluate performance
for query in test_queries:
    response = rag_system.ask_question(query)
    # Calculate metrics: faithfulness, relevance, etc.
```

---

## 🔍 Troubleshooting

### Common Issues & Solutions

#### **"Vectorstore not initialized"**
- **Solution**: The system now auto-initializes. If error persists, manually click "Initialize RAG System"

#### **"No Ollama models found"**
- **Check**: Is Ollama running? (`ollama serve`)
- **Pull models**: `ollama pull llama2`
- **Verify**: `ollama list` should show available models

#### **Memory issues with large documents**
- **Reduce chunk size**: Try 500-800 characters
- **Increase chunk overlap**: 100-200 characters
- **Use smaller embedding model**: `all-MiniLM-L6-v2`

#### **Slow performance**
- **Use ChromaDB**: Better for smaller datasets
- **Optimize chunking**: Larger chunks = fewer embeddings
- **CPU optimization**: Models run on CPU for macOS compatibility

#### **Tesseract OCR errors**
- **Install Tesseract**: Check system requirements above
- **Verify installation**: `tesseract --version`
- **Supported formats**: PNG, JPG, JPEG, BMP, TIFF

#### **Selenium webdriver issues**
- **Automatic handling**: webdriver-manager should resolve this
- **Manual install**: Download ChromeDriver manually if needed
- **Check Chrome version**: Must match ChromeDriver version

### Performance Tips

- **Chunk Size**: 1000 chars with 200 overlap works for most cases
- **Embedding Model**: `all-MiniLM-L6-v2` for speed, `all-mpnet-base-v2` for quality
- **Vector Store**: ChromaDB for development, FAISS for production
- **Memory**: Close other applications when processing large documents

### Getting Help

1. **Check the logs**: Look for error messages in the terminal
2. **Verify dependencies**: `pip list` to check installed packages
3. **Test components**: Try each tool individually to isolate issues
4. **Community**: Check GitHub issues for similar problems

---

## 🤝 Contributing

We welcome contributions! Here's how to get involved:

### Development Setup

```bash
# Fork and clone the repository
git clone https://github.com/your-username/advanced-rag.git
cd advanced-rag

# Create feature branch
git checkout -b feature/new-rag-technique

# Set up development environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Adding New Features

#### **New Document Extraction Tool**
1. Add extraction function in `app.py`
2. Update the tool selection dropdown
3. Add UI components and validation
4. Test with various document types

#### **New RAG Technique**
1. Implement in `rag_components.py`
2. Add configuration options
3. Create interactive demo
4. Update documentation

#### **New Evaluation Metric**
1. Add metric calculation function
2. Integrate with evaluation pipeline
3. Add visualization components
4. Update metrics dashboard

### Code Standards

- **Type hints**: Use type annotations for function parameters
- **Docstrings**: Comprehensive documentation for all functions
- **Error handling**: Graceful error handling with user-friendly messages
- **Testing**: Unit tests for core functionality
- **Performance**: Optimize for speed and memory usage

### Pull Request Process

1. **Fork** the repository
2. **Create** a feature branch
3. **Make** your changes with tests
4. **Ensure** all tests pass
5. **Update** documentation
6. **Submit** a pull request with description

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Ollama** for local LLM deployment
- **LangChain** for RAG framework components
- **Streamlit** for the web application framework
- **SentenceTransformers** for embedding models
- **ChromaDB** and **FAISS** for vector storage
- **Open-source community** for the amazing tools and libraries

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/srinipusuluri/advanced_rag/issues)
- **Discussions**: [GitHub Discussions](https://github.com/srinipusuluri/advanced_rag/discussions)
- **Documentation**: This README and inline code documentation

---

<div align="center">

**Built with ❤️ for the AI community**

*Empowering developers and researchers with advanced RAG techniques*

[⬆️ Back to Top](#-advanced-rag-chatbot-with-local-llama-models)

</div>

## Features

### 🤖 **Local Llama Model Integration**
- Uses Ollama to run Llama models locally
- No API keys or cloud dependencies required
- Supports multiple Llama model variants
- Configurable temperature for response creativity

### 📄 **Document Ingestion with Multiple Tools**
The application supports six different document extraction tools:

#### 1. **Beautiful Soup (HTML Parser)**
- Parse HTML and XML documents to extract structured data and text content
- Web scraping, HTML document processing, extracting text from web pages
- Handles malformed HTML gracefully with powerful searching and filtering

#### 2. **PyMuPDF (PDF Text Extractor)**
- Extract text, images, and layout information from PDF documents
- PDF document processing, text mining from PDFs, document analysis
- Fast PDF text extraction with encrypted PDF support

#### 3. **Tesseract (OCR - Optical Character Recognition)**
- Extract text from images using OCR technology
- Converting scanned documents, images with text, handwritten notes
- Supports 100+ languages and various image formats

#### 4. **Docling (Document Layout Analysis)**
- Analyze document layout and structure for comprehensive content extraction
- Complex document processing, layout-aware text extraction
- Extracts tables, figures, and text with markdown output

#### 5. **Selenium (Web Scraping)**
- Automate web browsers to extract content from dynamic web pages
- Scraping JavaScript-heavy websites, interacting with web applications
- Handles JavaScript execution and waits for page loading

#### 6. **Firecrawl (Advanced Web Scraping)**
- Advanced web scraping API for complex websites and structured data extraction
- Enterprise web scraping, content aggregation, data mining
- Removes ads and navigation elements automatically

### 🎛️ **Configurable Chunking & Embeddings**
- **Chunk Size Control**: Adjust text chunk sizes (100-2000 characters)
- **Chunk Overlap**: Configure overlap between chunks (0-500 characters)
- **Text Splitters**: Choose from recursive, character, or token-based splitting
- **Embedding Models**: Select from multiple sentence transformer models
  - `all-MiniLM-L6-v2` (fast, good balance)
  - `all-mpnet-base-v2` (higher quality, slower)
  - `paraphrase-MiniLM-L3-v2` (optimized for paraphrase detection)

### 🗄️ **Multiple Vector Store Options**
- **Chroma**: Open-source vector database, persistent storage, good for small to medium datasets
- **FAISS**: Facebook AI Similarity Search, optimized for large datasets, in-memory with disk persistence
- Automatic persistence and loading of vector stores
- Easy switching between different vector store types

### 💬 **Interactive Chat Interface**
- Chat with your documents using retrieved context
- Conversation history preservation
- Real-time responses with loading indicators
- Clear chat history functionality

### 🗄️ **Vector Store Management**
- ChromaDB for persistent vector storage
- Document count monitoring
- Clear all documents functionality
- Automatic metadata tagging

## Installation

1. **Clone or download** the project files
2. **Navigate** to the `rag-doc` directory
3. **Create virtual environment**:
   ```bash
   python -m venv venv
   ```
4. **Activate virtual environment**:
   - On macOS/Linux: `source venv/bin/activate`
   - On Windows: `venv\Scripts\activate`
5. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Prerequisites

### Ollama Setup
1. **Install Ollama**: Follow instructions at [ollama.ai](https://ollama.ai)
2. **Start Ollama service**:
   ```bash
   ollama serve
   ```
3. **Pull a Llama model** (in another terminal):
   ```bash
   ollama pull llama2
   # or other models like llama3, codellama, etc.
   ```

## Usage

### Getting Started
1. **Run the application**:
   ```bash
   streamlit run app.py
   ```

2. **Open your browser** and navigate to the displayed URL (typically `http://localhost:8501`)

### Three Main Tabs

#### 📄 **Document Ingestion Tab**
1. **Configure Processing Settings**:
   - **Chunk Size**: Set text chunk size (100-2000 characters)
   - **Chunk Overlap**: Configure overlap between chunks (0-500 characters)
   - **Text Splitter**: Choose splitting method (recursive/character/token)
   - **Embedding Model**: Select embedding model for vectorization

2. **Initialize RAG System**: Click "Initialize RAG System" to set up vector store

3. **Extract & Add Documents**:
   - **Beautiful Soup**: Enter URL or paste HTML content
   - **PyMuPDF**: Upload PDF files
   - **Tesseract**: Upload images for OCR
   - **Docling**: Upload complex documents (PDF, DOCX, PPTX, HTML)
   - **Selenium**: Scrape dynamic websites
   - **Firecrawl**: Advanced web scraping (requires API key)

4. **Add to RAG**: Each extraction automatically adds content to the vector store

#### 💬 **Chat with Documents Tab**
1. **Select Model**: Choose from available Ollama models
2. **Set Temperature**: Adjust response creativity (0.0-1.0)
3. **Initialize Chat Model**: Set up the selected LLM
4. **Start Chatting**: Ask questions about your ingested documents
5. **View History**: Conversation history is preserved

#### ⚙️ **Settings Tab**
1. **Monitor System**: View document count and system status
2. **Manage Documents**: Clear vector store if needed
3. **Advanced Configuration**: Access performance tips and troubleshooting

### Example Workflow
1. **Initialize**: Set up RAG system with desired chunking/embeddings
2. **Ingest**: Add documents using various extraction tools
3. **Chat**: Ask questions like "What are the main topics covered?" or "Summarize the key points"
4. **Iterate**: Adjust chunking settings and re-ingest if needed for better results

## Requirements

- Python 3.8+
- Tesseract OCR (for OCR functionality)
  - **macOS**: `brew install tesseract`
  - **Ubuntu/Debian**: `sudo apt-get install tesseract-ocr`
  - **Windows**: Download from [GitHub releases](https://github.com/UB-Mannheim/tesseract/wiki)

## Dependencies

- streamlit: Web application framework
- beautifulsoup4: HTML/XML parsing
- PyMuPDF: PDF processing
- pytesseract: OCR interface
- Pillow: Image processing
- docling: Document layout analysis
- selenium: Web browser automation
- webdriver-manager: Automatic driver management
- firecrawl-py: Advanced web scraping API
- requests: HTTP requests

## Project Structure

```
rag-doc/
├── app.py                 # Main Streamlit application with RAG chatbot
├── rag_components.py      # RAG system implementation (vector store, embeddings, LLM)
├── requirements.txt       # Python dependencies
├── README.md             # This documentation
├── chroma_db/            # ChromaDB vector store (created automatically)
├── faiss_index_rag_documents.faiss  # FAISS index file (created when using FAISS)
├── faiss_index_rag_documents.pkl    # FAISS metadata file (created when using FAISS)
└── venv/                 # Virtual environment (created during setup)
```

## How It Works

Each tool in the application demonstrates a different approach to document content extraction:

1. **Input Processing**: Files are uploaded or URLs are provided
2. **Content Extraction**: The selected tool processes the document
3. **Text Cleaning**: Unwanted elements (scripts, styles) are removed
4. **Output Formatting**: Clean, readable text is presented to the user

## Educational Value

This application serves as both a practical tool and an educational resource for understanding:

- Different document formats and their characteristics
- Various extraction techniques and their strengths
- When to use each tool based on the document type
- Best practices for document processing workflows

## Troubleshooting

### Common Issues:

1. **Tesseract not found**: Install Tesseract OCR on your system
2. **Chrome driver issues**: webdriver-manager should handle this automatically
3. **Docling import errors**: Some dependencies might be missing - check the error messages
4. **Memory issues with large files**: Try smaller files or increase system memory

### Performance Tips:

- Use appropriate file sizes for testing
- Close other applications when processing large documents
- Consider the processing time for complex documents

## Contributing

To extend the application:

1. Add new extraction tools in the `app.py` file
2. Update the sidebar dropdown with new tool names
3. Implement the extraction logic following the existing patterns
4. Add appropriate UI components and hints

## License

This project is provided as-is for educational and demonstration purposes.
