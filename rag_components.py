import os
import tempfile
import streamlit as st
from langchain_community.vectorstores import Chroma, FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter, TokenTextSplitter
from langchain_ollama import OllamaLLM
import chromadb
import ollama

# Fix PyTorch threading issues on macOS
try:
    import torch
    torch.set_num_threads(1)
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
except ImportError:
    pass

class RAGSystem:
    def __init__(self):
        self.vectorstore = None
        self.vectorstore_type = "chroma"  # Default vectorstore type
        self.embedding_model = None
        self.llm = None
        self.qa_chain = None
        self.documents = []
        self.collection_name = "rag_documents"
        self.chroma_client = None  # Initialize later when needed

    def _auto_initialize(self):
        """Auto-initialize the system with default settings"""
        try:
            # Initialize embedding model with default
            if not self.embedding_model:
                self.initialize_embedding_model("all-MiniLM-L6-v2")

            # Initialize vectorstore with default settings
            if not self.vectorstore:
                self.initialize_vectorstore("all-MiniLM-L6-v2", "chroma")
        except Exception:
            # Silently fail during auto-initialization to avoid errors during import
            pass

    def initialize_embedding_model(self, model_name="all-MiniLM-L6-v2"):
        """Initialize the embedding model"""
        try:
            # Force CPU usage and single thread to avoid macOS threading issues
            import os
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'
            os.environ['CUDA_VISIBLE_DEVICES'] = ''

            self.embedding_model = SentenceTransformerEmbeddings(
                model_name=model_name,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            return True
        except Exception as e:
            st.error(f"Error initializing embedding model: {str(e)}")
            return False

    def initialize_llm(self, model_name="llama2", temperature=0.7):
        """Initialize the Ollama LLM"""
        try:
            # Check if model is available
            available_models = ollama.list()
            model_names = [model.model for model in available_models.models]

            if model_name not in model_names:
                st.warning(f"Model '{model_name}' not found locally. Available models: {', '.join(model_names)}")
                # Try to pull the model
                with st.spinner(f"Pulling model '{model_name}'..."):
                    ollama.pull(model_name)

            self.llm = OllamaLLM(model=model_name, temperature=temperature)
            return True
        except Exception as e:
            st.error(f"Error initializing LLM: {str(e)}")
            return False

    def create_text_splitter(self, splitter_type="recursive", chunk_size=1000, chunk_overlap=200):
        """Create text splitter based on user preferences"""
        if splitter_type == "recursive":
            return RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", " ", ""]
            )
        elif splitter_type == "character":
            return CharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separator="\n"
            )
        elif splitter_type == "token":
            return TokenTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        else:
            return RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )

    def initialize_vectorstore(self, embedding_model_name="all-MiniLM-L6-v2", vectorstore_type="chroma"):
        """Initialize or load existing vectorstore based on type"""
        try:
            if not self.embedding_model:
                if not self.initialize_embedding_model(embedding_model_name):
                    return False

            self.vectorstore_type = vectorstore_type

            if vectorstore_type == "chroma":
                return self._initialize_chroma_vectorstore()
            elif vectorstore_type == "faiss":
                return self._initialize_faiss_vectorstore()
            else:
                st.error(f"Unsupported vectorstore type: {vectorstore_type}")
                return False

        except Exception as e:
            st.error(f"Error initializing vectorstore: {str(e)}")
            return False

    def _initialize_chroma_vectorstore(self):
        """Initialize Chroma vectorstore"""
        try:
            # Initialize ChromaDB client if not already done
            if self.chroma_client is None:
                self.chroma_client = chromadb.PersistentClient(path="./chroma_db")

            # Try to load existing collection
            try:
                self.vectorstore = Chroma(
                    client=self.chroma_client,
                    collection_name=self.collection_name,
                    embedding_function=self.embedding_model
                )
                # Check if collection has documents
                if self.vectorstore._collection.count() == 0:
                    st.info("Chroma vector store initialized but empty. Add documents to get started.")
                else:
                    st.success(f"Loaded existing Chroma vector store with {self.vectorstore._collection.count()} documents")
            except:
                # Create new collection
                self.vectorstore = Chroma(
                    client=self.chroma_client,
                    collection_name=self.collection_name,
                    embedding_function=self.embedding_model
                )
                st.info("Created new Chroma vector store")

            return True
        except Exception as e:
            st.error(f"Error initializing Chroma vectorstore: {str(e)}")
            return False

    def _initialize_faiss_vectorstore(self):
        """Initialize FAISS vectorstore"""
        try:
            # Check if FAISS index exists
            faiss_index_path = f"./faiss_index_{self.collection_name}"
            if os.path.exists(f"{faiss_index_path}.faiss") and os.path.exists(f"{faiss_index_path}.pkl"):
                # Load existing FAISS index
                self.vectorstore = FAISS.load_local(faiss_index_path, self.embedding_model, allow_dangerous_deserialization=True)
                st.success(f"Loaded existing FAISS vector store with {self.vectorstore.index.ntotal} documents")
            else:
                # Create new FAISS index
                self.vectorstore = FAISS.from_texts([""], self.embedding_model)  # Initialize with empty document
                # Remove the empty document
                self.vectorstore.delete([self.vectorstore.index_to_docstore_id[0]])
                st.info("Created new FAISS vector store")

            return True
        except Exception as e:
            st.error(f"Error initializing FAISS vectorstore: {str(e)}")
            return False

    def add_documents(self, texts, metadata=None, splitter_type="recursive", chunk_size=1000, chunk_overlap=200):
        """Add documents to the vectorstore"""
        try:
            if not self.vectorstore:
                # Auto-initialize if not already done
                if not self.initialize_vectorstore():
                    return False

            # Create text splitter
            text_splitter = self.create_text_splitter(splitter_type, chunk_size, chunk_overlap)

            # Split documents
            docs = []
            for i, text in enumerate(texts):
                splits = text_splitter.split_text(text)
                for j, split in enumerate(splits):
                    doc_metadata = {"source": f"document_{i}", "chunk": j}
                    if metadata and i < len(metadata):
                        doc_metadata.update(metadata[i])
                    docs.append((split, doc_metadata))

            # Add to vectorstore
            if docs:
                texts_to_add = [doc[0] for doc in docs]
                metadatas = [doc[1] for doc in docs]
                self.vectorstore.add_texts(texts_to_add, metadatas=metadatas)
                st.success(f"Added {len(docs)} document chunks to vector store")
                return True
            else:
                st.warning("No documents to add")
                return False

        except Exception as e:
            st.error(f"Error adding documents: {str(e)}")
            return False

    def ask_question(self, question):
        """Ask a question using the RAG system with manual retrieval"""
        try:
            if not self.vectorstore:
                # Auto-initialize vectorstore if not already done
                if not self.initialize_vectorstore():
                    return None
            if not self.llm:
                # Auto-initialize LLM with default model if not already done
                if not self.initialize_llm():
                    return None

            # Retrieve relevant documents
            docs = self.vectorstore.similarity_search(question, k=3)
            context = "\n\n".join([doc.page_content for doc in docs])

            # Create prompt
            prompt = f"""Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Answer:"""

            # Generate response
            response = self.llm.invoke(prompt)
            return response

        except Exception as e:
            st.error(f"Error asking question: {str(e)}")
            return None

    def get_document_count(self):
        """Get the number of documents in the vectorstore"""
        try:
            if self.vectorstore:
                if self.vectorstore_type == "chroma":
                    return self.vectorstore._collection.count()
                elif self.vectorstore_type == "faiss":
                    return self.vectorstore.index.ntotal
            return 0
        except Exception:
            return 0

    def clear_vectorstore(self):
        """Clear all documents from the vectorstore"""
        try:
            if self.vectorstore:
                if self.vectorstore_type == "chroma":
                    if self.chroma_client:
                        self.chroma_client.delete_collection(self.collection_name)
                    self.vectorstore = None
                    # Recreate empty vectorstore
                    self.initialize_vectorstore(vectorstore_type="chroma")
                elif self.vectorstore_type == "faiss":
                    # Remove FAISS index files
                    faiss_index_path = f"./faiss_index_{self.collection_name}"
                    for ext in ['.faiss', '.pkl']:
                        file_path = f"{faiss_index_path}{ext}"
                        if os.path.exists(file_path):
                            os.remove(file_path)
                    self.vectorstore = None
                    # Recreate empty vectorstore
                    self.initialize_vectorstore(vectorstore_type="faiss")

                st.success("Vector store cleared")
                return True
        except Exception as e:
            st.error(f"Error clearing vectorstore: {str(e)}")
            return False

    def get_available_vectorstore_types(self):
        """Get list of available vectorstore types"""
        return ["chroma", "faiss"]

    def get_available_ollama_models(self):
        """Get list of available Ollama models"""
        try:
            models = ollama.list()
            all_models = [model.model for model in models.models]
            return all_models
        except:
            return []

# Global RAG system instance
rag_system = RAGSystem()
