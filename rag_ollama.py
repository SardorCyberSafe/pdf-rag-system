"""
RAG (Retrieval-Augmented Generation) System for PDF Processing
Using LangChain + Ollama Models
"""

import os
from typing import List, Optional

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import torch


class PDFRAGSystem:
    """
    A complete RAG system for processing PDFs and answering questions.
    Uses Ollama for LLM.
    """
    
    def __init__(
        self,
        pdf_path: str,
        ollama_model: str = "qwen2.5:1.5b",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        ollama_base_url: str = "http://localhost:11434"
    ):
        """
        Initialize the RAG system.
        
        Args:
            pdf_path: Path to the PDF file
            ollama_model: Ollama model name to use
            embedding_model: HuggingFace model for embeddings
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            ollama_base_url: Ollama server URL
        """
        self.pdf_path = pdf_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.ollama_model = ollama_model
        
        # Initialize components
        self.documents = None
        self.vectorstore = None
        self.qa_chain = None
        
        # Initialize embeddings
        print("Loading embedding model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': device}
        )
        print(f"Embedding model loaded: {embedding_model}")
        
        # Initialize Ollama LLM
        print(f"Connecting to Ollama model: {ollama_model}...")
        self.llm = Ollama(
            model=ollama_model,
            base_url=ollama_base_url,
            temperature=0.7,
            num_predict=500
        )
        print(f"Ollama model connected: {ollama_model}")
    
    def load_and_process_pdf(self):
        """Load PDF and split into chunks."""
        if not os.path.exists(self.pdf_path):
            raise FileNotFoundError(f"PDF not found: {self.pdf_path}")
        
        print(f"\nLoading PDF: {self.pdf_path}")
        
        # Load PDF
        loader = PyPDFLoader(self.pdf_path)
        self.documents = loader.load()
        print(f"✓ Loaded {len(self.documents)} pages")
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        self.chunks = text_splitter.split_documents(self.documents)
        print(f"✓ Created {len(self.chunks)} text chunks")
        
        return self.chunks
    
    def create_vectorstore(self):
        """Create FAISS vector store from document chunks."""
        if self.documents is None:
            self.load_and_process_pdf()
        
        print("\nCreating vector store...")
        self.vectorstore = FAISS.from_documents(
            documents=self.chunks,
            embedding=self.embeddings
        )
        print("✓ Vector store created successfully!")
        
        return self.vectorstore
    
    def setup_qa_chain(self):
        """Set up the question-answering chain."""
        if self.vectorstore is None:
            self.create_vectorstore()
        
        # Custom prompt template for better responses
        prompt_template = """Savolingizga quyidagi kontekstdan foydalanib javob bering.
Agar javobni bilmasangiz, shunday deying.

Kontekst: {context}

Savol: {question}

Javob:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create retriever
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}  # Return top 3 most relevant chunks
        )
        
        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        print("✓ QA chain setup complete!")
        
        return self.qa_chain
    
    def query(self, question: str) -> dict:
        """
        Ask a question and get an answer.
        
        Args:
            question: The question to ask
            
        Returns:
            Dictionary with answer and source documents
        """
        if self.qa_chain is None:
            self.setup_qa_chain()
        
        print(f"\n{'='*60}")
        print(f"Savol: {question}")
        print(f"{'='*60}")
        
        result = self.qa_chain({"query": question})
        
        answer = result.get("result", "Javob yaratilmadi")
        source_docs = result.get("source_documents", [])
        
        print(f"\nJavob:\n{answer}")
        
        if source_docs:
            print(f"\n{'='*60}")
            print(f"Manba hujjatlar: {len(source_docs)} ta")
            for i, doc in enumerate(source_docs[:2], 1):
                print(f"\nManba {i} (Sahifa {doc.metadata.get('page', 'N/A')}):")
                print(doc.page_content[:200] + "...")
        
        return {
            "question": question,
            "answer": answer,
            "source_documents": source_docs
        }
    
    def save_vectorstore(self, path: str):
        """Save vector store to disk."""
        if self.vectorstore is None:
            raise ValueError("Vector store not created yet!")
        self.vectorstore.save_local(path)
        print(f"✓ Vector store saved to {path}")
    
    @classmethod
    def load_vectorstore(cls, path: str, ollama_model: str = "qwen2.5:1.5b", 
                        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                        ollama_base_url: str = "http://localhost:11434"):
        """Load vector store from disk and create QA chain."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': device}
        )
        vectorstore = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
        print(f"✓ Vector store loaded from {path}")
        
        # Create Ollama LLM
        llm = Ollama(
            model=ollama_model,
            base_url=ollama_base_url,
            temperature=0.7,
            num_predict=500
        )
        
        # Setup prompt
        prompt_template = """Savolingizga quyidagi kontekstdan foydalanib javob bering.
Agar javobni bilmasangiz, shunday deying.

Kontekst: {context}

Savol: {question}

Javob:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        
        return qa_chain


def select_ollama_model():
    """Interactive model selection."""
    import subprocess
    
    print("\n" + "="*60)
    print("Ollama Modellar:")
    print("="*60)
    
    # Get available models
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        models = []
        for line in result.stdout.split('\n')[1:]:
            if line.strip() and not line.startswith('NAME'):
                model_name = line.split()[0]
                models.append(model_name)
                print(f"{len(models)}. {model_name}")
    except:
        models = ["qwen2.5:1.5b", "llama3.2:latest", "deepseek-r1:1.5b"]
        print("Ollama ro'yxati olinmadi, default modellar:")
        for i, model in enumerate(models, 1):
            print(f"{i}. {model}")
    
    print("\nQaysi modelni ishlatasiz?")
    while True:
        try:
            choice = input(f"\nRaqam kiriting (1-{len(models)}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                selected_model = models[idx]
                print(f"✓ Tanlandi: {selected_model}")
                return selected_model
            else:
                print(f"1-{len(models)} oralig'ida raqam kiriting")
        except ValueError:
            print("Faqat raqam kiriting")


def main():
    """Main function with interactive model selection."""
    print("\n" + "="*60)
    print("PDF RAG Tizimi - Ollama bilan")
    print("="*60)
    
    # PDF path
    pdf_path = "/content/5b166cdc167e7.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"\n❌ PDF fayl topilmadi: {pdf_path}")
        pdf_path = input("PDF fayl yo'lini kiriting: ").strip()
        if not os.path.exists(pdf_path):
            print("❌ Fayl hali ham topilmadi!")
            return
    
    # Model selection
    selected_model = select_ollama_model()
    
    # Initialize RAG system
    print(f"\nRAG tizimi ishga tushirilmoqda: {selected_model}")
    rag = PDFRAGSystem(
        pdf_path=pdf_path,
        ollama_model=selected_model,
        chunk_size=500,
        chunk_overlap=50
    )
    
    # Process PDF
    rag.load_and_process_pdf()
    rag.create_vectorstore()
    rag.setup_qa_chain()
    
    # Save vector store
    save_path = "/content/rag_pdf/vectorstore"
    rag.save_vectorstore(save_path)
    
    # Interactive mode
    print("\n" + "="*60)
    print("🤖 INTERAKTIV REJIM - PDF haqida savol bering")
    print("Chiqish uchun 'quit' yoki 'exit' deb yozing")
    print("="*60)
    
    while True:
        print()
        question = input("💬 Savolingiz: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q', 'chiqish']:
            print("\n👋 Xayr!")
            break
        
        if question:
            rag.query(question)


if __name__ == "__main__":
    main()
