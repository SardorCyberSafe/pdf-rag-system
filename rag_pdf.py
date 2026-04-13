"""
RAG (Retrieval-Augmented Generation) System for PDF Processing
Using LangChain + HuggingFace Pre-trained Models
"""

import os
from typing import List, Optional

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import pipeline
import torch


class PDFRAGSystem:
    """
    A complete RAG system for processing PDFs and answering questions.
    Uses pre-trained models from HuggingFace.
    """
    
    def __init__(
        self,
        pdf_path: str,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model: str = "gpt2",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        device: Optional[str] = None
    ):
        """
        Initialize the RAG system.
        
        Args:
            pdf_path: Path to the PDF file
            embedding_model: HuggingFace model for embeddings
            llm_model: HuggingFace model for text generation
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            device: Device to run models on ('cuda', 'cpu', or None for auto)
        """
        self.pdf_path = pdf_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Initialize components
        self.documents = None
        self.vectorstore = None
        self.qa_chain = None
        
        # Initialize embeddings
        print("Loading embedding model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': self.device}
        )
        
        # Initialize LLM
        print(f"Loading LLM model: {llm_model}...")
        self._setup_llm(llm_model)
        
    def _setup_llm(self, model_name: str):
        """Set up the language model pipeline."""
        try:
            # Create text generation pipeline
            text_generation_pipeline = pipeline(
                "text-generation",
                model=model_name,
                tokenizer=model_name,
                max_new_tokens=500,
                temperature=0.7,
                top_p=0.95,
                repetition_penalty=1.2,
                device=0 if self.device == "cuda" else -1,
                trust_remote_code=True
            )
            
            self.llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
            print(f"LLM loaded successfully: {model_name}")
            
        except Exception as e:
            print(f"Warning: Could not load {model_name}. Error: {e}")
            print("Falling back to a simpler approach...")
            self.llm = None
    
    def load_and_process_pdf(self):
        """Load PDF and split into chunks."""
        if not os.path.exists(self.pdf_path):
            raise FileNotFoundError(f"PDF not found: {self.pdf_path}")
        
        print(f"Loading PDF: {self.pdf_path}")
        
        # Load PDF
        loader = PyPDFLoader(self.pdf_path)
        self.documents = loader.load()
        print(f"Loaded {len(self.documents)} pages")
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        self.chunks = text_splitter.split_documents(self.documents)
        print(f"Created {len(self.chunks)} text chunks")
        
        return self.chunks
    
    def create_vectorstore(self):
        """Create FAISS vector store from document chunks."""
        if self.documents is None:
            self.load_and_process_pdf()
        
        print("Creating vector store...")
        self.vectorstore = FAISS.from_documents(
            documents=self.chunks,
            embedding=self.embeddings
        )
        print("Vector store created successfully!")
        
        return self.vectorstore
    
    def setup_qa_chain(self):
        """Set up the question-answering chain."""
        if self.vectorstore is None:
            self.create_vectorstore()
        
        # Custom prompt template for better responses
        prompt_template = """Use the following context to answer the question. 
If you don't know the answer, just say that you don't know.

Context: {context}

Question: {question}

Answer:"""
        
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
        if self.llm is not None:
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": PROMPT},
                return_source_documents=True
            )
            print("QA chain with LLM setup complete!")
        else:
            # Fallback: just return retrieved documents
            self.qa_chain = retriever
            print("QA chain (retriever only) setup complete!")
        
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
        
        print(f"\nQuestion: {question}")
        
        if self.llm is not None:
            # Use full QA chain with LLM
            result = self.qa_chain({"query": question})
            
            answer = result.get("result", "No answer generated")
            source_docs = result.get("source_documents", [])
            
        else:
            # Fallback: return relevant chunks
            relevant_docs = self.qa_chain.get_relevant_documents(question)
            answer = "\n\n---\n\n".join([doc.page_content for doc in relevant_docs[:2]])
            source_docs = relevant_docs
        
        # Display results
        print(f"\nAnswer:\n{answer}")
        
        if source_docs:
            print(f"\n---\nSource Documents Used: {len(source_docs)}")
            for i, doc in enumerate(source_docs[:2], 1):
                print(f"\nSource {i} (Page {doc.metadata.get('page', 'N/A')}):")
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
        print(f"Vector store saved to {path}")
    
    @classmethod
    def load_vectorstore(cls, path: str, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Load vector store from disk."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': device}
        )
        vectorstore = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
        print(f"Vector store loaded from {path}")
        return vectorstore


def main():
    """Example usage of the RAG system."""
    # Path to your PDF file
    pdf_path = "/content/5b166cdc167e7.pdf"
    
    # Check if PDF exists
    if not os.path.exists(pdf_path):
        print(f"PDF file not found at: {pdf_path}")
        print("Please provide the correct path to your PDF file.")
        return
    
    # Initialize RAG system
    rag = PDFRAGSystem(
        pdf_path=pdf_path,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        llm_model="gpt2",  # You can change to other models like "tiiuae/falcon-7b-instruct"
        chunk_size=500,
        chunk_overlap=50
    )
    
    # Process PDF
    rag.load_and_process_pdf()
    
    # Create vector store
    rag.create_vectorstore()
    
    # Setup QA chain
    rag.setup_qa_chain()
    
    # Ask questions
    questions = [
        "What is the main topic of this document?",
        "What are the key points discussed?",
        "Summarize the content",
    ]
    
    print("\n" + "="*60)
    print("RAG System - Ask Questions About Your PDF")
    print("="*60)
    
    for question in questions:
        rag.query(question)
        print("\n" + "-"*60)
    
    # Interactive mode
    print("\nInteractive Mode (type 'quit' to exit)")
    while True:
        user_question = input("\nYour question: ").strip()
        if user_question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        if user_question:
            rag.query(user_question)


if __name__ == "__main__":
    main()
