"""
Example usage of the RAG system with different configurations
"""

from rag_pdf import PDFRAGSystem
import os


def example_basic_usage():
    """Basic example with default models."""
    print("=" * 60)
    print("BASIC USAGE EXAMPLE")
    print("=" * 60)
    
    pdf_path = "/content/5b166cdc167e7.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"PDF not found: {pdf_path}")
        return
    
    # Initialize with lightweight models (good for CPU)
    rag = PDFRAGSystem(
        pdf_path=pdf_path,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        llm_model="gpt2",
        chunk_size=500,
        chunk_overlap=50
    )
    
    # Process PDF
    rag.load_and_process_pdf()
    rag.create_vectorstore()
    rag.setup_qa_chain()
    
    # Ask a question
    result = rag.query("What is this document about?")
    
    # Save vector store for later use
    rag.save_vectorstore("/content/rag_pdf/vectorstore")
    
    return rag


def example_with_better_models():
    """Example with better quality models (requires more resources)."""
    print("=" * 60)
    print("ADVANCED USAGE EXAMPLE WITH BETTER MODELS")
    print("=" * 60)
    
    pdf_path = "/content/5b166cdc167e7.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"PDF not found: {pdf_path}")
        return
    
    # Use better models (requires GPU or more RAM)
    rag = PDFRAGSystem(
        pdf_path=pdf_path,
        embedding_model="sentence-transformers/all-mpnet-base-v2",  # Better embeddings
        llm_model="tiiuae/falcon-7b-instruct",  # Better LLM (requires ~14GB VRAM)
        chunk_size=1000,
        chunk_overlap=100
    )
    
    rag.load_and_process_pdf()
    rag.create_vectorstore()
    rag.setup_qa_chain()
    
    result = rag.query("Summarize the key findings in this document")
    
    return rag


def example_load_saved_vectorstore():
    """Example of loading a previously saved vector store."""
    print("=" * 60)
    print("LOADING SAVED VECTORSTORE EXAMPLE")
    print("=" * 60)
    
    vectorstore_path = "/content/rag_pdf/vectorstore"
    
    if not os.path.exists(vectorstore_path):
        print("No saved vector store found. Run basic example first.")
        return
    
    # Load the vector store
    vectorstore = PDFRAGSystem.load_vectorstore(vectorstore_path)
    print("Vector store loaded successfully!")
    
    # You can now use this vector store for similarity search
    query = "main topic"
    docs = vectorstore.similarity_search(query, k=2)
    
    print(f"\nQuery: {query}")
    print(f"\nMost relevant chunks:")
    for i, doc in enumerate(docs, 1):
        print(f"\n--- Chunk {i} ---")
        print(doc.page_content[:300])
        print(f"Metadata: {doc.metadata}")


def example_custom_questions():
    """Example of asking custom questions interactively."""
    print("=" * 60)
    print("CUSTOM QUESTIONS EXAMPLE")
    print("=" * 60)
    
    pdf_path = "/content/5b166cdc167e7.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"PDF not found: {pdf_path}")
        return
    
    rag = PDFRAGSystem(
        pdf_path=pdf_path,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        llm_model="gpt2"
    )
    
    rag.load_and_process_pdf()
    rag.create_vectorstore()
    rag.setup_qa_chain()
    
    # Ask specific questions
    questions = [
        "What are the main conclusions?",
        "What methods were used?",
        "What data or evidence is presented?",
        "What are the limitations mentioned?",
    ]
    
    for question in questions:
        rag.query(question)
        print("\n" + "-" * 60)


if __name__ == "__main__":
    import sys
    
    print("PDF RAG System - Example Usage")
    print("\nSelect an example to run:")
    print("1. Basic Usage (lightweight, good for CPU)")
    print("2. Advanced Usage (better models, requires GPU)")
    print("3. Load Saved Vector Store")
    print("4. Custom Questions")
    print("5. Run Basic + Interactive Mode")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    if choice == "1":
        example_basic_usage()
    elif choice == "2":
        example_with_better_models()
    elif choice == "3":
        example_load_saved_vectorstore()
    elif choice == "4":
        example_custom_questions()
    elif choice == "5":
        rag = example_basic_usage()
        if rag:
            # Interactive mode
            print("\n" + "=" * 60)
            print("INTERACTIVE MODE - Ask any question about the PDF")
            print("Type 'quit' to exit")
            print("=" * 60)
            
            while True:
                question = input("\nYour question: ").strip()
                if question.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                if question:
                    rag.query(question)
    else:
        print("Invalid choice. Running basic example...")
        example_basic_usage()
