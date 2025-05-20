from pymilvus import model
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
from typing import List, Tuple
import numpy as np
from dotenv import load_dotenv

class EmbeddingGenerator:
    def __init__(self):
        """Initialize the embedding generator with Gemini model"""
        load_dotenv()
        
    def read_pdf(self, pdf_path: str) -> List[str]:
        """Read PDF and extract text using LangChain's PyPDFLoader"""
        if not os.path.exists(pdf_path):
            print(f"PDF file not found at {pdf_path}, current working directory: {os.getcwd()}")
            raise FileNotFoundError(f"PDF file not found at {pdf_path}")
            
        # Load PDF using LangChain's PyPDFLoader
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        
        # Split documents using RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            separators=None,
            keep_separator=True,
            is_separator_regex=False,
            chunk_size=500,
            chunk_overlap=0,
        )
        split_docs = splitter.split_documents(docs)
        
        # Extract text from split documents
        texts = [doc.page_content for doc in split_docs]
        return texts
    
    def generate_embeddings(self, texts: List[str]) -> Tuple[List[str], List[np.ndarray]]:
        """Generate embeddings for a list of texts using Gemini embeddings"""
        print(f"Generating embeddings for {len(texts)} texts")
        
        gemini_ef = model.dense.GeminiEmbeddingFunction(
            model_name="gemini-embedding-exp-03-07",
            api_key=os.getenv("GEMINI_API_KEY"),
        )
        
        embeddings = gemini_ef.encode_documents(texts)
        return texts, embeddings
    
    def process_pdf(self, pdf_path: str) -> Tuple[List[str], List[np.ndarray]]:
        """Process PDF and generate embeddings for all chunks"""
        # Read PDF and get text chunks
        chunks = self.read_pdf(pdf_path)
        print(f"Generated {len(chunks)} text chunks")
        
        return self.generate_embeddings(chunks)

# Test the functionality
if __name__ == "__main__":
    print("Testing EmbeddingGenerator functionality")
    
    # Initialize the embedding generator
    generator = EmbeddingGenerator()
    
    # Test the process_pdf functionality
    pdf_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "constution_of_india.pdf")
    texts, embeddings = generator.process_pdf(pdf_path)
    
    # Print the results
    print(f"Generated {len(embeddings)} embeddings")
    print(f"First embedding shape: {len(embeddings[0])}")
    
    # Print the first text and embedding
    print(f"\nFirst text: {texts[0]}")
    print(f"First embedding: {embeddings[0]}")
    
    # Print the last text and embedding
    print(f"\nLast text: {texts[-1]}")
    print(f"Last embedding: {embeddings[-1]}")


    texts, embeddings = generator.generate_embeddings(["India, or any other instrument, treaty or agreement as envisaged under article 363 or otherwise."])
    print(texts)
    print(embeddings)