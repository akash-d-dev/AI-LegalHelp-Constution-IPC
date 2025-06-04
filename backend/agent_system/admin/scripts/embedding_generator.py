from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
from typing import List, Tuple, Dict
import numpy as np
from dotenv import load_dotenv

class EmbeddingGenerator:
    def __init__(self):
        """Initialize the embedding generator with HuggingFace embeddings"""
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
    
    def split_pdf_into_groups(self, pdf_path: str, pages_per_group: int = 100) -> Dict[int, List[str]]:
        """Split PDF into groups of pages and return text for each group"""
        if not os.path.exists(pdf_path):
            print(f"PDF file not found at {pdf_path}, current working directory: {os.getcwd()}")
            raise FileNotFoundError(f"PDF file not found at {pdf_path}")
            
        # Load PDF using LangChain's PyPDFLoader
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        
        # Group pages into chunks of specified size
        page_groups = {}
        current_group = []
        current_group_num = 1
        
        for i, doc in enumerate(docs):
            current_group.append(doc.page_content)
            
            # If we've reached the group size or this is the last page
            if len(current_group) == pages_per_group or i == len(docs) - 1:
                page_groups[current_group_num] = current_group
                current_group = []
                current_group_num += 1
        
        return page_groups
    
    def generate_embeddings(self, texts: List[str]) -> Tuple[List[str], List[np.ndarray]]:
        """Generate embeddings for a list of texts using HuggingFace embeddings"""
        print(f"Generating embeddings for {len(texts)} texts")
        
        model_name = "sentence-transformers/all-mpnet-base-v2"
        embeddings_model = HuggingFaceEmbeddings(model_name=model_name)
        embeddings = embeddings_model.embed_documents(texts)
        return texts, embeddings
    
    def process_pdf_group(self, texts: List[str]) -> Tuple[List[str], List[np.ndarray]]:
        """Process a group of texts and generate embeddings"""
        return self.generate_embeddings(texts)

# Test the functionality
if __name__ == "__main__":
    print("Testing EmbeddingGenerator functionality")
    
    # Initialize the embedding generator
    generator = EmbeddingGenerator()
    
    # Test the process_pdf functionality
    pdf_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "constution_of_india.pdf")
    
    # Test splitting into groups
    page_groups = generator.split_pdf_into_groups(pdf_path, pages_per_group=10)
    print(f"Split PDF into {len(page_groups)} groups")
    
    # Process first group as a test
    if page_groups:
        first_group = page_groups[1]
        texts, embeddings = generator.process_pdf_group(first_group)
        print(f"Generated {len(embeddings)} embeddings for first group")
        print(f"First embedding shape: {len(embeddings[0])}")

    texts, embeddings = generator.generate_embeddings(["India, or any other instrument, treaty or agreement as envisaged under article 363 or otherwise."])
    print(texts)
    print(embeddings)