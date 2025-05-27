"""Embedding generation utilities for the legal agent."""

from typing import List, Tuple
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from utils.Constants import Constants

class EmbeddingGenerator:
    """Generates embeddings for text using HuggingFace models."""

    def __init__(self):
        """Initialize the embedding generator with HuggingFace embeddings."""
        self.model_name = Constants.EMBEDDING_MODEL_NAME
        self.embeddings_model = HuggingFaceEmbeddings(model_name=self.model_name)

    def generate_embeddings(self, texts: List[str]) -> Tuple[List[str], List[np.ndarray]]:
        """Generate embeddings for a list of texts using HuggingFace embeddings.
        
        Args:
            texts: List of text strings to generate embeddings for
            
        Returns:
            Tuple containing:
            - List of original texts
            - List of numpy arrays containing the embeddings
        """
        embeddings = self.embeddings_model.embed_documents(texts)
        return texts, embeddings 