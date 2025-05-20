import os
from milvus_client import MilvusDBClient
from embedding_generator import EmbeddingGenerator
from dotenv import load_dotenv

def main():
    # Load environment variables
    load_dotenv()
    
    # Initialize clients
    milvus_client = MilvusDBClient()
    embedding_generator = EmbeddingGenerator()
    
    # Configuration
    collection_name = "constitution_of_india"
    pdf_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "constution_of_india.pdf")
    
    try:
        # Create collection if it doesn't exist
        try:
            milvus_client.create_collection(collection_name)
        except Exception as e:
            print(f"Collection might already exist: {str(e)}")
        
        # Process PDF and generate embeddings
        print("Processing PDF and generating embeddings...")
        texts, embeddings = embedding_generator.process_pdf(pdf_path)
        
        # Insert data into Zilliz Cloud
        print(f"Inserting {len(texts)} chunks into Zilliz Cloud...")
        milvus_client.insert_data(collection_name, texts, embeddings)
        
        print("Successfully processed and stored all data!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        # Close the connection
        milvus_client.close()

if __name__ == "__main__":
    main() 