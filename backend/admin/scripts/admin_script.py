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
    base_collection_name = "constitution_of_india"
    pdf_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "constution_of_india.pdf")
    pages_per_group = 85
    
    try:
        # Split PDF into groups
        print("Splitting PDF into groups...")
        page_groups = embedding_generator.split_pdf_into_groups(pdf_path, pages_per_group)

        # with open("./result.txt", "w") as f:
        #     f.write(str(page_groups))
        
        print(f"Split PDF into {len(page_groups)} groups")
        
        # Process each group
        for group_num, texts in page_groups.items():
            collection_name = f"{base_collection_name}_{group_num}"
            print(f"\nProcessing group {group_num}...")
            
            try:
                # Create collection for this group
                print(f"Creating collection: {collection_name}")
                milvus_client.create_collection(collection_name)
                
                # Generate embeddings for this group
                print(f"Generating embeddings for group {group_num}...")
                texts, embeddings = embedding_generator.process_pdf_group(texts)
                
                # Insert data into collection
                print(f"Inserting {len(texts)} chunks into collection {collection_name}...")
                milvus_client.insert_data(collection_name, texts, embeddings)
                
                print(f"Successfully processed and stored data for group {group_num}!")
                
            except Exception as e:
                print(f"Error processing group {group_num}: {str(e)}")
                continue
        
        print("\nAll groups processed!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        # Close the connection
        milvus_client.close()

if __name__ == "__main__":
    main() 