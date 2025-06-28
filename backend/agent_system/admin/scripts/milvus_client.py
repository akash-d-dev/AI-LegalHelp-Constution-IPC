from pymilvus import MilvusClient, DataType
import os
from dotenv import load_dotenv
import numpy as np

class MilvusDBClient:
    def __init__(self, uri: str | None = None, token: str | None = None):
        """Initialize the client with optional URI and token.

        If ``uri`` or ``token`` are not provided, values are read from the
        environment variables ``MILVUS_URI`` and ``MILVUS_TOKEN`` respectively.
        """
        load_dotenv()
        self.uri = uri or os.getenv("MILVUS_URI_DB_IPC")
        self.token = token or os.getenv("MILVUS_TOKEN_DB_IPC")

    def _get_client(self):
        """Create a new client connection"""
        return MilvusClient(
            uri=self.uri if self.uri else "",
            token=self.token if self.token else ""
        )

    def create_collection(self, collection_name, dimension=768):
        """Create a new collection with specified dimension"""
        try:
            client = self._get_client()
            
            # Create schema
            schema = MilvusClient.create_schema(
                auto_id=True,
                enable_dynamic_field=True
            )
            
            # Add fields to schema
            schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
            schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
            schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=dimension)
            
            # Create collection
            client.create_collection(
                collection_name=collection_name,
                schema=schema
            )
            
            # Prepare index parameters
            index_params = client.prepare_index_params()
            
            # Add index for vector field
            index_params.add_index(
                field_name="embedding",
                index_type="AUTOINDEX",
                metric_type="L2",
                params={"nlist": 1024}
            )
            
            # Create indexes
            client.create_index(
                collection_name=collection_name,
                index_params=index_params
            )
            
            # List indexes to verify
            indexes = client.list_indexes(collection_name=collection_name)
            print(f"Created indexes: {indexes}")
            print(f"Successfully created collection: {collection_name}")
        except Exception as e:
            print(f"Failed to create collection: {str(e)}")
            raise

    def delete_collection(self, collection_name):
        """Delete a collection"""
        try:
            client = self._get_client()
            client.drop_collection(collection_name)
            print(f"Successfully deleted collection: {collection_name}")
        except Exception as e:
            print(f"Failed to delete collection: {str(e)}")
            raise

    def insert_data(self, collection_name, texts, embeddings):
        """Insert data into collection"""
        try:
            client = self._get_client()
            # Create a list of dictionaries for insertion
            entities = [
                {"text": text, "embedding": embedding}
                for text, embedding in zip(texts, embeddings)
            ]
            
            result = client.insert(
                collection_name=collection_name,
                data=entities
            )
            print(f"Successfully inserted {result['insert_count']} records")
            return result
        except Exception as e:
            print(f"Failed to insert data: {str(e)}")
            raise

    def search_similar(self, collection_name, query_embedding, top_k=5):
        """Search for similar vectors"""
        try:
            client = self._get_client()
            
            # Load the collection before searching
            client.load_collection(collection_name)
            
            results = client.search(
                collection_name=collection_name,
                anns_field="embedding",
                data=[query_embedding],
                limit=top_k,
                search_params={"metric_type": "L2"}
            )
            
            # Process and return results in a more structured format
            processed_results = []
            for hits in results:
                hit_list = []
                for hit in hits:
                    hit_list.append({
                        "id": hit.get('id'),
                        "distance": hit.get('distance'),
                        "entity": hit.get('entity', {})
                    })
                processed_results.append(hit_list)
            
            return processed_results
        except Exception as e:
            print(f"Failed to search: {str(e)}")
            raise

    def close(self):
        """Close the client connection"""
        try:
            client = self._get_client()
            client.close()
            print("Successfully closed client connection")
        except Exception as e:
            print(f"Failed to close client connection: {str(e)}")
            raise

########################################################
# Test the MilvusDBClient functionality
########################################################
if __name__ == "__main__":
    # Test the MilvusDBClient functionality
    print("\n=== Testing MilvusDBClient ===")
    
    # Initialize client
    client = MilvusDBClient()
    
    # Test collection name
    test_collection = "test_collection"
    
    try:
        # Create a test collection
        print("\n1. Creating test collection...")
        client.create_collection(test_collection, dimension=768)
        print("✓ Collection created successfully")
        
        # Generate some test data
        print("\n2. Preparing test data...")
        test_texts = [
            "The Constitution of India is the supreme law of India.",
            "It lays down the framework defining fundamental political principles.",
            "It establishes the structure, procedures, powers and duties of government institutions."
        ]
        
        # Generate random embeddings for testing (768 dimensions)
        test_embeddings = [np.random.rand(768).tolist() for _ in range(len(test_texts))]
        
        # Insert test data
        print("\n3. Inserting test data...")
        result = client.insert_data(test_collection, test_texts, test_embeddings)
        print("✓ Data inserted successfully")
        
        # Test search functionality
        print("\n4. Testing search functionality...")
        query_embedding = test_embeddings[0]
        results = client.search_similar(test_collection, query_embedding, top_k=2)
        print("✓ Search completed successfully")
        print("Search results:")
        print(results)
        for hits in results:
            for hit in hits:
                print(f"ID: {hit['id']}, Distance: {hit['distance']}, Entity: {hit['entity']}")
        
        # Clean up - delete test collection
        print("\n5. Cleaning up - deleting test collection...")
        client.delete_collection(test_collection)
        print("✓ Test collection deleted successfully")
        
        print("\n=== All tests completed successfully! ===")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
    finally:
        # Close the client connection
        client.close() 
        
        
