# vector_store.py
"""
Vector storage and retrieval system for the Memory Orchestration Module.
Implements embedding-based memory retrieval using FAISS.

This component provides the foundation for semantic search and
similarity-based memory retrieval across all memory layers.
"""

import os
import json
import pickle
import numpy as np
import faiss
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime

from ..utils.vector_utils import embed_text, calculate_similarity
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class VectorStore:
    """
    FAISS-based vector storage system for memory embeddings.
    Provides methods for storing, querying, and managing vector embeddings.
    """
    
    def __init__(self, db_path: str = 'memory_store', embedding_dim: int = 512):
        """
        Initialize the vector store with a FAISS index.
        
        Args:
            db_path: Directory to store vector indices and metadata
            embedding_dim: Dimension of the embedding vectors
        """
        self.db_path = db_path
        self.embedding_dim = embedding_dim
        
        # Create storage directory if it doesn't exist
        os.makedirs(db_path, exist_ok=True)
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(embedding_dim)
        
        # Memory metadata store (key -> metadata including layer info)
        self.memory_store = {}
        
        # Load existing index and metadata if available
        self._load_store()
        
        logger.info(f"Vector store initialized with dimension {embedding_dim}")
    
    def embed(self, data: Union[str, Dict, List]) -> np.ndarray:
        """
        Convert data into vector embedding.
        
        Args:
            data: Text or structured data to embed
            
        Returns:
            numpy.ndarray: Vector embedding
        """
        # Convert non-string data to string for embedding
        if not isinstance(data, str):
            data = json.dumps(data)
            
        return embed_text(data, self.embedding_dim)
    
    def store(self, key: str, embedding: np.ndarray, metadata: Dict = None) -> bool:
        """
        Store embedding in FAISS index with metadata.
        
        Args:
            key: Unique identifier for the memory
            embedding: Vector embedding
            metadata: Additional information about the memory
            
        Returns:
            bool: Success status
        """
        try:
            # Ensure embedding is the right shape
            embedding = embedding.reshape(1, -1).astype(np.float32)
            
            # Add to FAISS index
            self.index.add(embedding)
            
            # Store metadata
            self.memory_store[key] = {
                'id': len(self.memory_store),  # Internal ID for FAISS
                'metadata': metadata or {},
                'created_at': datetime.now().isoformat()
            }
            
            # Periodically save the store
            if len(self.memory_store) % 100 == 0:
                self._save_store()
                
            return True
        except Exception as e:
            logger.error(f"Error storing embedding: {str(e)}")
            return False
    
    def query(self, query_vector: np.ndarray, top_k: int = 5, 
              threshold: float = 0.0, layer: Optional[str] = None) -> List[Dict]:
        """
        Query the FAISS index to find similar embeddings.
        
        Args:
            query_vector: Vector to search for
            top_k: Maximum number of results
            threshold: Minimum similarity threshold
            layer: Optional filter by memory layer
            
        Returns:
            List of matching memory entries with similarity scores
        """
        if self.index.ntotal == 0:
            logger.warning("Vector store is empty, no results returned")
            return []
            
        # Ensure query vector is the right shape
        query_vector = query_vector.reshape(1, -1).astype(np.float32)
        
        # Search the FAISS index
        distances, indices = self.index.search(query_vector, min(top_k * 2, self.index.ntotal))
        
        results = []
        for i, idx in enumerate(indices[0]):
            # Convert FAISS distance to similarity score (1.0 is perfect match)
            similarity = 1.0 / (1.0 + distances[0][i])
            
            # Skip results below threshold
            if similarity < threshold:
                continue
                
            # Find the key for this index
            key = None
            for k, v in self.memory_store.items():
                if v['id'] == idx:
                    key = k
                    break
                    
            if key is None:
                continue
                
            memory_data = self.memory_store[key]
            
            # Filter by layer if specified
            if layer and memory_data.get('metadata', {}).get('layer') != layer:
                continue
                
            results.append({
                'key': key,
                'similarity': float(similarity),
                'metadata': memory_data.get('metadata', {}),
                'created_at': memory_data.get('created_at')
            })
            
            # Stop once we have enough results
            if len(results) >= top_k:
                break
                
        return results
    
    def delete(self, key: str) -> bool:
        """
        Delete a memory entry.
        Note: FAISS doesn't support direct deletion, so we rebuild the index.
        
        Args:
            key: Key of memory to delete
            
        Returns:
            bool: Success status
        """
        if key not in self.memory_store:
            return False
            
        # Remove from metadata store
        del self.memory_store[key]
        
        # Rebuild the index (expensive operation - consider batching deletes)
        self._rebuild_index()
        
        return True
    
    def update_metadata(self, key: str, metadata_updates: Dict) -> bool:
        """
        Update metadata for a memory entry.
        
        Args:
            key: Key of memory to update
            metadata_updates: Metadata fields to update
            
        Returns:
            bool: Success status
        """
        if key not in self.memory_store:
            return False
            
        # Update metadata
        current_metadata = self.memory_store[key].get('metadata', {})
        current_metadata.update(metadata_updates)
        self.memory_store[key]['metadata'] = current_metadata
        
        return True
    
    def count_by_layer(self, layer: str) -> int:
        """
        Count memories in a specific layer.
        
        Args:
            layer: Memory layer to count
            
        Returns:
            int: Number of memories in the layer
        """
        count = 0
        for key, memory in self.memory_store.items():
            if memory.get('metadata', {}).get('layer') == layer:
                count += 1
        return count
    
    def _save_store(self) -> None:
        """Save vector index and metadata to disk."""
        try:
            # Save FAISS index
            index_path = os.path.join(self.db_path, 'faiss_index.bin')
            faiss.write_index(self.index, index_path)
            
            # Save metadata store
            metadata_path = os.path.join(self.db_path, 'metadata.pickle')
            with open(metadata_path, 'wb') as f:
                pickle.dump(self.memory_store, f)
                
            logger.info(f"Vector store saved to {self.db_path}")
        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}")
    
    def _load_store(self) -> None:
        """Load vector index and metadata from disk if available."""
        index_path = os.path.join(self.db_path, 'faiss_index.bin')
        metadata_path = os.path.join(self.db_path, 'metadata.pickle')
        
        # Load FAISS index if exists
        if os.path.exists(index_path):
            try:
                self.index = faiss.read_index(index_path)
                logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
            except Exception as e:
                logger.error(f"Error loading FAISS index: {str(e)}")
        
        # Load metadata store if exists
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'rb') as f:
                    self.memory_store = pickle.load(f)
                logger.info(f"Loaded metadata for {len(self.memory_store)} memories")
            except Exception as e:
                logger.error(f"Error loading metadata store: {str(e)}")
    
    def _rebuild_index(self) -> None:
        """Rebuild the FAISS index from scratch."""
        # Create new index
        new_index = faiss.IndexFlatL2(self.embedding_dim)
        
        # Get all vectors and rebuild IDs
        new_store = {}
        idx = 0
        
        for key, memory_data in self.memory_store.items():
            # Implement vector lookup from your storage
            # This is a placeholder - you would need to implement how to retrieve
            # the actual vector for a given key from your storage
            
            # For example:
            # vector = retrieve_vector_for_key(key)
            # new_index.add(np.array([vector]))
            
            # Update ID in metadata
            memory_data['id'] = idx
            new_store[key] = memory_data
            idx += 1
        
        # Replace old index and store
        self.index = new_index
        self.memory_store = new_store
        
        # Save the rebuilt store
        self._save_store()
