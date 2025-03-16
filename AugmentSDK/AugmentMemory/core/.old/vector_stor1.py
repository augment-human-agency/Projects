"""
Vector Storage Module for Augment SDK Memory Orchestration.

This module provides a comprehensive vector-based storage system for managing 
AI memory embeddings across multiple hierarchical memory layers. It integrates with
FAISS for efficient similarity search and supports dynamic memory weighting,
serialization, and retrieval optimization for different memory types.

Key capabilities:
- Hierarchical memory management across ephemeral, working, semantic, procedural, 
  reflective, and predictive memory layers
- Efficient vector storage and retrieval using FAISS
- Dynamic memory indexing with customizable embedding dimensions
- Memory tagging and metadata enrichment
- Serialization and persistence of vector storage
- Layer-specific retrieval optimization

Example:
    ```python
    from augment_sdk.memory.components import VectorStore
    from augment_sdk.utils.config import Config
    
    # Initialize vector store with default configuration
    config = Config()
    vector_store = VectorStore(config)
    
    # Store a memory in the semantic layer
    memory_id = vector_store.store(
        data="The recursive learning process improves AI memory retention over time",
        metadata={"source": "research", "confidence": 0.92},
        layer="semantic"
    )
    
    # Retrieve similar memories
    query_embedding = vector_store.embed("AI memory improvement techniques")
    results = vector_store.query(query_embedding, top_k=5, layer="semantic")
    ```
"""

import os
import json
import faiss
import logging
import numpy as np
import pickle
from typing import Dict, List, Optional, Union, Tuple, Any, Set
from enum import Enum, auto
from datetime import datetime
from pathlib import Path

from augment_sdk.memory.utils.vector_utils import get_embedding_model, embed_text
from augment_sdk.utils.logging import get_logger
from augment_sdk.utils.exceptions import VectorStoreError, EmbeddingError, LayerNotFoundError
from augment_sdk.utils.validation import validate_embedding_dimensions

# Configure logger for the vector store module
logger = get_logger(__name__)

class MemoryLayer(Enum):
    """Enumeration of memory layers in the hierarchical memory system."""
    EPHEMERAL = auto()    # Short-lived, temporary context (seconds to minutes)
    WORKING = auto()      # Task-relevant information (minutes to hours)
    SEMANTIC = auto()     # Factual, conceptual knowledge (persistent)
    PROCEDURAL = auto()   # Process knowledge, how-to information (persistent)
    REFLECTIVE = auto()   # Meta-cognitive insights, self-analysis (persistent)
    PREDICTIVE = auto()   # Forward-looking insights, future predictions (persistent)

class MemoryEntry:
    """Representation of a single memory entry in the vector store."""
    
    def __init__(
        self, 
        id: str, 
        embedding: np.ndarray, 
        data: Any, 
        layer: Union[MemoryLayer, str],
        metadata: Optional[Dict[str, Any]] = None,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None
    ):
        """
        Initialize a memory entry.
        
        Args:
            id: Unique identifier for the memory
            embedding: Vector representation of the memory
            data: Raw data content of the memory
            layer: The memory layer this entry belongs to
            metadata: Additional information about the memory
            created_at: Timestamp when memory was first created
            updated_at: Timestamp when memory was last updated
        """
        self.id = id
        self.embedding = embedding
        self.data = data
        
        # Convert string layer names to MemoryLayer enum
        if isinstance(layer, str):
            try:
                self.layer = MemoryLayer[layer.upper()]
            except KeyError:
                raise LayerNotFoundError(f"Invalid memory layer: {layer}")
        else:
            self.layer = layer
            
        self.metadata = metadata or {}
        self.created_at = created_at or datetime.now()
        self.updated_at = updated_at or datetime.now()
        
    def update(self, data: Optional[Any] = None, metadata: Optional[Dict[str, Any]] = None):
        """
        Update the memory entry with new data or metadata.
        
        Args:
            data: New data to replace the current data
            metadata: New metadata to merge with existing metadata
        """
        if data is not None:
            self.data = data
            
        if metadata is not None:
            self.metadata.update(metadata)
            
        self.updated_at = datetime.now()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert the memory entry to a dictionary representation."""
        return {
            "id": self.id,
            "data": self.data,
            "layer": self.layer.name,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any], embedding: np.ndarray) -> 'MemoryEntry':
        """
        Create a memory entry from a dictionary and embedding.
        
        Args:
            data: Dictionary containing memory entry data
            embedding: Vector embedding for the memory
            
        Returns:
            A new MemoryEntry object
        """
        return cls(
            id=data["id"],
            embedding=embedding,
            data=data["data"],
            layer=data["layer"],
            metadata=data["metadata"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"])
        )


class VectorStore:
    """
    Vector-based storage system for managing AI memory embeddings.
    
    Provides hierarchical memory management with FAISS integration for efficient 
    similarity search. Supports multiple memory layers with customizable embedding
    dimensions and persistence options.
    """
    
    def __init__(
        self, 
        config,
        embedding_dim: int = 768,
        storage_path: Optional[str] = None,
        load_existing: bool = True
    ):
        """
        Initialize the vector store.
        
        Args:
            config: Configuration object with settings for the vector store
            embedding_dim: Dimension of the embedding vectors
            storage_path: Path to store vector indices and memory data
            load_existing: Whether to load existing vector store data if available
            
        Raises:
            VectorStoreError: If there's an error initializing the vector store
        """
        self.config = config
        self.embedding_dim = embedding_dim
        self.storage_path = storage_path or config.get("VECTOR_STORE_PATH", "data/vector_store")
        
        # Initialize embedding model
        try:
            self.embedding_model = get_embedding_model(config)
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {str(e)}")
            raise VectorStoreError(f"Embedding model initialization failed: {str(e)}")
        
        # Initialize indices for each memory layer
        self.indices = {}
        self.memory_entries = {}
        self.id_to_index = {}  # Maps memory IDs to FAISS indices
        
        # Initialize layer-specific data structures
        for layer in MemoryLayer:
            self._init_layer(layer)
        
        # Load existing data if requested
        if load_existing:
            self._load_from_disk()
            
        logger.info(f"VectorStore initialized with embedding dimension {embedding_dim}")
    
    def _init_layer(self, layer: MemoryLayer):
        """
        Initialize data structures for a specific memory layer.
        
        Args:
            layer: The memory layer to initialize
        """
        # Create a flat L2 index for this layer
        self.indices[layer] = faiss.IndexFlatL2(self.embedding_dim)
        
        # Initialize empty collections for this layer
        if layer not in self.memory_entries:
            self.memory_entries[layer] = {}
            self.id_to_index[layer] = {}
            
        logger.debug(f"Initialized vector store layer: {layer.name}")
    
    def embed(self, text: str) -> np.ndarray:
        """
        Generate an embedding vector for the given text.
        
        Args:
            text: Text to embed
            
        Returns:
            Numpy array containing the embedding vector
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        try:
            embedding = embed_text(text, self.embedding_model)
            
            # Validate embedding dimensions
            validate_embedding_dimensions(embedding, self.embedding_dim)
            
            return embedding
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            raise EmbeddingError(f"Failed to generate embedding: {str(e)}")
    
    def store(
        self, 
        data: Any, 
        metadata: Optional[Dict[str, Any]] = None,
        layer: Union[MemoryLayer, str] = MemoryLayer.SEMANTIC,
        custom_id: Optional[str] = None,
        embedding: Optional[np.ndarray] = None
    ) -> str:
        """
        Store a memory entry in the vector store.
        
        Args:
            data: Data to store (typically text, but can be any serializable data)
            metadata: Additional metadata to store with the memory
            layer: Memory layer to store in (default: SEMANTIC)
            custom_id: Optional custom ID for the memory entry
            embedding: Optional pre-computed embedding vector
            
        Returns:
            ID of the stored memory entry
            
        Raises:
            VectorStoreError: If the memory storage operation fails
            LayerNotFoundError: If the specified layer is invalid
        """
        # Convert string layer to enum if needed
        if isinstance(layer, str):
            try:
                layer = MemoryLayer[layer.upper()]
            except KeyError:
                raise LayerNotFoundError(f"Invalid memory layer: {layer}")
        
        # Generate embedding if not provided
        if embedding is None:
            if isinstance(data, str):
                embedding = self.embed(data)
            else:
                # For non-string data, we need a text representation to embed
                text_repr = str(data)
                embedding = self.embed(text_repr)
        
        # Ensure embedding is in the correct format for FAISS
        embedding_vector = np.array(embedding).astype('float32').reshape(1, -1)
        
        # Generate a unique ID if not provided
        memory_id = custom_id or f"{layer.name.lower()}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{len(self.memory_entries[layer])}"
        
        # Create memory entry
        memory_entry = MemoryEntry(
            id=memory_id,
            embedding=embedding,
            data=data,
            layer=layer,
            metadata=metadata
        )
        
        # Get the current index size to track where this entry will be added
        current_index = self.indices[layer].ntotal
        
        # Add to FAISS index
        try:
            self.indices[layer].add(embedding_vector)
            self.memory_entries[layer][memory_id] = memory_entry
            self.id_to_index[layer][memory_id] = current_index
            logger.debug(f"Stored memory entry {memory_id} in layer {layer.name}")
        except Exception as e:
            logger.error(f"Failed to store memory in layer {layer.name}: {str(e)}")
            raise VectorStoreError(f"Memory storage failed: {str(e)}")
        
        return memory_id
    
    def batch_store(
        self, 
        items: List[Dict[str, Any]],
        layer: Union[MemoryLayer, str] = MemoryLayer.SEMANTIC
    ) -> List[str]:
        """
        Store multiple memory entries in batch.
        
        Args:
            items: List of dictionaries with keys 'data', optional 'metadata',
                  optional 'embedding', and optional 'custom_id'
            layer: Memory layer to store all items in
            
        Returns:
            List of memory IDs for the stored entries
        """
        # Convert string layer to enum if needed
        if isinstance(layer, str):
            try:
                layer = MemoryLayer[layer.upper()]
            except KeyError:
                raise LayerNotFoundError(f"Invalid memory layer: {layer}")
                
        # Process all embeddings first
        embeddings = []
        for item in items:
            if 'embedding' in item and item['embedding'] is not None:
                embeddings.append(item['embedding'])
            else:
                if isinstance(item['data'], str):
                    embeddings.append(self.embed(item['data']))
                else:
                    embeddings.append(self.embed(str(item['data'])))
        
        # Prepare a batch of embeddings for FAISS
        embeddings_array = np.array(embeddings).astype('float32')
        
        # Store all items
        memory_ids = []
        current_index = self.indices[layer].ntotal
        
        try:
            # Add all embeddings to FAISS at once
            self.indices[layer].add(embeddings_array)
            
            # Now create individual memory entries
            for i, item in enumerate(items):
                memory_id = item.get('custom_id') or f"{layer.name.lower()}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{current_index + i}"
                memory_entry = MemoryEntry(
                    id=memory_id,
                    embedding=embeddings[i],
                    data=item['data'],
                    layer=layer,
                    metadata=item.get('metadata')
                )
                
                self.memory_entries[layer][memory_id] = memory_entry
                self.id_to_index[layer][memory_id] = current_index + i
                memory_ids.append(memory_id)
            
            logger.debug(f"Batch stored {len(items)} memory entries in layer {layer.name}")
        except Exception as e:
            logger.error(f"Failed to batch store memories in layer {layer.name}: {str(e)}")
            raise VectorStoreError(f"Batch memory storage failed: {str(e)}")
            
        return memory_ids
    
    def query(
        self, 
        query_vector: np.ndarray,
        top_k: int = 5,
        layer: Union[MemoryLayer, str] = MemoryLayer.SEMANTIC,
        filter_fn: Optional[callable] = None,
        min_similarity: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Query the vector store for similar memories.
        
        Args:
            query_vector: Vector to find similar memories for
            top_k: Number of results to return
            layer: Memory layer to search in
            filter_fn: Optional function to filter results
            min_similarity: Minimum similarity threshold (0.0 to 1.0)
            
        Returns:
            List of dictionaries containing memory entries and their similarity scores
            
        Raises:
            LayerNotFoundError: If the specified layer is invalid
            VectorStoreError: If the query operation fails
        """
        # Convert string layer to enum if needed
        if isinstance(layer, str):
            try:
                layer = MemoryLayer[layer.upper()]
            except KeyError:
                raise LayerNotFoundError(f"Invalid memory layer: {layer}")
        
        # Ensure query vector is in the correct format
        query_vector = np.array(query_vector).astype('float32').reshape(1, -1)
        
        try:
            # Search the index for similar vectors
            distances, indices = self.indices[layer].search(query_vector, min(top_k * 2, self.indices[layer].ntotal))
            
            # Convert FAISS distances (L2 distance) to similarity scores (0-1)
            # Higher is better for similarity, so we use a simple conversion formula
            max_distance = 30.0  # Heuristic for max distance (adjust based on your embeddings)
            similarities = [max(0.0, 1.0 - (dist / max_distance)) for dist in distances[0]]
            
            # Filter by minimum similarity
            filtered_results = []
            for idx, similarity in zip(indices[0], similarities):
                if similarity < min_similarity:
                    continue
                    
                # Find the memory ID for this index
                memory_id = None
                for mid, index in self.id_to_index[layer].items():
                    if index == idx:
                        memory_id = mid
                        break
                
                if memory_id is None:
                    logger.warning(f"Could not find memory ID for index {idx} in layer {layer.name}")
                    continue
                    
                memory_entry = self.memory_entries[layer].get(memory_id)
                if memory_entry is None:
                    logger.warning(f"Memory entry {memory_id} not found in layer {layer.name}")
                    continue
                
                result = {
                    "id": memory_id,
                    "data": memory_entry.data,
                    "metadata": memory_entry.metadata,
                    "similarity": similarity,
                    "layer": layer.name,
                    "created_at": memory_entry.created_at,
                    "updated_at": memory_entry.updated_at
                }
                
                # Apply custom filter if provided
                if filter_fn is None or filter_fn(result):
                    filtered_results.append(result)
                    
            # Return the top K results after filtering
            return filtered_results[:top_k]
            
        except Exception as e:
            logger.error(f"Query operation failed in layer {layer.name}: {str(e)}")
            raise VectorStoreError(f"Memory query failed: {str(e)}")
    
    def query_by_text(
        self, 
        query_text: str,
        top_k: int = 5,
        layer: Union[MemoryLayer, str] = MemoryLayer.SEMANTIC,
        filter_fn: Optional[callable] = None,
        min_similarity: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Query the vector store using natural language text.
        
        Args:
            query_text: Text to search for
            top_k: Number of results to return
            layer: Memory layer to search in
            filter_fn: Optional function to filter results
            min_similarity: Minimum similarity threshold (0.0 to 1.0)
            
        Returns:
            List of dictionaries containing memory entries and their similarity scores
        """
        # Generate embedding for the query text
        query_vector = self.embed(query_text)
        
        # Use the vector query method
        return self.query(
            query_vector=query_vector,
            top_k=top_k,
            layer=layer,
            filter_fn=filter_fn,
            min_similarity=min_similarity
        )
    
    def multi_layer_query(
        self, 
        query_vector: np.ndarray,
        top_k: int = 5,
        layers: Optional[List[Union[MemoryLayer, str]]] = None,
        layer_weights: Optional[Dict[Union[MemoryLayer, str], float]] = None,
        filter_fn: Optional[callable] = None,
        min_similarity: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Query multiple memory layers and merge results.
        
        Args:
            query_vector: Vector to find similar memories for
            top_k: Number of total results to return
            layers: List of layers to query (defaults to all layers)
            layer_weights: Dictionary mapping layers to importance weights
            filter_fn: Optional function to filter results
            min_similarity: Minimum similarity threshold (0.0 to 1.0)
            
        Returns:
            List of dictionaries containing memory entries and their similarity scores,
            merged and ranked across specified layers
        """
        # Default to all layers if none specified
        if layers is None:
            layers = list(MemoryLayer)
        
        # Convert string layers to enums if needed
        parsed_layers = []
        for layer in layers:
            if isinstance(layer, str):
                try:
                    parsed_layers.append(MemoryLayer[layer.upper()])
                except KeyError:
                    raise LayerNotFoundError(f"Invalid memory layer: {layer}")
            else:
                parsed_layers.append(layer)
        
        # Default weights are equal across all specified layers
        if layer_weights is None:
            layer_weights = {layer: 1.0 for layer in parsed_layers}
        else:
            # Convert string keys to enums in layer_weights if needed
            normalized_weights = {}
            for layer, weight in layer_weights.items():
                if isinstance(layer, str):
                    try:
                        normalized_weights[MemoryLayer[layer.upper()]] = weight
                    except KeyError:
                        raise LayerNotFoundError(f"Invalid memory layer in weights: {layer}")
                else:
                    normalized_weights[layer] = weight
            layer_weights = normalized_weights
        
        # Query each layer
        all_results = []
        for layer in parsed_layers:
            # Skip layers that don't have weight specified
            if layer not in layer_weights:
                continue
                
            layer_results = self.query(
                query_vector=query_vector,
                top_k=top_k,  # Get top_k from each layer
                layer=layer,
                filter_fn=filter_fn,
                min_similarity=min_similarity
            )
            
            # Apply layer weight to similarity scores
            layer_weight = layer_weights.get(layer, 1.0)
            for result in layer_results:
                result["weighted_similarity"] = result["similarity"] * layer_weight
                all_results.append(result)
        
        # Sort by weighted similarity and return top_k overall
        all_results.sort(key=lambda x: x["weighted_similarity"], reverse=True)
        return all_results[:top_k]
    
    def get_by_id(self, memory_id: str, layer: Optional[Union[MemoryLayer, str]] = None) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific memory entry by ID.
        
        Args:
            memory_id: ID of the memory to retrieve
            layer: Optional layer to search in (searches all layers if None)
            
        Returns:
            Dictionary containing the memory entry or None if not found
        """
        # If layer is specified, only search that layer
        if layer is not None:
            # Convert string layer to enum if needed
            if isinstance(layer, str):
                try:
                    layer = MemoryLayer[layer.upper()]
                except KeyError:
                    raise LayerNotFoundError(f"Invalid memory layer: {layer}")
            
            memory_entry = self.memory_entries[layer].get(memory_id)
            if memory_entry is None:
                return None
                
            return {
                "id": memory_id,
                "data": memory_entry.data,
                "metadata": memory_entry.metadata,
                "layer": layer.name,
                "created_at": memory_entry.created_at,
                "updated_at": memory_entry.updated_at
            }
        
        # Search all layers
        for layer_enum in MemoryLayer:
            memory_entry = self.memory_entries[layer_enum].get(memory_id)
            if memory_entry is not None:
                return {
                    "id": memory_id,
                    "data": memory_entry.data,
                    "metadata": memory_entry.metadata,
                    "layer": layer_enum.name,
                    "created_at": memory_entry.created_at,
                    "updated_at": memory_entry.updated_at
                }
        
        return None
    
    def update(
        self, 
        memory_id: str,
        data: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[np.ndarray] = None,
        layer: Optional[Union[MemoryLayer, str]] = None
    ) -> bool:
        """
        Update an existing memory entry.
        
        Args:
            memory_id: ID of the memory to update
            data: New data to replace the current data
            metadata: New metadata to merge with existing metadata
            embedding: New embedding vector
            layer: Layer where the memory is stored (searches all if None)
            
        Returns:
            True if the update was successful, False if the memory wasn't found
            
        Raises:
            VectorStoreError: If the update operation fails
            LayerNotFoundError: If the specified layer is invalid
        """
        # Convert string layer to enum if needed
        if isinstance(layer, str):
            try:
                layer = MemoryLayer[layer.upper()]
            except KeyError:
                raise LayerNotFoundError(f"Invalid memory layer: {layer}")
        
        # Find the memory entry
        target_layer = None
        memory_entry = None
        
        if layer is not None:
            memory_entry = self.memory_entries[layer].get(memory_id)
            target_layer = layer
        else:
            # Search all layers
            for layer_enum in MemoryLayer:
                if memory_id in self.memory_entries[layer_enum]:
                    memory_entry = self.memory_entries[layer_enum][memory_id]
                    target_layer = layer_enum
                    break
        
        if memory_entry is None:
            logger.warning(f"Memory entry {memory_id} not found for update")
            return False
        
        try:
            # Update memory entry data and metadata
            memory_entry.update(data, metadata)
            
            # If embedding is provided, update it in the FAISS index
            if embedding is not None:
                # Ensure embedding is in the correct format
                embedding_vector = np.array(embedding).astype('float32').reshape(1, -1)
                
                # Get the index in the FAISS database
                index_pos = self.id_to_index[target_layer][memory_id]
                
                # Since FAISS doesn't support direct updates, we need to rebuild the index
                # This is a simplified approach - in production, consider more efficient methods
                # Extract all vectors
                all_ids = list(self.id_to_index[target_layer].keys())
                all_indices = list(self.id_to_index[target_layer].values())
                
                # Sort by index to maintain order
                sorted_pairs = sorted(zip(all_ids, all_indices), key=lambda pair: pair[1])
                sorted_ids = [pair[0] for pair in sorted_pairs]
                
                # Collect all embeddings
                embeddings = []
                for mid in sorted_ids:
                    if mid == memory_id:
                        # Use the new embedding for the updated entry
                        embeddings.append(embedding_vector[0])
                        # Update the stored embedding
                        memory_entry.embedding = embedding
                    else:
                        embeddings.append(self.memory_entries[target_layer][mid].embedding)
                
                # Recreate the index
                embeddings_array = np.array(embeddings).astype('float32')
                self.indices[target_layer] = faiss.IndexFlatL2(self.embedding_dim)
                self.indices[target_layer].add(embeddings_array)
            
            logger.debug(f"Updated memory entry {memory_id} in layer {target_layer.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update memory {memory_id}: {str(e)}")
            raise VectorStoreError(f"Memory update failed: {str(e)}")
    
    def delete(self, memory_id: str, layer: Optional[Union[MemoryLayer, str]] = None) -> bool:
        """
        Delete a memory entry.
        
        Args:
            memory_id: ID of the memory to delete
            layer: Layer where the memory is stored (searches all if None)
            
        Returns:
            True if the deletion was successful, False if the memory wasn't found
            
        Raises:
            VectorStoreError: If the deletion operation fails
            LayerNotFoundError: If the specified layer is invalid
        """
        # Convert string layer to enum if needed
        if isinstance(layer, str):
            try:
                layer = MemoryLayer[layer.upper()]
            except KeyError:
                raise LayerNotFoundError(f"Invalid memory layer: {layer}")
        
        # Find the memory entry
        target_layer = None
        
        if layer is not None:
            if memory_id in self.memory_entries[layer]:
                target_layer = layer
        else:
            # Search all layers
            for layer_enum in MemoryLayer:
                if memory_id in self.memory_entries[layer_enum]:
                    target_layer = layer_enum
                    break
        
        if target_layer is None:
            logger.warning(f"Memory entry {memory_id} not found for deletion")
            return False
        
        try:
            # Since FAISS doesn't support direct deletion, we need to rebuild the index
            # without the deleted vector
            
            # Get the index position to delete
            index_to_delete = self.id_to_index[target_layer][memory_id]
            
            # Extract all vectors except the one to delete
            all_ids = []
            all_embeddings = []
            
            # Update ID-to-index mapping
            new_id_to_index = {}
            new_index = 0
            
            for mid, idx in self.id_to_index[target_layer].items():
                if mid != memory_id:
                    all_ids.append(mid)
                    all_embeddings.append(self.memory_entries[target_layer][mid].embedding)
                    new_id_to_index[mid] = new_index
                    new_index += 1
            
            # Recreate the index if there are any remaining vectors
            if all_embeddings:
                embeddings_array = np.array(all_embeddings).astype('float32')
                self.indices[target_layer] = faiss.IndexFlatL2(self.embedding_dim)
                self.indices[target_layer].add(embeddings_array)
            else:
                # If no vectors remain, just reinitialize an empty index
                self.indices[target_layer] = faiss.IndexFlatL2(self.embedding_dim)
            
            # Update the ID-to-index mapping
            self.id_to_index[target_layer] = new_id_to_index
            
            # Remove the memory entry
            del self.memory_entries[target_layer][memory_id]
            
            logger.debug(f"Deleted memory entry {memory_id} from layer {target_layer.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete memory {memory_id}: {str(e)}")
            raise VectorStoreError(f"Memory deletion failed: {str(e)}")
    
    def clear_layer(self, layer: Union[MemoryLayer, str]):
        """
        Clear all memories from a specific layer.
        
        Args:
            layer: Layer to clear
            
        Raises:
            LayerNotFoundError: If the specified layer is invalid
        """
        # Convert string layer to enum if needed
        if isinstance(layer, str):
            try:
                layer = MemoryLayer[layer.upper()]
            except KeyError:
                raise LayerNotFoundError(f"Invalid memory layer: {layer}")
        
        # Reinitialize the layer
        self._init_layer(layer)
        logger.info(f"Cleared all memories from layer {layer.name}")
    
    def clear_all(self):
        """Clear all memories from all layers."""
        