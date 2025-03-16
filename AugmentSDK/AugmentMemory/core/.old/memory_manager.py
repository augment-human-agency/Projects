"""
Memory Manager for Augment SDK.

This module implements the core memory management functionality for hierarchical memory
systems in AI applications. It provides a unified interface for storing, retrieving,
and maintaining memories across multiple cognitive layers including:

- Ephemeral Memory: Short-term, temporary storage (seconds to minutes)
- Working Memory: Mid-term retention for active tasks (minutes to hours)
- Semantic Memory: Long-term storage of facts and concepts (persistent)
- Procedural Memory: Storage for processes and workflows (persistent)
- Reflective Memory: Self-analysis and past decision tracking (persistent)
- Predictive Memory: Anticipating future knowledge needs (dynamic)

The MemoryManager orchestrates interactions between these layers to enable
recursive knowledge refinement and context-aware recall in AI systems.

Typical usage:
    memory_manager = MemoryManager(config)
    memory_manager.store_memory("concept_123", "AI safety is critical", layer="semantic")
    results = memory_manager.retrieve_memory("AI safety considerations")
"""

import logging
import time
import uuid
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Union, TypeVar, Generic, Set

# Core SDK imports
from augment_sdk.core.exceptions import MemoryStoreError, MemoryRetrievalError
from augment_sdk.memory.vector_store import VectorStore
from augment_sdk.memory.cache_manager import CacheManager
from augment_sdk.utils.config import Config

# Configure logging
logger = logging.getLogger(__name__)


class MemoryLayer(Enum):
    """Enumeration of memory layers in the hierarchical memory system."""
    EPHEMERAL = "ephemeral"  # Short-lived, temporary data
    WORKING = "working"      # Mid-term retention for active tasks
    SEMANTIC = "semantic"    # Long-term storage of facts and concepts
    PROCEDURAL = "procedural"  # Step-by-step processes and workflows
    REFLECTIVE = "reflective"  # Self-analysis and past decision tracking
    PREDICTIVE = "predictive"  # Anticipation of future responses


class MemoryPriority(Enum):
    """Priority levels for memory storage and retrieval."""
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3


class MemoryEntry:
    """
    Represents a single memory entry in the memory system.
    
    Attributes:
        key (str): Unique identifier for the memory
        data (Any): The actual content of the memory
        layer (MemoryLayer): The memory layer this entry belongs to
        priority (MemoryPriority): Priority level of this memory
        created_at (float): Timestamp when the memory was created
        accessed_at (float): Timestamp when the memory was last accessed
        access_count (int): Number of times this memory has been accessed
        metadata (Dict): Additional information about this memory
    """
    
    def __init__(
        self,
        key: str,
        data: Any,
        layer: MemoryLayer,
        priority: MemoryPriority = MemoryPriority.MEDIUM,
        metadata: Optional[Dict] = None
    ):
        """
        Initialize a new memory entry.
        
        Args:
            key: Unique identifier for the memory
            data: The actual content of the memory
            layer: The memory layer this entry belongs to
            priority: Priority level of this memory
            metadata: Additional information about this memory
        """
        self.key = key
        self.data = data
        self.layer = layer
        self.priority = priority
        self.created_at = time.time()
        self.accessed_at = self.created_at
        self.access_count = 0
        self.metadata = metadata or {}
        self.embedding_id = None  # ID in vector store if applicable
    
    def access(self) -> None:
        """Update access metadata when memory is retrieved."""
        self.accessed_at = time.time()
        self.access_count += 1
    
    def to_dict(self) -> Dict:
        """
        Convert memory entry to dictionary representation.
        
        Returns:
            Dictionary representation of the memory entry
        """
        return {
            "key": self.key,
            "data": self.data,
            "layer": self.layer.value,
            "priority": self.priority.value,
            "created_at": self.created_at,
            "accessed_at": self.accessed_at,
            "access_count": self.access_count,
            "metadata": self.metadata,
            "embedding_id": self.embedding_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "MemoryEntry":
        """
        Create a memory entry from dictionary representation.
        
        Args:
            data: Dictionary representation of a memory entry
            
        Returns:
            A new MemoryEntry instance
        """
        entry = cls(
            key=data["key"],
            data=data["data"],
            layer=MemoryLayer(data["layer"]),
            priority=MemoryPriority(data["priority"]),
            metadata=data.get("metadata", {})
        )
        entry.created_at = data.get("created_at", time.time())
        entry.accessed_at = data.get("accessed_at", entry.created_at)
        entry.access_count = data.get("access_count", 0)
        entry.embedding_id = data.get("embedding_id")
        return entry


class MemoryManager:
    """
    Core memory management class for the Augment SDK.
    
    This class orchestrates interactions between different memory layers,
    providing a unified interface for storing, retrieving, and managing
    memories across different cognitive contexts.
    
    Attributes:
        vector_store (VectorStore): Storage for vector embeddings
        cache_manager (CacheManager): Manager for ephemeral memory cache
        memory_entries (Dict[str, MemoryEntry]): Dictionary of memory entries
        config (Config): Configuration settings
    """
    
    def __init__(self, config: Config):
        """
        Initialize the memory manager.
        
        Args:
            config: Configuration settings for the memory manager
        """
        self.config = config
        self.vector_store = VectorStore(config)
        self.cache_manager = CacheManager(config)
        self.memory_entries: Dict[str, MemoryEntry] = {}
        
        # Initialize memory stores for each layer
        self._init_memory_layers()
        
        # Track relationships between memories for recursive queries
        self._memory_relationships: Dict[str, Set[str]] = {}
        
        # Load any persistent memories from vector store
        self._load_persistent_memories()
        
        logger.info("Memory manager initialized with %s configuration", config.name)
    
    def _init_memory_layers(self) -> None:
        """Initialize memory stores for each memory layer."""
        # Time-to-live for each layer in seconds
        self.layer_ttl = {
            MemoryLayer.EPHEMERAL: self.config.get('memory.ephemeral.ttl', 300),  # 5 minutes
            MemoryLayer.WORKING: self.config.get('memory.working.ttl', 3600),     # 1 hour
            MemoryLayer.SEMANTIC: self.config.get('memory.semantic.ttl', None),   # Permanent
            MemoryLayer.PROCEDURAL: self.config.get('memory.procedural.ttl', None),  # Permanent
            MemoryLayer.REFLECTIVE: self.config.get('memory.reflective.ttl', None),  # Permanent
            MemoryLayer.PREDICTIVE: self.config.get('memory.predictive.ttl', 86400)  # 1 day
        }
        
        # Maximum capacity for each layer
        self.layer_capacity = {
            MemoryLayer.EPHEMERAL: self.config.get('memory.ephemeral.capacity', 1000),
            MemoryLayer.WORKING: self.config.get('memory.working.capacity', 5000),
            MemoryLayer.SEMANTIC: self.config.get('memory.semantic.capacity', 50000),
            MemoryLayer.PROCEDURAL: self.config.get('memory.procedural.capacity', 10000),
            MemoryLayer.REFLECTIVE: self.config.get('memory.reflective.capacity', 20000),
            MemoryLayer.PREDICTIVE: self.config.get('memory.predictive.capacity', 5000)
        }
        
        # Default embedding dimensions for vector representations
        self.embedding_dimensions = self.config.get('memory.embedding.dimensions', 768)
        
        # Threshold for similarity matching
        self.similarity_threshold = self.config.get('memory.similarity.threshold', 0.7)
        
        logger.debug("Memory layers initialized with configuration: %s", self.layer_capacity)
    
    def _load_persistent_memories(self) -> None:
        """Load persistent memories from vector store on initialization."""
        try:
            # Query vector store for all stored memories
            all_memories = self.vector_store.list_all()
            count = 0
            
            for memory_data in all_memories:
                try:
                    # Extract necessary data
                    key = memory_data.get("id")
                    content = memory_data.get("content")
                    metadata = memory_data.get("metadata", {})
                    
                    # Skip if missing essential data
                    if not key or not content:
                        continue
                    
                    # Extract layer from metadata
                    layer_str = metadata.get("layer", "semantic")
                    try:
                        layer = MemoryLayer(layer_str)
                    except ValueError:
                        # Default to semantic if invalid layer
                        layer = MemoryLayer.SEMANTIC
                    
                    # Extract priority from metadata
                    priority_val = metadata.get("priority", 1)
                    try:
                        priority = MemoryPriority(priority_val)
                    except ValueError:
                        # Default to medium if invalid priority
                        priority = MemoryPriority.MEDIUM
                    
                    # Create memory entry
                    entry = MemoryEntry(
                        key=key,
                        data=content,
                        layer=layer,
                        priority=priority,
                        metadata=metadata
                    )
                    entry.embedding_id = memory_data.get("embedding_id")
                    
                    # Store in memory entries dictionary
                    self.memory_entries[key] = entry
                    count += 1
                    
                except Exception as e:
                    logger.warning("Failed to load memory %s: %s", memory_data.get("id"), str(e))
            
            logger.info("Loaded %d persistent memories from vector store", count)
        except Exception as e:
            logger.error("Failed to load persistent memories: %s", str(e))
    
    def store_memory(
        self,
        key: str,
        data: Any,
        layer: Union[str, MemoryLayer] = MemoryLayer.SEMANTIC,
        priority: Union[int, MemoryPriority] = MemoryPriority.MEDIUM,
        metadata: Optional[Dict] = None,
        related_to: Optional[List[str]] = None
    ) -> str:
        """
        Store a memory in the specified memory layer.
        
        Args:
            key: Unique identifier for the memory (auto-generated if None)
            data: The content to store
            layer: The memory layer to store in (default: semantic)
            priority: Priority level for this memory
            metadata: Additional information about this memory
            related_to: List of keys of related memories to establish connections
        
        Returns:
            The key of the stored memory
            
        Raises:
            MemoryStoreError: If the memory cannot be stored
            ValueError: If an invalid memory layer is specified
        """
        try:
            # Generate key if not provided
            if not key:
                key = f"memory_{uuid.uuid4()}"
            
            # Convert string layer to enum if needed
            if isinstance(layer, str):
                try:
                    layer = MemoryLayer(layer)
                except ValueError:
                    valid_layers = [l.value for l in MemoryLayer]
                    raise ValueError(f"Invalid memory layer: {layer}. Valid layers: {valid_layers}")
            
            # Convert int priority to enum if needed
            if isinstance(priority, int):
                try:
                    priority = MemoryPriority(priority)
                except ValueError:
                    valid_priorities = [p.value for p in MemoryPriority]
                    raise ValueError(f"Invalid priority: {priority}. Valid priorities: {valid_priorities}")
            
            # Initialize or update metadata
            if metadata is None:
                metadata = {}
                
            # Add storage timestamp to metadata
            metadata["stored_at"] = time.time()
            
            # Create memory entry
            memory_entry = MemoryEntry(
                key=key,
                data=data,
                layer=layer,
                priority=priority,
                metadata=metadata
            )
            
            # Store in appropriate layer
            if layer == MemoryLayer.EPHEMERAL:
                self.cache_manager.store(key, memory_entry.to_dict())
            else:
                # For non-ephemeral memories, store in vector store and local dictionary
                embedding_id = self._store_in_vector_db(key, data, layer, priority, metadata)
                memory_entry.embedding_id = embedding_id
                self.memory_entries[key] = memory_entry
            
            # Establish relationships if specified
            if related_to:
                self._establish_relationships(key, related_to)
            
            # Perform capacity management
            self._manage_layer_capacity(layer)
            
            logger.info("Stored memory with key '%s' in %s layer", key, layer.value)
            return key
            
        except Exception as e:
            logger.error("Failed to store memory with key '%s': %s", key, str(e))
            raise MemoryStoreError(f"Failed to store memory: {str(e)}") from e
    
    def _store_in_vector_db(
        self, 
        key: str, 
        data: Any, 
        layer: MemoryLayer,
        priority: MemoryPriority,
        metadata: Dict
    ) -> Optional[str]:
        """
        Store data in the vector database.
        
        Args:
            key: Memory key
            data: Memory data
            layer: Memory layer
            priority: Memory priority
            metadata: Memory metadata
            
        Returns:
            Embedding ID if successful, None otherwise
        """
        try:
            # Convert data to string if it's not already
            if not isinstance(data, str):
                if hasattr(data, '__str__'):
                    data_str = str(data)
                else:
                    data_str = repr(data)
            else:
                data_str = data
            
            # Prepare metadata for vector store
            vector_metadata = metadata.copy()
            vector_metadata.update({
                "layer": layer.value,
                "priority": priority.value
            })
                
            # Store in vector database
            embedding_id = self.vector_store.store(
                key, 
                data_str, 
                metadata=vector_metadata
            )
            
            return embedding_id
            
        except Exception as e:
            logger.error("Vector store error for key '%s': %s", key, str(e))
            raise MemoryStoreError(f"Vector store error: {str(e)}") from e
    
    def _establish_relationships(self, key: str, related_keys: List[str]) -> None:
        """
        Establish relationships between memories.
        
        Args:
            key: Key of the memory to establish relationships for
            related_keys: Keys of related memories
        """
        # Initialize relationship set if needed
        if key not in self._memory_relationships:
            self._memory_relationships[key] = set()
        
        # Add relationships
        for related_key in related_keys:
            if related_key in self.memory_entries or self.cache_manager.exists(related_key):
                self._memory_relationships[key].add(related_key)
                
                # Create reverse relationship
                if related_key not in self._memory_relationships:
                    self._memory_relationships[related_key] = set()
                self._memory_relationships[related_key].add(key)
                
                # Update metadata for both memories
                self._update_relationship_metadata(key, related_key)
    
    def _update_relationship_metadata(self, key1: str, key2: str) -> None:
        """
        Update metadata to reflect the relationship between two memories.
        
        Args:
            key1: First memory key
            key2: Second memory key
        """
        # Update first memory metadata
        entry1 = self.memory_entries.get(key1)
        if entry1:
            if "related_memories" not in entry1.metadata:
                entry1.metadata["related_memories"] = []
            if key2 not in entry1.metadata["related_memories"]:
                entry1.metadata["related_memories"].append(key2)
        
        # Update second memory metadata
        entry2 = self.memory_entries.get(key2)
        if entry2:
            if "related_memories" not in entry2.metadata:
                entry2.metadata["related_memories"] = []
            if key1 not in entry2.metadata["related_memories"]:
                entry2.metadata["related_memories"].append(key1)
    
    def _manage_layer_capacity(self, layer: MemoryLayer) -> None:
        """
        Manage capacity for a memory layer.
        
        Args:
            layer: The memory layer to manage
        """
        capacity = self.layer_capacity.get(layer)
        if capacity is None:
            return  # No capacity limit for this layer
        
        # Count memories in this layer
        layer_memories = [m for m in self.memory_entries.values() if m.layer == layer]
        
        if len(layer_memories) <= capacity:
            return  # Capacity not exceeded
        
        # Sort by priority (highest first), then by recency (most recent first)
        # For equal priority and recency, prefer memories with more relationships
        layer_memories.sort(
            key=lambda m: (
                m.priority.value,  # Higher priority first
                m.accessed_at,     # More recently accessed first
                len(self._memory_relationships.get(m.key, set()))  # More relationships first
            ),
            reverse=True
        )
        
        # Remove lowest priority, least recently accessed memories
        memories_to_remove = layer_memories[capacity:]
        for memory in memories_to_remove:
            self._remove_memory(memory.key)
            
        logger.debug(
            "Removed %d memories from %s layer due to capacity limit", 
            len(memories_to_remove), 
            layer.value
        )
    
    def _remove_memory(self, key: str) -> None:
        """
        Remove a memory from all storage.
        
        Args:
            key: The key of the memory to remove
        """
        try:
            # Remove from cache
            self.cache_manager.remove(key)
            
            # Remove from vector store if we have an embedding ID
            entry = self.memory_entries.get(key)
            if entry and entry.embedding_id:
                self.vector_store.remove(entry.embedding_id)
            
            # Remove from memory entries
            if key in self.memory_entries:
                del self.memory_entries[key]
            
            # Remove relationships
            if key in self._memory_relationships:
                # Get all memories related to this one
                related_memories = self._memory_relationships[key]
                
                # Remove this memory from their relationship lists
                for related_key in related_memories:
                    if related_key in self._memory_relationships:
                        self._memory_relationships[related_key].discard(key)
                
                # Remove this memory's relationships
                del self._memory_relationships[key]
                
            logger.debug("Removed memory with key '%s'", key)
        except Exception as e:
            logger.warning("Error removing memory '%s': %s", key, str(e))
    
    def retrieve_memory(
        self, 
        query: str, 
        layer: Optional[Union[str, MemoryLayer]] = None,
        limit: int = 5,
        threshold: Optional[float] = None,
        include_related: bool = True,
        recursive_depth: int = 1
    ) -> List[Dict]:
        """
        Retrieve memories matching the query.
        
        Args:
            query: The search query
            layer: Optional layer to restrict search to
            limit: Maximum number of results to return
            threshold: Similarity threshold for results (0-1)
            include_related: Whether to include related memories
            recursive_depth: How many levels of related memories to include
        
        Returns:
            List of matching memory entries
            
        Raises:
            MemoryRetrievalError: If an error occurs during retrieval
            ValueError: If an invalid memory layer is specified
        """
        try:
            # Use default threshold if not provided
            if threshold is None:
                threshold = self.similarity_threshold
                
            # Convert string layer to enum if needed
            if isinstance(layer, str):
                try:
                    layer = MemoryLayer(layer)
                except ValueError:
                    valid_layers = [l.value for l in MemoryLayer]
                    raise ValueError(f"Invalid memory layer: {layer}. Valid layers: {valid_layers}")
            
            # First check ephemeral cache for exact matches
            cache_results = self._retrieve_from_cache(query)
            
            # Then search vector store for semantic matches
            vector_results = self._retrieve_from_vector_db(
                query, 
                layer, 
                limit, 
                threshold
            )
            
            # Combine and rank results
            combined_results = self._rank_results(
                cache_results, 
                vector_results, 
                limit
            )
            
            # Optionally include related memories
            if include_related and recursive_depth > 0:
                combined_results = self._include_related_memories(
                    combined_results,
                    limit,
                    recursive_depth
                )
            
            # Update access metadata for retrieved memories
            for result in combined_results:
                key = result.get("key")
                if key in self.memory_entries:
                    self.memory_entries[key].access()
                    
                    # Update the access timestamp in the result too
                    result["accessed_at"] = self.memory_entries[key].accessed_at
                    result["access_count"] = self.memory_entries[key].access_count
            
            logger.info("Retrieved %d memories for query '%s'", len(combined_results), query)
            return combined_results
            
        except Exception as e:
            logger.error("Failed to retrieve memories for query '%s': %s", query, str(e))
            raise MemoryRetrievalError(f"Failed to retrieve memories: {str(e)}") from e
    
    def _retrieve_from_cache(self, query: str) -> List[Dict]:
        """
        Retrieve memories from the cache.
        
        Args:
            query: The search query
            
        Returns:
            List of matching memory entries from cache
        """
        try:
            # Try exact key match first
            exact_match = self.cache_manager.get(query)
            if exact_match:
                return [exact_match]
            
            # Try searching in cache
            return self.cache_manager.search(query)
        except Exception as e:
            logger.warning("Cache retrieval error: %s", str(e))
            return []
    
    def _retrieve_from_vector_db(
        self, 
        query: str, 
        layer: Optional[MemoryLayer],
        limit: int,
        threshold: float
    ) -> List[Dict]:
        """
        Retrieve memories from vector database.
        
        Args:
            query: The search query
            layer: Optional layer to restrict search to
            limit: Maximum number of results to return
            threshold: Similarity threshold for results
            
        Returns:
            List of matching memory entries from vector store
        """
        try:
            # Prepare filter for specific layer if requested
            layer_filter = None
            if layer is not None:
                layer_filter = {"layer": layer.value}
            
            # Query vector store
            results = self.vector_store.search(
                query,
                limit=limit,
                metadata_filter=layer_filter,
                threshold=threshold
            )
            
            # Enhance results with full memory entries
            enhanced_results = []
            for result in results:
                key = result.get("id")
                if key in self.memory_entries:
                    # Combine vector search result with memory entry data
                    memory_dict = self.memory_entries[key].to_dict()
                    memory_dict["score"] = result.get("score", 0.0)
                    enhanced_results.append(memory_dict)
            
            return enhanced_results
        except Exception as e:
            logger.warning("Vector retrieval error: %s", str(e))
            return []
    
    def _include_related_memories(
        self, 
        results: List[Dict], 
        limit: int,
        depth: int = 1
    ) -> List[Dict]:
        """
        Include related memories in the results.
        
        Args:
            results: Initial results from retrieval
            limit: Maximum total results to return
            depth: How many levels of related memories to include
            
        Returns:
            Extended results with related memories
        """
        if depth <= 0:
            return results
        
        all_keys = set()
        extended_results = []
        
        # First, collect all initial keys and add to results
        for result in results:
            key = result.get("key")
            if key:
                all_keys.add(key)
                extended_results.append(result)
                
        # For each depth level
        current_keys = list(all_keys)
        for level in range(depth):
            next_level_keys = set()
            
            # For each key at current level
            for key in current_keys:
                # Get related memories
                related_keys = self._memory_relationships.get(key, set())
                
                # Add any new related keys
                for related_key in related_keys:
                    if related_key not in all_keys:
                        next_level_keys.add(related_key)
                        all_keys.add(related_key)
                        
                        # Get memory entry for this related key
                        entry = self.memory_entries.get(related_key)
                        if entry:
                            # Add to results with relationship metadata
                            result_dict = entry.to_dict()
                            result_dict["related_to"] = key
                            result_dict["relation_depth"] = level + 1
                            extended_results.append(result_dict)
            
            # If we've reached the limit, stop adding more
            if len(extended_results) >= limit:
                break
                
            # Set up next level
            current_keys = list(next_level_keys)
            if not current_keys:
                break
                
        # Apply limit and return
        return extended_results[:limit]
    
    def _rank_results(
        self, 
        cache_results: List[Dict], 
        vector_results: List[Dict],
        limit: int
    ) -> List[Dict]:
        """
        Combine and rank results from different sources.
        
        Args:
            cache_results: Results from cache retrieval
            vector_results: Results from vector database retrieval
            limit: Maximum number of results to return
            
        Returns:
            Combined and ranked results list
        """
        # Convert cache results to consistent format
        formatted_cache = []
        for result in cache_results:
            if isinstance(result, dict) and "key" in result:
                # Add a perfect score for exact cache matches
                result["score"] = 1.0
                formatted_cache.append(result)
        
        # Combine results
        all_results = formatted_cache + vector_results
        
        # Remove duplicates based on key
        seen_keys = set()
        unique_results = []
        for result in all_results:
            key = result.get("key")
            if key and key not in seen_keys:
                seen_keys.add(key)
                unique_results.append(result)
        
        # Rank by score and priority
        ranked_results = sorted(
            unique_results, 
            key=lambda x: (
                x.get("score", 0.0),  # Higher score first
                x.get("priority", 1),  # Higher priority first
                x.get("accessed_at", 0)  # More recently accessed first
            ),
            reverse=True
        )
        
        # Apply limit
        return ranked_results[:limit]
    
    def get_memory(self, key: str) -> Optional[Dict]:
        """
        Get a specific memory by key.
        
        Args:
            key: The key of the memory to retrieve
            
        Returns:
            The memory entry or None if not found
        """
        # First check cache
        cache_result = self.cache_manager.get(key)
        if cache_result:
            return cache_result
        
        # Then check memory entries
        if key in self.memory_entries:
            memory_entry = self.memory_entries[key]
            memory_entry.access()  # Update access metadata
            return memory_entry.to_dict()
        
        # Not found
        logger.debug("Memory with key '%s' not found", key)
        return None
    
    def update_memory(
        self,
        key: str,
        data: Any = None,
        layer: Optional[Union[str, MemoryLayer]] = None,
        priority: Optional[Union[int, MemoryPriority]] = None,
        metadata: Optional[Dict] = None,
        related_to: Optional[List[str]] = None
    ) -> bool:
        """
        Update an existing memory.
        
        Args:
            key: The key of the memory to update
            data: New data (if None, keeps existing)
            layer: New layer (if None, keeps existing)
            priority: New priority (if None, keeps existing)
            metadata: Metadata to update or add (merges with existing)
            related_to: List of keys of memories to establish relationships with
            
        Returns:
            True if successful, False if memory not found
            
        Raises:
            MemoryStoreError: If an error occurs during update
            ValueError: If an invalid memory layer or priority is specified
        """
        try:
            # Get existing memory
            existing_dict = self.get_memory(key)
            if not existing_dict:
                logger.warning("Cannot update memory '%s': not found", key)
                return False
            
            # Parse layer if provided
            if isinstance(layer, str):
                try:
                    layer = MemoryLayer(layer)
                except ValueError:
                    valid_layers = [l.value for l in MemoryLayer]
                    raise ValueError(f"Invalid memory layer: {layer}. Valid layers: {valid_layers}")
            
            # Parse priority if provided
            if isinstance(priority, int):
                try:
                    priority = MemoryPriority(priority)
                except ValueError:
                    valid_priorities = [p.value for p in MemoryPriority]
                    raise ValueError(f"Invalid priority: {priority}. Valid priorities: {valid_priorities}")
            
            # Prepare updated memory entry
            update_data = data if data is not None else existing_dict.get("data")
            update_layer = layer if layer is not None else MemoryLayer(existing_dict.get("layer"))
            update_priority = priority if priority is not None else MemoryPriority(existing_dict.get("priority", 1))
            
            # Merge metadata if provided
            update_metadata = existing_dict.get("metadata", {}).copy()
            if metadata:
                update_metadata.update(metadata)
            
            # Add update timestamp to metadata
            update_metadata["updated_at"] = time.time()
            
            # First remove the old memory
            self._remove_memory(key)
            
            # Then store the updated memory
            self.store_memory(
                key=key,
                data=update_data,
                layer=update_layer,
                priority=update_priority,
                metadata=update_metadata,
                related_to=related_to
            )
            
            logger.info("Updated memory with key '%s'", key)
            return True
            
        except Exception as e:
            logger.error("Failed to update memory '%s': %s", key, str(e))
            raise MemoryStoreError(f"Failed to update memory: {str(e)}") from e
    
    def delete_memory(self, key: str) -> bool:
        """
        Delete a memory.
        
        Args:
            key: The key of the memory to delete
            
        Returns:
            True if memory was found and deleted, False otherwise
        """
        if key in self.memory_entries or self.cache_manager.exists(key):
            self._remove_memory(key)
            logger.info("Deleted memory with key '%s'", key)
            return True
        else:
            logger.warning("Cannot delete memory '%s': not found", key)
            return False
    
    def clear_layer(self, layer: Union[str, MemoryLayer]) -> int:
        """
        Clear all memories in a specific layer.
        
        Args:
            layer: The layer to clear
            
        Returns:
            Number of memories cleared
            
        Raises:
            ValueError: If an invalid memory layer is specified
        """
        # Convert string layer to enum if needed
        if isinstance(layer, str):
            try:
                layer = MemoryLayer(layer)
            except ValueError:
                valid_layers = [l.value for l in MemoryLayer]
                raise ValueError(f"Invalid memory layer: {layer}. Valid layers: {valid_layers}")
        
        # Clear ephemeral layer via cache manager
        if layer == MemoryLayer.EPHEMERAL:
            count = self.cache_manager.clear()
            logger.info("Cleared %d memories from ephemeral layer", count)
            return count
        
        # For other layers, find all matching memories and delete them
        keys_to_delete = [
            key for key, entry in self.memory_entries.items() 
            if entry.layer == layer
        ]
        
        for key in keys_to_delete:
            self._remove_memory(key)
        
        logger.info("Cleared %d memories from %s layer", len(keys_to_delete), layer.value)
        return len(keys_to_delete)
    
    def clean_expired_memories(self) -> int:
        """
        Remove expired memories based on TTL configuration.
        
        Returns:
            Number of memories removed
        """
        current_time = time.time()
        expired_count = 0
        
        # Check all non-permanent layers
        for layer, ttl in self.layer_ttl.items():
            if ttl is None:
                continue  # Skip permanent layers
            
            # Find expired memories in this layer
            expired_keys = [
                key for key, entry in self.memory_entries.items()
                if (entry.layer == layer and 
                    current_time - entry.accessed_at > ttl)
            ]
            
            # Remove expired memories
            for key in expired_keys:
                self._remove_memory(key)
                expired_count += 1
        
        # Also clean cache manager
        cache_expired = self.cache_manager.clean_expired()
        expired_count += cache_expired
        
        if expired_count > 0:
            logger.info("Removed %d expired memories", expired_count)
        
        return expired_count
    
    def merge_memories(
        self, 
        source_keys: List[str], 
        target_key: Optional[str] = None,
        merge_strategy: str = "concatenate"
    ) -> Optional[str]:
        """
        Merge multiple memories into a single memory.
        
        Args:
            source_keys: Keys of memories to merge
            target_key: Key for the merged memory (auto-generated if None)
            merge_strategy: Strategy for merging ("concatenate", "summarize", etc.)
            
        Returns:
            Key of the merged memory, or None if unsuccessful
            
        Raises:
            MemoryStoreError: If memories cannot be merged
        """
        try:
            # Get source memories
            source_memories = [self.get_memory(key) for key in source_keys]
            source_memories = [m for m in source_memories if m]  # Filter out None values
            
            if not source_memories:
                logger.warning("No valid source memories found for merging")
                return None
            
            # Generate target key if not provided
            if not target_key:
                target_key = f"merged_{uuid.uuid4()}"
            
            # Determine merged layer and priority
            # Use highest priority and most persistent layer from source memories
            priority_values = [MemoryPriority(m.get("priority", 1)).value for m in source_memories]
            merged_priority = MemoryPriority(max(priority_values))
            
            layer_persistence = {
                MemoryLayer.EPHEMERAL: 0,
                MemoryLayer.WORKING: 1,
                MemoryLayer.PREDICTIVE: 2,
                MemoryLayer.REFLECTIVE: 3,
                MemoryLayer.PROCEDURAL: 4,
                MemoryLayer.SEMANTIC: 5
            }
            
            layer_values = [MemoryLayer(m.get("layer")).value for m in source_memories]
            layer_scores = [layer_persistence.get(MemoryLayer(layer), 0) for layer in layer_values]
            merged_layer = MemoryLayer(layer_values[layer_scores.index(max(layer_scores))])
            
            # Merge metadata
            merged_metadata = {"source_memories": source_keys, "merge_strategy": merge_strategy}
            
            # Collect all existing metadata
            for memory in source_memories:
                metadata = memory.get("metadata", {})
                for key, value in metadata.items():
                    if key not in merged_metadata:
                        merged_metadata[key] = value
            
            # Merge data based on strategy
            if merge_strategy == "concatenate":
                merged_data = self._concatenate_memories(source_memories)
            elif merge_strategy == "summarize":
                merged_data = self._summarize_memories(source_memories)
            else:
                merged_data = self._concatenate_memories(source_memories)
                
            # Store merged memory
            self.store_memory(
                key=target_key,
                data=merged_data,
                layer=merged_layer,
                priority=merged_priority,
                metadata=merged_metadata,
                related_to=source_keys  # Establish relationships with source memories
            )
            
            logger.info(
                "Merged %d memories into new memory '%s' using strategy '%s'",
                len(source_memories),
                target_key,
                merge_strategy
            )
            
            return target_key
            
        except Exception as e:
            logger.error("Failed to merge memories: %s", str(e))
            raise MemoryStoreError(f"Failed to merge memories: {str(e)}") from e
    
    def _concatenate_memories(self, memories: List[Dict]) -> str:
        """
        Concatenate memory data.
        
        Args:
            memories: List of memory dictionaries
            
        Returns:
            Concatenated memory data
        """
        # Extract and concatenate data with separators
        data_parts = []
        for memory in memories:
            data = memory.get("data", "")
            if data:
                data_parts.append(str(data))
        
        return "\n\n".join(data_parts)
    
    def _summarize_memories(self, memories: List[Dict]) -> str:
        """
        Summarize memory data.
        
        This is a placeholder implementation. In a real system, this would use an
        LLM or other summarization technique to create a compact representation.
        
        Args:
            memories: List of memory dictionaries
            
        Returns:
            Summarized memory data
        """
        # Simple concatenation with a header for now
        # In a real implementation, this would use an LLM or other summarization method
        data_parts = []
        for memory in memories:
            data = memory.get("data", "")
            key = memory.get("key", "unknown")
            if data:
                data_parts.append(f"Memory '{key}': {str(data)}")
        
        return "Summary of merged memories:\n\n" + "\n\n".join(data_parts)
    
    def reflect_on_memories(self, query: Optional[str] = None) -> Dict:
        """
        Perform self-reflection on memories to enable recursive refinement.
        
        Args:
            query: Optional query to focus reflection on specific memories
            
        Returns:
            Dictionary with reflection results
        """
        # Get memories to reflect on (all or query-specific)
        if query:
            memories = self.retrieve_memory(query, limit=50)
        else:
            # Use a sample of memories, prioritizing more recently accessed ones
            memories = sorted(
                [entry.to_dict() for entry in self.memory_entries.values()],
                key=lambda m: m.get("accessed_at", 0),
                reverse=True
            )[:100]  # Limit to 100 most recent
        
        # Calculate basic memory statistics
        layer_counts = {}
        for memory in memories:
            layer = memory.get("layer")
            if layer:
                layer_counts[layer] = layer_counts.get(layer, 0) + 1
        
        # Identify most frequently accessed memories
        frequent_memories = sorted(
            memories,
            key=lambda m: m.get("access_count", 0),
            reverse=True
        )[:10]  # Top 10
        
        # Identify most connected memories
        connected_memories = sorted(
            memories,
            key=lambda m: len(self._memory_relationships.get(m.get("key", ""), set())),
            reverse=True
        )[:10]  # Top 10
        
        # Store reflection in reflective layer
        reflection_data = {
            "timestamp": time.time(),
            "query": query,
            "memory_count": len(memories),
            "layer_distribution": layer_counts,
            "frequent_memories": [m.get("key") for m in frequent_memories],
            "connected_memories": [m.get("key") for m in connected_memories],
        }
        
        # Store reflection as a new memory
        reflection_key = f"reflection_{uuid.uuid4()}"
        self.store_memory(
            key=reflection_key,
            data=f"Memory reflection: {reflection_data}",
            layer=MemoryLayer.REFLECTIVE,
            metadata={
                "reflection_type": "memory_analysis",
                "query": query,
                "memory_count": len(memories)
            }
        )
        
        return reflection_data
    
    def get_stats(self) -> Dict:
        """
        Get statistics about the memory system.
        
        Returns:
            Dictionary with memory statistics
        """
        stats = {
            "total_memories": len(self.memory_entries),
            "layers": {},
            "cache": self.cache_manager.get_stats(),
            "relationships": {
                "total": sum(len(related) for related in self._memory_relationships.values()),
                "most_connected": None,
                "average_connections": 0
            }
        }
        
        # Count memories per layer
        for layer in MemoryLayer:
            layer_count = sum(1 for entry in self.memory_entries.values() if entry.layer == layer)
            stats["layers"][layer.value] = {
                "count": layer_count,
                "capacity": self.layer_capacity.get(layer),
                "ttl": self.layer_ttl.get(layer)
            }
        
        # Find most connected memory
        if self._memory_relationships:
            most_connected_key = max(
                self._memory_relationships.items(),
                key=lambda item: len(item[1]),
                default=(None, set())
            )[0]
            
            if most_connected_key:
                connections = len(self._memory_relationships[most_connected_key])
                stats["relationships"]["most_connected"] = {
                    "key": most_connected_key,
                    "connections": connections
                }
        
        # Calculate average connections
        if self._memory_relationships:
            total_connections = sum(len(related) for related in self._memory_relationships.values())
            stats["relationships"]["average_connections"] = total_connections / len(self._memory_relationships)
        
        return stats