# memory_manager.py
from typing import Dict, List, Any, Optional
from .vector_store import VectorStore
from .memory_retrieval import MemoryRetrieval
from .meta_cognition import MetaCognition
from .memory_decay import MemoryDecay
from .cache_manager import CacheManager
from ..utils.logger import get_logger

logger = get_logger(__name__)

class MemoryManager:
    """
    Core controller for memory operations in the AugmentSDK.
    
    The MemoryManager orchestrates all memory-related functionality,
    including storage, retrieval, reflection, and memory maintenance.
    It serves as the primary interface for interacting with the memory system.
    
    Attributes:
        vector_store: Handles embedding-based storage and retrieval
        memory_retrieval: Specialized component for memory querying
        meta_cognition: Handles self-reflection and memory weighting
        memory_decay: Manages memory aging and pruning
        cache_manager: Manages ephemeral (short-term) memory
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Memory Manager with configuration.
        
        Args:
            config: Configuration dictionary containing paths and settings
        """
        logger.info("Initializing Memory Manager")
        self.config = config
        self.vector_store = VectorStore(
            db_path=config.get('VECTOR_DB_PATH', './vector_db'),
            embedding_dim=config.get('EMBEDDING_DIM', 512)
        )
        self.memory_retrieval = MemoryRetrieval(self.vector_store)
        self.meta_cognition = MetaCognition()
        self.memory_decay = MemoryDecay(
            decay_rate=config.get('MEMORY_DECAY_RATE', 0.05),
            threshold=config.get('MEMORY_DECAY_THRESHOLD', 0.3)
        )
        self.cache_manager = CacheManager(
            ttl=config.get('CACHE_TTL', 3600),
            max_size=config.get('CACHE_MAX_SIZE', 1000)
        )
        logger.info("Memory Manager initialized successfully")

    def store_memory(self, key: str, data: Any, layer: str = 'semantic', metadata: Optional[Dict] = None) -> bool:
        """
        Store memory with optional meta-cognition weighting.
        
        Args:
            key: Unique identifier for the memory
            data: Content to store (text, embedding, or structured data)
            layer: Memory layer ('ephemeral', 'working', 'semantic', 'procedural', 'reflective')
            metadata: Additional information about the memory
            
        Returns:
            bool: Success status
        """
        logger.debug(f"Storing memory: {key} in layer: {layer}")
        
        # Handle ephemeral memory differently
        if layer == 'ephemeral':
            return self.cache_manager.store(key, data, metadata)
            
        # For persistent memory layers
        try:
            # Get embedding from text or use provided embedding
            if isinstance(data, str):
                embedding = self.vector_store.embed(data)
            else:
                embedding = data
                
            # Store in vector database
            self.vector_store.store(key, embedding, layer, metadata)
            
            # Evaluate memory for meta-cognitive weighting
            self.meta_cognition.evaluate_memory(key, data)
            
            logger.info(f"Successfully stored memory: {key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store memory: {key}. Error: {str(e)}")
            return False

    def retrieve_memory(self, query: str, layer: str = 'semantic', top_k: int = 5) -> List[Dict]:
        """
        Retrieve memory using vector similarity or direct key lookup.
        
        Args:
            query: Search query or memory key
            layer: Memory layer to search in
            top_k: Number of results to return
            
        Returns:
            List of memory items with content and metadata
        """
        logger.debug(f"Retrieving memory with query: {query}, layer: {layer}")
        
        # Check cache first for ephemeral memories
        if layer == 'ephemeral':
            cache_result = self.cache_manager.retrieve(query)
            if cache_result:
                return [cache_result]
        
        # Use memory retrieval for vector search
        results = self.memory_retrieval.query_memory(query, layer, top_k)
        
        # Log retrieval metrics
        logger.info(f"Retrieved {len(results)} memories for query: {query}")
        
        return results

    def prune_memory(self, threshold: Optional[float] = None) -> int:
        """
        Apply memory decay to maintain relevance and remove outdated memories.
        
        Args:
            threshold: Optional override for pruning threshold
            
        Returns:
            Number of memories pruned
        """
        logger.info("Beginning memory pruning process")
        pruned_count = self.memory_decay.apply_decay(self.vector_store, threshold)
        logger.info(f"Pruned {pruned_count} memories")
        return pruned_count

    def reflect(self, depth: int = 1) -> Dict[str, Any]:
        """
        Self-reflective analysis to reweight stored knowledge.
        
        Args:
            depth: How deep to perform the reflection (1-3)
            
        Returns:
            Dictionary with reflection metrics and insights
        """
        logger.info(f"Performing self-reflection with depth: {depth}")
        results = self.meta_cognition.self_reflect(self.vector_store, depth)
        return results
        
    def merge_related_memories(self, key: str, similarity_threshold: float = 0.85) -> bool:
        """
        Find and merge highly similar memories to reduce redundancy.
        
        Args:
            key: Key of the memory to find similar items for
            similarity_threshold: Minimum similarity score for merging
            
        Returns:
            Success status
        """
        logger.info(f"Attempting to merge memories related to: {key}")
        try:
            # Get the original memory
            original = self.vector_store.get(key)
            if not original:
                logger.warning(f"Memory with key {key} not found")
                return False
                
            # Find similar memories
            similar = self.memory_retrieval.find_similar(
                original['embedding'], 
                threshold=similarity_threshold
            )
            
            # Merge if found
            if similar:
                merged = self.meta_cognition.merge_memories(original, similar)
                self.vector_store.update(key, merged)
                logger.info(f"Successfully merged {len(similar)} memories with {key}")
                return True
            else:
                logger.info(f"No similar memories found for {key}")
                return False
                
        except Exception as e:
            logger.error(f"Error merging memories: {str(e)}")
            return False
