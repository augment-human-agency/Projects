# memory_manager.py
"""
Core controller for the Memory Orchestration Module (MOM).
Manages memory storage, retrieval, self-reflection, and memory decay.

This component implements the hierarchical memory structure concept with:
- Ephemeral Memory (short-term)
- Working Memory (task-oriented)
- Semantic Memory (long-term knowledge)
- Procedural Memory (methods and workflows)
- Reflective Memory (self-optimization)
- Predictive Memory (future knowledge pathways)
"""

import logging
from typing import Dict, List, Any, Optional, Union

from .vector_store import VectorStore
from .memory_retrieval import MemoryRetrieval
from .meta_cognition import MetaCognition
from .memory_decay import MemoryDecay
from .dynamic_adapter import DynamicAdapter
from ..utils.config import load_config
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class MemoryManager:
    """
    Central controller for all memory operations in the Augment SDK.
    Implements the hierarchical memory structure with multiple memory layers.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the Memory Manager with all required components.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        self.config = load_config(config_path)
        
        # Initialize component dependencies
        self.vector_store = VectorStore(
            db_path=self.config.get('VECTOR_DB_PATH', 'memory_store'),
            embedding_dim=self.config.get('EMBEDDING_DIM', 512)
        )
        
        self.memory_retrieval = MemoryRetrieval(self.vector_store)
        self.meta_cognition = MetaCognition()
        self.memory_decay = MemoryDecay()
        self.dynamic_adapter = DynamicAdapter(self.vector_store, self.meta_cognition)
        
        # Memory layer definitions with default TTL values
        self.memory_layers = {
            'ephemeral': {'ttl': 60 * 10},  # 10 minutes
            'working': {'ttl': 60 * 60 * 24},  # 1 day
            'semantic': {'ttl': None},  # Permanent
            'procedural': {'ttl': None},  # Permanent
            'reflective': {'ttl': None},  # Permanent
            'predictive': {'ttl': 60 * 60 * 24 * 7}  # 1 week
        }
        
        logger.info("Memory Manager initialized with %d memory layers", len(self.memory_layers))

    def store_memory(self, key: str, data: Any, layer: str = 'semantic', metadata: Dict = None) -> bool:
        """
        Store memory with optional meta-cognition weighting.
        
        Args:
            key: Unique identifier for the memory
            data: The memory data to store
            layer: Memory layer to store in (default: semantic)
            metadata: Additional information about the memory
            
        Returns:
            bool: Success status
        """
        if layer not in self.memory_layers:
            logger.error(f"Invalid memory layer: {layer}")
            return False
            
        try:
            # Process data into embeddings
            embedding = self.vector_store.embed(data)
            
            # Add metadata if not provided
            if metadata is None:
                metadata = {}
                
            # Include memory layer information and creation timestamp
            metadata['layer'] = layer
            metadata['created_at'] = self.meta_cognition.get_timestamp()
            
            # Store in vector database
            self.vector_store.store(key, embedding, metadata)
            
            # Evaluate memory for initial weighting
            self.meta_cognition.evaluate_memory(key, data)
            
            # Apply domain-specific adaptation if context is available
            if 'domain' in metadata:
                self.dynamic_adapter.adjust_memory_weights(metadata['domain'], data)
                
            logger.info(f"Memory stored successfully: {key} in layer {layer}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store memory: {str(e)}")
            return False
    
    def retrieve_memory(self, query: str, layer: str = 'semantic', 
                        limit: int = 10, threshold: float = 0.7) -> List[Dict]:
        """
        Retrieve memory using vector similarity search.
        
        Args:
            query: Search query text or embedding
            layer: Memory layer to search in
            limit: Maximum number of results to return
            threshold: Minimum similarity score threshold
            
        Returns:
            List of matching memory entries with similarity scores
        """
        try:
            results = self.memory_retrieval.query_memory(
                query=query, 
                layer=layer, 
                limit=limit,
                threshold=threshold
            )
            
            # Track this retrieval in meta-cognition for memory reinforcement
            self.meta_cognition.record_retrieval(query, results)
            
            # Enhance future retrievals through dynamic adaptation
            self.dynamic_adapter.learn_from_retrieval(query, results)
            
            return results
        except Exception as e:
            logger.error(f"Error retrieving memory: {str(e)}")
            return []
    
    def prune_memory(self, decay_rate: float = 0.05) -> int:
        """
        Apply memory decay to maintain relevance and remove obsolete memories.
        
        Args:
            decay_rate: Rate at which to decay memory (0.0-1.0)
            
        Returns:
            Number of memories pruned
        """
        try:
            pruned_count = self.memory_decay.apply_decay(
                self.vector_store, 
                decay_rate=decay_rate
            )
            logger.info(f"Memory pruning complete: {pruned_count} memories pruned")
            return pruned_count
        except Exception as e:
            logger.error(f"Error during memory pruning: {str(e)}")
            return 0
    
    def reflect(self) -> Dict[str, Any]:
        """
        Perform self-reflective analysis to reweight stored knowledge.
        Implements the "Reflective Memory" concept from the Augment Baby framework.
        
        Returns:
            Dict containing reflection statistics and insights
        """
        try:
            reflection_results = self.meta_cognition.self_reflect(self.vector_store)
            
            # Schedule predictive memory generation based on reflection insights
            self._generate_predictive_memory(reflection_results)
            
            logger.info("Memory reflection complete with %d adjustments", 
                       reflection_results.get('adjustments_count', 0))
            
            return reflection_results
        except Exception as e:
            logger.error(f"Error during memory reflection: {str(e)}")
            return {'error': str(e), 'success': False}
    
    def _generate_predictive_memory(self, reflection_data: Dict) -> None:
        """
        Generate predictive memory based on reflection insights.
        This implements the "Predictive Memory" layer from the Augment Baby framework.
        
        Args:
            reflection_data: Data from the reflection process
        """
        # Implementation details would depend on your specific approach
        # This could involve pattern analysis, trend detection, etc.
        pass
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current memory store.
        
        Returns:
            Dict with memory statistics
        """
        total_memories = 0
        stats = {'layers': {}}
        
        for layer in self.memory_layers:
            layer_count = self.vector_store.count_by_layer(layer)
            stats['layers'][layer] = {
                'count': layer_count,
                'ttl': self.memory_layers[layer]['ttl']
            }
            total_memories += layer_count
            
        stats['total_memories'] = total_memories
        return stats
