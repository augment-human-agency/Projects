# meta_cognition.py
"""
Self-reflective memory analysis and reweighting system.

This component implements the "Reflective Memory" concept from the Augment Baby 
framework, enabling the AI to review past responses, adjust reasoning, and
improve its approach over time.
"""

import time
import math
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union

from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class MetaCognition:
    """
    Provides self-reflective capabilities for memory analysis and reweighting.
    Implements memory scoring, confidence assessment, and recursive refinement.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the meta-cognition system.
        
        Args:
            config: Configuration settings
        """
        self.config = config or {}
        
        # Memory score tracking
        self.memory_scores = {}
        
        # Retrieval history for reinforcement learning
        self.retrieval_history = {}
        
        # Access frequency tracking
        self.access_counts = {}
        
        # Confidence scoring
        self.confidence_thresholds = {
            'high': 0.8,
            'medium': 0.5,
            'low': 0.2
        }
        
        logger.info("Meta-cognition system initialized")
    
    def evaluate_memory(self, key: str, data: Any) -> float:
        """
        Assign an initial weighting to new memory.
        
        Args:
            key: Memory key
            data: Memory data
            
        Returns:
            float: Initial score assigned to the memory
        """
        # Calculate initial score based on data complexity and content
        if isinstance(data, str):
            # For text: base score on length and complexity
            complexity = min(1.0, len(data) / 1000)  # Cap at 1.0
            score = 0.5 + (complexity * 0.5)
        else:
            # For structured data: use a default score
            score = 0.7
            
        # Store the score
        self.memory_scores[key] = score
        self.access_counts[key] = 0
        
        return score
    
    def record_retrieval(self, query: str, results: List[Dict]) -> None:
        """
        Record a memory retrieval event for learning and reinforcement.
        
        Args:
            query: Search query
            results: Retrieved results
        """
        timestamp = self.get_timestamp()
        
        # Record this retrieval
        retrieval_id = f"retrieval_{timestamp}"
        self.retrieval_history[retrieval_id] = {
            'query': query,
            'timestamp': timestamp,
            'results': [r['key'] for r in results]
        }
        
        # Update access counts for retrieved memories
        for result in results:
            key = result['key']
            self.access_counts[key] = self.access_counts.get(key, 0) + 1
    
    def self_reflect(self, vector_store) -> Dict[str, Any]:
        """
        Perform self-reflective analysis to reweight stored knowledge.
        
        Args:
            vector_store: Vector store instance
            
        Returns:
            Dict with reflection results
        """
        start_time = time.time()
        adjustments_count = 0
        
        # Analyze memory access patterns
        access_patterns = self._analyze_access_patterns()
        
        # Analyze temporal relevance
        temporal_relevance = self._analyze_temporal_relevance()
        
        # Adjust memory weights based on analysis
        for key in list(self.memory_scores.keys()):
            # Skip if memory no longer exists
            if key not in vector_store.memory_store:
                del self.memory_scores[key]
                continue
                
            # Get memory metadata
            memory_data = vector_store.memory_store[key]
            memory_layer = memory_data.get('metadata', {}).get('layer', 'semantic')
            
            # Calculate new score based on multiple factors
            old_score = self.memory_scores[key]
            
            # Factor 1: Access frequency (higher access = higher score)
            access_factor = min(1.0, self.access_counts.get(key, 0) / 10)
            
            # Factor 2: Temporal relevance (newer = higher score)
            created_at = memory_data.get('created_at', '')
            if created_at:
                try:
                    memory_age = (datetime.now() - datetime.fromisoformat(created_at)).days
                    time_factor = math.exp(-0.01 * memory_age)  # Exponential decay
                except:
                    time_factor = 0.5  # Default if date parsing fails
            else:
                time_factor = 0.5
                
            # Factor 3: Memory layer importance
            layer_weights = {
                'ephemeral': 0.6,
                'working': 0.7,
                'semantic': 0.9,
                'procedural': 0.8,
                'reflective': 0.9,
                'predictive': 0.7
            }
            layer_factor = layer_weights.get(memory_layer, 0.7)
            
            # Combine factors for new score
            new_score = 0.4 * access_factor + 0.3 * time_factor + 0.3 * layer_factor
            
            # Apply score adjustment if significant
            if abs(new_score - old_score) > 0.1:
                self.memory_scores[key] = new_score
                
                # Update metadata in vector store
                confidence_level = self._get_confidence_level(new_score)
                vector_store.update_metadata(key, {
                    'confidence': new_score,
                    'confidence_level': confidence_level,
                    'last_evaluated': self.get_timestamp()
                })
                
                adjustments_count += 1
        
        elapsed_time = time.time() - start_time
        
        # Return reflection statistics
        return {
            'reflection_time': elapsed_time,
            'memories_analyzed': len(self.memory_scores),
            'adjustments_count': adjustments_count,
            'access_patterns': access_patterns
        }
    
    def reweight_memory(self, key: str, adjustment: float) -> float:
        """
        Manually adjust the