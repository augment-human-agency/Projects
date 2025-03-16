"""
Augment SDK - Meta-Cognition Module

This module implements self-reflective capabilities for AI memory systems,
enabling dynamic memory evaluation, reweighting, and recursive knowledge refinement.
The meta-cognition component is responsible for evaluating memory quality,
prioritizing relevant memories, and enabling AI systems to reflect on and
improve their understanding over time.

The module supports all memory layers in the hierarchical memory model:
- Ephemeral Memory: Short-term, temporary data
- Working Memory: Mid-term, task-focused retention
- Semantic Memory: Long-term factual knowledge
- Procedural Memory: Process and workflow knowledge
- Reflective Memory: Self-analysis and improvement tracking
- Predictive Memory: Anticipation of future knowledge needs

Classes:
    MetaCognitionError: Base exception for meta-cognition operations
    MemoryEvaluationError: Exception for memory evaluation failures
    MetaCognition: Main class implementing self-reflective capabilities
    MemoryMetrics: Helper class for tracking memory performance metrics
    
Functions:
    calculate_memory_confidence: Utility function for confidence scoring
"""

import logging
import time
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Union, Any, Set, Callable
import numpy as np
from datetime import datetime, timedelta

# Try to import internal Augment SDK components
try:
    from augment_sdk.memory.types import MemoryLayer, MemoryRecord
    from augment_sdk.utils.logging import get_logger
    from augment_sdk.memory.exceptions import MemoryError
    from augment_sdk.utils.config import Configuration
except ImportError:
    # Fallback for testing or standalone usage
    from enum import Enum, auto
    class MemoryLayer(Enum):
        """Memory layer types in the hierarchical memory model."""
        EPHEMERAL = auto()
        WORKING = auto()
        SEMANTIC = auto()
        PROCEDURAL = auto()
        REFLECTIVE = auto()
        PREDICTIVE = auto()
    
    class MemoryRecord:
        """Simple memory record class for standalone usage."""
        def __init__(self, key: str, data: Any, layer: MemoryLayer, 
                     embedding: Optional[np.ndarray] = None,
                     metadata: Optional[Dict[str, Any]] = None):
            self.key = key
            self.data = data
            self.layer = layer
            self.embedding = embedding
            self.metadata = metadata or {}
            self.created_at = datetime.now()
            self.last_accessed = datetime.now()
            self.access_count = 0
    
    class MemoryError(Exception):
        """Base exception for memory operations."""
        pass
    
    def get_logger(name: str) -> logging.Logger:
        """Simple logger setup for standalone usage."""
        logger = logging.getLogger(name)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger
    
    class Configuration:
        """Simple configuration class for standalone usage."""
        def __init__(self):
            self.meta_cognition = {
                "confidence_threshold": 0.7,
                "relevance_decay_rate": 0.95,
                "reflection_interval": 3600,  # seconds
                "min_reflection_memories": 10,
                "max_reflection_memories": 100,
                "context_weight": 0.6,
                "recency_weight": 0.2,
                "importance_weight": 0.2,
                "enable_auto_reflection": True
            }
        
        def get(self, section: str, key: str, default: Any = None) -> Any:
            """Get configuration value."""
            return getattr(self, section, {}).get(key, default)


# Set up module logger
logger = get_logger(__name__)


class MetaCognitionError(MemoryError):
    """Base exception for meta-cognition related errors."""
    pass


class MemoryEvaluationError(MetaCognitionError):
    """Exception raised when memory evaluation fails."""
    pass


class ReflectionError(MetaCognitionError):
    """Exception raised when self-reflection process fails."""
    pass


class ConfidenceLevel(Enum):
    """Confidence levels for memory evaluations."""
    VERY_LOW = 0.2
    LOW = 0.4
    MEDIUM = 0.6
    HIGH = 0.8
    VERY_HIGH = 1.0


class MemoryRelevance(Enum):
    """Relevance levels for memory items."""
    IRRELEVANT = 0.1
    TANGENTIAL = 0.3
    RELEVANT = 0.6
    HIGHLY_RELEVANT = 0.9
    CRITICAL = 1.0


class MemoryMetrics:
    """Helper class for tracking memory performance metrics."""
    
    def __init__(self):
        """Initialize memory metrics tracking."""
        self.recalls = 0
        self.successful_recalls = 0
        self.failed_recalls = 0
        self.reflection_cycles = 0
        self.reweighted_memories = 0
        self.consolidated_memories = 0
        self.last_reflection_time = None
        self.reflection_duration_sum = 0
        self.access_patterns: Dict[str, int] = {}
        self.memory_confidence_history: Dict[str, List[float]] = {}
        
    def record_recall(self, success: bool) -> None:
        """
        Record a memory recall attempt.
        
        Args:
            success: Whether the recall was successful
        """
        self.recalls += 1
        if success:
            self.successful_recalls += 1
        else:
            self.failed_recalls += 1
    
    def record_reflection(self, duration: float, memories_reweighted: int) -> None:
        """
        Record a reflection cycle.
        
        Args:
            duration: Duration of the reflection cycle in seconds
            memories_reweighted: Number of memories that were reweighted
        """
        self.reflection_cycles += 1
        self.last_reflection_time = datetime.now()
        self.reflection_duration_sum += duration
        self.reweighted_memories += memories_reweighted
    
    def record_memory_access(self, memory_key: str) -> None:
        """
        Record access to a specific memory.
        
        Args:
            memory_key: Identifier of the accessed memory
        """
        self.access_patterns[memory_key] = self.access_patterns.get(memory_key, 0) + 1
    
    def record_confidence(self, memory_key: str, confidence: float) -> None:
        """
        Record confidence score for a memory.
        
        Args:
            memory_key: Identifier of the memory
            confidence: Confidence score (0.0 to 1.0)
        """
        if memory_key not in self.memory_confidence_history:
            self.memory_confidence_history[memory_key] = []
        self.memory_confidence_history[memory_key].append(confidence)
    
    def get_average_reflection_duration(self) -> float:
        """
        Calculate the average duration of reflection cycles.
        
        Returns:
            Average duration in seconds
        """
        if self.reflection_cycles == 0:
            return 0.0
        return self.reflection_duration_sum / self.reflection_cycles
    
    def get_recall_success_rate(self) -> float:
        """
        Calculate the success rate of memory recalls.
        
        Returns:
            Success rate as a percentage
        """
        if self.recalls == 0:
            return 0.0
        return (self.successful_recalls / self.recalls) * 100
    
    def get_frequent_memories(self, top_n: int = 10) -> List[Tuple[str, int]]:
        """
        Get the most frequently accessed memories.
        
        Args:
            top_n: Number of top memories to return
            
        Returns:
            List of (memory_key, access_count) tuples
        """
        return sorted(self.access_patterns.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    def get_confidence_trend(self, memory_key: str) -> List[float]:
        """
        Get the confidence trend for a specific memory.
        
        Args:
            memory_key: Identifier of the memory
            
        Returns:
            List of confidence scores over time
        """
        return self.memory_confidence_history.get(memory_key, [])


def calculate_memory_confidence(
    content_length: int, 
    consistency: float, 
    source_reliability: float,
    confirmation_count: int
) -> float:
    """
    Calculate confidence score for a memory based on various factors.
    
    Args:
        content_length: Length of the memory content
        consistency: Consistency with other memories (0.0 to 1.0)
        source_reliability: Reliability of the memory source (0.0 to 1.0)
        confirmation_count: Number of times this memory has been confirmed
        
    Returns:
        Confidence score between 0.0 and 1.0
    """
    # Normalize content length (longer content tends to be more detailed)
    norm_length = min(1.0, content_length / 1000)
    
    # Calculate weighted score
    length_weight = 0.2
    consistency_weight = 0.3
    reliability_weight = 0.3
    confirmation_weight = 0.2
    
    # Normalize confirmation count
    norm_confirmation = min(1.0, confirmation_count / 5)
    
    confidence = (
        length_weight * norm_length +
        consistency_weight * consistency +
        reliability_weight * source_reliability +
        confirmation_weight * norm_confirmation
    )
    
    return min(1.0, max(0.0, confidence))


class MetaCognition:
    """
    Main class implementing self-reflective capabilities for AI memory.
    
    This class enables an AI system to evaluate the quality of its memories,
    reflect on its knowledge, and dynamically adjust memory weights to improve
    recall relevance over time.
    """
    
    def __init__(self, config: Optional[Configuration] = None):
        """
        Initialize the MetaCognition system.
        
        Args:
            config: Configuration instance (optional)
        """
        self.config = config or Configuration()
        self.logger = logger
        self.metrics = MemoryMetrics()
        self.memory_scores: Dict[str, float] = {}
        self.memory_confidence: Dict[str, float] = {}
        self.memory_contexts: Dict[str, Set[str]] = {}
        self.memory_importance: Dict[str, float] = {}
        self.reflection_history: List[Dict[str, Any]] = []
        self.last_reflection_time = datetime.now()
        
        # Load configuration values
        self._confidence_threshold = self.config.get(
            "meta_cognition", "confidence_threshold", 0.7
        )
        self._relevance_decay_rate = self.config.get(
            "meta_cognition", "relevance_decay_rate", 0.95
        )
        self._reflection_interval = self.config.get(
            "meta_cognition", "reflection_interval", 3600  # 1 hour
        )
        self._min_reflection_memories = self.config.get(
            "meta_cognition", "min_reflection_memories", 10
        )
        self._max_reflection_memories = self.config.get(
            "meta_cognition", "max_reflection_memories", 100
        )
        self._context_weight = self.config.get(
            "meta_cognition", "context_weight", 0.6
        )
        self._recency_weight = self.config.get(
            "meta_cognition", "recency_weight", 0.2
        )
        self._importance_weight = self.config.get(
            "meta_cognition", "importance_weight", 0.2
        )
        self._enable_auto_reflection = self.config.get(
            "meta_cognition", "enable_auto_reflection", True
        )
        
        self.logger.info("MetaCognition initialized with confidence threshold: %s", 
                         self._confidence_threshold)
    
    def evaluate_memory(self, memory_record: MemoryRecord) -> float:
        """
        Evaluate a new memory and assign an initial confidence score.
        
        Args:
            memory_record: The memory record to evaluate
            
        Returns:
            Initial confidence score (0.0 to 1.0)
            
        Raises:
            MemoryEvaluationError: If memory evaluation fails
        """
        try:
            self.logger.debug("Evaluating memory: %s", memory_record.key)
            
            # Extract basic properties for evaluation
            content = memory_record.data
            content_length = len(str(content)) if content else 0
            
            # Get source reliability from metadata (default to medium)
            source_reliability = memory_record.metadata.get("source_reliability", 0.5)
            
            # Initial consistency is set to medium (will be refined during reflection)
            consistency = 0.5
            
            # Get confirmation count from metadata
            confirmation_count = memory_record.metadata.get("confirmation_count", 0)
            
            # Calculate initial confidence
            confidence = calculate_memory_confidence(
                content_length, 
                consistency, 
                source_reliability,
                confirmation_count
            )
            
            # Store confidence score
            self.memory_confidence[memory_record.key] = confidence
            
            # Initialize memory score based on confidence and layer
            base_score = confidence
            
            # Adjust score based on memory layer
            layer_weight_map = {
                MemoryLayer.EPHEMERAL: 0.7,
                MemoryLayer.WORKING: 0.8,
                MemoryLayer.SEMANTIC: 1.0,
                MemoryLayer.PROCEDURAL: 0.9,
                MemoryLayer.REFLECTIVE: 0.85,
                MemoryLayer.PREDICTIVE: 0.75
            }
            layer_weight = layer_weight_map.get(memory_record.layer, 1.0)
            
            # Calculate final score
            final_score = base_score * layer_weight
            
            # Store memory score
            self.memory_scores[memory_record.key] = final_score
            
            # Initialize context set
            self.memory_contexts[memory_record.key] = set()
            if "context_tags" in memory_record.metadata:
                self.memory_contexts[memory_record.key].update(
                    memory_record.metadata["context_tags"]
                )
            
            # Initialize importance
            self.memory_importance[memory_record.key] = memory_record.metadata.get("importance", 0.5)
            
            # Track metrics
            self.metrics.record_confidence(memory_record.key, confidence)
            
            self.logger.debug("Memory %s evaluated with confidence %s and score %s", 
                             memory_record.key, confidence, final_score)
            
            return final_score
            
        except Exception as e:
            error_msg = f"Failed to evaluate memory {memory_record.key}: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise MemoryEvaluationError(error_msg) from e
    
    def update_memory_context(self, memory_key: str, contexts: Set[str]) -> None:
        """
        Update the context associations for a memory.
        
        Args:
            memory_key: Identifier of the memory
            contexts: Set of context tags to associate with the memory
            
        Raises:
            MetaCognitionError: If the memory key doesn't exist
        """
        if memory_key not in self.memory_scores:
            raise MetaCognitionError(f"Cannot update context for unknown memory: {memory_key}")
        
        existing_contexts = self.memory_contexts.get(memory_key, set())
        updated_contexts = existing_contexts.union(contexts)
        self.memory_contexts[memory_key] = updated_contexts
        
        self.logger.debug("Updated context for memory %s: %s", 
                         memory_key, updated_contexts)
    
    def update_memory_importance(self, memory_key: str, importance: float) -> None:
        """
        Update the importance score for a memory.
        
        Args:
            memory_key: Identifier of the memory
            importance: New importance score (0.0 to 1.0)
            
        Raises:
            MetaCognitionError: If the memory key doesn't exist
        """
        if memory_key not in self.memory_scores:
            raise MetaCognitionError(f"Cannot update importance for unknown memory: {memory_key}")
        
        importance = min(1.0, max(0.0, importance))
        self.memory_importance[memory_key] = importance
        
        self.logger.debug("Updated importance for memory %s: %s", 
                         memory_key, importance)
    
    def get_memory_confidence(self, memory_key: str) -> float:
        """
        Get the confidence score for a memory.
        
        Args:
            memory_key: Identifier of the memory
            
        Returns:
            Confidence score (0.0 to 1.0)
            
        Raises:
            MetaCognitionError: If the memory key doesn't exist
        """
        if memory_key not in self.memory_confidence:
            raise MetaCognitionError(f"Unknown memory key: {memory_key}")
        
        return self.memory_confidence[memory_key]
    
    def get_memory_score(self, memory_key: str) -> float:
        """
        Get the current score for a memory.
        
        Args:
            memory_key: Identifier of the memory
            
        Returns:
            Memory score (0.0 to 1.0)
            
        Raises:
            MetaCognitionError: If the memory key doesn't exist
        """
        if memory_key not in self.memory_scores:
            raise MetaCognitionError(f"Unknown memory key: {memory_key}")
        
        return self.memory_scores[memory_key]
    
    def record_memory_access(self, memory_key: str) -> None:
        """
        Record access to a memory to improve its relevance.
        
        Args:
            memory_key: Identifier of the accessed memory
            
        Raises:
            MetaCognitionError: If the memory key doesn't exist
        """
        if memory_key not in self.memory_scores:
            raise MetaCognitionError(f"Cannot record access for unknown memory: {memory_key}")
        
        # Boost the memory score slightly (recency effect)
        current_score = self.memory_scores[memory_key]
        boost_factor = 1.05  # 5% boost
        new_score = min(1.0, current_score * boost_factor)
        self.memory_scores[memory_key] = new_score
        
        # Record access in metrics
        self.metrics.record_memory_access(memory_key)
        
        self.logger.debug("Recorded access for memory %s, score adjusted from %s to %s", 
                         memory_key, current_score, new_score)
    
    def calculate_memory_relevance(
        self, 
        memory_key: str, 
        query_context: Optional[Set[str]] = None
    ) -> float:
        """
        Calculate the relevance of a memory for a specific query context.
        
        Args:
            memory_key: Identifier of the memory
            query_context: Set of context tags for the query (optional)
            
        Returns:
            Relevance score (0.0 to 1.0)
            
        Raises:
            MetaCognitionError: If the memory key doesn't exist
        """
        if memory_key not in self.memory_scores:
            raise MetaCognitionError(f"Cannot calculate relevance for unknown memory: {memory_key}")
        
        # Base score from memory evaluation
        base_score = self.memory_scores[memory_key]
        
        # Context relevance (if query_context is provided)
        context_relevance = 0.5  # Default medium relevance
        if query_context and memory_key in self.memory_contexts:
            memory_contexts = self.memory_contexts[memory_key]
            if memory_contexts and query_context:
                # Calculate overlap between memory contexts and query context
                overlap = len(memory_contexts.intersection(query_context))
                total = len(memory_contexts.union(query_context))
                if total > 0:
                    context_relevance = overlap / total
        
        # Importance factor
        importance = self.memory_importance.get(memory_key, 0.5)
        
        # Calculate weighted relevance
        relevance = (
            self._context_weight * context_relevance +
            self._importance_weight * importance +
            (1.0 - self._context_weight - self._importance_weight) * base_score
        )
        
        return min(1.0, max(0.0, relevance))
    
    def reflect(self, vector_store, max_memories: Optional[int] = None) -> Dict[str, Any]:
        """
        Perform self-reflection to improve memory quality and relevance.
        
        This process:
        1. Analyzes recent memory access patterns
        2. Identifies potential inconsistencies
        3. Updates memory confidence scores
        4. Adjusts memory weights based on relevance
        
        Args:
            vector_store: Vector database containing memories
            max_memories: Maximum number of memories to analyze (optional)
            
        Returns:
            Dictionary with reflection results and statistics
            
        Raises:
            ReflectionError: If reflection process fails
        """
        try:
            self.logger.info("Starting self-reflection cycle")
            start_time = time.time()
            
            # Determine how many memories to analyze
            max_count = max_memories or self._max_reflection_memories
            min_count = min(max_count, self._min_reflection_memories)
            
            if not hasattr(vector_store, 'memory_store') or not vector_store.memory_store:
                self.logger.warning("Vector store has no memories, skipping reflection")
                return {
                    "status": "skipped",
                    "reason": "no_memories",
                    "duration": 0,
                    "memories_analyzed": 0,
                    "memories_reweighted": 0
                }
            
            # Get memories to analyze (focusing on semantic layer for deep reflection)
            memories_to_analyze = {}
            semantic_count = 0
            
            for key, memory_data in vector_store.memory_store.items():
                if key in self.memory_scores:
                    layer = memory_data.get('layer', None)
                    if layer == MemoryLayer.SEMANTIC:
                        memories_to_analyze[key] = memory_data
                        semantic_count += 1
                        if semantic_count >= max_count:
                            break
            
            # If we don't have enough semantic memories, add from other layers
            if semantic_count < min_count:
                for key, memory_data in vector_store.memory_store.items():
                    if key not in memories_to_analyze and key in self.memory_scores:
                        memories_to_analyze[key] = memory_data
                        if len(memories_to_analyze) >= max_count:
                            break
            
            # Skip if not enough memories
            if len(memories_to_analyze) < min_count:
                self.logger.warning(
                    "Not enough memories to reflect on (%s < %s), skipping reflection",
                    len(memories_to_analyze), min_count
                )
                return {
                    "status": "skipped",
                    "reason": "insufficient_memories",
                    "duration": 0,
                    "memories_analyzed": len(memories_to_analyze),
                    "memories_reweighted": 0
                }
            
            # Analyze memory consistency across similar vectors
            memory_updates = {}
            consistency_scores = {}
            
            # For each memory, find similar memories and check consistency
            for key, memory_data in memories_to_analyze.items():
                if 'embedding' not in memory_data:
                    continue
                
                # Find similar memories
                query_vector = memory_data['embedding']
                similar_results = vector_store.query(query_vector, top_k=5)
                
                # Skip self-comparison and analyze only valid results
                similar_memories = [r for r in similar_results if r and r.get('key', '') != key]
                
                if not similar_memories:
                    continue
                
                # Check consistency with similar memories
                consistency_sum = 0
                comparison_count = 0
                
                for similar in similar_memories:
                    similar_key = similar.get('key', '')
                    if not similar_key or similar_key not in self.memory_confidence:
                        continue
                    
                    # Use similarity as a proxy for consistency
                    similarity = 1.0 - similar.get('distance', 0.5)  # Convert distance to similarity
                    similarity = max(0.0, min(1.0, similarity))
                    
                    consistency_sum += similarity
                    comparison_count += 1
                    
                    # Update context associations
                    if similar_key in self.memory_contexts and key in self.memory_contexts:
                        self.memory_contexts[key].update(self.memory_contexts[similar_key])
                
                # Calculate average consistency
                consistency = consistency_sum / max(1, comparison_count)
                consistency_scores[key] = consistency
                
                # Update confidence based on consistency
                current_confidence = self.memory_confidence.get(key, 0.5)
                content_length = len(str(memory_data.get('data', '')))
                source_reliability = memory_data.get('metadata', {}).get('source_reliability', 0.5)
                confirmation_count = memory_data.get('metadata', {}).get('confirmation_count', 0)
                
                new_confidence = calculate_memory_confidence(
                    content_length,
                    consistency,
                    source_reliability,
                    confirmation_count
                )
                
                # Blend old and new confidence (gives stability to the system)
                blended_confidence = (current_confidence * 0.7) + (new_confidence * 0.3)
                
                # Record the update
                memory_updates[key] = {
                    'old_confidence': current_confidence,
                    'new_confidence': blended_confidence,
                    'consistency': consistency,
                    'similar_count': comparison_count
                }
                
                # Update the confidence
                self.memory_confidence[key] = blended_confidence
            
            # Apply decay to memories that haven't been accessed recently
            decay_count = 0
            for key in self.memory_scores:
                if key not in memory_updates:
                    # Apply gentle decay to memories not examined in this reflection
                    self.memory_scores[key] *= self._relevance_decay_rate
                    decay_count += 1
            
            # Update memory scores based on new confidence values
            reweight_count = 0
            for key, update in memory_updates.items():
                old_score = self.memory_scores.get(key, 0.5)
                
                # Use confidence as a factor in the score
                confidence_factor = update['new_confidence'] / max(0.1, update['old_confidence'])
                
                # Apply the confidence factor to the score
                new_score = old_score * confidence_factor
                
                # Ensure score is in valid range
                new_score = min(1.0, max(0.1, new_score))
                
                # Update score if it changed significantly
                if abs(new_score - old_score) > 0.05:
                    self.memory_scores[key] = new_score
                    reweight_count += 1
                    
                    # Record the confidence history
                    self.metrics.record_confidence(key, update['new_confidence'])
            
            # Update last reflection time
            self.last_reflection_time = datetime.now()
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Record metrics
            self.metrics.record_reflection(duration, reweight_count)
            
            # Compile reflection results
            results = {
                "status": "completed",
                "timestamp": datetime.now().isoformat(),
                "duration": duration,
                "memories_analyzed": len(memories_to_analyze),
                "memories_reweighted": reweight_count,
                "memories_decayed": decay_count,
                "average_consistency": sum(consistency_scores.values()) / max(1, len(consistency_scores)),
                "average_confidence": sum(self.memory_confidence.values()) / max(1, len(self.memory_confidence))
            }
            
            # Add to reflection history
            self.reflection_history.append(results)
            
            self.logger.info(
                "Self-reflection completed in %.2f seconds. Analyzed: %d, Reweighted: %d, Decayed: %d",
                duration, len(memories_to_analyze), reweight_count, decay_count
            )
            
            return results
            
        except Exception as e:
            error_msg = f"Self-reflection failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise ReflectionError(error_msg) from e
    
    def should_reflect(self) -> bool:
        """
        Determine if it's time to perform reflection based on time interval and memory changes.
        
        Returns:
            True if reflection should be performed, False otherwise
        """
        if not self._enable_auto_reflection:
            return False
        
        # Check if enough time has passed since last reflection
        time_since_last = datetime.now() - self.last_reflection_time
        interval_seconds = timedelta(seconds=self._reflection_interval)
        
        return time_since_last >= interval_seconds
    
    def check_memory_health(self) -> Dict[str, Any]:
        """
        Check overall health of the memory system.
        
        Returns:
            Dictionary with memory health metrics
        """
        total_memories = len(self.memory_scores)
        if total_memories == 0:
            return {
                "status": "empty",
                "total_memories": 0
            }
        
        # Calculate average scores and confidence
        avg_score = sum(self.memory_scores.values()) / total_memories
        avg_confidence = sum(self.memory_confidence.values()) / total_memories
        
        # Count memories by confidence level
        confidence_levels = {
            "very_low": 0,
            "low": 0,
            "medium": 0,
            "high": 0,
            "very_high": 0
        }
        
        for confidence in self.memory_confidence.values():
            if confidence < 0.3:
                confidence_levels["very_low"] += 1
            elif confidence < 0.5:
                confidence_levels["low"] += 1
            elif confidence < 0.7:
                confidence_levels["medium"] += 1
            elif confidence < 0.9:
                confidence_levels["high"] += 1
            else:
                confidence_levels["very_high"] += 1
        
        # Calculate confidence distribution percentages
        confidence_distribution = {}
        for level, count in confidence_levels.items():
            confidence_distribution[level] = (count / total_memories) * 100
        
        # Get reflection metrics
        last_reflection = self.reflection_history[-1] if self.reflection_history else None
        reflection_count = len(self.reflection_history)
        
        return {
            "status": "healthy" if avg_confidence > self._confidence_threshold else "needs_improvement",
            "total_memories": total_memories,
            "average_score": avg_score,
            "average_confidence": avg_confidence,
            "confidence_distribution": confidence_distribution,
            "reflection_count": reflection_count,
            "last_reflection": last_reflection,
            "recall_success_rate": self.metrics.get_recall_success_rate(),
            "common_contexts": self._get_common_contexts(top_n=5)
        }
    
    def reweight_memories_by_context(self, context_tags: Set[str], weight_adjustment: float) -> int:
        """
        Adjust weights for memories associated with specific contexts.
        
        Args:
            context_tags: Set of context tags to match
            weight_adjustment: Adjustment factor to apply (e.g., 1.2 for 20% increase)
            
        Returns:
            Number of memories adjusted
        """
        if not context_tags:
            return 0
        
        adjusted_count = 0
        
        for key, contexts in self.memory_contexts.items():
            if not contexts.isdisjoint(context_tags):  # Check for any overlap
                if key in self.memory_scores:
                    # Apply adjustment with bounds checking
                    current_score = self.memory_scores[key]
                    new_score = current_score * weight_adjustment
                    new_score = min(1.0, max(0.1, new_score))  # Keep within bounds
                    
                    if abs(new_score - current_score) > 0.01:  # Only count significant changes
                        self.memory_scores[key] = new_score
                        adjusted_count += 1
                        
                        self.logger.debug(
                            "Reweighted memory %s by context match: %s -> %s", 
                            key, current_score, new_score
                        )
        
        return adjusted_count
    
    def merge_similar_memories(self, vector_store, similarity_threshold: float = 0.9) -> Dict[str, Any]:
        """
        Identify and consolidate highly similar memories.
        
        Args:
            vector_store: Vector database containing memories
            similarity_threshold: Threshold for considering memories similar enough to merge
            
        Returns:
            Dictionary with merging results and statistics
        """
        if not hasattr(vector_store, 'memory_store') or not vector_store.memory_store:
            return {"status": "skipped", "reason": "no_memories", "merged_count": 0}
        
        try:
            self.logger.info("Starting memory consolidation process")
            
            # Get semantic memories (most suitable for merging)
            semantic_memories = {}
            for key, memory_data in vector_store.memory_store.items():
                if memory_data.get('layer') == MemoryLayer.SEMANTIC:
                    if 'embedding' in memory_data:
                        semantic_memories[key] = memory_data
            
            if len(semantic_memories) < 2:
                return {"status": "skipped", "reason": "insufficient_memories", "merged_count": 0}
            
            # Find clusters of similar memories
            memory_clusters = []
            processed_keys = set()
            
            for key, memory_data in semantic_memories.items():
                if key in processed_keys:
                    continue
                
                # Query for similar memories
                query_vector = memory_data['embedding']
                similar_results = vector_store.query(query_vector, top_k=10)
                
                # Convert results to similarity scores
                similar_with_scores = []
                for result in similar_results:
                    if not result or 'key' not in result:
                        continue
                    
                    similar_key = result['key']
                    if similar_key == key or similar_key in processed_keys:
                        continue
                    
                    # Convert distance to similarity
                    similarity = 1.0 - result.get('distance', 0.0)
                    if similarity >= similarity_threshold:
                        similar_with_scores.append((similar_key, similarity))
                
                if similar_with_scores:
                    # Form a cluster with the original memory and similar ones
                    cluster = [(key, 1.0)] + similar_with_scores
                    memory_clusters.append(cluster)
                    
                    # Mark all keys in this cluster as processed
                    processed_keys.add(key)
                    processed_keys.update(k for k, _ in similar_with_scores)
            
            # Merge the clusters
            merged_count = 0
            for cluster in memory_clusters:
                if len(cluster) < 2:
                    continue
                
                # Sort by similarity (highest first)
                cluster.sort(key=lambda x: x[1], reverse=True)
                
                # Use the highest similarity memory as the primary
                primary_key = cluster[0][0]
                
                # Merge context tags
                merged_contexts = set(self.memory_contexts.get(primary_key, set()))
                
                # Track confidence improvements
                confidence_sum = self.memory_confidence.get(primary_key, 0.5)
                confidence_count = 1
                
                # Process other memories in the cluster
                for similar_key, similarity in cluster[1:]:
                    # Merge contexts
                    merged_contexts.update(self.memory_contexts.get(similar_key, set()))
                    
                    # Aggregate confidence scores
                    if similar_key in self.memory_confidence:
                        confidence_sum += self.memory_confidence[similar_key]
                        confidence_count += 1
                
                # Update the primary memory with merged data
                if primary_key in self.memory_contexts:
                    self.memory_contexts[primary_key] = merged_contexts
                
                # Update confidence (slightly higher due to confirmation)
                if primary_key in self.memory_confidence:
                    avg_confidence = confidence_sum / confidence_count
                    # Boost confidence slightly since multiple similar memories confirm each other
                    boosted_confidence = min(1.0, avg_confidence * 1.1)  
                    self.memory_confidence[primary_key] = boosted_confidence
                
                # Update memory score based on new confidence
                if primary_key in self.memory_scores:
                    self.memory_scores[primary_key] = min(
                        1.0, self.memory_scores[primary_key] * 1.05
                    )
                
                # Count the merge
                merged_count += 1
            
            # Record metrics
            self.metrics.consolidated_memories += merged_count
            
            return {
                "status": "completed",
                "clusters_found": len(memory_clusters),
                "merged_count": merged_count
            }
            
        except Exception as e:
            self.logger.error("Memory consolidation failed: %s", str(e), exc_info=True)
            return {
                "status": "failed",
                "error": str(e),
                "merged_count": 0
            }
    
    def predict_relevant_contexts(self, recent_contexts: List[Set[str]], max_predictions: int = 5) -> List[str]:
        """
        Predict which contexts might be relevant for future queries based on recent patterns.
        
        Args:
            recent_contexts: List of context sets from recent queries
            max_predictions: Maximum number of contexts to predict
            
        Returns:
            List of predicted relevant context tags
        """
        if not recent_contexts:
            return []
        
        # Flatten and count context occurrences
        context_counts = {}
        for context_set in recent_contexts:
            for context in context_set:
                context_counts[context] = context_counts.get(context, 0) + 1
        
        # Sort by frequency
        sorted_contexts = sorted(
            context_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Return the most frequent contexts
        return [context for context, _ in sorted_contexts[:max_predictions]]
    
    def _get_common_contexts(self, top_n: int = 10) -> List[Tuple[str, int]]:
        """
        Get the most common context tags across all memories.
        
        Args:
            top_n: Number of top contexts to return
            
        Returns:
            List of (context_tag, occurrence_count) tuples
        """
        context_counts = {}
        
        for contexts in self.memory_contexts.values():
            for context in contexts:
                context_counts[context] = context_counts.get(context, 0) + 1
        
        # Sort by frequency
        sorted_contexts = sorted(
            context_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return sorted_contexts[:top_n]
    
    def get_reflection_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the history of reflection cycles.
        
        Args:
            limit: Maximum number of history entries to return
            
        Returns:
            List of reflection result dictionaries
        """
        return list(reversed(self.reflection_history[-limit:]))
    
    def export_metrics(self) -> Dict[str, Any]:
        """
        Export all metrics and statistics for analysis.
        
        Returns:
            Dictionary with all metrics and statistics
        """
        return {
            "memory_count": len(self.memory_scores),
            "average_confidence": sum(self.memory_confidence.values()) / max(1, len(self.memory_confidence)),
            "average_score": sum(self.memory_scores.values()) / max(1, len(self.memory_scores)),
            "reflection_count": len(self.reflection_history),
            "reflection_frequency": timedelta(seconds=self._reflection_interval).total_seconds() / 3600,
            "recall_success_rate": self.metrics.get_recall_success_rate(),
            "frequent_memories": self.metrics.get_frequent_memories(top_n=10),
            "average_reflection_duration": self.metrics.get_average_reflection_duration(),
            "common_contexts": self._get_common_contexts(top_n=10)
        }