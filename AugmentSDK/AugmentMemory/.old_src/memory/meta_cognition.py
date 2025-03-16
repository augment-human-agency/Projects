"""
Meta-Cognition Module for Augment SDK.

This module implements the self-reflective capabilities that enable AI systems 
to evaluate, reweight, and evolve their stored knowledge over time. The Meta-Cognition
system functions as the "thinking about thinking" layer in the Augment memory hierarchy,
providing mechanisms for memory quality assessment, relevance scoring, confidence 
calibration, and knowledge refinement.

The module supports multiple memory layers including:
- Ephemeral Memory: Short-lived context and temporary information
- Working Memory: Task-oriented, active information
- Semantic Memory: Factual, conceptual knowledge
- Procedural Memory: Process-based knowledge and skills
- Reflective Memory: Self-assessment and historical decision records
- Predictive Memory: Forward-looking knowledge anticipation

Through continuous self-reflection cycles, AI systems using this module can:
1. Identify and correct inconsistencies in stored knowledge
2. Prioritize memories based on relevance, recency, and utility
3. Adapt memory weights dynamically based on changing contexts
4. Generate meta-knowledge about the system's own learning patterns

This forms a core component of the recursive learning capabilities in the Augment SDK.
"""

import enum
import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
from dataclasses import dataclass, field

# Import from other Augment SDK modules
from augment_sdk.memory.vector_store import VectorStore
from augment_sdk.utils.config import Config
from augment_sdk.utils.exceptions import (
    MetaCognitionError, 
    MemoryLayerError,
    ReflectionError
)

# Configure logging
logger = logging.getLogger(__name__)

class MemoryLayer(enum.Enum):
    """Enumeration of memory layers supported by the system."""
    EPHEMERAL = "ephemeral"
    WORKING = "working"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    REFLECTIVE = "reflective"
    PREDICTIVE = "predictive"


class ConfidenceLevel(enum.Enum):
    """Confidence levels for memory assessments."""
    VERY_LOW = 0.1
    LOW = 0.3
    MEDIUM = 0.5
    HIGH = 0.7
    VERY_HIGH = 0.9


@dataclass
class MemoryMetadata:
    """Metadata structure for memory entries."""
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    confidence: float = 0.5
    relevance_score: float = 0.5
    utility_score: float = 0.5
    contradiction_score: float = 0.0
    layer: MemoryLayer = MemoryLayer.SEMANTIC
    tags: Set[str] = field(default_factory=set)
    relations: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary for serialization.
        
        Returns:
            Dict[str, Any]: Dictionary representation of metadata
        """
        return {
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count,
            "confidence": self.confidence,
            "relevance_score": self.relevance_score,
            "utility_score": self.utility_score,
            "contradiction_score": self.contradiction_score,
            "layer": self.layer.value,
            "tags": list(self.tags),
            "relations": self.relations
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryMetadata':
        """Create metadata instance from dictionary.
        
        Args:
            data: Dictionary containing metadata fields
            
        Returns:
            MemoryMetadata: New metadata instance
        """
        metadata = cls(
            created_at=data.get("created_at", time.time()),
            last_accessed=data.get("last_accessed", time.time()),
            access_count=data.get("access_count", 0),
            confidence=data.get("confidence", 0.5),
            relevance_score=data.get("relevance_score", 0.5),
            utility_score=data.get("utility_score", 0.5),
            contradiction_score=data.get("contradiction_score", 0.0),
            layer=MemoryLayer(data.get("layer", "semantic")),
        )
        
        metadata.tags = set(data.get("tags", []))
        metadata.relations = data.get("relations", {})
        return metadata


class MetaCognition:
    """
    Core class for AI self-reflection and memory assessment.
    
    This class implements the self-reflective capabilities that allow
    an AI system to evaluate and refine its stored knowledge over time.
    It supports meta-cognitive operations across all memory layers.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the MetaCognition system.
        
        Args:
            config: Configuration parameters for meta-cognition.
                   Defaults to None, which uses default settings.
        """
        self.config = config or {}
        self.memory_metadata: Dict[str, MemoryMetadata] = {}
        self.reflection_history: List[Dict[str, Any]] = []
        self.reflection_interval = self.config.get("reflection_interval", 3600)  # Default: 1 hour
        self.last_reflection_time = time.time()
        self.min_confidence_threshold = self.config.get("min_confidence", 0.2)
        self.max_contradiction_threshold = self.config.get("max_contradiction", 0.7)
        self.relevance_decay_rate = self.config.get("relevance_decay", 0.05)
        self.reflection_callbacks = []
        
        # Initialize layer-specific weights
        self.layer_weights = {
            MemoryLayer.EPHEMERAL: self.config.get("ephemeral_weight", 0.3),
            MemoryLayer.WORKING: self.config.get("working_weight", 0.6),
            MemoryLayer.SEMANTIC: self.config.get("semantic_weight", 0.8),
            MemoryLayer.PROCEDURAL: self.config.get("procedural_weight", 0.7),
            MemoryLayer.REFLECTIVE: self.config.get("reflective_weight", 0.5),
            MemoryLayer.PREDICTIVE: self.config.get("predictive_weight", 0.4),
        }
        
        logger.info("MetaCognition module initialized with %d configuration parameters", 
                   len(self.config))
    
    def register_reflection_callback(self, callback: callable) -> None:
        """
        Register a callback function to be called after reflection cycles.
        
        Args:
            callback: Function to call after reflection
        """
        if not callable(callback):
            raise ValueError("Callback must be callable")
        self.reflection_callbacks.append(callback)
        logger.debug("Registered new reflection callback. Total callbacks: %d", 
                     len(self.reflection_callbacks))
    
    def evaluate_memory(self, key: str, data: Any, metadata: Optional[Dict[str, Any]] = None,
                        layer: Union[str, MemoryLayer] = MemoryLayer.SEMANTIC) -> MemoryMetadata:
        """
        Evaluate a memory item and assign initial meta-cognitive scores.
        
        Args:
            key: Unique identifier for the memory
            data: Content of the memory
            metadata: Optional pre-existing metadata
            layer: Memory layer for this item
            
        Returns:
            MemoryMetadata: Updated metadata for the memory item
        
        Raises:
            MemoryLayerError: If the specified layer is invalid
        """
        # Convert string layer to enum if necessary
        if isinstance(layer, str):
            try:
                layer = MemoryLayer(layer)
            except ValueError:
                supported = ", ".join([l.value for l in MemoryLayer])
                logger.error("Invalid memory layer: %s. Supported layers: %s", layer, supported)
                raise MemoryLayerError(f"Invalid memory layer: {layer}. Supported layers: {supported}")
        
        if key in self.memory_metadata:
            # Update existing metadata
            memory_meta = self.memory_metadata[key]
            memory_meta.last_accessed = time.time()
            memory_meta.access_count += 1
            
            # Update layer if provided
            if layer != memory_meta.layer:
                logger.info("Changing memory layer for '%s' from %s to %s", 
                           key, memory_meta.layer.value, layer.value)
                memory_meta.layer = layer
        else:
            # Create new metadata
            if metadata:
                try:
                    memory_meta = MemoryMetadata.from_dict(metadata)
                except (ValueError, KeyError) as e:
                    logger.warning("Invalid metadata format: %s. Creating new metadata.", str(e))
                    memory_meta = self._create_initial_metadata(data, layer)
            else:
                memory_meta = self._create_initial_metadata(data, layer)
                
            self.memory_metadata[key] = memory_meta
            
        logger.debug("Evaluated memory '%s': confidence=%.2f, relevance=%.2f", 
                    key, memory_meta.confidence, memory_meta.relevance_score)
        return memory_meta
    
    def _create_initial_metadata(self, data: Any, layer: MemoryLayer) -> MemoryMetadata:
        """
        Create initial metadata for a new memory item.
        
        Args:
            data: Content of the memory
            layer: Memory layer for this item
            
        Returns:
            MemoryMetadata: New metadata for the memory item
        """
        # Initial confidence based on data characteristics
        confidence = self._assess_initial_confidence(data)
        
        # Initial relevance based on layer and recency
        relevance = self.layer_weights[layer] 
        
        # Extract tags from data if possible
        tags = self._extract_tags(data) 
        
        metadata = MemoryMetadata(
            created_at=time.time(),
            last_accessed=time.time(),
            access_count=1,
            confidence=confidence,
            relevance_score=relevance,
            utility_score=0.5,  # Default initial utility
            layer=layer,
            tags=tags
        )
        
        return metadata
    
    def _assess_initial_confidence(self, data: Any) -> float:
        """
        Assess initial confidence score for a memory item.
        
        Args:
            data: Content of the memory
            
        Returns:
            float: Initial confidence score between 0 and 1
        """
        # Base confidence level
        confidence = 0.5
        
        # Adjust based on data type and content
        if isinstance(data, str):
            # Text-based confidence heuristics
            if len(data) < 10:
                confidence -= 0.1  # Short texts less reliable
            elif len(data) > 1000:
                confidence += 0.1  # Detailed texts more reliable
                
            # Check for uncertainty markers
            uncertainty_markers = ["maybe", "perhaps", "possibly", "might", "could"]
            for marker in uncertainty_markers:
                if marker in data.lower():
                    confidence -= 0.05
                    
            # Check for certainty markers
            certainty_markers = ["definitely", "certainly", "absolutely", "always", "never"]
            for marker in certainty_markers:
                if marker in data.lower():
                    confidence += 0.05
        
        elif isinstance(data, dict):
            # Structured data tends to have higher confidence
            confidence += 0.1
            
            # Check for confidence field
            if "confidence" in data:
                try:
                    explicit_confidence = float(data["confidence"])
                    if 0 <= explicit_confidence <= 1:
                        confidence = explicit_confidence
                except (ValueError, TypeError):
                    pass
        
        # Ensure confidence is within bounds
        return max(0.1, min(0.9, confidence))
    
    def _extract_tags(self, data: Any) -> Set[str]:
        """
        Extract relevant tags from memory data.
        
        Args:
            data: Content of the memory
            
        Returns:
            Set[str]: Set of extracted tags
        """
        tags = set()
        
        if isinstance(data, str):
            # Extract potential keywords from text
            import re
            from collections import Counter
            
            # Remove punctuation and convert to lowercase
            text = re.sub(r'[^\w\s]', '', data.lower())
            words = text.split()
            
            # Remove common stop words (simplified list)
            stop_words = {
                "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", 
                "in", "on", "at", "to", "for", "with", "by", "about", "of"
            }
            filtered_words = [word for word in words if word not in stop_words]
            
            # Get most common words as tags
            word_counts = Counter(filtered_words)
            tags = {word for word, count in word_counts.most_common(5) if count > 1}
            
        elif isinstance(data, dict):
            # Extract tags from dictionary keys or explicit tag field
            if "tags" in data and isinstance(data["tags"], (list, set)):
                tags = set(data["tags"])
            else:
                # Use top-level keys as tags
                tags = {key for key in data.keys() if isinstance(key, str)}
        
        return tags
    
    def self_reflect(self, vector_store: Optional[VectorStore] = None) -> Dict[str, Any]:
        """
        Perform a self-reflection cycle to update memory metadata.
        
        This function evaluates all stored memories, updates their scores based on
        various factors, and identifies potential inconsistencies or gaps in knowledge.
        
        Args:
            vector_store: Optional vector store to query for related memories
            
        Returns:
            Dict[str, Any]: Report of the reflection cycle
        """
        start_time = time.time()
        reflection_report = {
            "timestamp": datetime.now().isoformat(),
            "metrics": {},
            "updated_memories": 0,
            "identified_contradictions": 0,
            "knowledge_gaps": [],
            "recommendations": []
        }
        
        try:
            # Only perform reflection if enough time has passed since last reflection
            if (time.time() - self.last_reflection_time) < self.reflection_interval:
                logger.debug("Skipping reflection cycle - insufficient time elapsed since last reflection")
                reflection_report["status"] = "skipped"
                return reflection_report
            
            logger.info("Starting self-reflection cycle with %d stored memories", 
                        len(self.memory_metadata))
            
            # Step 1: Apply time-based decay to relevance scores
            self._apply_relevance_decay()
            
            # Step 2: Update utility scores based on access patterns
            self._update_utility_scores()
            
            # Step 3: Identify and assess contradictions
            if vector_store:
                contradictions = self._identify_contradictions(vector_store)
                reflection_report["identified_contradictions"] = len(contradictions)
                
                # Update contradiction scores
                for memory_key, contradiction_info in contradictions.items():
                    if memory_key in self.memory_metadata:
                        self.memory_metadata[memory_key].contradiction_score = contradiction_info["score"]
            
            # Step 4: Generate memory recommendations
            recommendations = self._generate_recommendations()
            reflection_report["recommendations"] = recommendations
            
            # Count updated memories
            reflection_report["updated_memories"] = len(self.memory_metadata)
            
            # Update last reflection time
            self.last_reflection_time = time.time()
            
            # Calculate metrics for report
            reflection_report["metrics"] = self._calculate_reflection_metrics()
            reflection_report["duration"] = time.time() - start_time
            reflection_report["status"] = "completed"
            
            # Record reflection in history
            self.reflection_history.append(reflection_report)
            
            # Call registered callbacks
            for callback in self.reflection_callbacks:
                try:
                    callback(reflection_report)
                except Exception as e:
                    logger.error("Error in reflection callback: %s", str(e))
            
            logger.info("Completed self-reflection cycle in %.2f seconds", 
                        reflection_report["duration"])
            
            return reflection_report
            
        except Exception as e:
            error_msg = f"Error during self-reflection: {str(e)}"
            logger.error(error_msg, exc_info=True)
            reflection_report["status"] = "error"
            reflection_report["error"] = error_msg
            
            raise ReflectionError(error_msg) from e
    
    def _apply_relevance_decay(self) -> None:
        """
        Apply time-based decay to relevance scores of all memories.
        """
        current_time = time.time()
        decay_count = 0
        
        for key, metadata in self.memory_metadata.items():
            # Calculate time since last access
            time_since_access = current_time - metadata.last_accessed
            
            # Only decay memories that haven't been accessed recently
            if time_since_access > (self.reflection_interval / 2):
                # Calculate decay factor based on time elapsed and layer
                layer_decay_modifier = {
                    MemoryLayer.EPHEMERAL: 2.0,    # Fastest decay
                    MemoryLayer.WORKING: 1.5,
                    MemoryLayer.SEMANTIC: 0.5,     # Slower decay
                    MemoryLayer.PROCEDURAL: 0.3,
                    MemoryLayer.REFLECTIVE: 0.7,
                    MemoryLayer.PREDICTIVE: 0.8
                }
                
                decay_factor = self.relevance_decay_rate * layer_decay_modifier[metadata.layer]
                decay_amount = decay_factor * (time_since_access / self.reflection_interval)
                
                # Apply decay with a lower bound
                old_score = metadata.relevance_score
                metadata.relevance_score = max(0.1, metadata.relevance_score - decay_amount)
                
                if old_score != metadata.relevance_score:
                    decay_count += 1
        
        logger.debug("Applied relevance decay to %d memories", decay_count)
    
    def _update_utility_scores(self) -> None:
        """
        Update utility scores based on access patterns and relevance.
        """
        # Find max access count to normalize scores
        max_access = max([meta.access_count for meta in self.memory_metadata.values()], default=1)
        
        for key, metadata in self.memory_metadata.items():
            # Calculate normalized access frequency
            access_factor = metadata.access_count / max_access if max_access > 0 else 0
            
            # Update utility as weighted combination of access and relevance
            metadata.utility_score = (0.7 * access_factor) + (0.3 * metadata.relevance_score)
    
    def _identify_contradictions(self, vector_store: VectorStore) -> Dict[str, Dict[str, Any]]:
        """
        Identify potential contradictions in semantic memory.
        
        Args:
            vector_store: Vector store to query for similar memories
            
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of contradiction assessments
        """
        contradictions = {}
        
        # Only check semantic memory items for contradictions
        semantic_keys = [
            key for key, meta in self.memory_metadata.items() 
            if meta.layer == MemoryLayer.SEMANTIC
        ]
        
        for key in semantic_keys:
            # Skip if already processed
            if key in contradictions:
                continue
                
            # Get vector for this memory
            try:
                vector = vector_store.get_vector(key)
                if vector is None:
                    continue
                    
                # Find similar memories
                similar = vector_store.query(vector, top_k=5)
                
                # Check each similar memory for potential contradiction
                for similar_key, similarity in similar:
                    # Skip self-comparison
                    if similar_key == key:
                        continue
                        
                    # Skip already processed or non-semantic memories
                    if similar_key not in self.memory_metadata:
                        continue
                        
                    if self.memory_metadata[similar_key].layer != MemoryLayer.SEMANTIC:
                        continue
                    
                    # Calculate contradiction likelihood based on similarity and confidence difference
                    confidence_diff = abs(
                        self.memory_metadata[key].confidence - 
                        self.memory_metadata[similar_key].confidence
                    )
                    
                    # High similarity + high confidence difference suggests contradiction
                    contradiction_score = similarity * confidence_diff * 2
                    
                    if contradiction_score > 0.3:  # Threshold for recording contradiction
                        contradictions[key] = {
                            "contradicts": similar_key,
                            "score": contradiction_score,
                            "similarity": similarity
                        }
                        
                        # Also record for the other memory
                        contradictions[similar_key] = {
                            "contradicts": key,
                            "score": contradiction_score,
                            "similarity": similarity
                        }
            
            except Exception as e:
                logger.warning("Error checking contradictions for memory '%s': %s", key, str(e))
                continue
        
        return contradictions
    
    def _generate_recommendations(self) -> List[Dict[str, Any]]:
        """
        Generate recommendations for memory management.
        
        Returns:
            List[Dict[str, Any]]: List of recommendations
        """
        recommendations = []
        
        # Recommend pruning low utility memories
        low_utility_memories = [
            key for key, meta in self.memory_metadata.items()
            if meta.utility_score < 0.2 and meta.access_count > 3
        ]
        
        if low_utility_memories:
            recommendations.append({
                "type": "prune",
                "target": low_utility_memories[:5],  # Limit to 5 examples
                "count": len(low_utility_memories),
                "reason": "Low utility score despite multiple accesses"
            })
        
        # Recommend strengthening high relevance, low confidence memories
        reinforce_candidates = [
            key for key, meta in self.memory_metadata.items()
            if meta.relevance_score > 0.7 and meta.confidence < 0.4
        ]
        
        if reinforce_candidates:
            recommendations.append({
                "type": "reinforce",
                "target": reinforce_candidates[:5],  # Limit to 5 examples
                "count": len(reinforce_candidates),
                "reason": "High relevance but low confidence"
            })
        
        # Recommend resolving high contradiction memories
        contradiction_candidates = [
            key for key, meta in self.memory_metadata.items()
            if meta.contradiction_score > 0.6
        ]
        
        if contradiction_candidates:
            recommendations.append({
                "type": "resolve_contradiction",
                "target": contradiction_candidates[:5],  # Limit to 5 examples
                "count": len(contradiction_candidates),
                "reason": "High contradiction score with other memories"
            })
        
        return recommendations
    
    def _calculate_reflection_metrics(self) -> Dict[str, Any]:
        """
        Calculate metrics summarizing the current memory state.
        
        Returns:
            Dict[str, Any]: Dictionary of metrics
        """
        metrics = {}
        
        # Count memories by layer
        layer_counts = {layer: 0 for layer in MemoryLayer}
        for meta in self.memory_metadata.values():
            layer_counts[meta.layer] += 1
        
        metrics["memory_counts"] = {layer.value: count for layer, count in layer_counts.items()}
        
        # Average confidence by layer
        layer_confidence = {layer: [] for layer in MemoryLayer}
        for meta in self.memory_metadata.values():
            layer_confidence[meta.layer].append(meta.confidence)
        
        metrics["avg_confidence"] = {
            layer.value: sum(scores)/len(scores) if scores else 0
            for layer, scores in layer_confidence.items()
        }
        
        # Overall memory health score (weighted average of confidence, relevance, and utility)
        if self.memory_metadata:
            avg_confidence = sum(meta.confidence for meta in self.memory_metadata.values()) / len(self.memory_metadata)
            avg_relevance = sum(meta.relevance_score for meta in self.memory_metadata.values()) / len(self.memory_metadata)
            avg_utility = sum(meta.utility_score for meta in self.memory_metadata.values()) / len(self.memory_metadata)
            avg_contradiction = sum(meta.contradiction_score for meta in self.memory_metadata.values()) / len(self.memory_metadata)
            
            metrics["memory_health"] = {
                "confidence": avg_confidence,
                "relevance": avg_relevance,
                "utility": avg_utility,
                "contradiction": avg_contradiction,
                "overall": (0.3 * avg_confidence + 0.3 * avg_relevance + 0.3 * avg_utility - 0.1 * avg_contradiction)
            }
        else:
            metrics["memory_health"] = {
                "confidence": 0,
                "relevance": 0,
                "utility": 0,
                "contradiction": 0,
                "overall": 0
            }
        
        return metrics
    
    def get_metadata(self, key: str) -> Optional[MemoryMetadata]:
        """
        Get metadata for a specific memory.
        
        Args:
            key: Memory identifier
            
        Returns:
            Optional[MemoryMetadata]: Metadata if found, None otherwise
        """
        return self.memory_metadata.get(key)
    
    def update_metadata(self, key: str, 
                        metadata_updates: Dict[str, Any]) -> Optional[MemoryMetadata]:
        """
        Update specific metadata fields for a memory.
        
        Args:
            key: Memory identifier
            metadata_updates: Dictionary of metadata fields to update
            
        Returns:
            Optional[MemoryMetadata]: Updated metadata if found, None otherwise
            
        Raises:
            MetaCognitionError: If metadata update fails
        """
        if key not in self.memory_metadata:
            logger.warning("Attempted to update metadata for unknown memory: %s", key)
            return None
        
        try:
            metadata = self.memory_metadata[key]
            
            # Update each provided field
            for field, value in metadata_updates.items():
                if hasattr(metadata, field):
                    # Handle special case for layer
                    if field == "layer" and isinstance(value, str):
                        try:
                            value = MemoryLayer(value)
                        except ValueError:
                            raise MetaCognitionError(f"Invalid memory layer: {value}")
                    
                    setattr(metadata, field, value)
                else:
                    logger.warning("Unknown metadata field '%s' for memory '%s'", field, key)
            
            # Always update last_accessed when metadata is modified
            metadata.last_accessed = time.time()
            
            logger.debug("Updated metadata for memory '%s': %s", key, list(metadata_updates.keys()))
            return metadata
            
        except Exception as e:
            error_msg = f"Failed to update metadata for memory '{key}': {str(e)}"
            logger.error(error_msg)
            raise MetaCognitionError(error_msg) from e
    
    def get_last_reflection_report(self) -> Optional[Dict[str, Any]]:
        """
        Get the most recent reflection report.
        
        Returns:
            Optional[Dict[str, Any]]: Most recent reflection report or None
        """
        if not self.reflection_history:
            return None
        return self.reflection_history[-1]
    
    def export_metadata(self) -> Dict[str, Dict[str, Any]]:
        """
        Export all memory metadata for persistence.
        
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of serialized metadata
        """
        return {
            key: metadata.to_dict() 
            for key, metadata in self.memory_metadata.items()
        }
    
    def import_metadata(self, metadata_dict: Dict[str, Dict[str, Any]]) -> int:
        """
        Import memory metadata from persistent storage.
        
        Args:
            metadata_dict: Dictionary of serialized metadata
            
        Returns:
            int: Number of metadata items imported
            
        Raises:
            MetaCognitionError: If metadata import fails
        """
        try:
            imported_count = 0
            for key, metadata in metadata_dict.items():
                try:
                    self.memory_metadata[key] = MemoryMetadata.from_dict(metadata)
                    imported_count += 1
                except (ValueError, KeyError) as e:
                    logger.warning("Failed to import metadata for '%s': %s", key, str(e))
            
            logger.info("Imported metadata for %d memories", imported_count)
            return imported_count
            
        except Exception as e:
            error_msg = f"Failed to import metadata: {str(e)}"
            logger.error(error_msg)
            raise MetaCognitionError(error_msg) from e
    
    def assess_memory_health(self) -> Dict[str, Any]:
        """
        Perform a comprehensive assessment of memory health.
        
        Returns:
            Dict[str, Any]: Memory health assessment report
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_memories": len(self.memory_metadata),
            "layer_distribution": {},
            "confidence_distribution": {
                "very_low": 0,
                "low": 0,
                "medium": 0,
                "high": 0,
                "very_high": 0
            },
            "age_distribution": {
                "recent": 0,      # < 1 day
                "moderate": 0,    # 1-7 days
                "old": 0          # > 7 days
            },
            "health_scores": {},
            "recommendations": []
        }
        
        current_time = time.time()
        
        # Count memories by layer
        layer_counts = {layer: 0 for layer in MemoryLayer}
        
        for key, meta in self.memory_metadata.items():
            # Layer distribution
            layer_counts[meta.layer] += 1
            
            # Confidence distribution
            if meta.confidence < 0.2:
                report["confidence_distribution"]["very_low"] += 1
            elif meta.confidence < 0.4:
                report["confidence_distribution"]["low"] += 1
            elif meta.confidence < 0.6:
                report["confidence_distribution"]["medium"] += 1
            elif meta.confidence < 0.8:
                report["confidence_distribution"]["high"] += 1
            else:
                report["confidence_distribution"]["very_high"] += 1
            
            # Age distribution
            age_in_days = (current_time - meta.created_at) / (24 * 3600)
            if age_in_days < 1:
                report["age_distribution"]["recent"] += 1
            elif age_in_days < 7:
                report["age_distribution"]["moderate"] += 1
            else:
                report["age_distribution"]["old"] += 1
        
        # Convert layer counts to percentages
        total = len(self.memory_metadata) or 1  # Avoi