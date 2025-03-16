"""
Ethical Memory System for the Augment SDK.

This module implements an ethical governance layer for memory management in AI systems,
providing oversight, bias detection, fairness assessment, and compliance tracking for
all memory operations. It enforces ethical guidelines across memory layers including
ephemeral, working, semantic, procedural, reflective, and predictive memory.

The EthicalMemory class works alongside the Memory Orchestration Module (MOM) to ensure
that memories stored, retrieved, and processed adhere to ethical principles and avoid
harmful biases or unfair representations.

Attributes:
    None

Example:
    ```python
    from augment_sdk.memory import MemoryManager
    from augment_sdk.ethics import EthicalMemory
    
    memory_manager = MemoryManager(config)
    ethical_memory = EthicalMemory(memory_manager)
    
    # Store memory with ethical oversight
    ethical_memory.store("user_data", data, ethical_context={"purpose": "personalization"})
    
    # Retrieve memory with fairness assessment
    results = ethical_memory.retrieve("sensitive_query", fairness_threshold=0.8)
    ```
"""

import enum
import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

# Internal Augment SDK imports
from augment_sdk.ethics.bias_detection import BiasDetector
from augment_sdk.ethics.fairness import FairnessAssessor
from augment_sdk.memory.base import MemoryManager, MemoryLayer
from augment_sdk.utils.config import ConfigManager
from augment_sdk.utils.exceptions import EthicalConstraintError, MemoryAccessError
from augment_sdk.utils.logging import get_logger

# Constants
DEFAULT_FAIRNESS_THRESHOLD = 0.7
DEFAULT_BIAS_THRESHOLD = 0.3
MAX_PERMISSION_CACHE_SIZE = 1000
ETHICAL_AUDIT_INTERVAL = 3600  # 1 hour in seconds


class EthicalPrinciple(enum.Enum):
    """Ethical principles applied to memory operations."""
    
    FAIRNESS = "fairness"
    TRANSPARENCY = "transparency"
    PRIVACY = "privacy"
    BENEFICENCE = "beneficence"
    NON_MALEFICENCE = "non_maleficence"
    ACCOUNTABILITY = "accountability"
    AUTONOMY = "autonomy"
    JUSTICE = "justice"


class DataCategory(enum.Enum):
    """Categories of data with different ethical sensitivities."""
    
    GENERAL = "general"
    PERSONAL = "personal"
    SENSITIVE = "sensitive"
    BIOMETRIC = "biometric"
    PROTECTED = "protected"
    CLINICAL = "clinical"
    FINANCIAL = "financial"


class MemoryOperation(enum.Enum):
    """Types of memory operations requiring ethical oversight."""
    
    STORE = "store"
    RETRIEVE = "retrieve"
    UPDATE = "update"
    DELETE = "delete"
    SHARE = "share"
    ANALYZE = "analyze"
    INFER = "infer"


class EthicalAssessment:
    """Container for ethical assessments of memory operations."""
    
    def __init__(
        self,
        operation: MemoryOperation,
        memory_key: str,
        timestamp: float,
        principles_scores: Dict[EthicalPrinciple, float],
        passed: bool,
        explanation: str,
        recommendations: List[str]
    ):
        """
        Initialize an ethical assessment.
        
        Args:
            operation: Type of memory operation assessed
            memory_key: Identifier for the memory being assessed
            timestamp: Unix timestamp when assessment was performed
            principles_scores: Dictionary mapping ethical principles to compliance scores
            passed: Whether the operation passed ethical assessment
            explanation: Human-readable explanation of the assessment
            recommendations: List of recommendations to improve ethical compliance
        """
        self.operation = operation
        self.memory_key = memory_key
        self.timestamp = timestamp
        self.principles_scores = principles_scores
        self.passed = passed
        self.explanation = explanation
        self.recommendations = recommendations
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the assessment to a dictionary.
        
        Returns:
            Dictionary representation of the assessment
        """
        return {
            "operation": self.operation.value,
            "memory_key": self.memory_key,
            "timestamp": self.timestamp,
            "datetime": datetime.fromtimestamp(self.timestamp).isoformat(),
            "principles_scores": {k.value: v for k, v in self.principles_scores.items()},
            "passed": self.passed,
            "explanation": self.explanation,
            "recommendations": self.recommendations
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EthicalAssessment':
        """
        Create an EthicalAssessment from a dictionary.
        
        Args:
            data: Dictionary representation of an assessment
        
        Returns:
            EthicalAssessment instance
        """
        return cls(
            operation=MemoryOperation(data["operation"]),
            memory_key=data["memory_key"],
            timestamp=data["timestamp"],
            principles_scores={EthicalPrinciple(k): v for k, v in data["principles_scores"].items()},
            passed=data["passed"],
            explanation=data["explanation"],
            recommendations=data["recommendations"]
        )
    
    def __str__(self) -> str:
        """Return string representation of the assessment."""
        return (
            f"EthicalAssessment({self.operation.value}, {self.memory_key}, "
            f"passed={self.passed}, scores={self.principles_scores})"
        )


class EthicalMemory:
    """
    Ethical governance layer for AI memory operations.
    
    This class implements ethical oversight, bias detection, fairness assessment,
    and compliance tracking for all memory operations, ensuring that AI systems
    maintain ethical standards while storing and retrieving information.
    """
    
    def __init__(
        self,
        memory_manager: MemoryManager,
        config: Optional[Dict[str, Any]] = None,
        bias_detector: Optional[BiasDetector] = None,
        fairness_assessor: Optional[FairnessAssessor] = None
    ):
        """
        Initialize the ethical memory system.
        
        Args:
            memory_manager: The memory manager to apply ethical oversight to
            config: Configuration settings for ethical governance
            bias_detector: Custom bias detector implementation (optional)
            fairness_assessor: Custom fairness assessor implementation (optional)
        """
        self.memory_manager = memory_manager
        self.config = config or ConfigManager().get_config("ethical_memory")
        self.logger = get_logger(__name__)
        
        # Initialize ethical components
        self.bias_detector = bias_detector or BiasDetector()
        self.fairness_assessor = fairness_assessor or FairnessAssessor()
        
        # Initialize tracking and audit systems
        self._audit_log: List[EthicalAssessment] = []
        self._permission_cache: Dict[str, Tuple[bool, float]] = {}  # (key, op) -> (permitted, timestamp)
        self._sensitive_keys: Set[str] = set()
        self._last_audit_time = time.time()
        
        # Load ethical constraints
        self._ethical_constraints = self._load_ethical_constraints()
        
        self.logger.info("Ethical Memory System initialized")
    
    def store(
        self,
        key: str,
        data: Any,
        layer: MemoryLayer = MemoryLayer.SEMANTIC,
        ethical_context: Optional[Dict[str, Any]] = None,
        data_category: DataCategory = DataCategory.GENERAL,
        fairness_threshold: Optional[float] = None,
        skip_assessment: bool = False
    ) -> bool:
        """
        Store memory with ethical oversight.
        
        Args:
            key: Identifier for the memory
            data: Data to store
            layer: Memory layer to store the data in
            ethical_context: Additional context for ethical assessment
            data_category: Category of data being stored
            fairness_threshold: Minimum fairness score required (0.0-1.0)
            skip_assessment: Whether to skip ethical assessment (for non-sensitive data)
        
        Returns:
            Boolean indicating success
            
        Raises:
            EthicalConstraintError: If the operation fails ethical assessment
        """
        self.logger.debug(f"Requested memory store operation for key: {key}")
        
        # Check if we can skip assessment for non-sensitive data
        if skip_assessment and data_category == DataCategory.GENERAL:
            return self.memory_manager.store(key, data, layer)
        
        # Check permission cache for recent decisions
        cache_key = f"{key}:{MemoryOperation.STORE.value}"
        cached_decision = self._check_permission_cache(cache_key)
        if cached_decision is not None:
            if not cached_decision:
                raise EthicalConstraintError(f"Memory operation STORE denied for key: {key} (cached decision)")
            return self.memory_manager.store(key, data, layer)
        
        # Perform ethical assessment
        assessment = self._assess_operation(
            MemoryOperation.STORE,
            key,
            data,
            ethical_context or {},
            data_category,
            fairness_threshold or DEFAULT_FAIRNESS_THRESHOLD
        )
        
        # Log the assessment
        self._audit_log.append(assessment)
        self._update_permission_cache(cache_key, assessment.passed)
        
        # Check if operation is permitted
        if not assessment.passed:
            self.logger.warning(
                f"Memory operation STORE denied for key: {key}. "
                f"Explanation: {assessment.explanation}"
            )
            raise EthicalConstraintError(
                f"Memory operation failed ethical assessment: {assessment.explanation}"
            )
        
        # If sensitive data, track in sensitive keys
        if data_category in (DataCategory.SENSITIVE, DataCategory.PROTECTED, 
                             DataCategory.BIOMETRIC, DataCategory.CLINICAL):
            self._sensitive_keys.add(key)
        
        # Perform the memory operation
        result = self.memory_manager.store(key, data, layer)
        
        # Run audit if it's time
        self._run_periodic_audit()
        
        return result
    
    def retrieve(
        self,
        query: str,
        layer: MemoryLayer = MemoryLayer.SEMANTIC,
        ethical_context: Optional[Dict[str, Any]] = None,
        fairness_threshold: Optional[float] = None,
        top_k: int = 5,
        skip_assessment: bool = False
    ) -> List[Any]:
        """
        Retrieve memory with ethical oversight.
        
        Args:
            query: Query string to retrieve memory
            layer: Memory layer to retrieve from
            ethical_context: Additional context for ethical assessment
            fairness_threshold: Minimum fairness score required (0.0-1.0)
            top_k: Maximum number of results to return
            skip_assessment: Whether to skip ethical assessment 
        
        Returns:
            List of memory results
            
        Raises:
            EthicalConstraintError: If the operation fails ethical assessment
        """
        self.logger.debug(f"Requested memory retrieve operation for query: {query}")
        
        # Check if we can skip assessment
        if skip_assessment and not any(key in query for key in self._sensitive_keys):
            return self.memory_manager.retrieve(query, layer, top_k)
        
        # Check permission cache for recent decisions
        cache_key = f"{query}:{MemoryOperation.RETRIEVE.value}"
        cached_decision = self._check_permission_cache(cache_key)
        if cached_decision is not None:
            if not cached_decision:
                raise EthicalConstraintError(f"Memory operation RETRIEVE denied for query: {query} (cached decision)")
            return self.memory_manager.retrieve(query, layer, top_k)
        
        # Perform ethical assessment
        assessment = self._assess_operation(
            MemoryOperation.RETRIEVE,
            query,
            None,  # No data for retrieve
            ethical_context or {},
            DataCategory.GENERAL,  # Default category for retrieval
            fairness_threshold or DEFAULT_FAIRNESS_THRESHOLD
        )
        
        # Log the assessment
        self._audit_log.append(assessment)
        self._update_permission_cache(cache_key, assessment.passed)
        
        # Check if operation is permitted
        if not assessment.passed:
            self.logger.warning(
                f"Memory operation RETRIEVE denied for query: {query}. "
                f"Explanation: {assessment.explanation}"
            )
            raise EthicalConstraintError(
                f"Memory operation failed ethical assessment: {assessment.explanation}"
            )
        
        # Perform the memory operation
        results = self.memory_manager.retrieve(query, layer, top_k)
        
        # Post-process results for fairness
        processed_results = self._ensure_fair_representation(results, fairness_threshold)
        
        # Run audit if it's time
        self._run_periodic_audit()
        
        return processed_results
    
    def update(
        self,
        key: str,
        data: Any,
        layer: MemoryLayer = MemoryLayer.SEMANTIC,
        ethical_context: Optional[Dict[str, Any]] = None,
        data_category: DataCategory = DataCategory.GENERAL,
        fairness_threshold: Optional[float] = None
    ) -> bool:
        """
        Update memory with ethical oversight.
        
        Args:
            key: Identifier for the memory
            data: Updated data
            layer: Memory layer containing the data
            ethical_context: Additional context for ethical assessment
            data_category: Category of data being updated
            fairness_threshold: Minimum fairness score required (0.0-1.0)
        
        Returns:
            Boolean indicating success
            
        Raises:
            EthicalConstraintError: If the operation fails ethical assessment
        """
        self.logger.debug(f"Requested memory update operation for key: {key}")
        
        # Perform ethical assessment
        assessment = self._assess_operation(
            MemoryOperation.UPDATE,
            key,
            data,
            ethical_context or {},
            data_category,
            fairness_threshold or DEFAULT_FAIRNESS_THRESHOLD
        )
        
        # Log the assessment
        self._audit_log.append(assessment)
        
        # Check if operation is permitted
        if not assessment.passed:
            self.logger.warning(
                f"Memory operation UPDATE denied for key: {key}. "
                f"Explanation: {assessment.explanation}"
            )
            raise EthicalConstraintError(
                f"Memory operation failed ethical assessment: {assessment.explanation}"
            )
        
        # If sensitive data, track in sensitive keys
        if data_category in (DataCategory.SENSITIVE, DataCategory.PROTECTED, 
                             DataCategory.BIOMETRIC, DataCategory.CLINICAL):
            self._sensitive_keys.add(key)
        
        # Perform the memory operation
        result = self.memory_manager.update(key, data, layer)
        
        # Run audit if it's time
        self._run_periodic_audit()
        
        return result
    
    def delete(
        self,
        key: str,
        layer: MemoryLayer = MemoryLayer.SEMANTIC,
        ethical_context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Delete memory with ethical oversight.
        
        Args:
            key: Identifier for the memory
            layer: Memory layer containing the data
            ethical_context: Additional context for ethical assessment
        
        Returns:
            Boolean indicating success
            
        Raises:
            EthicalConstraintError: If the operation fails ethical assessment
        """
        self.logger.debug(f"Requested memory delete operation for key: {key}")
        
        # For deletion, we generally prioritize privacy rights, but still log the operation
        assessment = self._assess_operation(
            MemoryOperation.DELETE,
            key,
            None,  # No data for delete
            ethical_context or {},
            DataCategory.GENERAL,  # Default category
            DEFAULT_FAIRNESS_THRESHOLD
        )
        
        # Log the assessment
        self._audit_log.append(assessment)
        
        # Perform the memory operation (allow deletion even if assessment fails)
        # This prioritizes the "right to be forgotten"
        result = self.memory_manager.delete(key, layer)
        
        # Remove from sensitive keys tracking if present
        if key in self._sensitive_keys:
            self._sensitive_keys.remove(key)
        
        # Run audit if it's time
        self._run_periodic_audit()
        
        return result
    
    def get_audit_log(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        operation_types: Optional[List[MemoryOperation]] = None,
        passed_only: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve ethical audit logs for memory operations.
        
        Args:
            start_time: Filter logs after this Unix timestamp
            end_time: Filter logs before this Unix timestamp
            operation_types: Filter logs by operation types
            passed_only: If True, only return operations that passed assessment
        
        Returns:
            List of audit log entries as dictionaries
        """
        filtered_logs = self._audit_log
        
        # Apply time filters
        if start_time is not None:
            filtered_logs = [log for log in filtered_logs if log.timestamp >= start_time]
        if end_time is not None:
            filtered_logs = [log for log in filtered_logs if log.timestamp <= end_time]
        
        # Apply operation type filter
        if operation_types:
            filtered_logs = [log for log in filtered_logs if log.operation in operation_types]
        
        # Apply passed/failed filter
        if passed_only is not None:
            filtered_logs = [log for log in filtered_logs if log.passed == passed_only]
        
        # Convert to dictionary format
        return [log.to_dict() for log in filtered_logs]
    
    def clear_audit_log(self) -> None:
        """Clear the audit log."""
        self._audit_log = []
        self.logger.info("Ethical memory audit log cleared")
    
    def export_audit_log(self, filepath: str) -> bool:
        """
        Export the audit log to a JSON file.
        
        Args:
            filepath: Path to the output file
            
        Returns:
            Boolean indicating success
        """
        try:
            log_data = [log.to_dict() for log in self._audit_log]
            with open(filepath, 'w') as f:
                json.dump(log_data, f, indent=2)
            return True
        except Exception as e:
            self.logger.error(f"Failed to export audit log: {str(e)}")
            return False
    
    def analyze_ethical_trends(self) -> Dict[str, Any]:
        """
        Analyze trends in ethical assessments.
        
        Returns:
            Dictionary with ethical trend analysis
        """
        if not self._audit_log:
            return {"message": "No audit data available for trend analysis"}
        
        # Group by operation type
        operations = {}
        for assessment in self._audit_log:
            if assessment.operation.value not in operations:
                operations[assessment.operation.value] = []
            operations[assessment.operation.value].append(assessment)
        
        # Calculate statistics by operation
        results = {
            "total_operations": len(self._audit_log),
            "operations_by_type": {k: len(v) for k, v in operations.items()},
            "pass_rate_by_operation": {},
            "principle_scores": {}
        }
        
        # Calculate pass rates
        for op_type, assessments in operations.items():
            passed = sum(1 for a in assessments if a.passed)
            results["pass_rate_by_operation"][op_type] = passed / len(assessments)
        
        # Calculate principle scores
        all_principles = [p for assessment in self._audit_log for p in assessment.principles_scores.keys()]
        unique_principles = set([p.value for p in all_principles])
        
        for principle in unique_principles:
            principle_values = [
                assessment.principles_scores.get(EthicalPrinciple(principle), 0)
                for assessment in self._audit_log
                if EthicalPrinciple(principle) in assessment.principles_scores
            ]
            if principle_values:
                results["principle_scores"][principle] = {
                    "mean": np.mean(principle_values),
                    "median": np.median(principle_values),
                    "min": min(principle_values),
                    "max": max(principle_values),
                    "std": np.std(principle_values)
                }
        
        return results
    
    def _assess_operation(
        self,
        operation: MemoryOperation,
        key: str,
        data: Any,
        ethical_context: Dict[str, Any],
        data_category: DataCategory,
        fairness_threshold: float
    ) -> EthicalAssessment:
        """
        Perform ethical assessment of a memory operation.
        
        Args:
            operation: Type of memory operation
            key: Memory identifier
            data: Data being operated on (None for retrieve/delete)
            ethical_context: Additional context for assessment
            data_category: Category of data being operated on
            fairness_threshold: Minimum fairness score required
            
        Returns:
            EthicalAssessment object with the assessment results
        """
        timestamp = time.time()
        
        # Initialize scores for each principle
        principles_scores = {principle: 1.0 for principle in EthicalPrinciple}
        explanations = []
        recommendations = []
        
        # Check constraints based on operation type
        constraints = self._ethical_constraints.get(operation.value, [])
        for constraint in constraints:
            # Skip constraints that don't apply to this data category
            if "applicable_categories" in constraint and data_category.value not in constraint["applicable_categories"]:
                continue
                
            # Apply constraint check
            if operation in (MemoryOperation.STORE, MemoryOperation.UPDATE) and data is not None:
                # Check for bias in data
                if "check_bias" in constraint and constraint["check_bias"]:
                    bias_result = self.bias_detector.detect_bias(
                        data, 
                        sensitive_attributes=constraint.get("sensitive_attributes", [])
                    )
                    if bias_result["bias_detected"]:
                        principles_scores[EthicalPrinciple.FAIRNESS] *= (1 - bias_result["bias_score"])
                        explanations.append(f"Bias detected: {bias_result['explanation']}")
                        recommendations.extend(bias_result["recommendations"])
                
                # Check for fairness
                if "check_fairness" in constraint and constraint["check_fairness"]:
                    fairness_result = self.fairness_assessor.assess_fairness(
                        data,
                        context=ethical_context
                    )
                    principles_scores[EthicalPrinciple.FAIRNESS] *= fairness_result["fairness_score"]
                    if fairness_result["fairness_score"] < fairness_threshold:
                        explanations.append(f"Fairness concern: {fairness_result['explanation']}")
                        recommendations.extend(fairness_result["recommendations"])
            
            # For retrieval operations, assess query
            if operation == MemoryOperation.RETRIEVE:
                # Check for sensitive terms in query
                if "check_sensitive_terms" in constraint and constraint["check_sensitive_terms"]:
                    sensitive_terms = constraint.get("sensitive_terms", [])
                    found_terms = [term for term in sensitive_terms if term.lower() in key.lower()]
                    if found_terms:
                        principles_scores[EthicalPrinciple.PRIVACY] *= 0.6
                        explanations.append(f"Query contains sensitive terms: {', '.join(found_terms)}")
                        recommendations.append("Consider revising query to avoid sensitive terms")
        
        # Calculate overall ethical compliance
        passed = all(score >= fairness_threshold for score in principles_scores.values())
        
        # Prepare explanation
        if not explanations:
            explanation = "Operation meets ethical requirements"
        else:
            explanation = "; ".join(explanations)
        
        # Create assessment
        assessment = EthicalAssessment(
            operation=operation,
            memory_key=key,
            timestamp=timestamp,
            principles_scores=principles_scores,
            passed=passed,
            explanation=explanation,
            recommendations=recommendations
        )
        
        return assessment
    
    def _ensure_fair_representation(
        self,
        results: List[Any],
        fairness_threshold: Optional[float] = None
    ) -> List[Any]:
        """
        Post-process retrieval results to ensure fair representation.
        
        Args:
            results: Original retrieval results
            fairness_threshold: Minimum fairness score required
            
        Returns:
            Potentially modified results ensuring fairness
        """
        # If no results or no threshold specified, return as is
        if not results or fairness_threshold is None:
            return results
        
        # For basic implementation, we'll just use the fairness assessor
        # In a real implementation, this would apply more sophisticated balancing
        fairness_results = self.fairness_assessor.assess_result_set(results)
        
        if fairness_results["fairness_score"] >= fairness_threshold:
            return results
        
        # Apply fairness corrections
        corrected_results = self.fairness_assessor.balance_results(
            results, 
            target_fairness=fairness_threshold
        )
        
        return corrected_results
    
    def _load_ethical_constraints(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load ethical constraints from configuration.
        
        Returns:
            Dictionary mapping operation types to lists of constraints
        """
        # Default constraints if none provided in config
        default_constraints = {
            MemoryOperation.STORE.value: [
                {
                    "name": "bias_detection",
                    "check_bias": True,
                    "bias_threshold": DEFAULT_BIAS_THRESHOLD,
                    "sensitive_attributes": ["gender", "race", "religion", "nationality"],
                    "applicable_categories": [
                        DataCategory.PERSONAL.value,
                        DataCategory.SENSITIVE.value,
                        DataCategory.PROTECTED.value
                    ]
                },
                {
                    "name": "fairness_assessment",
                    "check_fairness": True,
                    "fairness_threshold": DEFAULT_FAIRNESS_THRESHOLD,
                    "applicable_categories": [
                        DataCategory.PERSONAL.value,
                        DataCategory.SENSITIVE.value,
                        DataCategory.PROTECTED.value
                    ]
                }
            ],
            MemoryOperation.RETRIEVE.value: [
                {
                    "name": "sensitive_query_check",
                    "check_sensitive_terms": True,
                    "sensitive_terms": [
                        "password", "ssn", "social security", "credit card", 
                        "bank account", "medical", "health", "disease"
                    ]
                }
            ]
        }
        
        # Get constraints from config if available
        config_constraints = self.config.get("ethical_constraints", {})
        
        # Merge default with config constraints
        merged_constraints = default_constraints.copy()
        for op_type, constraints in config_constraints.items():
            if op_type in merged_constraints:
                merged_constraints[op_type].extend(constraints)
            else:
                merged_constraints[op_type] = constraints
        
        return merged_constraints
    
    def _check_permission_cache(self, cache_key: str) -> Optional[bool]:
        """
        Check if a permission decision is cached.
        
        Args:
            cache_key: Cache key combining operation and memory key
            
        Returns:
            Boolean permission decision if cached, None otherwise
        """
        if cache_key in self._permission_cache:
            permission, timestamp = self._permission_cache[cache_key]
            # Check if cache entry is still valid (10 second TTL)
            if time.time() - timestamp < 10:
                return permission
            # Remove expired entry
            del self._permission_cache[cache_key]
        return None
    
    def _update_permission_cache(self, cache_key: str, permission: bool) -> None:
        """
        Update the permission cache.
        
        Args:
            cache_key: Cache key combining operation and memory key
            permission: Whether the operation is permitted
        """
        # Add new cache entry
        self._permission_cache[cache_key] = (permission, time.time())
        
        # Manage cache size
        if len(self._permission_cache) > MAX_PERMISSION_CACHE_SIZE:
            # Remove oldest entries
            sorted_items = sorted(
                self._permission_cache.items(),
                key=lambda x: x[1][1]  # Sort by timestamp
            )
            # Keep the most recent half of entries
            self._permission_cache = dict(sorted_items[len(sorted_items)//2:])
    
    def _run_periodic_audit(self) -> None:
        """Run a periodic ethical audit if it's time."""
        current_time = time.time()
        if current_time - self._last_audit_time >= ETHICAL_AUDIT_INTERVAL:
            self.logger.info("Running periodic ethical audit")
            
            # Analyze ethical trends
            trends = self.analyze_ethical_trends()
            
            # Log a summary of the trends
            self.logger.info(
                f"Ethical audit summary: {trends['total_operations']} operations, "
                f"average pass rate: {np.mean(list(trends['pass_rate_by_operation'].values())):.2f}"
            )
            
            # Check if any principle has concerningly low scores
            for principle, stats in trends.get("principle_scores", {}).items():
                if stats.get("mean", 1.0) < 0.6:  # Arbitrary threshold for demonstration
                    self.logger.warning(
                        f"Potential ethical concern: Principle '{principle}' has low average score: {stats['mean']:.2f}"
                    )
            
            # Update last audit time
            self._last_audit_time = current_time


# Helper functions for module-level usage

def assess_ethical_implications(
    data: Any,
    context: Optional[Dict[str, Any]] = None,
    sensitivity: float = 0.5
) -> Dict[str, Any]:
    """
    Standalone function to assess ethical implications of data.
    
    Args:
        data: Data to assess
        context: Additional context for the assessment
        sensitivity: How sensitive the assessment should be (0.0-1.0)
        
    Returns:
        Dictionary with ethical assessment results
    """
    context = context or {}
    bias_detector = BiasDetector()
    fairness_assessor = FairnessAssessor()
    
    bias_result = bias_detector.detect_bias(data)
    fairness_result = fairness_assessor.assess_fairness(data, context)
    
    # Adjust thresholds based on sensitivity
    bias_threshold = DEFAULT_BIAS_THRESHOLD * (1 - sensitivity)
    fairness_threshold = DEFAULT_FAIRNESS_THRESHOLD * sensitivity
    
    passed = (
        bias_result["bias_score"] <= bias_threshold and
        fairness_result["fairness_score"] >= fairness_threshold
    )
    
    return {
        "passed": passed,
        "bias_assessment": bias_result