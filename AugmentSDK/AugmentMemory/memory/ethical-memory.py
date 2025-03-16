"""
Ethical Memory Module for Augment SDK.

This module provides a comprehensive ethical memory management system that enforces 
ethical constraints on memory operations within the Augment SDK. It integrates with 
the Memory Orchestration Module (MOM) to ensure that memory storage, retrieval, and 
refinement adhere to ethical guidelines and principles.

Key features:
- Ethical assessment of memory entries before storage and retrieval
- Bias detection and mitigation in memory operations
- Fairness evaluation across different memory layers
- Ethical filtering of memory based on configurable guidelines
- Integration with the memory orchestration system

The EthicalMemory class serves as a wrapper around memory operations, providing
ethical guardrails while maintaining the full functionality of the underlying
memory system.

Typical usage:
    ```python
    from augment_sdk.ethics import EthicalMemory
    from augment_sdk.memory import MemoryManager
    
    # Initialize the memory manager
    memory_manager = MemoryManager(config)
    
    # Wrap it with ethical memory
    ethical_memory = EthicalMemory(memory_manager, 
                                  bias_threshold=0.7,
                                  fairness_threshold=0.8)
    
    # Store memory with ethical assessment
    ethical_memory.store("important_insight", 
                        "AI systems should prioritize human autonomy", 
                        layer="semantic")
    
    # Retrieve memory with ethical filtering
    results = ethical_memory.retrieve("AI ethics guidelines")
    ```
"""

import enum
import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field, validator

# Augment SDK imports
from augment_sdk.memory.components.memory_manager import MemoryManager
from augment_sdk.memory.components.vector_store import VectorStore
from augment_sdk.memory.utils.config import Config
from augment_sdk.memory.utils.logger import get_logger
from augment_sdk.memory.utils.vector_utils import cosine_similarity

# Configure logger
logger = get_logger(__name__)


class EthicalCategory(str, Enum):
    """Categories for ethical assessment of memory entries."""
    FAIRNESS = "fairness"
    TRANSPARENCY = "transparency"
    ACCOUNTABILITY = "accountability"
    PRIVACY = "privacy"
    SAFETY = "safety"
    INCLUSIVITY = "inclusivity"
    HUMAN_AUTONOMY = "human_autonomy"
    ENVIRONMENTAL_IMPACT = "environmental_impact"


class BiasType(str, Enum):
    """Types of bias that can be detected in memory entries."""
    DEMOGRAPHIC = "demographic"
    REPRESENTATION = "representation"
    LANGUAGE = "language"
    HISTORICAL = "historical"
    MEASUREMENT = "measurement"
    ALGORITHMIC = "algorithmic"
    AGGREGATION = "aggregation"
    SELECTION = "selection"


class EthicalSeverity(str, Enum):
    """Severity levels for ethical concerns."""
    CRITICAL = "critical"  # Severe ethical violation requiring immediate attention
    HIGH = "high"          # Significant ethical concern
    MEDIUM = "medium"      # Moderate ethical concern
    LOW = "low"            # Minor ethical concern
    NONE = "none"          # No ethical concern


class EthicalMemoryException(Exception):
    """Base exception for ethical memory module."""
    pass


class EthicalViolationError(EthicalMemoryException):
    """Raised when a memory operation violates ethical constraints."""
    def __init__(self, message: str, category: EthicalCategory, severity: EthicalSeverity):
        self.category = category
        self.severity = severity
        super().__init__(f"{severity.value.upper()} ethical violation in {category.value}: {message}")


class BiasDetectionError(EthicalMemoryException):
    """Raised when bias detection fails."""
    pass


@dataclass
class EthicalScoreResult:
    """Result of an ethical assessment for a memory entry."""
    overall_score: float
    category_scores: Dict[EthicalCategory, float]
    bias_scores: Dict[BiasType, float]
    risk_level: EthicalSeverity
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class EthicalMemoryConfig(BaseModel):
    """Configuration for the ethical memory system."""
    bias_threshold: float = Field(0.7, description="Threshold for bias detection (0.0-1.0)")
    fairness_threshold: float = Field(0.8, description="Threshold for fairness evaluation (0.0-1.0)")
    minimum_ethical_score: float = Field(0.6, description="Minimum overall ethical score for memory operations")
    
    sensitive_categories: List[str] = Field(
        default=[
            "race", "gender", "religion", "nationality", "sexuality", 
            "disability", "socioeconomic", "political", "age"
        ],
        description="Categories to monitor for sensitive content and bias"
    )
    
    enable_fairness_evaluation: bool = Field(
        True, description="Enable fairness evaluation for memory operations"
    )
    enable_bias_detection: bool = Field(
        True, description="Enable bias detection for memory operations"
    )
    enable_filtering: bool = Field(
        True, description="Enable filtering of memories based on ethical scores"
    )
    
    log_ethical_assessments: bool = Field(
        True, description="Log details of ethical assessments"
    )
    
    critical_severity_action: str = Field(
        "block", description="Action to take for critical severity issues (block, warn, log)"
    )
    high_severity_action: str = Field(
        "warn", description="Action to take for high severity issues (block, warn, log)"
    )
    
    @validator("bias_threshold", "fairness_threshold", "minimum_ethical_score")
    def validate_threshold(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Threshold must be between 0.0 and 1.0, got {v}")
        return v
    
    @validator("critical_severity_action", "high_severity_action")
    def validate_action(cls, v):
        valid_actions = {"block", "warn", "log"}
        if v not in valid_actions:
            raise ValueError(f"Action must be one of {valid_actions}, got {v}")
        return v


class EthicalMemory:
    """
    Ethical Memory system that wraps around memory operations to enforce ethical constraints.
    
    This class provides ethical guardrails for memory operations, including storage, retrieval,
    and refinement. It detects bias, evaluates fairness, and applies ethical filtering based
    on configurable guidelines.
    
    Attributes:
        memory_manager: The underlying memory manager to wrap.
        config: Configuration for ethical memory operations.
        bias_patterns: Compiled regex patterns for detecting bias in text.
        sensitive_terms: Dictionary mapping sensitive categories to terms.
    """
    
    def __init__(
        self, 
        memory_manager: MemoryManager, 
        config: Optional[Union[Dict, EthicalMemoryConfig]] = None
    ):
        """
        Initialize the ethical memory system.
        
        Args:
            memory_manager: The memory manager to wrap with ethical constraints.
            config: Configuration for ethical memory operations. Can be a dictionary
                or an EthicalMemoryConfig object.
        """
        self.memory_manager = memory_manager
        
        # Initialize configuration
        if config is None:
            self.config = EthicalMemoryConfig()
        elif isinstance(config, dict):
            self.config = EthicalMemoryConfig(**config)
        else:
            self.config = config
            
        # Load sensitive terms for bias detection
        self.sensitive_terms = self._load_sensitive_terms()
        
        # Compile regex patterns for bias detection
        self.bias_patterns = self._compile_bias_patterns()
        
        logger.info(f"Initialized EthicalMemory with config: {self.config}")
    
    def store(
        self, 
        key: str, 
        data: str, 
        layer: str = "semantic",
        metadata: Optional[Dict[str, Any]] = None,
        bypass_ethics: bool = False
    ) -> Dict[str, Any]:
        """
        Store memory with ethical assessment.
        
        Args:
            key: Unique identifier for the memory.
            data: Content to store.
            layer: Memory layer to store in (e.g., "semantic", "procedural").
            metadata: Additional metadata for the memory entry.
            bypass_ethics: If True, bypass ethical assessment (for system use only).
            
        Returns:
            Dict with storage status and ethical assessment results.
            
        Raises:
            EthicalViolationError: If the memory fails ethical assessment and
                the configuration specifies blocking such violations.
        """
        if metadata is None:
            metadata = {}
            
        if not bypass_ethics and self.config.enable_filtering:
            # Perform ethical assessment
            assessment = self.assess_ethical_implications(data)
            
            # Store assessment in metadata
            metadata["ethical_assessment"] = {
                "overall_score": assessment.overall_score,
                "risk_level": assessment.risk_level.value,
                "issues": assessment.issues
            }
            
            # Check if we should block based on severity
            if assessment.risk_level == EthicalSeverity.CRITICAL and self.config.critical_severity_action == "block":
                raise EthicalViolationError(
                    f"Memory content has critical ethical issues: {assessment.issues}",
                    EthicalCategory.FAIRNESS,  # Default category, could be refined
                    EthicalSeverity.CRITICAL
                )
            
            if assessment.risk_level == EthicalSeverity.HIGH and self.config.high_severity_action == "block":
                raise EthicalViolationError(
                    f"Memory content has high-severity ethical issues: {assessment.issues}",
                    EthicalCategory.FAIRNESS,  # Default category, could be refined
                    EthicalSeverity.HIGH
                )
            
            # Log warnings if configured
            if assessment.risk_level == EthicalSeverity.CRITICAL and self.config.critical_severity_action == "warn":
                logger.warning(f"CRITICAL ethical issue in memory storage: {assessment.issues}")
            
            if assessment.risk_level == EthicalSeverity.HIGH and self.config.high_severity_action == "warn":
                logger.warning(f"HIGH-severity ethical issue in memory storage: {assessment.issues}")
        
        # Store memory
        result = self.memory_manager.store_memory(key, data, layer, metadata=metadata)
        
        # Return combined result
        return {
            "status": "success",
            "memory_key": key,
            "ethical_assessment": getattr(assessment, "overall_score", None) if not bypass_ethics else None,
            "storage_result": result
        }
    
    def retrieve(
        self, 
        query: str, 
        layer: str = "semantic", 
        top_k: int = 5,
        filter_criteria: Optional[Dict[str, Any]] = None,
        ethical_minimum_score: Optional[float] = None,
        bypass_ethics: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Retrieve memory with ethical filtering.
        
        Args:
            query: Query string to search for.
            layer: Memory layer to retrieve from.
            top_k: Maximum number of results to return.
            filter_criteria: Additional criteria to filter results.
            ethical_minimum_score: Minimum ethical score for retrieved memories.
            bypass_ethics: If True, bypass ethical filtering (for system use only).
            
        Returns:
            List of memory entries that pass ethical filtering.
        """
        # Set default ethical minimum score if not provided
        if ethical_minimum_score is None:
            ethical_minimum_score = self.config.minimum_ethical_score
        
        # Retrieve memories
        results = self.memory_manager.retrieve_memory(query, layer, top_k)
        
        # Apply ethical filtering if enabled
        if not bypass_ethics and self.config.enable_filtering:
            filtered_results = []
            for result in results:
                # Check if memory has ethical assessment metadata
                ethical_score = None
                if (
                    result.get("metadata") 
                    and result["metadata"].get("ethical_assessment")
                    and "overall_score" in result["metadata"]["ethical_assessment"]
                ):
                    ethical_score = result["metadata"]["ethical_assessment"]["overall_score"]
                
                # If no previous assessment, assess now
                if ethical_score is None and "data" in result:
                    assessment = self.assess_ethical_implications(result["data"])
                    ethical_score = assessment.overall_score
                
                # Apply filtering based on ethical score
                if ethical_score is None or ethical_score >= ethical_minimum_score:
                    filtered_results.append(result)
            
            return filtered_results
        
        return results
    
    def assess_ethical_implications(self, content: str) -> EthicalScoreResult:
        """
        Perform a comprehensive ethical assessment of memory content.
        
        Args:
            content: Text content to assess.
            
        Returns:
            EthicalScoreResult with detailed assessment results.
        """
        # Initialize category scores with default values
        category_scores = {category: 1.0 for category in EthicalCategory}
        
        # Detect bias if enabled
        bias_scores = {}
        if self.config.enable_bias_detection:
            bias_scores = self._detect_bias(content)
            
            # Impact fairness score based on bias detection
            if bias_scores:
                avg_bias = sum(bias_scores.values()) / len(bias_scores)
                category_scores[EthicalCategory.FAIRNESS] -= avg_bias
        
        # Evaluate inclusivity
        inclusivity_score = self._evaluate_inclusivity(content)
        category_scores[EthicalCategory.INCLUSIVITY] = inclusivity_score
        
        # Evaluate privacy concerns
        privacy_score = self._evaluate_privacy(content)
        category_scores[EthicalCategory.PRIVACY] = privacy_score
        
        # Evaluate human autonomy
        autonomy_score = self._evaluate_human_autonomy(content)
        category_scores[EthicalCategory.HUMAN_AUTONOMY] = autonomy_score
        
        # Calculate overall score (average of category scores)
        overall_score = sum(category_scores.values()) / len(category_scores)
        
        # Determine risk level based on overall score
        risk_level = self._determine_risk_level(overall_score)
        
        # Identify issues and recommendations
        issues, recommendations = self._generate_issues_and_recommendations(
            category_scores, bias_scores, overall_score
        )
        
        # Create result
        result = EthicalScoreResult(
            overall_score=overall_score,
            category_scores=category_scores,
            bias_scores=bias_scores,
            risk_level=risk_level,
            issues=issues,
            recommendations=recommendations
        )
        
        # Log assessment if configured
        if self.config.log_ethical_assessments:
            logger.info(f"Ethical assessment: score={overall_score:.2f}, risk={risk_level.value}")
            if issues:
                logger.info(f"Ethical issues: {issues}")
        
        return result
    
    def _detect_bias(self, content: str) -> Dict[BiasType, float]:
        """
        Detect various types of bias in content.
        
        Args:
            content: Text content to analyze.
            
        Returns:
            Dictionary mapping bias types to severity scores (0.0-1.0).
        """
        bias_scores = {}
        
        # Detect demographic bias
        demographic_bias_score = self._detect_demographic_bias(content)
        if demographic_bias_score > 0:
            bias_scores[BiasType.DEMOGRAPHIC] = demographic_bias_score
        
        # Detect representation bias
        representation_bias_score = self._detect_representation_bias(content)
        if representation_bias_score > 0:
            bias_scores[BiasType.REPRESENTATION] = representation_bias_score
        
        # Detect language bias
        language_bias_score = self._detect_language_bias(content)
        if language_bias_score > 0:
            bias_scores[BiasType.LANGUAGE] = language_bias_score
        
        # Only return bias types that exceed the threshold
        return {
            bias_type: score 
            for bias_type, score in bias_scores.items() 
            if score >= self.config.bias_threshold
        }
    
    def _detect_demographic_bias(self, content: str) -> float:
        """
        Detect bias related to demographic groups.
        
        Args:
            content: Text content to analyze.
            
        Returns:
            Bias severity score (0.0-1.0).
        """
        # Lower case for case-insensitive matching
        content_lower = content.lower()
        
        # Check for explicit demographic terms with potentially biased language
        bias_indicators = []
        
        # Check for generalization patterns about demographic groups
        for category, terms in self.sensitive_terms.items():
            for term in terms:
                # Look for generalizing statements about groups
                generalization_patterns = [
                    rf"{term}s are",
                    rf"all {term}s",
                    rf"{term}s always",
                    rf"{term}s never",
                    rf"typical {term}",
                    rf"those {term}s",
                    rf"these {term}s"
                ]
                
                for pattern in generalization_patterns:
                    if pattern in content_lower:
                        bias_indicators.append(f"Generalization about {category}: '{pattern}'")
        
        # Calculate score based on number of indicators
        if not bias_indicators:
            return 0.0
        
        # Calculate severity based on number and type of bias indicators
        # More indicators = higher severity
        severity = min(1.0, len(bias_indicators) * 0.2)
        
        return severity
    
    def _detect_representation_bias(self, content: str) -> float:
        """
        Detect bias related to representation of different groups.
        
        Args:
            content: Text content to analyze.
            
        Returns:
            Bias severity score (0.0-1.0).
        """
        # This is a simplified implementation
        # A real implementation would have more sophisticated analysis
        
        # Detect patterns of erasure or over-emphasis
        erasure_patterns = [
            r"only ([^\s]+) can",
            r"([^\s]+) are not capable of",
            r"([^\s]+) don't belong",
            r"([^\s]+) should not be included",
        ]
        
        erasure_count = 0
        for pattern in erasure_patterns:
            matches = re.findall(pattern, content.lower())
            erasure_count += len(matches)
        
        # Calculate score based on detected patterns
        if erasure_count == 0:
            return 0.0
        
        return min(1.0, erasure_count * 0.25)
    
    def _detect_language_bias(self, content: str) -> float:
        """
        Detect bias embedded in language usage.
        
        Args:
            content: Text content to analyze.
            
        Returns:
            Bias severity score (0.0-1.0).
        """
        # Look for loaded language, coded terms, and microaggressions
        biased_language_patterns = [
            # Gendered language when not necessary
            r"\b(mankind|manpower|manmade)\b",
            
            # Terms that could reinforce stereotypes
            r"\bcredit to (his|her|their) race\b", 
            r"\barticulate for a\b",
            
            # Coded language
            r"\burban\b problem", 
            r"\binner city\b issue",
            r"\bthug\b",
            
            # Dismissive or minimizing language
            r"\bplaying the (race|gender|victim) card\b",
            r"\bpolitically correct\b",
            
            # Othering language
            r"\bthose people\b", 
            r"\byou people\b",
            r"\bthat's so gay\b",
            
            # Implicit assumptions
            r"\bnormal people\b", 
            r"\bregular families\b",
        ]
        
        # Count pattern matches
        match_count = 0
        for pattern in biased_language_patterns:
            matches = re.findall(pattern, content.lower())
            match_count += len(matches)
        
        # Calculate bias score
        if match_count == 0:
            return 0.0
        
        return min(1.0, match_count * 0.3)
    
    def _evaluate_inclusivity(self, content: str) -> float:
        """
        Evaluate how inclusive the content is.
        
        Args:
            content: Text content to analyze.
            
        Returns:
            Inclusivity score (0.0-1.0), higher is better.
        """
        # This is a simplified implementation
        
        # Check for inclusive language patterns
        inclusive_patterns = [
            r"\bdiverse\b", 
            r"\binclusive\b",
            r"\bequity\b", 
            r"\baccessib(le|ility)\b",
            r"\bfair(ness)?\b"
        ]
        
        inclusive_count = 0
        for pattern in inclusive_patterns:
            matches = re.findall(pattern, content.lower())
            inclusive_count += len(matches)
        
        # Check for exclusionary language
        exclusionary_patterns = [
            r"\bonly for\b", 
            r"\bexclud(e|ing)\b",
            r"\bnot allowed\b"
        ]
        
        exclusionary_count = 0
        for pattern in exclusionary_patterns:
            matches = re.findall(pattern, content.lower())
            exclusionary_count += len(matches)
        
        # Start with a default score
        base_score = 0.8
        
        # Adjust based on inclusive and exclusionary language
        adjusted_score = base_score + (inclusive_count * 0.05) - (exclusionary_count * 0.1)
        
        # Ensure score is within bounds
        return max(0.0, min(1.0, adjusted_score))
    
    def _evaluate_privacy(self, content: str) -> float:
        """
        Evaluate privacy concerns in the content.
        
        Args:
            content: Text content to analyze.
            
        Returns:
            Privacy score (0.0-1.0), higher is better.
        """
        # Check for potential personally identifiable information (PII)
        pii_patterns = [
            r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b",  # SSN
            r"\b\d{4}[-.\s]?\d{4}[-.\s]?\d{4}[-.\s]?\d{4}\b",  # Credit card
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",  # Email
            r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"  # Phone number
        ]
        
        pii_count = 0
        for pattern in pii_patterns:
            matches = re.findall(pattern, content)
            pii_count += len(matches)
        
        # Start with perfect score and reduce based on PII
        privacy_score = 1.0 - (pii_count * 0.2)
        
        # Check for data protection language
        protection_patterns = [
            r"\banonymized\b", 
            r"\bconfidential\b",
            r"\bencrypted\b", 
            r"\bsecure\b",
            r"\bprivacy\b"
        ]
        
        protection_count = 0
        for pattern in protection_patterns:
            matches = re.findall(pattern, content.lower())
            protection_count += len(matches)
        
        # Improve score slightly based on protection language
        privacy_score += (protection_count * 0.05)
        
        # Ensure score is within bounds
        return max(0.0, min(1.0, privacy_score))
    
    def _evaluate_human_autonomy(self, content: str) -> float:
        """
        Evaluate respect for human autonomy in the content.
        
        Args:
            content: Text content to analyze.
            
        Returns:
            Autonomy score (0.0-1.0), higher is better.
        """
        # Check for language that respects human agency and autonomy
        autonomy_patterns = [
            r"\bchoice\b", 
            r"\bconsent\b",
            r"\bopt(-| )in\b", 
            r"\bopt(-| )out\b",
            r"\bcontrol\b", 
            r"\buser preference\b",
            r"\bdecision\b"
        ]
        
        autonomy_count = 0
        for pattern in autonomy_patterns:
            matches = re.findall(pattern, content.lower())
            autonomy_count += len(matches)
        
        # Check for language that could undermine autonomy
        control_patterns = [
            r"\bforce\b", 
            r"\bcompel\b",
            r"\bmust\b", 
            r"\brequired\b",
            r"\bmandate\b", 
            r"\bautomatically\b",
            r"\bwithout consent\b", 
            r"\bno choice\b"
        ]
        
        control_count = 0
        for pattern in control_patterns:
            matches = re.findall(pattern, content.lower())
            control_count += len(matches)
        
        # Start with a default score
        base_score = 0.7
        
        # Adjust based on autonomy and control language
        adjusted_score = base_score + (autonomy_count * 0.05) - (control_count * 0.1)
        
        # Ensure score is within bounds
        return max(0.0, min(1.0, adjusted_score))
    
    def _determine_risk_level(self, overall_score: float) -> EthicalSeverity:
        """
        Determine the ethical risk level based on the overall score.
        
        Args:
            overall_score: The overall ethical score (0.0-1.0).
            
        Returns:
            EthicalSeverity enum value.
        """
        if overall_score < 0.3:
            return EthicalSeverity.CRITICAL
        elif overall_score < 0.5:
            return EthicalSeverity.HIGH
        elif overall_score < 0.7:
            return EthicalSeverity.MEDIUM
        elif overall_score < 0.9:
            return EthicalSeverity.LOW
        else:
            return EthicalSeverity.NONE
    
    def _generate_issues_and_recommendations(
        self, 
        category_scores: Dict[EthicalCategory, float],
        bias_scores: Dict[BiasType, float],
        overall_score: float
    ) -> Tuple[List[str], List[str]]:
        """
        Generate lists of issues and recommendations based on assessment results.
        
        Args:
            category_scores: Dictionary mapping ethical categories to scores.
            bias_scores: Dictionary mapping bias types to scores.
            overall_score: The overall ethical score.
            
        Returns:
            Tuple of (issues list, recommendations list).
        """
        issues = []
        recommendations = []
        
        # Check for low category scores
        for category, score in category_scores.items():
            if score < 0.5:
                issues.append(f"Low {category.value} score ({score:.2f})")
                
                # Add category-specific recommendations
                if category == EthicalCategory.FAIRNESS:
                    recommendations.append("Review content for potential bias and unfair representation")
                elif category == EthicalCategory.INCLUSIVITY:
                    recommendations.append("Use more inclusive language and consider diverse perspectives")
                elif category == EthicalCategory.PRIVACY:
                    recommendations.append("Remove or anonymize potentially sensitive personal information")
                elif category == EthicalCategory.HUMAN_AUTONOMY:
                    recommendations.append("Emphasize human choice and agency in decision-making processes")
        
        # Check for bias issues
        for bias_type, score in bias_scores.items():
            issues.append(f"Detected {bias_type.value} bias (severity: {score:.2f})")
            
            # Add bias-specific recommendations
            if bias_type == BiasType.DEMOGRAPHIC:
                recommendations.append("Avoid generalizations about demographic groups")
            elif bias_type == BiasType.REPRESENTATION:
                recommendations.append("Ensure fair and balanced representation of different groups")
            elif bias_type == BiasType.LANGUAGE:
                recommendations.append("Use neutral language that avoids reinforcing stereotypes")
        
        # Add general recommendations based on overall score
        if overall_score < 0.6:
            recommendations.append("Perform a comprehensive ethical review before proceeding")
        
        return issues, recommendations
    
    def _load_sensitive_terms(self) -> Dict[str, List[str]]:
        """
        Load sensitive terms for bias detection.
        
        Returns:
            Dictionary mapping categories to lists of sensitive terms.
        """
        # This is a simplified implementation
        # A real implementation would load from a more comprehensive database
        
        sensitive_terms = {
            "race": [
                "black", "white", "asian", "hispanic", "latino", "latinx",
                "african american", "caucasian", "pacific islander", "indigenous",
                "native american", "middle eastern"
            ],
            "gender": [
                "man", "woman", "boy", "girl", "male", "female", 
                "transgender", "nonbinary", "genderqueer", "gender fluid"
            ],
            "religion": [
                "christian", "muslim", "jewish", "hindu", "buddhist", 
                "atheist", "agnostic", "sikh", "jain", "pagan"
            ],
            "nationality": [
                "american", "canadian", "mexican", "chinese", "russian", 
                "indian", "british", "german", "french", "japanese"
            ],
            "disability": [
                "disabled", "blind", "deaf", "wheelchair", "neurodivergent", 
                "autistic", "adhd", "mobility impaired", "learning disability"
            ],
            "age": [
                "young", "old", "elderly", "senior", "boomer", 
                "millennial", "gen z", "gen x", "teenager", "child"
            ]
        }
        
        return sensitive_terms
    
    def _compile_bias_patterns(self) -> Dict[str, re.Pattern]:
        """
        Compile regex patterns for bias detection.
        
        Returns:
            Dictionary mapping pattern names to compiled regex patterns.
        """
        patterns = {
            "generalization": re.compile(r"(all|every|always|never|typical|those|these) ([^\s]+)"),
            "stereotyping": re.compile(r"([^\s]+) are ([^\s]+)"),
            "undermining": re.compile(r"(articulate|well-spoken|credit to) ([^\s]+)"),
            "othering": re.compile(r"(those|these|you) people"),
        }
        
        return patterns
    
    def reweight_memory(self, key: str, new_weight: float) -> bool:
        """
        Adjust the ethical weight of a memory entry.
        
        Args:
            key: Memory key to adjust.
            new_weight: New ethical weight to assign.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            # Retrieve the memory entry
            memory = self.memory_manager.get_memory(key)
            
            if not memory:
                logger.warning(f"Memory key '{key}' not found for reweighting")
                return False
            
            # Update ethical assessment in metadata
            if "metadata" not in memory:
                memory["metadata"] = {}
                
            if "ethical_assessment" not in memory["metadata"]:
                memory["metadata"]["ethical_assessment"] = {}
                
            memory["metadata"]["ethical_assessment"]["overall_score"] = new_weight
            
            # Update the memory with new metadata
            success = self.memory_manager.update_memory(key, memory)
            
            return success
        except Exception as e:
            logger.error(f"Failed to reweight memory '{key}': {str(e)}")
            return False
    
    def get_ethical_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics about ethical assessments across memories.
        
        Returns:
            Dictionary with ethical summary statistics.
        """
        try:
            # Get all memories
            all_memories = self.memory_manager.list_memories()
            
            # Initialize stats
            stats = {
                "total_memories": len(all_memories),
                "memories_with_assessment": 0,
                "average_ethical_score": 0.0,
                "risk_level_counts": {
                    level.value: 0 for level in EthicalSeverity
                },
                "category_average_scores": {
                    category.value: 0.0 for category in EthicalCategory
                },
                "bias_type_counts": {
                    bias_type.value: 0 for bias_type in BiasType
                }
            }
            
            # Collect assessment data
            ethical_scores = []
            
            for memory in all_memories:
                if (
                    memory.get("metadata") 
                    and memory["metadata"].get("ethical_assessment")
                    and "overall_score" in memory["metadata"]["ethical_assessment"]
                ):
                    stats["memories_with_assessment"] += 1
                    score = memory["metadata"]["ethical_assessment"]["overall_score"]
                    ethical_scores.append(score)
                    
                    # Count risk levels
                    if "risk_level" in memory["metadata"]["ethical_assessment"]:
                        risk_level = memory["metadata"]["ethical_assessment"]["risk_level"]
                        stats["risk_level_counts"][risk_level] += 1
                    
                    # Collect category scores if available
                    if "category_scores" in memory["metadata"]["ethical_assessment"]:
                        for category, category_score in memory["metadata"]["ethical_assessment"]["category_scores"].items():
                            stats["category_average_scores"][category] += category_score
                    
                    # Count bias types if available
                    if "bias_scores" in memory["metadata"]["ethical_assessment"]:
                        for bias_type in memory["metadata"]["ethical_assessment"]["bias_scores"]:
                            stats["bias_type_counts"][bias_type] += 1
            
            # Calculate averages
            if ethical_scores:
                stats["average_ethical_score"] = sum(ethical_scores) / len(ethical_scores)
                
                # Average category scores
                for category in EthicalCategory:
                    if stats["memories_with_assessment"] > 0:
                        stats["category_average_scores"][category.value] /= stats["memories_with_assessment"]
            
            return stats
        
        except Exception as e:
            logger.error(f"Failed to generate ethical summary: {str(e)}")
            return {"error": str(e)}
    
    def export_ethical_guidelines(self, format: str = "json") -> str:
        """
        Export the ethical guidelines and configuration used by the system.
        
        Args:
            format: Output format ("json" or "text").
            
        Returns:
            Formatted guidelines as a string.
        """
        guidelines = {
            "config": self.config.dict(),
            "categories": [category.value for category in EthicalCategory],
            "bias_types": [bias_type.value for bias_type in BiasType],
            "severity_levels": [severity.value for severity in EthicalSeverity],
            "sensitive_categories": list(self.sensitive_terms.keys())
        }
        
        if format.lower() == "json":
            return json.dumps(guidelines, indent=2)
        else:
            # Text format
            text = "ETHICAL MEMORY GUIDELINES\n"
            text += "=======================\n\n"
            
            text += "Configuration:\n"
            for key, value in self.config.dict().items():
                text += f"  - {key}: {value}\n"
            
            text += "\nEthical Categories:\n"
            for category in guidelines["categories"]:
                text += f"  - {category}\n"
            
            text += "\nBias Types:\n"
            for bias_type in guidelines["bias_types"]:
                text += f"  - {bias_type}\n"
            
            text += "\nSeverity Levels:\n"
            for level in guidelines["severity_levels"]:
                text += f"  - {level}\n"
            
            text += "\nSensitive Categories Monitored:\n"
            for category in guidelines["sensitive_categories"]:
                text += f"  - {category}\n"
            
            return text
