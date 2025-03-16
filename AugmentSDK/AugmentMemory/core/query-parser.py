"""
Query Parser for the Memory Orchestration Module (MOM) in Augment SDK.

This module provides functionality for parsing, validating, and preparing memory queries
before they are processed by the memory retrieval system. It handles complex query syntax,
including filters, memory layer targeting, relevance scoring, and metadata extraction.

The QueryParser supports various memory layers (ephemeral, working, semantic, procedural,
reflective, and predictive) and enables context-aware memory recall through specialized
query operators.

Example usage:
    query = "concept:AI ethics context:healthcare layer:semantic limit:5"
    parser = QueryParser()
    parsed_query = parser.parse(query)
    # Use parsed_query with the memory retrieval system

Classes:
    QueryParser: Main class for parsing and processing memory queries
    QueryType: Enum for different types of memory queries
    QueryOperator: Enum for different query operations
    ParsedQuery: Data class to hold parsed query information
"""

import re
import enum
import logging
from typing import Dict, List, Optional, Set, Tuple, Union, Any
from dataclasses import dataclass, field

# Configure logger
logger = logging.getLogger(__name__)

class QueryType(enum.Enum):
    """Enum defining the types of memory queries supported by the system."""
    KEYWORD = "keyword"     # Simple keyword-based search
    SEMANTIC = "semantic"   # Semantic similarity search
    HYBRID = "hybrid"       # Combined keyword and semantic search
    TEMPORAL = "temporal"   # Time-based memory search
    RELATIONAL = "relational"  # Query based on relationships between memories
    RECURSIVE = "recursive"    # Recursive memory exploration


class QueryOperator(enum.Enum):
    """Enum defining the operators used in query filters."""
    EQUALS = "equals"           # Exact match (=)
    CONTAINS = "contains"       # Contains substring (contains:)
    GREATER_THAN = "gt"         # Greater than (>)
    LESS_THAN = "lt"            # Less than (<)
    GREATER_EQUAL = "gte"       # Greater than or equal (>=)
    LESS_EQUAL = "lte"          # Less than or equal (<=)
    NOT_EQUALS = "not"          # Not equal (!=)
    AND = "and"                 # Logical AND (&)
    OR = "or"                   # Logical OR (|)
    SIMILAR = "similar"         # Semantic similarity (~)
    BEFORE = "before"           # Temporal before (before:)
    AFTER = "after"             # Temporal after (after:)
    CONNECTED = "connected"     # Relational connection (connected:)


@dataclass
class ParsedQuery:
    """Data class to hold the parsed query information."""
    raw_query: str
    query_type: QueryType = QueryType.SEMANTIC
    search_text: str = ""
    memory_layer: str = "semantic"
    filters: Dict[str, Any] = field(default_factory=dict)
    limit: int = 10
    offset: int = 0
    min_similarity: float = 0.7
    sort_by: str = "relevance"
    sort_order: str = "desc"
    context: Optional[str] = None
    temporal_range: Optional[Tuple[str, str]] = None
    metadata_requirements: Dict[str, Any] = field(default_factory=dict)
    related_concepts: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the parsed query to a dictionary representation."""
        return {
            "raw_query": self.raw_query,
            "query_type": self.query_type.value,
            "search_text": self.search_text,
            "memory_layer": self.memory_layer,
            "filters": self.filters,
            "limit": self.limit,
            "offset": self.offset,
            "min_similarity": self.min_similarity,
            "sort_by": self.sort_by,
            "sort_order": self.sort_order,
            "context": self.context,
            "temporal_range": self.temporal_range,
            "metadata_requirements": self.metadata_requirements,
            "related_concepts": self.related_concepts
        }


class QueryParseError(Exception):
    """Exception raised when a query cannot be parsed correctly."""
    pass


class UnsupportedOperatorError(Exception):
    """Exception raised when an unsupported operator is used in a query."""
    pass


class QueryParser:
    """
    Parser for memory queries in the Augment SDK.
    
    This class provides methods to parse, validate and prepare queries for use
    with the memory retrieval system. It can handle complex query syntax and
    supports various filters and operators.
    """
    
    # Define valid memory layers
    VALID_MEMORY_LAYERS = {
        "ephemeral", "working", "semantic", "procedural", 
        "reflective", "predictive", "all"
    }
    
    # Define common query patterns
    QUERY_PATTERNS = {
        # Layer specification: layer:semantic
        "layer": re.compile(r'layer:(\w+)'),
        
        # Limit specification: limit:10
        "limit": re.compile(r'limit:(\d+)'),
        
        # Offset specification: offset:5
        "offset": re.compile(r'offset:(\d+)'),
        
        # Context specification: context:healthcare
        "context": re.compile(r'context:([a-zA-Z0-9_-]+)'),
        
        # Concept specification: concept:ethics
        "concept": re.compile(r'concept:([a-zA-Z0-9_-]+)'),
        
        # Similarity threshold: similarity:0.8
        "similarity": re.compile(r'similarity:(0\.\d+)'),
        
        # Metadata filter: metadata.author:John
        "metadata": re.compile(r'metadata\.([a-zA-Z0-9_]+):([^:\s]+)'),
        
        # Temporal filters
        "before": re.compile(r'before:([^:\s]+)'),
        "after": re.compile(r'after:([^:\s]+)'),
        
        # Related concepts: related:concept1,concept2
        "related": re.compile(r'related:([^:\s]+(?:,[^:\s]+)*)'),
        
        # Sort specification: sort:timestamp:desc
        "sort": re.compile(r'sort:([a-zA-Z0-9_]+)(?::(\w+))?'),
    }

    def __init__(self, default_memory_layer: str = "semantic"):
        """
        Initialize the QueryParser with default settings.
        
        Args:
            default_memory_layer (str): The default memory layer to use if 
                                        not specified in query. Defaults to "semantic".
        """
        if default_memory_layer not in self.VALID_MEMORY_LAYERS:
            raise ValueError(f"Invalid default memory layer: {default_memory_layer}. "
                            f"Must be one of {self.VALID_MEMORY_LAYERS}")
                            
        self.default_memory_layer = default_memory_layer
        logger.debug(f"Initialized QueryParser with default_memory_layer={default_memory_layer}")

    def parse(self, query: str) -> ParsedQuery:
        """
        Parse a query string into a structured ParsedQuery object.
        
        This method processes the query string to extract search terms, filters,
        operators, and other query parameters.
        
        Args:
            query (str): The raw query string to parse
            
        Returns:
            ParsedQuery: A structured representation of the parsed query
            
        Raises:
            QueryParseError: If the query cannot be parsed correctly
        """
        if not query or not isinstance(query, str):
            raise QueryParseError("Query must be a non-empty string")
            
        logger.debug(f"Parsing query: {query}")
        
        # Create a new ParsedQuery object with the raw query
        parsed_query = ParsedQuery(raw_query=query)
        
        # Extract known patterns from the query
        self._extract_patterns(query, parsed_query)
        
        # Determine the query type based on patterns and operators
        parsed_query.query_type = self._determine_query_type(query, parsed_query)
        
        # Extract the main search text (removing special operators)
        parsed_query.search_text = self._extract_search_text(query)
        
        # Validate the final parsed query
        self._validate_parsed_query(parsed_query)
        
        logger.info(f"Successfully parsed query into {parsed_query.query_type.value} type")
        return parsed_query

    def _extract_patterns(self, query: str, parsed_query: ParsedQuery) -> None:
        """
        Extract known patterns from the query string and update the parsed_query object.
        
        Args:
            query (str): The raw query string
            parsed_query (ParsedQuery): The parsed query object to update
            
        Raises:
            QueryParseError: If a pattern is found but cannot be correctly interpreted
        """
        # Check for layer specification
        layer_match = self.QUERY_PATTERNS["layer"].search(query)
        if layer_match:
            layer = layer_match.group(1).lower()
            if layer not in self.VALID_MEMORY_LAYERS:
                raise QueryParseError(f"Invalid memory layer: {layer}. "
                                    f"Must be one of {self.VALID_MEMORY_LAYERS}")
            parsed_query.memory_layer = layer
        
        # Check for limit specification
        limit_match = self.QUERY_PATTERNS["limit"].search(query)
        if limit_match:
            try:
                parsed_query.limit = int(limit_match.group(1))
                if parsed_query.limit < 1:
                    raise QueryParseError("Limit must be a positive integer")
            except ValueError:
                raise QueryParseError(f"Invalid limit value: {limit_match.group(1)}")
        
        # Check for offset specification
        offset_match = self.QUERY_PATTERNS["offset"].search(query)
        if offset_match:
            try:
                parsed_query.offset = int(offset_match.group(1))
                if parsed_query.offset < 0:
                    raise QueryParseError("Offset cannot be negative")
            except ValueError:
                raise QueryParseError(f"Invalid offset value: {offset_match.group(1)}")
        
        # Check for context specification
        context_match = self.QUERY_PATTERNS["context"].search(query)
        if context_match:
            parsed_query.context = context_match.group(1)
        
        # Check for similarity threshold
        similarity_match = self.QUERY_PATTERNS["similarity"].search(query)
        if similarity_match:
            try:
                parsed_query.min_similarity = float(similarity_match.group(1))
                if not 0 <= parsed_query.min_similarity <= 1:
                    raise QueryParseError("Similarity threshold must be between 0 and 1")
            except ValueError:
                raise QueryParseError(f"Invalid similarity value: {similarity_match.group(1)}")
        
        # Extract metadata filters
        metadata_matches = self.QUERY_PATTERNS["metadata"].finditer(query)
        for match in metadata_matches:
            key, value = match.groups()
            parsed_query.metadata_requirements[key] = value
        
        # Extract temporal range if specified
        before_match = self.QUERY_PATTERNS["before"].search(query)
        after_match = self.QUERY_PATTERNS["after"].search(query)
        
        if before_match or after_match:
            before_date = before_match.group(1) if before_match else None
            after_date = after_match.group(1) if after_match else None
            parsed_query.temporal_range = (after_date, before_date)
        
        # Extract related concepts
        related_match = self.QUERY_PATTERNS["related"].search(query)
        if related_match:
            related_concepts = related_match.group(1).split(',')
            parsed_query.related_concepts = [concept.strip() for concept in related_concepts]
        
        # Extract sort specifications
        sort_match = self.QUERY_PATTERNS["sort"].search(query)
        if sort_match:
            parsed_query.sort_by = sort_match.group(1)
            if sort_match.group(2):
                order = sort_match.group(2).lower()
                if order not in ["asc", "desc"]:
                    raise QueryParseError(f"Invalid sort order: {order}. Must be 'asc' or 'desc'")
                parsed_query.sort_order = order

    def _determine_query_type(self, query: str, parsed_query: ParsedQuery) -> QueryType:
        """
        Determine the type of query based on its content and patterns.
        
        Args:
            query (str): The raw query string
            parsed_query (ParsedQuery): The partially parsed query
            
        Returns:
            QueryType: The determined query type
        """
        # Default to semantic search
        query_type = QueryType.SEMANTIC
        
        # Check for temporal indicators
        if parsed_query.temporal_range or "before:" in query or "after:" in query:
            query_type = QueryType.TEMPORAL
        
        # Check for relational indicators
        if parsed_query.related_concepts or "connected:" in query:
            query_type = QueryType.RELATIONAL
        
        # Check for recursive exploration indicators
        if "recursive:" in query or "depth:" in query:
            query_type = QueryType.RECURSIVE
            
        # Check for hybrid search indicators (both semantic and keyword)
        if "~" in query or "similar:" in query:
            # If we already have a different type, make it hybrid
            if query_type != QueryType.SEMANTIC:
                query_type = QueryType.HYBRID
        
        # If query has simple keywords without operators, it's a keyword search
        has_operators = any(pattern in query for pattern in ["layer:", "context:", "concept:",
                                                          "before:", "after:", "metadata.",
                                                          "related:", "similarity:"])
        if not has_operators and query_type == QueryType.SEMANTIC:
            query_type = QueryType.KEYWORD
            
        return query_type

    def _extract_search_text(self, query: str) -> str:
        """
        Extract the main search text from the query, removing all special operators.
        
        Args:
            query (str): The raw query string
            
        Returns:
            str: The extracted search text
        """
        # Make a copy of the query to modify
        search_text = query
        
        # Remove all known patterns/operators
        for pattern_name, pattern in self.QUERY_PATTERNS.items():
            search_text = pattern.sub('', search_text)
        
        # Clean up any leftover artifacts and normalize whitespace
        search_text = re.sub(r'\s+', ' ', search_text).strip()
        
        return search_text

    def _validate_parsed_query(self, parsed_query: ParsedQuery) -> None:
        """
        Validate the final parsed query for consistency and correctness.
        
        Args:
            parsed_query (ParsedQuery): The parsed query to validate
            
        Raises:
            QueryParseError: If the parsed query is invalid
        """
        # Check if we have at least some search criteria
        has_criteria = (parsed_query.search_text or 
                      parsed_query.metadata_requirements or 
                      parsed_query.related_concepts or 
                      parsed_query.temporal_range or
                      parsed_query.context)
                      
        if not has_criteria:
            raise QueryParseError("Query must contain at least some search criteria")
            
        # For temporal queries, ensure we have a temporal range
        if parsed_query.query_type == QueryType.TEMPORAL and not parsed_query.temporal_range:
            logger.warning("Query is classified as temporal but no explicit temporal range was provided")
            
        # For relational queries, ensure we have related concepts
        if parsed_query.query_type == QueryType.RELATIONAL and not parsed_query.related_concepts:
            logger.warning("Query is classified as relational but no explicit related concepts were provided")

    def decompose_complex_query(self, query: str) -> List[ParsedQuery]:
        """
        Decompose a complex query with multiple parts into a list of simpler queries.
        
        Args:
            query (str): The complex query string
            
        Returns:
            List[ParsedQuery]: A list of parsed queries, one for each part
            
        Raises:
            QueryParseError: If the query cannot be decomposed
        """
        # Split on logical operators (if properly formatted)
        parts = []
        
        # Check if the query is using explicit logical operators
        if " AND " in query.upper():
            parts = query.split(" AND ")
            logger.debug(f"Decomposed query using AND into {len(parts)} parts")
        elif " OR " in query.upper():
            parts = query.split(" OR ")
            logger.debug(f"Decomposed query using OR into {len(parts)} parts")
        elif ";" in query:
            # Semicolon can be used as a query separator
            parts = query.split(";")
            logger.debug(f"Decomposed query using semicolons into {len(parts)} parts")
        
        # If no explicit decomposition markers found, treat as a single query
        if not parts:
            return [self.parse(query)]
            
        # Parse each part separately
        parsed_parts = []
        for part in parts:
            part = part.strip()
            if part:  # Skip empty parts
                parsed_parts.append(self.parse(part))
                
        return parsed_parts

    def rewrite_query_for_layer(self, parsed_query: ParsedQuery, target_layer: str) -> ParsedQuery:
        """
        Rewrite a query to target a specific memory layer.
        
        Args:
            parsed_query (ParsedQuery): The original parsed query
            target_layer (str): The target memory layer
            
        Returns:
            ParsedQuery: A new parsed query targeting the specified layer
            
        Raises:
            ValueError: If the target layer is invalid
        """
        if target_layer not in self.VALID_MEMORY_LAYERS:
            raise ValueError(f"Invalid target memory layer: {target_layer}")
            
        # Create a copy of the parsed query
        new_query = ParsedQuery(
            raw_query=parsed_query.raw_query,
            query_type=parsed_query.query_type,
            search_text=parsed_query.search_text,
            memory_layer=target_layer,  # Set the new layer
            filters=parsed_query.filters.copy(),
            limit=parsed_query.limit,
            offset=parsed_query.offset,
            min_similarity=parsed_query.min_similarity,
            sort_by=parsed_query.sort_by,
            sort_order=parsed_query.sort_order,
            context=parsed_query.context,
            temporal_range=parsed_query.temporal_range,
            metadata_requirements=parsed_query.metadata_requirements.copy(),
            related_concepts=parsed_query.related_concepts.copy()
        )
        
        logger.debug(f"Rewrote query to target {target_layer} layer")
        return new_query

    def optimize_query(self, parsed_query: ParsedQuery) -> ParsedQuery:
        """
        Optimize a parsed query for better performance.
        
        Args:
            parsed_query (ParsedQuery): The original parsed query
            
        Returns:
            ParsedQuery: An optimized version of the query
        """
        # Create a copy to avoid modifying the original
        optimized = ParsedQuery(
            raw_query=parsed_query.raw_query,
            query_type=parsed_query.query_type,
            search_text=parsed_query.search_text,
            memory_layer=parsed_query.memory_layer,
            filters=parsed_query.filters.copy(),
            limit=parsed_query.limit,
            offset=parsed_query.offset,
            min_similarity=parsed_query.min_similarity,
            sort_by=parsed_query.sort_by,
            sort_order=parsed_query.sort_order,
            context=parsed_query.context,
            temporal_range=parsed_query.temporal_range,
            metadata_requirements=parsed_query.metadata_requirements.copy(),
            related_concepts=parsed_query.related_concepts.copy()
        )
        
        # Apply optimizations based on query type
        if optimized.query_type == QueryType.SEMANTIC and not optimized.search_text.strip():
            # For semantic queries without explicit search text, try to extract meaning from context
            if optimized.context:
                optimized.search_text = optimized.context
                logger.debug("Using context as search text for semantic query")
                
        # Set reasonable limit if not specified or too large
        if optimized.limit > 100:
            optimized.limit = 100
            logger.debug("Limited query results to 100 for performance")
            
        # Ensure at least some minimum similarity for semantic queries
        if optimized.query_type in [QueryType.SEMANTIC, QueryType.HYBRID] and optimized.min_similarity < 0.5:
            optimized.min_similarity = 0.5
            logger.debug("Increased minimum similarity threshold to 0.5")
            
        return optimized

    def format_for_vector_search(self, parsed_query: ParsedQuery) -> Dict[str, Any]:
        """
        Format a parsed query specifically for use with vector databases/search.
        
        Args:
            parsed_query (ParsedQuery): The parsed query to format
            
        Returns:
            Dict[str, Any]: A dictionary representation suitable for vector search
        """
        vector_query = {
            "text": parsed_query.search_text,
            "top_k": parsed_query.limit,
            "min_score": parsed_query.min_similarity,
        }
        
        # Add filters based on metadata requirements
        if parsed_query.metadata_requirements:
            vector_query["filter"] = {
                "metadata": parsed_query.metadata_requirements
            }
            
        # Add layer filtering if not "all"
        if parsed_query.memory_layer != "all":
            if "filter" not in vector_query:
                vector_query["filter"] = {}
            vector_query["filter"]["layer"] = parsed_query.memory_layer
            
        # Add temporal constraints if specified
        if parsed_query.temporal_range:
            after_date, before_date = parsed_query.temporal_range
            if "filter" not in vector_query:
                vector_query["filter"] = {}
                
            vector_query["filter"]["timestamp"] = {}
            
            if after_date:
                vector_query["filter"]["timestamp"]["gte"] = after_date
                
            if before_date:
                vector_query["filter"]["timestamp"]["lte"] = before_date
                
        logger.debug(f"Formatted query for vector search: {vector_query}")
        return vector_query

    def build_query_string(self, parsed_query: ParsedQuery) -> str:
        """
        Build a query string from a ParsedQuery object.
        
        This is essentially the reverse of parse(), converting a structured query
        back into a string representation.
        
        Args:
            parsed_query (ParsedQuery): The parsed query to convert to string
            
        Returns:
            str: The reconstructed query string
        """
        parts = []
        
        # Add the main search text
        if parsed_query.search_text:
            parts.append(parsed_query.search_text)
            
        # Add layer specification
        if parsed_query.memory_layer != self.default_memory_layer:
            parts.append(f"layer:{parsed_query.memory_layer}")
            
        # Add limit if not default
        if parsed_query.limit != 10:
            parts.append(f"limit:{parsed_query.limit}")
            
        # Add offset if not default
        if parsed_query.offset != 0:
            parts.append(f"offset:{parsed_query.offset}")
            
        # Add context if specified
        if parsed_query.context:
            parts.append(f"context:{parsed_query.context}")
            
        # Add similarity threshold if not default
        if parsed_query.min_similarity != 0.7:
            parts.append(f"similarity:{parsed_query.min_similarity:.1f}")
            
        # Add metadata requirements
        for key, value in parsed_query.metadata_requirements.items():
            parts.append(f"metadata.{key}:{value}")
            
        # Add temporal range if specified
        if parsed_query.temporal_range:
            after_date, before_date = parsed_query.temporal_range
            if after_date:
                parts.append(f"after:{after_date}")
            if before_date:
                parts.append(f"before:{before_date}")
                
        # Add related concepts
        if parsed_query.related_concepts:
            concepts_str = ",".join(parsed_query.related_concepts)
            parts.append(f"related:{concepts_str}")
            
        # Add sort specification if not default
        if parsed_query.sort_by != "relevance" or parsed_query.sort_order != "desc":
            parts.append(f"sort:{parsed_query.sort_by}:{parsed_query.sort_order}")
            
        # Join all parts with spaces
        query_string = " ".join(parts)
        
        logger.debug(f"Built query string: {query_string}")
        return query_string
