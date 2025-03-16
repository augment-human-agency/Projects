"""
Advanced logging system for Augment SDK.

This module provides a comprehensive logging infrastructure for the Augment SDK,
offering hierarchical logging capabilities with context awareness, memory layer tracking,
and intelligent log management. The logger is designed to support the various memory
layers (ephemeral, working, semantic, procedural, reflective, and predictive) with
appropriate verbosity levels and formatting.

The logger implements both file and console handlers with configurable rotation policies,
log levels per component, and thread-safe operations. It also provides specialized
logging methods for memory operations, self-reflection events, and system diagnostics.

Typical usage:
    ```python
    from augment_sdk.utils.logger import get_logger

    # Get a logger for a specific component
    logger = get_logger("memory.semantic")
    
    # Log with context
    logger.info("Storing new concept", extra={"memory_id": "concept_123", "layer": "semantic"})
    
    # Log a memory operation
    logger.memory_operation(
        operation="store",
        memory_layer="semantic", 
        success=True, 
        metadata={"vector_id": "vec_456", "relevance_score": 0.89}
    )
    ```

The logger also supports structured logging for analytics and monitoring integrations.
"""

import os
import sys
import json
import time
import logging
import datetime
import threading
import traceback
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Union, List, Tuple, Callable
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import uuid

# Define constants
DEFAULT_LOG_LEVEL = logging.INFO
DEFAULT_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
MAX_LOG_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
LOG_BACKUP_COUNT = 5


class MemoryLayer(Enum):
    """Enum representing the different memory layers in Augment SDK."""
    EPHEMERAL = "ephemeral"
    WORKING = "working"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    REFLECTIVE = "reflective"
    PREDICTIVE = "predictive"
    SYSTEM = "system"  # For system-level logs


class LogLevel(Enum):
    """Enum mapping Augment SDK log levels to standard logging levels."""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL
    MEMORY = 15  # Custom level between DEBUG and INFO for memory operations
    REFLECTION = 25  # Custom level between INFO and WARNING for reflection logs


class AugmentLogRecord(logging.LogRecord):
    """Extended LogRecord with additional fields for Augment SDK context."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory_layer = getattr(self, 'memory_layer', None)
        self.operation_id = getattr(self, 'operation_id', None)
        self.memory_id = getattr(self, 'memory_id', None)
        self.execution_time = getattr(self, 'execution_time', None)
        self.metadata = getattr(self, 'metadata', {})


class StructuredFormatter(logging.Formatter):
    """Formatter that outputs JSON formatted logs."""
    
    def __init__(self, fmt=None, datefmt=None):
        super().__init__(fmt, datefmt)
    
    def format(self, record) -> str:
        """Format the record as JSON.
        
        Args:
            record: The log record to format.
            
        Returns:
            A JSON string representing the log record.
        """
        log_data = {
            'timestamp': self.formatTime(record, self.datefmt),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'thread_id': record.thread,
            'thread_name': record.threadName,
            'process_id': record.process
        }
        
        # Add Augment SDK specific fields if available
        if hasattr(record, 'memory_layer') and record.memory_layer:
            log_data['memory_layer'] = record.memory_layer
            
        if hasattr(record, 'operation_id') and record.operation_id:
            log_data['operation_id'] = record.operation_id
            
        if hasattr(record, 'memory_id') and record.memory_id:
            log_data['memory_id'] = record.memory_id
            
        if hasattr(record, 'execution_time') and record.execution_time:
            log_data['execution_time'] = record.execution_time
            
        if hasattr(record, 'metadata') and record.metadata:
            log_data['metadata'] = record.metadata
            
        # Add exception info if available
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
            
        return json.dumps(log_data)


class ConsoleColorFormatter(logging.Formatter):
    """Formatter for console output with color coding based on log level."""
    
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[41m',   # Red background
        'MEMORY': '\033[35m',     # Magenta
        'REFLECTION': '\033[34m', # Blue
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record) -> str:
        """Format the record with color coding.
        
        Args:
            record: The log record to format.
            
        Returns:
            A color-formatted string for console output.
        """
        log_message = super().format(record)
        level_name = record.levelname
        
        # Handle custom log levels
        if record.levelno == LogLevel.MEMORY.value:
            level_name = "MEMORY"
        elif record.levelno == LogLevel.REFLECTION.value:
            level_name = "REFLECTION"
            
        # Apply color if the level has a color defined
        if level_name in self.COLORS:
            return f"{self.COLORS[level_name]}{log_message}{self.COLORS['RESET']}"
        
        return log_message


class AugmentLogger(logging.Logger):
    """Extended Logger with additional methods for Augment SDK operations."""
    
    def __init__(self, name: str, level: int = logging.NOTSET):
        """Initialize the AugmentLogger.
        
        Args:
            name: The name of the logger.
            level: The log level.
        """
        super().__init__(name, level)
        
        # Register custom log levels
        logging.addLevelName(LogLevel.MEMORY.value, "MEMORY")
        logging.addLevelName(LogLevel.REFLECTION.value, "REFLECTION")
    
    def memory_operation(
        self, 
        operation: str, 
        memory_layer: Union[str, MemoryLayer], 
        success: bool, 
        metadata: Optional[Dict[str, Any]] = None,
        execution_time: Optional[float] = None
    ) -> None:
        """Log a memory operation with structured metadata.
        
        Args:
            operation: The operation being performed (e.g., 'store', 'retrieve').
            memory_layer: The memory layer involved.
            success: Whether the operation was successful.
            metadata: Additional metadata about the operation.
            execution_time: How long the operation took in seconds.
        """
        if isinstance(memory_layer, MemoryLayer):
            memory_layer = memory_layer.value
        
        if metadata is None:
            metadata = {}
            
        metadata.update({
            'operation': operation,
            'success': success,
            'timestamp': datetime.datetime.now().isoformat()
        })
        
        extra = {
            'memory_layer': memory_layer,
            'operation_id': str(uuid.uuid4()),
            'metadata': metadata,
            'execution_time': execution_time
        }
        
        # Use the custom MEMORY level
        self.log(LogLevel.MEMORY.value, 
                 f"Memory {operation} on {memory_layer} layer: {'success' if success else 'failed'}", 
                 extra=extra)
    
    def reflection(
        self, 
        message: str, 
        memory_ids: Optional[List[str]] = None, 
        insights: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log a self-reflection event.
        
        Args:
            message: The reflection message.
            memory_ids: Associated memory IDs involved in the reflection.
            insights: Any insights or conclusions from the reflection.
        """
        extra = {
            'memory_layer': MemoryLayer.REFLECTIVE.value,
            'operation_id': str(uuid.uuid4()),
            'metadata': {
                'memory_ids': memory_ids or [],
                'insights': insights or {}
            }
        }
        
        # Use the custom REFLECTION level
        self.log(LogLevel.REFLECTION.value, message, extra=extra)
    
    def performance(
        self, 
        operation: str, 
        execution_time: float, 
        resource_usage: Optional[Dict[str, float]] = None
    ) -> None:
        """Log performance metrics.
        
        Args:
            operation: The operation being measured.
            execution_time: Time taken in seconds.
            resource_usage: Dictionary of resource usage metrics (CPU, memory, etc.).
        """
        extra = {
            'execution_time': execution_time,
            'metadata': {
                'operation': operation,
                'resource_usage': resource_usage or {}
            }
        }
        
        self.info(f"Performance metrics for {operation}: {execution_time:.4f}s", extra=extra)


class LoggerFactory:
    """Factory for creating and managing loggers for the Augment SDK."""
    
    _instance = None
    _lock = threading.Lock()
    _loggers = {}
    _default_config = {
        'log_dir': 'logs',
        'default_level': DEFAULT_LOG_LEVEL,
        'console_logging': True,
        'file_logging': True,
        'structured_logging': False,
        'component_levels': {}
    }
    
    def __new__(cls):
        """Ensure the LoggerFactory is a singleton.
        
        Returns:
            The singleton instance of LoggerFactory.
        """
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(LoggerFactory, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        """Initialize the LoggerFactory if not already initialized."""
        if not self._initialized:
            self._config = self._default_config.copy()
            self._configure_logging()
            # Register our custom logger class
            logging.setLoggerClass(AugmentLogger)
            self._initialized = True
    
    def _configure_logging(self) -> None:
        """Configure the logging system based on current settings."""
        # Create log directory if needed
        if self._config['file_logging']:
            log_dir = Path(self._config['log_dir'])
            log_dir.mkdir(parents=True, exist_ok=True)
    
    def configure(self, config: Dict[str, Any]) -> None:
        """Configure the logging system.
        
        Args:
            config: Configuration dictionary with logger settings.
        """
        with self._lock:
            self._config.update(config)
            
            # Reset existing loggers if configuration changes
            for logger in self._loggers.values():
                self._configure_logger(logger)
    
    def _get_log_file_path(self, logger_name: str) -> Path:
        """Get the log file path for a specific logger.
        
        Args:
            logger_name: The name of the logger.
            
        Returns:
            The path to the log file.
        """
        # Create component-specific log files
        base_name = logger_name.split('.')[0]
        return Path(self._config['log_dir']) / f"{base_name}.log"
    
    def _get_logger_level(self, logger_name: str) -> int:
        """Get the appropriate log level for a logger.
        
        Args:
            logger_name: The name of the logger.
            
        Returns:
            The log level as an integer.
        """
        # Check for component-specific level
        for component, level in self._config['component_levels'].items():
            if logger_name.startswith(component):
                if isinstance(level, str):
                    return getattr(logging, level.upper())
                return level
        
        return self._config['default_level']
    
    def _configure_logger(self, logger: logging.Logger) -> None:
        """Configure handlers and formatters for a logger.
        
        Args:
            logger: The logger to configure.
        """
        # Reset handlers
        logger.handlers = []
        
        # Set level
        logger.setLevel(self._get_logger_level(logger.name))
        
        # Add console handler if enabled
        if self._config['console_logging']:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logger.level)
            
            # Use color formatter for console
            formatter = ConsoleColorFormatter(DEFAULT_LOG_FORMAT, DEFAULT_DATE_FORMAT)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # Add file handler if enabled
        if self._config['file_logging']:
            log_file = self._get_log_file_path(logger.name)
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=MAX_LOG_FILE_SIZE,
                backupCount=LOG_BACKUP_COUNT
            )
            file_handler.setLevel(logger.level)
            
            # Use structured or standard formatter based on configuration
            if self._config['structured_logging']:
                formatter = StructuredFormatter()
            else:
                formatter = logging.Formatter(DEFAULT_LOG_FORMAT, DEFAULT_DATE_FORMAT)
                
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    
    def get_logger(self, name: str) -> AugmentLogger:
        """Get or create a logger with the specified name.
        
        Args:
            name: The name for the logger, typically using dot notation for hierarchy.
            
        Returns:
            An AugmentLogger instance.
        """
        with self._lock:
            if name not in self._loggers:
                logger = logging.getLogger(name)
                
                # Ensure it's an AugmentLogger
                if not isinstance(logger, AugmentLogger):
                    # Create a new AugmentLogger with the same name
                    logger = AugmentLogger(name, level=self._get_logger_level(name))
                
                self._configure_logger(logger)
                self._loggers[name] = logger
                
            return self._loggers[name]


# Module-level factory instance
_logger_factory = LoggerFactory()


def get_logger(name: str) -> AugmentLogger:
    """Get a logger instance for the specified name.
    
    This is the main entry point for obtaining loggers in the Augment SDK.
    
    Args:
        name: The name for the logger, typically following dot notation to indicate
            the component hierarchy (e.g., 'memory.semantic' or 'core.orchestration').
            
    Returns:
        An AugmentLogger instance configured according to the current settings.
    """
    return _logger_factory.get_logger(name)


def configure_logging(config: Dict[str, Any]) -> None:
    """Configure the logging system.
    
    Args:
        config: Configuration dictionary with logger settings.
            Supported keys:
            - log_dir: Directory to store log files
            - default_level: Default log level
            - console_logging: Enable/disable console logs
            - file_logging: Enable/disable file logs
            - structured_logging: Enable/disable JSON structured logs
            - component_levels: Dict mapping component names to log levels
    """
    _logger_factory.configure(config)


def log_execution_time(logger: Optional[AugmentLogger] = None, level: int = logging.INFO):
    """Decorator to log execution time of a function.
    
    Args:
        logger: The logger to use. If None, a logger will be created based on the module name.
        level: The log level to use.
        
    Returns:
        The decorated function.
    """
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            nonlocal logger
            
            if logger is None:
                # Create a logger based on the module name
                module_name = func.__module__
                logger = get_logger(module_name)
                
            # Record start time
            start_time = time.time()
            
            try:
                # Execute the function
                result = func(*args, **kwargs)
                
                # Calculate execution time
                execution_time = time.time() - start_time
                
                # Log execution time
                logger.log(level, f"Function '{func.__name__}' executed in {execution_time:.4f} seconds",
                          extra={'execution_time': execution_time})
                
                return result
            except Exception as e:
                # Calculate execution time up to the exception
                execution_time = time.time() - start_time
                
                # Log exception with execution time
                logger.error(f"Exception in '{func.__name__}' after {execution_time:.4f} seconds: {str(e)}",
                            exc_info=True,
                            extra={'execution_time': execution_time})
                
                # Re-raise the exception
                raise
                
        return wrapper
    
    return decorator
