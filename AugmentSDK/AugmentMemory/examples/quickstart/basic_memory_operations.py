This example demonstrates the fundamental operations of the Augment SDK:
- Initializing the memory manager
- Storing memories in different layers
- Retrieving memories using queries
- Working with memory meta-information

This serves as a quick introduction to get you started with the SDK.
"""

import asyncio
import logging
from pprint import pprint

from augment_sdk.memory.components.memory_manager import MemoryManager
from augment_sdk.memory.utils.config import load_config


# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def main():
    """
    Main function demonstrating basic memory operations.
    """
    logger.info("Initializing Memory Manager")
    
    # Load the default configuration
    config = load_config()
    
    # Initialize the memory manager
    memory_manager = MemoryManager(config)
    await memory_manager.initialize()
    
    try:
        # Example 1: Storing memories in different layers
        logger.info("\n--- Example 1: Storing memories in different layers ---")
        
        # Ephemeral memory (short-term)
        memory_manager.store_memory(
            key="current_user_request",
            data="The user asked about the weather in Paris",
            layer="ephemeral"
        )
        logger.info("Stored ephemeral memory")
        
        # Working memory (task-oriented)
        memory_manager.store_memory(
            key="active_project",
            data="Building a client presentation for Company XYZ",
            layer="working"
        )
        logger.info("Stored working memory")
        
        # Semantic memory (long-term knowledge)
        memory_manager.store_memory(
            key="paris_weather_patterns",
            data="Paris has a temperate climate with mild, wet winters and warm summers. "
                 "The average temperature ranges from 5°C in winter to 20°C in summer.",
            layer="semantic"
        )
        logger.info("Stored semantic memory")
        
        # Procedural memory (how-to knowledge)
        memory_manager.store_memory(
            key="weather_api_procedure",
            data="To get weather information: 1) Connect to the API, 2) Send location query, "
                 "3) Parse the JSON response, 4) Extract temperature and conditions",
            layer="procedural"
        )
        logger.info("Stored procedural memory")
        
        # Example 2: Retrieving memories with queries
        logger.info("\n--- Example 2: Retrieving memories with queries ---")
        
        # Simple retrieval from a specific layer
        result = memory_manager.retrieve_memory(
            query="weather in Paris",
            layer="semantic"
        )
        logger.info("Semantic memory retrieval results:")
        pprint(result)
        
        # Retrieval from procedural memory
        result = memory_manager.retrieve_memory(
            query="how to get weather information",
            layer="procedural"
        )
        logger.info("Procedural memory retrieval results:")
        pprint(result)
        
        # Example 3: Memory meta-operations
        logger.info("\n--- Example 3: Memory meta-operations ---")
        
        # Trigger reflection process (meta-cognition)
        memory_manager.reflect()
        logger.info("Completed memory reflection process")
        
        # Pruning old memories (apply decay)
        memory_manager.prune_memory()
        logger.info("Completed memory pruning process")
        
        # Example 4: Memory health check
        logger.info("\n--- Example 4: Memory health check ---")
        
        health_status = await memory_manager.health_check()
        logger.info("Memory system health status:")
        pprint(health_status)

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
    
    finally:
        # Clean up resources
        await memory_manager.shutdown()
        logger.info("Memory manager shut down")


if __name__ == "__main__":
    # Run the async example
    asyncio.run(main())
