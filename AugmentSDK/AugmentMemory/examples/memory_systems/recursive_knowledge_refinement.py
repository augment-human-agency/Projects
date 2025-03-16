#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recursive Knowledge Refinement Example

This example demonstrates how Augment SDK enables AI systems to 
recursively refine knowledge over time. This is a key feature that allows
AI to evolve its understanding, correct misconceptions, and build more
nuanced knowledge representations through self-reflection.

The example shows:
1. Initial knowledge storage
2. Retrieval and analysis of that knowledge
3. Refinement based on new information
4. Self-reflection to improve quality
5. Progressive improvement over multiple iterations
"""

import asyncio
import logging
from pprint import pprint
import time

from augment_sdk.memory.components.memory_manager import MemoryManager
from augment_sdk.memory.utils.config import load_config
from augment_sdk.memory.components.meta_cognition import MetaCognition


# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def simulate_knowledge_evolution(memory_manager):
    """
    Simulates the evolution of knowledge over multiple iterations.
    
    Args:
        memory_manager: Initialized memory manager instance
    """
    # Initial knowledge about a topic (simplified example)
    initial_knowledge = {
        "topic": "climate_change",
        "content": "Climate change is primarily caused by human activities, especially burning fossil fuels.",
        "confidence": 0.8,
        "iteration": 1
    }
    
    # Store initial knowledge
    memory_manager.store_memory(
        key="knowledge_climate_change_v1",
        data=str(initial_knowledge),
        layer="semantic"
    )
    logger.info("Stored initial knowledge about climate change")
    
    # First refinement: Adding more detail
    time.sleep(1)  # Simulate time passing
    
    # Retrieve existing knowledge
    current_knowledge = memory_manager.retrieve_memory(
        query="climate change causes",
        layer="semantic"
    )
    logger.info("Retrieved current knowledge:")
    pprint(current_knowledge)
    
    # Refine with additional information
    refined_knowledge = {
        "topic": "climate_change",
        "content": "Climate change is primarily caused by human activities, especially burning fossil fuels. "
                  "This increases greenhouse gas concentrations in the atmosphere, particularly CO2, methane, "
                  "and nitrous oxide. These gases trap heat, causing global warming.",
        "confidence": 0.85,
        "supporting_evidence": ["IPCC Report 2021", "NASA Climate Data"],
        "iteration": 2
    }
    
    # Store refined knowledge
    memory_manager.store_memory(
        key="knowledge_climate_change_v2",
        data=str(refined_knowledge),
        layer="semantic"
    )
    logger.info("Stored refined knowledge (iteration 2)")
    
    # Second refinement: Adding nuance and correction
    time.sleep(1)  # Simulate time passing
    
    # Retrieve refined knowledge
    current_knowledge = memory_manager.retrieve_memory(
        query="climate change greenhouse gases",
        layer="semantic"
    )
    logger.info("Retrieved refined knowledge:")
    pprint(current_knowledge)
    
    # Apply self-reflection
    memory_manager.reflect()
    logger.info("Applied self-reflection to evaluate knowledge quality")
    
    # Further refine with corrections and nuance
    further_refined_knowledge = {
        "topic": "climate_change",
        "content": "Climate change is primarily caused by human activities, especially burning fossil fuels, "
                  "industrial processes, and land use changes. Greenhouse gases (CO2, methane, N2O, and others) "
                  "trap heat in the atmosphere, causing global warming. While natural factors like solar cycles "
                  "and volcanic activity exist, they cannot explain current warming trends without human influence.",
        "confidence": 0.92,
        "supporting_evidence": ["IPCC AR6 Report", "NASA Climate Data", "NOAA Global Temperature Analysis"],
        "counter_arguments_addressed": ["solar cycle variations", "natural warming periods"],
        "iteration": 3
    }
    
    # Store further refined knowledge
    memory_manager.store_memory(
        key="knowledge_climate_change_v3",
        data=str(further_refined_knowledge),
        layer="semantic"
    )
    logger.info("Stored further refined knowledge (iteration 3)")
    
    # Final state: Comprehensive knowledge
    time.sleep(1)  # Simulate time passing
    
    # Apply memory decay to deprioritize older versions
    memory_manager.prune_memory()
    logger.info("Applied memory pruning to deprioritize outdated information")
    
    # Retrieve final state of knowledge
    final_knowledge = memory_manager.retrieve_memory(
        query="comprehensive climate change causes and evidence",
        layer="semantic"
    )
    logger.info("Retrieved final state of knowledge:")
    pprint(final_knowledge)
    
    # Demonstrate meta-cognition analysis
    meta_cognition = MetaCognition()
    knowledge_analysis = meta_cognition.analyze_knowledge_evolution(final_knowledge)
    logger.info("Meta-cognitive analysis of knowledge evolution:")
    pprint(knowledge_analysis)
    
    logger.info("\nKey takeaways from recursive knowledge refinement:")
    logger.info("1. Knowledge evolved from basic facts to comprehensive understanding")
    logger.info("2. Confidence increased as supporting evidence was added")
    logger.info("3. Self-reflection helped identify and address gaps")
    logger.info("4. Older, less complete knowledge was automatically deprioritized")
    logger.info("5. The system maintains a history of knowledge evolution")


async def main():
    """
    Main function demonstrating recursive knowledge refinement.
    """
    logger.info("Initializing Memory Manager for knowledge refinement demo")
    
    # Load configuration and initialize memory manager
    config = load_config()
    memory_manager = MemoryManager(config)
    await memory_manager.initialize()
    
    try:
        # Run the knowledge evolution simulation
        await simulate_knowledge_evolution(memory_manager)
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
    
    finally:
        # Clean up resources
        await memory_manager.shutdown()
        logger.info("Memory manager shut down")


if __name__ == "__main__":
    # Run the async example
    asyncio.run(main())