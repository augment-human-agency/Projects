# AugmentSDK Quickstart Guide

This guide will help you get started with AugmentSDK, a comprehensive toolkit for building AI systems with persistent, evolving memory and cognition.

## Installation

```bash
# Clone the repository
git clone https://github.com/augment-human-agency/augment-sdk.git
cd augment-sdk

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Basic Usage

### 1. Initialize the Memory Manager

```python
from augment_sdk.memory import MemoryManager
from augment_sdk.utils.config import load_config

# Load configuration
config = load_config("config.yaml")

# Initialize memory manager
memory_manager = MemoryManager(config)
```

### 2. Store and Retrieve Memories

```python
# Store memory in the semantic layer
memory_manager.store_memory(
    key="concept:ai_ethics", 
    data="AI ethics involves ensuring fairness, transparency, and preventing harm in AI systems.",
    layer="semantic",
    metadata={"source": "textbook", "importance": "high"}
)

# Retrieve memories related to a concept
results = memory_manager.retrieve_memory(
    query="What are the principles of AI ethics?",
    layer="semantic",
    top_k=3
)

# Display results
for i, result in enumerate(results):
    print(f"Result {i+1}:")
    print(f"Content: {result['content']}")
    print(f"Relevance: {result['score']:.2f}")
    print(f"Metadata: {result['metadata']}")
    print("---")
```

### 3. Working with Different Memory Layers

```python
# Store ephemeral (short-term) memory
memory_manager.store_memory(
    key="user:last_query", 
    data="What are the principles of AI ethics?",
    layer="ephemeral"
)

# Store working memory for an ongoing task
memory_manager.store_memory(
    key="task:research_ethics", 
    data={"status": "in_progress", "findings": ["fairness is key", "transparency builds trust"]},
    layer="working"
)

# Store procedural memory
memory_manager.store_memory(
    key="procedure:embedding_generation", 
    data="1. Tokenize text\n2. Pass tokens through encoder\n3. Normalize vectors",
    layer="procedural"
)
```

### 4. Using Meta-Cognition

```python
# Perform self-reflection to improve memory weighting
reflection_results = memory_manager.reflect(depth=2)

print(f"Memories analyzed: {reflection_results['memories_analyzed']}")
print(f"Weights adjusted: {reflection_results['weights_adjusted']}")
print(f"Top insights: {reflection_results['top_insights']}")
```

### 5. Memory Maintenance

```python
# Apply memory decay to prune less relevant memories
pruned_count = memory_manager.prune_memory(threshold=0.3)
print(f"Pruned {pruned_count} outdated or low-relevance memories")

# Merge similar memories to reduce redundancy
merged = memory_manager.merge_related_memories(
    key="concept:ai_ethics", 
    similarity_threshold=0.85
)
```

## Using the API

### 1. Run the API Server

```bash
# Start the FastAPI server
cd augment_sdk
python -m api.main
```

The API will be available at `http://localhost:8000`.

### 2. API Endpoints

- **Store Memory**: `POST /memory/store`
- **Retrieve Memory**: `GET /memory/retrieve`
- **Perform Reflection**: `POST /memory/reflect`
- **Prune Memory**: `POST /memory/prune`

### 3. Example API Request

```python
import requests
import json

# Store memory
response = requests.post(
    "http://localhost:8000/memory/store",
    json={
        "key": "concept:machine_learning",
        "data": "Machine learning is a branch of AI focused on building systems that learn from data.",
        "layer": "semantic",
        "metadata": {"domain": "AI", "importance": "high"}
    }
)
print(response.json())

# Retrieve memory
response = requests.get(
    "http://localhost:8000/memory/retrieve",
    params={"query": "What is machine learning?", "layer": "semantic", "top_k": 3}
)
print(response.json())
```

## Advanced Features

### Dynamic Memory Optimization Kit (DMOK)

```python
from augment_sdk.dmok import MemoryOrchestrator

# Initialize memory orchestrator
orchestrator = MemoryOrchestrator()

# Store with dynamic weighting
orchestrator.store(
    layer='semantic', 
    key='concept:reinforcement_learning', 
    data="Reinforcement learning is learning what to do to maximize reward."
)

# Retrieve with context weighting
results = orchestrator.retrieve(
    layer='semantic', 
    query='How does an agent learn in AI?',
    context={'recent_topics': ['machine learning', 'rewards']}
)
```

### Self-Auditing Reflection Cycles

```python
from augment_sdk.meta import ReflectionEngine

# Initialize reflection engine
reflection = ReflectionEngine()

# Add memories to reflect on
reflection.add_memory("AI can make biased decisions if trained on biased data.")
reflection.add_memory("Fairness in AI requires careful dataset curation.")
reflection.add_memory("Some AI developers prioritize performance over fairness.")

# Run reflection cycle
insights = reflection.reflect(question="How can we ensure AI fairness?")

# Get reflection insights
for insight in insights:
    print(f"Insight: {insight['content']}")
    print(f"Confidence: {insight['confidence']}")
    print(f"Supporting evidence: {insight['evidence']}")
    print("---")
```

## Next Steps

- Explore the [full documentation](docs/README.md) for detailed information
- Check out the [examples directory](examples/) for more complex use cases
- Contribute to the project by following our [contribution guidelines](CONTRIBUTING.md)

For more information, visit [https://augmenthumanagency.com/](https://augmenthumanagency.com/) or contact us at dev@augmenthumanagency.com.
