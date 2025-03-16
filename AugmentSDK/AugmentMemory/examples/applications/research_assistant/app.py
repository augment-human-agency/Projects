#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Research Assistant Application

This example demonstrates a practical application of the Augment SDK: 
a research assistant that helps organize, recall, and evolve knowledge 
for research projects. The assistant uses hierarchical memory layers to
differentiate between short-term context, active research topics, and
long-term knowledge.

Key features:
1. Context-aware information retrieval
2. Automatic knowledge organization into research topics
3. Knowledge evolution as new information is added
4. Memory prioritization based on relevance and recency
5. Citation tracking and source management

This showcases how Augment SDK can be used in a practical application
for knowledge workers and researchers.
"""

import asyncio
import logging
import os
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field
from datetime import datetime
import json

from augment_sdk.memory.components.memory_manager import MemoryManager
from augment_sdk.memory.utils.config import load_config


# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ResearchNote:
    """Class representing a research note with metadata."""
    content: str
    source: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    importance: float = 1.0
    project: Optional[str] = None
    citations: List[Dict[str, str]] = field(default_factory=list)


class ResearchAssistant:
    """
    Research assistant application using Augment SDK for advanced memory management.
    
    This class provides functionality to store, retrieve, and evolve research notes
    using hierarchical memory layers.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the research assistant.
        
        Args:
            config: Configuration dictionary for the memory manager
        """
        self.memory_manager = MemoryManager(config)
        self.current_project: Optional[str] = None
        self.initialized = False
    
    async def initialize(self):
        """Initialize the memory systems."""
        if not self.initialized:
            await self.memory_manager.initialize()
            self.initialized = True
            logger.info("Research Assistant initialized")
    
    async def shutdown(self):
        """Shut down the memory systems."""
        if self.initialized:
            await self.memory_manager.shutdown()
            self.initialized = False
            logger.info("Research Assistant shut down")
    
    def set_active_project(self, project_name: str):
        """
        Set the active research project.
        
        Args:
            project_name: Name of the project to set as active
        """
        self.current_project = project_name
        
        # Store this in working memory
        self.memory_manager.store_memory(
            key=f"active_project_{datetime.now().isoformat()}",
            data=f"Current active project: {project_name}",
            layer="working"
        )
        
        logger.info(f"Active project set to: {project_name}")
    
    def add_note(self, note: ResearchNote):
        """
        Add a research note to memory.
        
        The note is stored in multiple memory layers:
        - Ephemeral: For immediate context
        - Working: For active project work
        - Semantic: For long-term knowledge retention
        
        Args:
            note: Research note to add
        """
        # Ensure the note has the current project if not specified
        if not note.project and self.current_project:
            note.project = self.current_project
        
        # Convert to JSON for storage
        note_json = json.dumps(note.__dict__)
        
        # Store in ephemeral memory for immediate context
        self.memory_manager.store_memory(
            key=f"recent_note_{datetime.now().isoformat()}",
            data=note_json,
            layer="ephemeral"
        )
        
        # Store in working memory for active project work
        if note.project:
            self.memory_manager.store_memory(
                key=f"project_{note.project}_note_{datetime.now().isoformat()}",
                data=note_json,
                layer="working"
            )
        
        # Store in semantic memory for long-term retention
        unique_id = f"{note.project}_{len(note.content)%100}_{datetime.now().isoformat()}"
        self.memory_manager.store_memory(
            key=f"research_note_{unique_id}",
            data=note_json,
            layer="semantic"
        )
        
        # Also store tags as procedural memory for better retrieval
        for tag in note.tags:
            self.memory_manager.store_memory(
                key=f"tag_{tag}_{datetime.now().isoformat()}",
                data=json.dumps({
                    "tag": tag,
                    "note_id": unique_id,
                    "project": note.project,
                    "created_at": note.created_at
                }),
                layer="procedural"
            )
        
        logger.info(f"Added research note: {note.content[:50]}...")
        
        # Trigger reflection to update knowledge connections
        self.memory_manager.reflect()
    
    def search_notes(self, query: str, project: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for research notes matching the query.
        
        Args:
            query: Search query
            project: Optional project filter
            
        Returns:
            List of matching notes
        """
        # First try project-specific search if project is specified
        results = []
        
        if project:
            # Search in working memory first for active project
            working_results = self.memory_manager.retrieve_memory(
                query=f"{project} {query}",
                layer="working"
            )
            
            # Then search semantic memory
            semantic_results = self.memory_manager.retrieve_memory(
                query=f"{project} {query}",
                layer="semantic"
            )
            
            # Combine results
            results = working_results + semantic_results
        else:
            # General search across semantic memory
            results = self.memory_manager.retrieve_memory(
                query=query,
                layer="semantic"
            )
        
        # Parse JSON results back to dictionaries
        parsed_results = []
        for result in results:
            try:
                # Extract the note content from the memory result
                if isinstance(result, dict) and "data" in result:
                    note_data = json.loads(result["data"])
                    parsed_results.append(note_data)
                elif isinstance(result, str):
                    note_data = json.loads(result)
                    parsed_results.append(note_data)
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Failed to parse note data: {e}")
        
        # Deduplicate results based on content
        unique_results = []
        seen_content = set()
        
        for note in parsed_results:
            content = note.get("content", "")
            if content and content not in seen_content:
                seen_content.add(content)
                unique_results.append(note)
        
        logger.info(f"Found {len(unique_results)} unique research notes for query: {query}")
        return unique_results
    
    def search_by_tag(self, tag: str) -> List[Dict[str, Any]]:
        """
        Search for research notes by tag.
        
        Args:
            tag: Tag to search for
            
        Returns:
            List of matching notes
        """
        # Search procedural memory for tag references
        tag_results = self.memory_manager.retrieve_memory(
            query=f"tag {tag}",
            layer="procedural"
        )
        
        # Extract note IDs from tag results
        note_ids = []
        for result in tag_results:
            try:
                if isinstance(result, dict) and "data" in result:
                    tag_data = json.loads(result["data"])
                    if tag_data.get("tag") == tag and "note_id" in tag_data:
                        note_ids.append(tag_data["note_id"])
                elif isinstance(result, str):
                    tag_data = json.loads(result)
                    if tag_data.get("tag") == tag and "note_id" in tag_data:
                        note_ids.append(tag_data["note_id"])
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Failed to parse tag data: {e}")
        
        # Fetch the actual notes from semantic memory
        notes = []
        for note_id in note_ids:
            note_results = self.memory_manager.retrieve_memory(
                query=f"research_note_{note_id}",
                layer="semantic"
            )
            
            for result in note_results:
                try:
                    if isinstance(result, dict) and "data" in result:
                        note_data = json.loads(result["data"])
                        notes.append(note_data)
                    elif isinstance(result, str):
                        note_data = json.loads(result)
                        notes.append(note_data)
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"Failed to parse note data: {e}")
        
        logger.info(f"Found {len(notes)} research notes with tag: {tag}")
        return notes
    
    def get_project_summary(self, project: str) -> Dict[str, Any]:
        """
        Generate a summary of the specified project.
        
        Args:
            project: Project name
            
        Returns:
            Project summary including key notes, tags, and sources
        """
        # Collect all notes for the project
        project_notes = self.search_notes(project, project=project)
        
        # Extract key information
        all_tags = set()
        all_sources = set()
        all_citations = []
        
        for note in project_notes:
            if "tags" in note and isinstance(note["tags"], list):
                all_tags.update(note["tags"])
            
            if "source" in note and note["source"]:
                all_sources.add(note["source"])
            
            if "citations" in note and isinstance(note["citations"], list):
                all_citations.extend(note["citations"])
        
        # Generate summary
        summary = {
            "project": project,
            "note_count": len(project_notes),
            "tags": list(all_tags),
            "sources": list(all_sources),
            "citations": all_citations,
            "last_updated": datetime.now().isoformat(),
            "recent_notes": project_notes[:5] if len(project_notes) > 5 else project_notes
        }
        
        # Store summary in working memory
        self.memory_manager.store_memory(
            key=f"project_summary_{project}_{datetime.now().isoformat()}",
            data=json.dumps(summary),
            layer="working"
        )
        
        logger.info(f"Generated summary for project: {project}")
        return summary
    
    def evolve_knowledge(self, project: Optional[str] = None):
        """
        Trigger a knowledge evolution cycle for the project or all projects.
        
        This demonstrates the recursive knowledge refinement capabilities by:
        1. Analyzing existing notes
        2. Identifying connections and patterns
        3. Generating insights
        4. Updating the knowledge representation
        
        Args:
            project: Optional project to evolve knowledge for
        """
        logger.info(f"Starting knowledge evolution for {'all projects' if not project else project}")
        
        # First, apply reflection to update connections
        self.memory_manager.reflect()
        
        # Then apply memory pruning to prioritize relevant information
        self.memory_manager.prune_memory()
        
        # Finally, generate insights based on existing knowledge
        # (In a real implementation, this would involve more complex analysis)
        if project:
            project_notes = self.search_notes("", project=project)
            
            # Create an insight note
            if project_notes:
                insight = ResearchNote(
                    content=f"Knowledge evolution insight: Project {project} has {len(project_notes)} notes "
                            f"covering {len(set([n.get('tags', []) for n in project_notes]))} unique tags.",
                    source="Augment SDK Knowledge Evolution",
                    tags=["insight", "meta-analysis", project],
                    importance=0.9,
                    project=project
                )
                
                self.add_note(insight)
        
        logger.info("Knowledge evolution complete")


async def run_demo():
    """Run a demonstration of the research assistant."""
    config = load_config()
    assistant = ResearchAssistant(config)
    await assistant.initialize()
    
    try:
        # Set up a research project
        assistant.set_active_project("Climate Change Research")
        
        # Add some research notes
        assistant.add_note(ResearchNote(
            content="Global temperatures have increased by approximately 1.1°C since the pre-industrial era.",
            source="IPCC Report 2021",
            tags=["climate", "temperature", "statistics"],
            importance=0.9,
            citations=[{"title": "IPCC Sixth Assessment Report", "year": "2021"}]
        ))
        
        assistant.add_note(ResearchNote(
            content="Sea levels rose by 0.2 meters during the 20th century and are projected to rise by up to 1 meter by 2100.",
            source="NASA Global Climate Change",
            tags=["climate", "sea level", "projections"],
            importance=0.85,
            citations=[{"title": "NASA Sea Level Research", "year": "2022"}]
        ))
        
        assistant.add_note(ResearchNote(
            content="Renewable energy capacity increased by 45% globally between 2015 and 2020.",
            source="International Energy Agency",
            tags=["energy", "renewables", "solutions"],
            importance=0.75,
            citations=[{"title": "IEA Renewables Report", "year": "2021"}]
        ))
        
        # Search for information
        temperature_notes = assistant.search_notes("temperature climate")
        logger.info(f"Found {len(temperature_notes)} notes about temperature")
        
        energy_notes = assistant.search_by_tag("energy")
        logger.info(f"Found {len(energy_notes)} notes tagged with energy")
        
        # Generate project summary
        summary = assistant.get_project_summary("Climate Change Research")
        logger.info(f"Project has {summary['note_count']} notes and {len(summary['tags'])} unique tags")
        
        # Evolve knowledge to find new connections
        assistant.evolve_knowledge("Climate Change Research")
        
        # Add a note to another project to demonstrate multiple project handling
        assistant.set_active_project("AI Ethics Research")
        assistant.add_note(ResearchNote(
            content="AI systems should be designed to be transparent, fair, and accountable.",
            source="IEEE Ethics Guidelines",
            tags=["AI", "ethics", "principles"],
            importance=0.9
        ))
        
        # Search across all projects
        all_results = assistant.search_notes("research importance")
        logger.info(f"Found {len(all_results)} notes across all projects related to research importance")
        
        # Get project summary for the new project
        ai_summary = assistant.get_project_summary("AI Ethics Research")
        logger.info(f"AI Ethics project has {ai_summary['note_count']} notes")
        
    finally:
        await assistant.shutdown()


if __name__ == "__main__":
    asyncio.run(run_demo())