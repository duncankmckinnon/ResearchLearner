"""
Async tool wrappers for knowledge graph functions to be used by LangGraph agent
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Union
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from agent.knowledge_graph import get_knowledge_graph_manager

logger = logging.getLogger("knowledge_tools")


# Input schemas for tools
class SearchKnowledgeInput(BaseModel):
    """Input for searching the knowledge graph"""
    query: str = Field(description="Search query for the knowledge graph")
    limit: int = Field(default=10, description="Maximum number of results to return")


class GetRelatedPapersInput(BaseModel):
    """Input for getting related papers"""
    topic: str = Field(description="Topic to search for related papers")
    limit: int = Field(default=15, description="Maximum number of papers to return (use 10-20 for comprehensive research)")


class GetResearchInsightsInput(BaseModel):
    """Input for getting research insights"""
    topic: str = Field(description="Topic to search for research insights")
    limit: int = Field(default=10, description="Maximum number of insights to return")


class AddResearchPaperInput(BaseModel):
    """Input for adding a research paper"""
    paper_data: Dict[str, Any] = Field(description="Research paper data to store")


class AddResearchInsightInput(BaseModel):
    """Input for adding a research insight"""
    insight: str = Field(description="The research insight content")
    topic: str = Field(description="Topic of the insight")
    paper_ids: Optional[List[str]] = Field(default=None, description="IDs of the papers the insight is about")
    context: Dict[str, Any] = Field(default_factory=dict, description="Context information")


class GetKnowledgeSummaryInput(BaseModel):
    """Input for getting knowledge summary"""
    topic: str = Field(description="Topic for knowledge summary")


# Async tool implementations
class SearchKnowledgeTool(BaseTool):
    """Tool for searching the knowledge graph"""
    name: str = "search_knowledge"
    description: str = "Search the knowledge graph for relevant information using semantic similarity"
    args_schema: type = SearchKnowledgeInput

    def _run(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Synchronous version (fallback)"""
        return asyncio.run(self._arun(query, limit))

    async def _arun(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search the knowledge graph asynchronously"""
        try:
            logger.info(f"Executing search_knowledge tool: query='{query}', limit={limit}")
            
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            kg_manager = get_knowledge_graph_manager()
            
            def search_sync():
                return kg_manager.search_knowledge(query, limit)
            
            results = await loop.run_in_executor(None, search_sync)
            
            logger.info(f"search_knowledge tool completed: found {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error in search_knowledge tool: {str(e)}")
            return []


class GetRelatedPapersTool(BaseTool):
    """Tool for getting related research papers"""
    name: str = "get_related_papers"
    description: str = "Get research papers related to a specific topic from knowledge graph and ArXiv. Start by gathering a few papers to get a sense of the topic and then widen the search if necessary to find the information requested."
    args_schema: type = GetRelatedPapersInput

    def _run(self, topic: str, limit: int = 5) -> Union[List[Union[Dict[str, Any], None]], None]:
        """Synchronous version (fallback)"""
        return asyncio.run(self._arun(topic, limit))

    async def _arun(self, topic: str, limit: int = 5) -> Union[List[Union[Dict[str, Any], None]], None]:
        """Get related papers asynchronously"""
        try:
            logger.info(f"Executing get_related_papers tool: topic='{topic}', limit={limit}")
            
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            kg_manager = get_knowledge_graph_manager()
            
            def get_papers_sync():
                return kg_manager.get_related_papers(topic, limit)
            
            results = await loop.run_in_executor(None, get_papers_sync)
            
            if results:
                logger.info(f"get_related_papers tool completed: found {len(results)} papers")
                return results
            else:
                logger.warning(f"get_related_papers tool completed: found no papers for topic: {topic}")
                return None

        except Exception as e:
            logger.error(f"Error in get_related_papers tool: {str(e)}")
            return []


class GetResearchInsightsTool(BaseTool):
    """Tool for getting research insights"""
    name: str = "get_research_insights"
    description: str = "Get research insights for a specific topic from the knowledge graph"
    args_schema: type = GetResearchInsightsInput

    def _run(self, topic: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Synchronous version (fallback)"""
        return asyncio.run(self._arun(topic, limit))

    async def _arun(self, topic: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get research insights asynchronously"""
        try:
            logger.info(f"Executing get_research_insights tool: topic='{topic}', limit={limit}")
            
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            kg_manager = get_knowledge_graph_manager()
            
            def get_insights_sync():
                return kg_manager.get_research_insights(topic, limit)
            
            results = await loop.run_in_executor(None, get_insights_sync)
            
            logger.info(f"get_research_insights tool completed: found {len(results)} insights")
            return results
            
        except Exception as e:
            logger.error(f"Error in get_research_insights tool: {str(e)}")
            return []


class AddResearchPaperTool(BaseTool):
    """Tool for adding research papers to knowledge graph"""
    name: str = "add_research_paper"
    description: str = "Add a research paper to the knowledge graph for future retrieval"
    args_schema: type = AddResearchPaperInput

    def _run(self, paper_data: Dict[str, Any]) -> bool:
        """Synchronous version (fallback)"""
        return asyncio.run(self._arun(paper_data))

    async def _arun(self, paper_data: Dict[str, Any]) -> bool:
        """Add research paper asynchronously"""
        try:
            paper_title = paper_data.get("title", "Unknown")
            logger.info(f"Executing add_research_paper tool: paper='{paper_title}'")
            
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            kg_manager = get_knowledge_graph_manager()
            
            def add_paper_sync():
                return kg_manager.add_research_paper(paper_data)
            
            success = await loop.run_in_executor(None, add_paper_sync)
            
            logger.info(f"add_research_paper tool completed: success={success}")
            return success
            
        except Exception as e:
            logger.error(f"Error in add_research_paper tool: {str(e)}")
            return False


class AddResearchInsightTool(BaseTool):
    """Tool for adding research insights to knowledge graph"""
    name: str = "add_research_insight"
    description: str = "Add a research insight to the knowledge graph for future retrieval. CALL THIS UP TO 3 TIMES - extract distinct insights from each paper set. Be comprehensive and detailed."
    args_schema: type = AddResearchInsightInput

    def _run(self, insight: str, topic: str, paper_ids: Optional[List[str]] = None, context: Optional[Dict[str, Any]] = None) -> bool:
        """Synchronous version (fallback)"""
        return asyncio.run(self._arun(insight, topic, paper_ids or [], context or {}))

    async def _arun(self, insight: str, topic: str, paper_ids: Optional[List[str]] = None, context: Optional[Dict[str, Any]] = None) -> bool:
        """Add research insight asynchronously"""
        try:
            logger.info(f"Executing add_research_insight tool: topic='{topic}'")
            
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            kg_manager = get_knowledge_graph_manager()
            
            def add_insight_sync():
                return kg_manager.add_research_insight(insight, topic, paper_ids or [], context or {})
            
            success = await loop.run_in_executor(None, add_insight_sync)
            
            logger.info(f"add_research_insight tool completed: success={success}")
            return success
            
        except Exception as e:
            logger.error(f"Error in add_research_insight tool: {str(e)}")
            return False


class GetKnowledgeSummaryTool(BaseTool):
    """Tool for getting comprehensive knowledge summary"""
    name: str = "get_knowledge_summary"
    description: str = "Get a comprehensive knowledge summary including papers, insights, and general knowledge for a topic"
    args_schema: type = GetKnowledgeSummaryInput

    def _run(self, topic: str) -> Dict[str, Any]:
        """Synchronous version (fallback)"""
        return asyncio.run(self._arun(topic))

    async def _arun(self, topic: str) -> Dict[str, Any]:
        """Get knowledge summary asynchronously"""
        try:
            logger.info(f"Executing get_knowledge_summary tool: topic='{topic}'")
            
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            kg_manager = get_knowledge_graph_manager()
            
            def get_summary_sync():
                return kg_manager.get_knowledge_summary(topic)
            
            results = await loop.run_in_executor(None, get_summary_sync)
            
            logger.info(f"get_knowledge_summary tool completed: {results.get('total_papers', 0)} papers, {results.get('total_insights', 0)} insights")
            return results
            
        except Exception as e:
            logger.error(f"Error in get_knowledge_summary tool: {str(e)}")
            return {"error": str(e)}


# Tool registry for easy access
KNOWLEDGE_TOOLS = [
    SearchKnowledgeTool(),
    GetRelatedPapersTool(),
    GetResearchInsightsTool(),
    AddResearchPaperTool(),
    AddResearchInsightTool(),
    GetKnowledgeSummaryTool(),
]

# Tool mapping for quick lookup
KNOWLEDGE_TOOL_MAP = {tool.name: tool for tool in KNOWLEDGE_TOOLS}


def get_knowledge_tools() -> List[BaseTool]:
    """Get all knowledge graph tools"""
    return KNOWLEDGE_TOOLS


def get_knowledge_tool(name: str) -> Optional[BaseTool]:
    """Get a specific knowledge graph tool by name"""
    return KNOWLEDGE_TOOL_MAP.get(name)