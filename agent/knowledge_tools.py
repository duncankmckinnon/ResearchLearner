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


class DownloadAndProcessPaperInput(BaseModel):
    """Input for downloading and processing a paper with full PDF extraction"""
    arxiv_id: str = Field(description="ArXiv ID of the paper to download and process")
    title: str = Field(default="", description="Title of the paper")
    authors: List[str] = Field(default_factory=list, description="Authors of the paper")
    categories: List[str] = Field(default_factory=list, description="Categories of the paper")


class FindStoredPaperInput(BaseModel):
    """Input for finding a paper already stored in the knowledge graph"""
    query: str = Field(description="Title, ArXiv ID, or other identifying information of the paper to find")
    limit: int = Field(default=5, description="Maximum number of results to return")


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


class DownloadAndProcessPaperTool(BaseTool):
    """Tool for downloading papers and extracting full PDF content"""
    name: str = "download_and_process_paper"
    description: str = "Download a paper by ArXiv ID and extract full PDF content for storage in knowledge graph"
    args_schema: type = DownloadAndProcessPaperInput

    def _run(self, arxiv_id: str, title: str = "", authors: List[str] = None, categories: List[str] = None) -> Dict[str, Any]:
        """Synchronous version (fallback)"""
        return asyncio.run(self._arun(arxiv_id, title, authors or [], categories or []))

    async def _arun(self, arxiv_id: str, title: str = "", authors: List[str] = None, categories: List[str] = None) -> Dict[str, Any]:
        """Download and process paper asynchronously"""
        try:
            from agent.arxiv_client import SimpleArxivClient

            logger.info(f"Executing download_and_process_paper tool: arxiv_id='{arxiv_id}'")

            # Initialize ArXiv client
            arxiv_client = SimpleArxivClient()

            # Download the paper
            download_result = await arxiv_client.download_paper(arxiv_id)
            if not download_result.success:
                logger.error(f"Failed to download paper {arxiv_id}: {download_result.error}")
                return {"success": False, "error": f"Download failed: {download_result.error}"}

            # Read and extract full content
            read_result = await arxiv_client.read_paper(arxiv_id)
            if not read_result.success:
                logger.error(f"Failed to read paper {arxiv_id}: {read_result.error}")
                return {"success": False, "error": f"Read failed: {read_result.error}"}

            paper_data = read_result.data

            # Store in knowledge graph
            loop = asyncio.get_event_loop()
            kg_manager = get_knowledge_graph_manager()

            def add_paper_sync():
                return kg_manager.add_research_paper(paper_data)

            storage_success = await loop.run_in_executor(None, add_paper_sync)

            if storage_success:
                logger.info(f"Successfully processed and stored paper: {arxiv_id}")
                return {
                    "success": True,
                    "arxiv_id": arxiv_id,
                    "title": paper_data.get("title", ""),
                    "content_length": len(paper_data.get("content", "")),
                    "urls_found": len(paper_data.get("pdf_metadata", {}).get("urls", [])),
                    "dois_found": len(paper_data.get("pdf_metadata", {}).get("dois", []))
                }
            else:
                return {"success": False, "error": "Failed to store in knowledge graph"}

        except Exception as e:
            logger.error(f"Error in download_and_process_paper tool: {str(e)}")
            return {"success": False, "error": str(e)}


class FindStoredPaperTool(BaseTool):
    """Tool for finding papers already stored in the knowledge graph"""
    name: str = "find_stored_paper"
    description: str = "Find papers already stored in the knowledge graph by title, ArXiv ID, or other identifying information. Use this BEFORE downloading new papers to check if they're already available with full content."
    args_schema: type = FindStoredPaperInput

    def _run(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Synchronous version (fallback)"""
        return asyncio.run(self._arun(query, limit))

    async def _arun(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Find stored papers asynchronously"""
        try:
            logger.info(f"Executing find_stored_paper tool: query='{query}', limit={limit}")

            loop = asyncio.get_event_loop()
            kg_manager = get_knowledge_graph_manager()

            def search_papers_sync():
                papers = []

                # Try multiple search strategies
                search_queries = [
                    query,
                    f"Title: {query}",
                    f"research paper {query}",
                    f"paper titled {query}"
                ]

                seen_arxiv_ids = set()

                for search_query in search_queries:
                    results = kg_manager.search_knowledge(search_query, limit * 2)

                    for result in results:
                        metadata = result.get("metadata", {})
                        content = result.get("content", "")

                        # Check if this is a research paper by metadata or content
                        is_research_paper = (
                            metadata.get("type") == "research_paper" or
                            "Title:" in content or
                            "Authors:" in content or
                            "ArXiv ID:" in content or
                            "Abstract:" in content
                        )

                        if is_research_paper:
                            arxiv_id = metadata.get("arxiv_id", "")
                            title = metadata.get("title", "")

                            # Check for title match or ArXiv ID match
                            query_lower = query.lower()
                            title_match = query_lower in title.lower() if title else False
                            arxiv_match = query_lower in arxiv_id.lower() if arxiv_id else False
                            content_match = query_lower in content.lower() if content else False

                            # Be more permissive - include if any match or if it's a research paper
                            should_include = (title_match or arxiv_match or content_match or
                                            (is_research_paper and not papers)) and arxiv_id not in seen_arxiv_ids

                            if should_include:
                                seen_arxiv_ids.add(arxiv_id)
                                papers.append({
                                    "title": title,
                                    "authors": [author.strip() for author in metadata.get("authors", "").split(", ") if author.strip()],
                                    "arxiv_id": arxiv_id,
                                    "categories": [cat.strip() for cat in metadata.get("categories", "").split(", ") if cat.strip()],
                                    "urls": [url.strip() for url in metadata.get("urls", "").split(", ") if url.strip()],
                                    "dois": [doi.strip() for doi in metadata.get("dois", "").split(", ") if doi.strip()],
                                    "content": content,
                                    "relevance_score": result.get("relevance_score", 0),
                                    "source": "knowledge_graph_stored",
                                    "has_full_content": "Full Paper Content:" in content,
                                    "match_type": "title" if title_match else "arxiv_id" if arxiv_match else "content"
                                })

                                if len(papers) >= limit:
                                    break

                    if len(papers) >= limit:
                        break

                return papers[:limit]

            papers = await loop.run_in_executor(None, search_papers_sync)
            logger.info(f"find_stored_paper tool completed: found {len(papers)} stored papers")

            # Debug logging
            if not papers:
                logger.info(f"No papers found for query: '{query}'. Checking raw search results...")
                debug_results = kg_manager.search_knowledge(query, 5)
                logger.info(f"Raw search returned {len(debug_results)} results")
                for i, result in enumerate(debug_results[:3]):
                    metadata = result.get("metadata", {})
                    content_preview = result.get("content", "")[:200] + "..." if len(result.get("content", "")) > 200 else result.get("content", "")
                    logger.info(f"Result {i}: type={metadata.get('type')}, title={metadata.get('title')}, content_preview={content_preview}")

            return papers

        except Exception as e:
            logger.error(f"Error in find_stored_paper tool: {str(e)}")
            return []


# Tool registry for easy access
KNOWLEDGE_TOOLS = [
    SearchKnowledgeTool(),
    GetRelatedPapersTool(),
    GetResearchInsightsTool(),
    AddResearchInsightTool(),
    GetKnowledgeSummaryTool(),
    DownloadAndProcessPaperTool(),
    FindStoredPaperTool(),
]

# Tool mapping for quick lookup
KNOWLEDGE_TOOL_MAP = {tool.name: tool for tool in KNOWLEDGE_TOOLS}


def get_knowledge_tools() -> List[BaseTool]:
    """Get all knowledge graph tools"""
    return KNOWLEDGE_TOOLS


def get_knowledge_tool(name: str) -> Optional[BaseTool]:
    """Get a specific knowledge graph tool by name"""
    return KNOWLEDGE_TOOL_MAP.get(name)