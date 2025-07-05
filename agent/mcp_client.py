import asyncio
import json
import logging
import subprocess
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import os

logger = logging.getLogger("mcp_client")

@dataclass
class MCPToolResult:
    """Result from an MCP tool call"""
    success: bool
    content: Any
    error: Optional[str] = None

class ArxivMCPClient:
    """Client for communicating with the ArXiv MCP server"""
    
    def __init__(self, storage_path: str = None):
        self.storage_path = storage_path or os.path.expanduser("~/.arxiv-mcp-server/papers")
        self.server_process = None
        self.is_running = False
        
    async def start_server(self) -> bool:
        """Start the ArXiv MCP server"""
        try:
            # Create storage directory if it doesn't exist
            os.makedirs(self.storage_path, exist_ok=True)
            
            # For now, we'll assume the server is running externally
            # In production, you'd start: uv tool run arxiv-mcp-server --storage-path <path>
            self.is_running = True
            logger.info(f"ArXiv MCP server assumed running with storage path: {self.storage_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start ArXiv MCP server: {str(e)}")
            return False
    
    async def stop_server(self):
        """Stop the ArXiv MCP server"""
        if self.server_process and self.is_running:
            try:
                self.server_process.terminate()
                await self.server_process.wait()
                self.is_running = False
                logger.info("ArXiv MCP server stopped")
            except Exception as e:
                logger.error(f"Error stopping ArXiv MCP server: {str(e)}")
    
    async def call_tool(self, tool_name: str, params: Dict[str, Any]) -> MCPToolResult:
        """Call a tool on the MCP server"""
        if not self.is_running:
            return MCPToolResult(
                success=False,
                content=None,
                error="MCP server is not running"
            )
        
        try:
            # Format the MCP request
            request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": params
                }
            }
            
            # Send request to server
            request_json = json.dumps(request) + "\n"
            self.server_process.stdin.write(request_json.encode())
            await self.server_process.stdin.drain()
            
            # Read response
            response_line = await self.server_process.stdout.readline()
            if not response_line:
                return MCPToolResult(
                    success=False,
                    content=None,
                    error="No response from server"
                )
            
            response = json.loads(response_line.decode().strip())
            
            if "error" in response:
                return MCPToolResult(
                    success=False,
                    content=None,
                    error=response["error"].get("message", "Unknown error")
                )
            
            return MCPToolResult(
                success=True,
                content=response.get("result", {})
            )
            
        except Exception as e:
            logger.error(f"Error calling MCP tool {tool_name}: {str(e)}")
            return MCPToolResult(
                success=False,
                content=None,
                error=str(e)
            )
    
    async def search_papers(self, query: str, max_results: int = 10, 
                          date_from: str = None, categories: List[str] = None) -> MCPToolResult:
        """Search for papers on ArXiv"""
        params = {
            "query": query,
            "max_results": max_results
        }
        
        if date_from:
            params["date_from"] = date_from
        if categories:
            params["categories"] = categories
            
        return await self.call_tool("search_papers", params)
    
    async def download_paper(self, paper_id: str) -> MCPToolResult:
        """Download a paper by ArXiv ID"""
        return await self.call_tool("download_paper", {"paper_id": paper_id})
    
    async def list_papers(self) -> MCPToolResult:
        """List all downloaded papers"""
        return await self.call_tool("list_papers", {})
    
    async def read_paper(self, paper_id: str) -> MCPToolResult:
        """Read the content of a downloaded paper"""
        return await self.call_tool("read_paper", {"paper_id": paper_id})
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.start_server()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.stop_server()

class ResearchAgent:
    """Enhanced research agent with ArXiv MCP integration"""
    
    def __init__(self, storage_path: str = None):
        self.arxiv_client = ArxivMCPClient(storage_path)
        
    async def research_topic(self, topic: str, max_papers: int = 10) -> Dict[str, Any]:
        """Research a topic using ArXiv papers"""
        try:
            async with self.arxiv_client as client:
                # Search for papers
                search_result = await client.search_papers(topic, max_results=max_papers)
                
                if not search_result.success:
                    return {
                        "topic": topic,
                        "error": search_result.error,
                        "papers": []
                    }
                
                papers = search_result.content.get("papers", [])
                
                # Download top papers
                downloaded_papers = []
                for paper in papers[:5]:  # Download top 5 papers
                    paper_id = paper.get("id", "").replace("http://arxiv.org/abs/", "")
                    if paper_id:
                        download_result = await client.download_paper(paper_id)
                        if download_result.success:
                            downloaded_papers.append(paper_id)
                
                return {
                    "topic": topic,
                    "papers_found": len(papers),
                    "papers_downloaded": len(downloaded_papers),
                    "papers": papers,
                    "downloaded_paper_ids": downloaded_papers
                }
                
        except Exception as e:
            logger.error(f"Error researching topic {topic}: {str(e)}")
            return {
                "topic": topic,
                "error": str(e),
                "papers": []
            }
    
    async def analyze_paper(self, paper_id: str) -> Dict[str, Any]:
        """Analyze a specific paper"""
        try:
            async with self.arxiv_client as client:
                # First download the paper if not already downloaded
                download_result = await client.download_paper(paper_id)
                if not download_result.success:
                    return {
                        "paper_id": paper_id,
                        "error": f"Failed to download paper: {download_result.error}"
                    }
                
                # Read the paper content
                read_result = await client.read_paper(paper_id)
                if not read_result.success:
                    return {
                        "paper_id": paper_id,
                        "error": f"Failed to read paper: {read_result.error}"
                    }
                
                paper_content = read_result.content
                
                return {
                    "paper_id": paper_id,
                    "title": paper_content.get("title", ""),
                    "authors": paper_content.get("authors", []),
                    "abstract": paper_content.get("abstract", ""),
                    "content": paper_content.get("content", ""),
                    "categories": paper_content.get("categories", []),
                    "published": paper_content.get("published", "")
                }
                
        except Exception as e:
            logger.error(f"Error analyzing paper {paper_id}: {str(e)}")
            return {
                "paper_id": paper_id,
                "error": str(e)
            }
    
    async def get_research_summary(self, topic: str) -> Dict[str, Any]:
        """Get a comprehensive research summary on a topic"""
        try:
            # Research the topic
            research_result = await self.research_topic(topic)
            
            if "error" in research_result:
                return research_result
            
            # Analyze downloaded papers
            analyzed_papers = []
            for paper_id in research_result.get("downloaded_paper_ids", []):
                analysis = await self.analyze_paper(paper_id)
                if "error" not in analysis:
                    analyzed_papers.append(analysis)
            
            return {
                "topic": topic,
                "research_summary": research_result,
                "analyzed_papers": analyzed_papers,
                "total_papers_found": research_result.get("papers_found", 0),
                "total_papers_analyzed": len(analyzed_papers)
            }
            
        except Exception as e:
            logger.error(f"Error getting research summary for {topic}: {str(e)}")
            return {
                "topic": topic,
                "error": str(e)
            }