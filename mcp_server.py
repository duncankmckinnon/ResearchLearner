#!/usr/bin/env python3
"""
MCP Server for Research Agent Integration with Claude
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional
from mcp import ClientSession, StdioServerParameters
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    Resource,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
    LoggingLevel
)
from agent.langgraph_agent import LangGraphResearchAgent
from agent.knowledge_graph import KnowledgeGraphManager
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("research-mcp-server")

# Create MCP server instance
server = Server("research-agent")

# Initialize research agent and knowledge graph
research_agent = LangGraphResearchAgent()
knowledge_graph = KnowledgeGraphManager()

@server.list_tools()
async def handle_list_tools() -> List[Tool]:
    """List available tools for the research agent"""
    return [
        Tool(
            name="research_topic",
            description="Research a specific topic using ArXiv papers and knowledge graph",
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "The research topic or query to investigate"
                    },
                    "max_papers": {
                        "type": "integer",
                        "description": "Maximum number of papers to analyze (default: 5)",
                        "default": 5
                    }
                },
                "required": ["topic"]
            }
        ),
        Tool(
            name="query_knowledge",
            description="Query the existing knowledge graph for information",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query to search in the knowledge base"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 10)",
                        "default": 10
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="analyze_paper",
            description="Analyze a specific ArXiv paper by ID",
            inputSchema={
                "type": "object",
                "properties": {
                    "paper_id": {
                        "type": "string",
                        "description": "ArXiv paper ID (e.g., '2301.12345')"
                    }
                },
                "required": ["paper_id"]
            }
        ),
        Tool(
            name="get_knowledge_summary",
            description="Get a comprehensive knowledge summary for a topic",
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "The topic to get knowledge summary for"
                    }
                },
                "required": ["topic"]
            }
        ),
        Tool(
            name="add_research_insight",
            description="Add a research insight to the knowledge graph",
            inputSchema={
                "type": "object",
                "properties": {
                    "insight": {
                        "type": "string",
                        "description": "The research insight to store"
                    },
                    "topic": {
                        "type": "string",
                        "description": "The topic this insight relates to"
                    },
                    "context": {
                        "type": "object",
                        "description": "Additional context for the insight",
                        "default": {}
                    }
                },
                "required": ["insight", "topic"]
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: Dict[str, Any] | None) -> List[TextContent]:
    """Handle tool calls from Claude"""
    try:
        if not arguments:
            arguments = {}
        
        session_id = str(uuid.uuid4())
        
        if name == "research_topic":
            topic = arguments.get("topic", "")
            max_papers = arguments.get("max_papers", 5)
            
            logger.info(f"Researching topic: {topic}")
            
            # Use the LangGraph agent to process the research request
            result = await research_agent.process_request(
                f"Research papers about {topic} and analyze up to {max_papers} papers",
                session_id
            )
            
            # Format the response
            response_text = f"Research Results for '{topic}':\n\n"
            response_text += result.get("response", "No response generated")
            
            # Add additional context if available
            if result.get("research_data"):
                research_data = result["research_data"]
                papers_found = research_data.get("search_results", {}).get("papers_found", 0)
                analyzed_papers = len(research_data.get("analyzed_papers", []))
                response_text += f"\n\nSummary: Found {papers_found} papers, analyzed {analyzed_papers} papers"
            
            return [TextContent(type="text", text=response_text)]
        
        elif name == "query_knowledge":
            query = arguments.get("query", "")
            limit = arguments.get("limit", 10)
            
            logger.info(f"Querying knowledge graph: {query}")
            
            results = knowledge_graph.search_knowledge(query, limit=limit)
            
            if not results:
                response_text = f"No knowledge found for query: {query}"
            else:
                response_text = f"Knowledge Search Results for '{query}':\n\n"
                for i, result in enumerate(results, 1):
                    response_text += f"{i}. {result['content'][:300]}...\n"
                    response_text += f"   Relevance Score: {result['relevance_score']:.2f}\n\n"
            
            return [TextContent(type="text", text=response_text)]
        
        elif name == "analyze_paper":
            paper_id = arguments.get("paper_id", "")
            
            logger.info(f"Analyzing paper: {paper_id}")
            
            # Use the LangGraph agent to analyze the paper
            result = await research_agent.process_request(
                f"Analyze the ArXiv paper {paper_id} in detail",
                session_id
            )
            
            response_text = f"Analysis of Paper {paper_id}:\n\n"
            response_text += result.get("response", "No analysis generated")
            
            return [TextContent(type="text", text=response_text)]
        
        elif name == "get_knowledge_summary":
            topic = arguments.get("topic", "")
            
            logger.info(f"Getting knowledge summary for: {topic}")
            
            summary = knowledge_graph.get_knowledge_summary(topic)
            
            if "error" in summary:
                response_text = f"Error getting knowledge summary: {summary['error']}"
            else:
                response_text = f"Knowledge Summary for '{topic}':\n\n"
                response_text += f"Related Papers: {summary.get('total_papers', 0)}\n"
                response_text += f"Research Insights: {summary.get('total_insights', 0)}\n"
                response_text += f"Knowledge Items: {summary.get('total_knowledge_items', 0)}\n\n"
                
                # Add paper summaries
                for paper in summary.get("related_papers", [])[:3]:
                    response_text += f"📄 {paper.get('title', 'Unknown Title')}\n"
                    response_text += f"   Authors: {', '.join(paper.get('authors', []))}\n\n"
                
                # Add insights
                for insight in summary.get("research_insights", [])[:3]:
                    response_text += f"💡 {insight.get('insight', '')[:200]}...\n\n"
            
            return [TextContent(type="text", text=response_text)]
        
        elif name == "add_research_insight":
            insight = arguments.get("insight", "")
            topic = arguments.get("topic", "")
            context = arguments.get("context", {})
            
            logger.info(f"Adding research insight for topic: {topic}")
            
            success = knowledge_graph.add_research_insight(insight, topic, context)
            
            if success:
                response_text = f"Successfully added research insight for topic: {topic}"
            else:
                response_text = f"Failed to add research insight for topic: {topic}"
            
            return [TextContent(type="text", text=response_text)]
        
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
    
    except Exception as e:
        logger.error(f"Error in tool call {name}: {str(e)}")
        return [TextContent(type="text", text=f"Error: {str(e)}")]

@server.list_resources()
async def handle_list_resources() -> List[Resource]:
    """List available resources"""
    return [
        Resource(
            uri="knowledge://papers",
            name="Research Papers",
            description="Collection of analyzed research papers",
            mimeType="application/json"
        ),
        Resource(
            uri="knowledge://insights",
            name="Research Insights",
            description="Collection of research insights and findings",
            mimeType="application/json"
        )
    ]

@server.read_resource()
async def handle_read_resource(uri: str) -> str:
    """Read resource content"""
    try:
        if uri == "knowledge://papers":
            memories = knowledge_graph.get_all_memories(limit=50)
            papers = [m for m in memories if m.get("metadata", {}).get("type") == "research_paper"]
            return json.dumps(papers, indent=2)
        
        elif uri == "knowledge://insights":
            memories = knowledge_graph.get_all_memories(limit=50)
            insights = [m for m in memories if m.get("metadata", {}).get("type") == "research_insight"]
            return json.dumps(insights, indent=2)
        
        else:
            raise ValueError(f"Unknown resource: {uri}")
    
    except Exception as e:
        logger.error(f"Error reading resource {uri}: {str(e)}")
        raise

async def main():
    """Main entry point for the MCP server"""
    # Run the stdio server
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="research-agent",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=None,
                    experimental_capabilities=None
                )
            )
        )

if __name__ == "__main__":
    asyncio.run(main())