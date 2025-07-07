from typing import Dict, List, Any, Optional
import logging
import os
from datetime import datetime
import json
from opentelemetry import trace
from openinference.semconv.trace import SpanAttributes

logger = logging.getLogger("knowledge_graph")

class KnowledgeGraphManager:
    """Manager for mem0 knowledge graph integration"""
    
    def __init__(self, user_id: str = "default", tracer: trace.Tracer = None):
        self.user_id = user_id
        # Don't store tracer instance - use global tracer for proper context propagation
        self.memory = None
        self._initialize_memory()
    
    def _initialize_memory(self):
        """Initialize mem0 memory instance"""
        try:
            from mem0 import Memory
            
            config = {
                "vector_store": {
                    "provider": "chroma",
                    "config": {
                        "collection_name": "research_knowledge",
                        "path": os.path.expanduser("~/.research_learner/knowledge_db")
                    }
                },
                "llm": {
                    "provider": "openai",
                    "config": {
                        "model": os.getenv("OPENAI_MODEL", "gpt-4o"),
                        "temperature": 0.1
                    }
                }
            }
            
            self.memory = Memory.from_config(config)
            logger.info("Mem0 knowledge graph initialized successfully")
            
        except ImportError:
            logger.error("mem0 library not installed. Install with: pip install mem0ai")
            self.memory = None
        except Exception as e:
            logger.error(f"Error initializing mem0: {str(e)}")
            self.memory = None
    
    def add_research_paper(self, paper_data: Dict[str, Any]) -> bool:
        """Add a research paper to the knowledge graph"""
        if not self.memory:
            logger.error("Memory not initialized")
            return False
        
        try:
            # Create comprehensive paper description
            paper_text = self._format_paper_for_storage(paper_data)
            
            # Add to memory with metadata (flatten complex types for ChromaDB)
            metadata = {
                "type": "research_paper",
                "arxiv_id": paper_data.get("paper_id", ""),
                "title": paper_data.get("title", ""),
                "published": paper_data.get("published", ""),
                "added_date": datetime.now().isoformat()
            }
            
            # Convert list fields to comma-separated strings for ChromaDB
            authors = paper_data.get("authors", [])
            if isinstance(authors, list):
                metadata["authors"] = ", ".join(str(author) for author in authors)
            else:
                metadata["authors"] = str(authors)
                
            categories = paper_data.get("categories", [])
            if isinstance(categories, list):
                metadata["categories"] = ", ".join(str(cat) for cat in categories)
            else:
                metadata["categories"] = str(categories)
            
            result = self.memory.add(paper_text, user_id=self.user_id, metadata=metadata)
            logger.info(f"Added paper to knowledge graph: {paper_data.get('title', 'Unknown')}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding paper to knowledge graph: {str(e)}")
            return False
    
    def add_research_insight(self, insight: str, topic: str, context: Dict[str, Any] = {}) -> bool:
        """Add a research insight to the knowledge graph"""
        if not self.memory:
            logger.error("Memory not initialized")
            return False
        
        try:
            # Format insight with context
            insight_text = f"Research insight on {topic}: {insight}"
            
            # Flatten context into the metadata (ChromaDB only accepts simple types)
            metadata = {
                "type": "research_insight",
                "topic": topic,
                "added_date": datetime.now().isoformat()
            }
            
            # Add context as flat key-value pairs with string values only
            if context:
                insight_text += f"\n\nContext: {json.dumps(context, indent=2)}"
                for key, value in context.items():
                    # Convert all values to strings for ChromaDB compatibility
                    safe_key = f"context_{key}"
                    if isinstance(value, (str, int, float, bool)):
                        metadata[safe_key] = str(value)
                    elif value is None:
                        metadata[safe_key] = "None"
                    else:
                        # Convert complex types to JSON strings
                        metadata[safe_key] = json.dumps(value)
            
            result = self.memory.add(insight_text, user_id=self.user_id, metadata=metadata)
            logger.info(f"Added research insight for topic: {topic}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding research insight: {str(e)}")
            return False
    
    def search_knowledge(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search the knowledge graph"""
        if not self.memory:
            logger.error("Memory not initialized")
            return []
        
        try:
            with trace.get_tracer(__name__).start_as_current_span(
                "search_knowledge",
                attributes={
                    SpanAttributes.OPENINFERENCE_SPAN_KIND: "TOOL",
                    SpanAttributes.TOOL_NAME: "search_knowledge"
                }
            ) as tool_span:
                input_data = {"query": query, "limit": limit}
                tool_span.set_attribute(SpanAttributes.INPUT_VALUE, json.dumps(input_data))
                
                results = self.memory.search(query, user_id=self.user_id, limit=limit)
                
                # Format results for easier consumption
                formatted_results = []
                for result in results:
                    # Handle both dict and string results from mem0
                    if isinstance(result, dict):
                        formatted_results.append({
                            "content": result.get("memory", ""),
                            "metadata": result.get("metadata", {}),
                            "relevance_score": result.get("score", 0),
                            "id": result.get("id", "")
                        })
                    else:
                        # Handle string results
                        formatted_results.append({
                            "content": str(result),
                            "metadata": {},
                            "relevance_score": 0,
                            "id": ""
                        })
                
                logger.info(f"Found {len(formatted_results)} results for query: {query}")
                
                tool_span.set_attribute(SpanAttributes.OUTPUT_VALUE, json.dumps({"results": formatted_results, "query": query}))
                return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching knowledge graph: {str(e)}")
            return []
    
    def get_related_papers(self, topic: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get papers related to a specific topic"""
        if not self.memory:
            return []
        
        try:
            with trace.get_tracer(__name__).start_as_current_span(
                "get_related_papers",
                attributes={
                    SpanAttributes.OPENINFERENCE_SPAN_KIND: "TOOL",
                    SpanAttributes.TOOL_NAME: "get_related_papers"
                }
            ) as tool_span:
                input_data = {"topic": topic, "limit": limit}
                tool_span.set_attribute(SpanAttributes.INPUT_VALUE, json.dumps(input_data))
                
                # First check memory for existing papers
                memory_results = self.memory.search(
                    f"research papers about {topic}", 
                    user_id=self.user_id, 
                    limit=limit
                )
                
                # Filter for research papers from memory
                papers = []
                for result in memory_results:
                    try:
                        if isinstance(result, str):
                            result = json.loads(result)
                        elif isinstance(result, dict):
                            pass
                        else:
                            raise ValueError(f"Unknown result type: {type(result)}")
                        
                        metadata = result.get("metadata", {})
                        if metadata.get("type") == "research_paper":
                            papers.append({
                                "title": metadata.get("title", ""),
                                "authors": metadata.get("authors", "").split(", ") if metadata.get("authors") else [],
                                "arxiv_id": metadata.get("arxiv_id", ""),
                                "categories": metadata.get("categories", "").split(", ") if metadata.get("categories") else [],
                                "relevance_score": result.get("score", 0),
                                "content": result.get("memory", ""),
                                "source": "knowledge_graph"
                            })
                    except ValueError:
                        logger.error(f"Error processing memory result: {str(result)}")
                        continue
                    
                # If we don't have enough papers in memory, search ArXiv
                if len(papers) < limit:
                    try:
                        import arxiv
                        search = arxiv.Search(
                            query=topic,
                            max_results=limit - len(papers),
                            sort_by=arxiv.SortCriterion.Relevance
                        )
                        client = arxiv.Client()
                        
                        for result in client.results(search):
                            papers.append({
                                "title": result.title,
                                "authors": [str(author) for author in result.authors],
                                "arxiv_id": result.entry_id.split("/")[-1],
                                "categories": result.categories,
                                "relevance_score": 1.0,  # ArXiv results don't have scores
                                "content": result.summary,
                                "source": "arxiv_search"
                            })
                            
                    except ImportError:
                        logger.warning("arxiv library not available for searching additional papers")
                    except Exception as e:
                        logger.error(f"Error searching ArXiv for additional papers: {str(e)}")
                
                tool_span.set_attribute(SpanAttributes.OUTPUT_VALUE, json.dumps({"papers": papers, "topic": topic}))
                return papers[:limit]  # Ensure we don't exceed the limit
            
        except Exception as e:
            logger.error(f"Error getting related papers: {str(e)}")
            return []
    
    def get_research_insights(self, topic: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get research insights for a specific topic"""
        if not self.memory:
            return []
        
        try:
            with trace.get_tracer(__name__).start_as_current_span(
                "get_research_insights",
                attributes={
                    SpanAttributes.OPENINFERENCE_SPAN_KIND: "TOOL",
                    SpanAttributes.TOOL_NAME: "get_research_insights"
                }
            ) as tool_span:
                # Search for insights specifically
                tool_span.set_attribute(SpanAttributes.INPUT_VALUE, json.dumps({"topic": topic, "limit": limit}))
                results = self.memory.search(
                    f"research insights about {topic}", 
                    user_id=self.user_id, 
                    limit=limit
                )
                logger.info(f"Results: {results}")
                
                # Filter for research insights
                insights = []
                for result in results:
                    try:
                        if isinstance(result, str):
                            result = json.loads(result)
                        elif isinstance(result, dict):
                            pass
                        else:
                            raise ValueError(f"Unknown result type: {type(result)}")
                        
                    
                        metadata = result.get("metadata", {})
                        if metadata.get("type") == "research_insight":
                            insights.append({
                                "insight": result.get("memory", ""),
                                "topic": metadata.get("topic", ""),
                                "context": metadata.get("context", {}),
                                "relevance_score": result.get("score", 0),
                                "added_date": metadata.get("added_date", "")
                            })
                    except ValueError as e:
                        logger.error(f"Error processing memory result: {str(e)}")
                        continue
                tool_span.set_attribute(SpanAttributes.OUTPUT_VALUE, json.dumps({"insights_count": len(insights), "topic": topic}))
            return insights
            
        except Exception as e:
            logger.error(f"Error getting research insights: {str(e)}")
            return []
    
    def get_all_memories(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get all memories from the knowledge graph"""
        if not self.memory:
            return []
        
        try:
            # Get all memories for the user
            memories = self.memory.get_all(user_id=self.user_id, limit=limit)
            formatted_memories = []
            for memory in memories:
                memory = json.loads(memory)
                formatted_memories.append({
                    "id": memory.get("id", ""),
                    "content": memory.get("memory", ""),
                    "metadata": memory.get("metadata", {}),
                    "created_at": memory.get("created_at", ""),
                    "updated_at": memory.get("updated_at", "")
                })
            
            return formatted_memories
            
        except Exception as e:
            logger.error(f"Error getting all memories: {str(e)}")
            return []
    
    def delete_memory(self, memory_id: str) -> bool:
        """Delete a specific memory"""
        if not self.memory:
            return False
        
        try:
            self.memory.delete(memory_id=memory_id)
            logger.info(f"Deleted memory: {memory_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting memory: {str(e)}")
            return False
    
    def update_memory(self, memory_id: str, new_content: str) -> bool:
        """Update a specific memory"""
        if not self.memory:
            return False
        
        try:
            self.memory.update(memory_id=memory_id, data=new_content)
            logger.info(f"Updated memory: {memory_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating memory: {str(e)}")
            return False
    
    def _format_paper_for_storage(self, paper_data: Dict[str, Any]) -> str:
        """Format paper data for storage in the knowledge graph"""
        title = paper_data.get("title", "Unknown Title")
        authors = paper_data.get("authors", [])
        abstract = paper_data.get("abstract", "")
        content = paper_data.get("content", "")
        categories = paper_data.get("categories", [])
        paper_id = paper_data.get("paper_id", "")
        
        paper_text = f"""
Title: {title}

Authors: {', '.join(authors)}

ArXiv ID: {paper_id}

Categories: {', '.join(categories)}

Abstract: {abstract}

Key Content: {content[:2000]}...
        """.strip()
        
        return paper_text
    
    def get_knowledge_summary(self, topic: str) -> Dict[str, Any]:
        """Get a comprehensive knowledge summary for a topic"""
        if not self.memory:
            return {"error": "Knowledge graph not initialized"}
        
        try:
            # Get related papers
            papers = self.get_related_papers(topic, limit=5)
            
            # Get research insights
            insights = self.get_research_insights(topic, limit=10)
            
            # Get general knowledge
            general_knowledge = self.search_knowledge(topic, limit=10)
            
            summary = {
                "topic": topic,
                "related_papers": papers,
                "research_insights": insights,
                "general_knowledge": general_knowledge,
                "total_papers": len(papers),
                "total_insights": len(insights),
                "total_knowledge_items": len(general_knowledge)
            }
            
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting knowledge summary: {str(e)}")
            return {"error": str(e)}