from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from agent.agent import Agent
from phoenix.otel import register
from openinference.instrumentation.openai import OpenAIInstrumentor
from openinference.instrumentation.langchain import LangChainInstrumentor
from agent.schema import (
    RequestFormat, 
    ResponseFormat, 
    KnowledgeSearchRequest, 
    KnowledgeSearchResult, 
    ResearchPaper, 
    ResearchInsight, 
    KnowledgeSummary
)
from agent.caching import LRUCache
from agent.constants import PROJECT_NAME
from agent.knowledge_graph import get_knowledge_graph_manager
import logging
import json
import asyncio
import time
from typing import Generator
import uuid
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

logger.info("Initializing FastAPI application...")

tracer_provider = register(
    project_name=PROJECT_NAME,
)

# Template uses OpenAI, but any LLM provider or agentic framework can be plugged in
LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

app = FastAPI()
agent = Agent()
knowledge_graph = get_knowledge_graph_manager()

# Store for tracking ongoing processes
active_processes = {}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/clear_cache")
def clear_cache():
    cache.clear()
    return {"message": "Cache cleared"}

@app.post("/agent", response_model=ResponseFormat)
def process_request(request: RequestFormat):
    try:
        response = agent.handle_request(request)
        
        return response
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/agent/stream")
async def process_request_stream(request: RequestFormat):
    """Stream agent processing with real-time updates"""
    try:
        process_id = str(uuid.uuid4())
        
        def generate_stream():
            try:
                # Store process info
                active_processes[process_id] = {
                    "status": "starting",
                    "timestamp": datetime.now().isoformat(),
                    "request": request.dict()
                }
                
                # Send initial status
                yield f"data: {json.dumps({'type': 'status', 'message': 'Starting analysis...', 'process_id': process_id})}\n\n"
                time.sleep(0.2)  # Small delay for UI
                
                # Update process status
                active_processes[process_id]["status"] = "processing"
                yield f"data: {json.dumps({'type': 'status', 'message': 'Processing your request...', 'process_id': process_id})}\n\n"
                time.sleep(0.2)
                
                # Step 1: Detect intent and create plan
                yield f"data: {json.dumps({'type': 'progress', 'message': 'Analyzing request intent...', 'step': 1, 'total_steps': 5})}\n\n"
                time.sleep(1.0)  # Realistic delay for intent detection
                
                # Step 2: Show that we're starting the main processing
                yield f"data: {json.dumps({'type': 'progress', 'message': 'Initializing agent workflow...', 'step': 2, 'total_steps': 5})}\n\n"
                time.sleep(0.8)
                
                # Step 3: Start actual processing (this is where the real work happens)
                yield f"data: {json.dumps({'type': 'progress', 'message': 'Processing with research agent...', 'step': 3, 'total_steps': 5})}\n\n"
                
                # Process the request with the agent (this is the main work)
                logger.info("Starting agent request processing...")
                response = agent.handle_request(request)
                logger.info(f"Agent processing completed. Response type: {type(response)}")
                logger.info(f"Response object attributes: {dir(response)}")
                logger.info(f"Response content preview: {getattr(response, 'response', 'NO RESPONSE ATTR')[:100]}...")
                
                # Step 4: Post-processing
                yield f"data: {json.dumps({'type': 'progress', 'message': 'Finalizing results...', 'step': 4, 'total_steps': 5})}\n\n"
                time.sleep(0.8)
                
                # Step 5: Send additional progress based on detected intent
                if hasattr(response, 'intent'):
                    if response.intent == "research":
                        yield f"data: {json.dumps({'type': 'progress', 'message': 'Research completed - papers analyzed and knowledge updated', 'step': 5, 'total_steps': 5})}\n\n"
                    elif response.intent == "knowledge_query":
                        yield f"data: {json.dumps({'type': 'progress', 'message': 'Knowledge base searched and results compiled', 'step': 5, 'total_steps': 5})}\n\n"
                    else:
                        yield f"data: {json.dumps({'type': 'progress', 'message': 'Response generated successfully', 'step': 5, 'total_steps': 5})}\n\n"
                else:
                    yield f"data: {json.dumps({'type': 'progress', 'message': 'Processing completed', 'step': 5, 'total_steps': 5})}\n\n"
                
                time.sleep(0.5)
                
                # Send final response
                active_processes[process_id]["status"] = "completed"
                
                # Debug: Log the response being sent
                try:
                    # Use dict access for Pydantic models or getattr for regular objects
                    if hasattr(response, 'dict'):
                        # It's a Pydantic model
                        response_dict = response.dict()
                        response_text = response_dict.get('response', 'No response generated')
                        response_intent = response_dict.get('intent', None)
                        response_plan = response_dict.get('plan', None)
                        logger.info(f"Pydantic response dict: {response_dict}")
                    else:
                        # It's a regular object
                        response_text = getattr(response, 'response', 'No response generated')
                        response_intent = getattr(response, 'intent', None)
                        response_plan = getattr(response, 'plan', None)
                        logger.info(f"Regular object response - text: {response_text[:100]}...")
                    
                    response_data = {
                        'type': 'response', 
                        'response': response_text, 
                        'intent': response_intent, 
                        'plan': response_plan
                    }
                    logger.info(f"Sending response event: {response_data}")
                    
                    yield f"data: {json.dumps(response_data)}\n\n"
                except Exception as e:
                    logger.error(f"Error formatting response: {str(e)}, response object: {response}")
                    logger.error(f"Response type: {type(response)}, Response attrs: {dir(response) if hasattr(response, '__dict__') else 'No __dict__'}")
                    # Send a fallback response
                    fallback_response = {
                        'type': 'response',
                        'response': 'I apologize, but there was an issue formatting the response. Please try again.',
                        'intent': None,
                        'plan': None
                    }
                    yield f"data: {json.dumps(fallback_response)}\n\n"
                
                # Send completion status
                yield f"data: {json.dumps({'type': 'complete', 'message': 'Process completed successfully', 'process_id': process_id})}\n\n"
                
            except Exception as e:
                logger.error(f"Error in stream processing: {str(e)}")
                if process_id in active_processes:
                    active_processes[process_id]["status"] = "error"
                yield f"data: {json.dumps({'type': 'error', 'message': f'Error: {str(e)}', 'process_id': process_id})}\n\n"
            finally:
                # Clean up process tracking
                if process_id in active_processes:
                    del active_processes[process_id]
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type",
            }
        )
        
    except Exception as e:
        logger.error(f"Error initializing stream: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agent/status/{process_id}")
def get_process_status(process_id: str):
    """Get the status of a specific process"""
    if process_id in active_processes:
        return active_processes[process_id]
    else:
        return {"error": "Process not found"}

@app.get("/agent/processes")
def list_active_processes():
    """List all active processes"""
    return {"active_processes": list(active_processes.keys()), "count": len(active_processes)}


# Knowledge Store Endpoints

@app.post("/knowledge/search")
def search_knowledge(request: KnowledgeSearchRequest):
    """Search the knowledge store for relevant information"""
    try:
        kg_manager = get_knowledge_graph_manager()
        results = kg_manager.search_knowledge(request.query, limit=request.limit)
        
        # Convert to Pydantic models
        search_results = [
            KnowledgeSearchResult(
                content=result.get("content", ""),
                metadata=result.get("metadata", {}),
                relevance_score=result.get("relevance_score", 0.0),
                id=result.get("id", "")
            )
            for result in results
        ]
        
        return {
            "results": search_results,
            "total_results": len(search_results),
            "query": request.query
        }
        
    except Exception as e:
        logger.error(f"Error searching knowledge store: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/knowledge/papers/{topic}")
def get_related_papers(topic: str, limit: int = 5):
    """Get research papers related to a specific topic"""
    try:
        kg_manager = get_knowledge_graph_manager()
        papers_data = kg_manager.get_related_papers(topic, limit=limit)
        
        # Convert to Pydantic models
        papers = [
            ResearchPaper(
                title=paper.get("title", ""),
                authors=paper.get("authors", []),
                arxiv_id=paper.get("arxiv_id", ""),
                categories=paper.get("categories", []),
                relevance_score=paper.get("relevance_score", 0.0),
                content=paper.get("content", ""),
                source=paper.get("source", "unknown")
            )
            for paper in papers_data
        ]
        
        return {
            "papers": papers,
            "total_papers": len(papers),
            "topic": topic
        }
        
    except Exception as e:
        logger.error(f"Error retrieving related papers: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/knowledge/insights/{topic}")
def get_research_insights(topic: str, limit: int = 10):
    """Get research insights for a specific topic"""
    try:
        kg_manager = get_knowledge_graph_manager()
        insights_data = kg_manager.get_research_insights(topic, limit=limit)
        
        # Convert to Pydantic models
        insights = [
            ResearchInsight(
                insight=insight.get("insight", ""),
                topic=insight.get("topic", ""),
                context=insight.get("context", {}),
                relevance_score=insight.get("relevance_score", 0.0),
                added_date=insight.get("added_date", "")
            )
            for insight in insights_data
        ]
        
        return {
            "insights": insights,
            "total_insights": len(insights),
            "topic": topic
        }
        
    except Exception as e:
        logger.error(f"Error retrieving research insights: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/knowledge/summary/{topic}")
def get_knowledge_summary(topic: str):
    """Get a comprehensive knowledge summary for a topic"""
    try:
        kg_manager = get_knowledge_graph_manager()
        summary_data = kg_manager.get_knowledge_summary(topic)
        
        if "error" in summary_data:
            raise HTTPException(status_code=500, detail=summary_data["error"])
        
        # Convert papers to Pydantic models
        papers = [
            ResearchPaper(
                title=paper.get("title", ""),
                authors=paper.get("authors", []),
                arxiv_id=paper.get("arxiv_id", ""),
                categories=paper.get("categories", []),
                relevance_score=paper.get("relevance_score", 0.0),
                content=paper.get("content", ""),
                source=paper.get("source", "unknown")
            )
            for paper in summary_data.get("related_papers", [])
        ]
        
        # Convert insights to Pydantic models
        insights = [
            ResearchInsight(
                insight=insight.get("insight", ""),
                topic=insight.get("topic", ""),
                context=insight.get("context", {}),
                relevance_score=insight.get("relevance_score", 0.0),
                added_date=insight.get("added_date", "")
            )
            for insight in summary_data.get("research_insights", [])
        ]
        
        # Convert general knowledge to Pydantic models
        general_knowledge = [
            KnowledgeSearchResult(
                content=item.get("content", ""),
                metadata=item.get("metadata", {}),
                relevance_score=item.get("relevance_score", 0.0),
                id=item.get("id", "")
            )
            for item in summary_data.get("general_knowledge", [])
        ]
        
        summary = KnowledgeSummary(
            topic=summary_data.get("topic", topic),
            related_papers=papers,
            research_insights=insights,
            general_knowledge=general_knowledge,
            total_papers=summary_data.get("total_papers", len(papers)),
            total_insights=summary_data.get("total_insights", len(insights)),
            total_knowledge_items=summary_data.get("total_knowledge_items", len(general_knowledge))
        )
        
        return summary
        
    except Exception as e:
        logger.error(f"Error retrieving knowledge summary: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/knowledge/memories")
def get_all_memories(limit: int = 50):
    """Get all memories from the knowledge store"""
    try:
        kg_manager = get_knowledge_graph_manager()
        memories = kg_manager.get_all_memories(limit=limit)
        
        return {
            "memories": memories,
            "total_memories": len(memories),
        }
        
    except Exception as e:
        logger.error(f"Error retrieving all memories: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/knowledge/memory/{memory_id}")
def delete_memory(memory_id: str):
    """Delete a specific memory from the knowledge store"""
    try:
        kg_manager = get_knowledge_graph_manager()
        success = kg_manager.delete_memory(memory_id)
        
        if success:
            return {"message": f"Memory {memory_id} deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Memory not found or could not be deleted")
            
    except Exception as e:
        logger.error(f"Error deleting memory: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))