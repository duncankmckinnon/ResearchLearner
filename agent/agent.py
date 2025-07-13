from typing import Dict
from openinference.instrumentation import using_session
from openinference.semconv.trace import SpanAttributes
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from agent.schema import RequestFormat, ResponseFormat
from agent.prompts import Prompts
from agent.caching import LRUCache
from agent.langgraph_agent import LangGraphResearchAgent
from dotenv import load_dotenv
from agent.constants import PROJECT_NAME, AGENT_NAME, SPAN_TYPE
import os
import logging
import asyncio

logger = logging.getLogger("agent_demo")

tracer = trace.get_tracer(PROJECT_NAME)

load_dotenv()

class Agent:

    def __init__(self):
        self.langgraph_agent = LangGraphResearchAgent()
        self.cache = LRUCache()

    @tracer.start_as_current_span(
        name=AGENT_NAME, attributes={SpanAttributes.OPENINFERENCE_SPAN_KIND: SPAN_TYPE}
    )
    def analyze_request(self, message: RequestFormat) -> Dict:
        """Analyze the request and determine the appropriate response using LangGraph"""
        conversation_hash = message.conversation_hash
        request = message.customer_message
        
        # Get or create session context
        stuff = self.cache.get(conversation_hash)
        if not stuff:
            context = "start"
            session_id = self.cache.set(conversation_hash, [{"context": context}])
        else:
            context = stuff[0]
            session_id = stuff[1]
        
        current_span = trace.get_current_span()
        current_span.set_attribute(SpanAttributes.SESSION_ID, str(session_id))
        current_span.set_attribute(SpanAttributes.INPUT_VALUE, request)
        
        try:
            with using_session(session_id):
                # Use LangGraph agent to process the request
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(
                        self.langgraph_agent.process_request(request, session_id, context)
                    )
                    
                    response = result.get("response", "No response generated")
                    
                    # Cache the interaction
                    self.cache.add_interaction(conversation_hash, request, response)
                    
                    # Set span attributes with additional context
                    current_span.set_attribute(SpanAttributes.OUTPUT_VALUE, str(response))
                    current_span.set_attribute("intent", result.get("intent", "unknown"))
                    current_span.set_attribute("plan", str(result.get("plan", [])))
                    current_span.set_status(Status(StatusCode.OK))
                    
                    return {
                        "response": response,
                        "intent": result.get("intent"),
                        "plan": result.get("plan"),
                        "research_data": result.get("research_data")
                    }
                    
                finally:
                    loop.close()
                    
        except Exception as e:
            error_response = {
                "response": "I apologize, but I'm having trouble processing your request. Please try again."
            }
            logger.error(f"Error processing request with LangGraph: {str(e)}")
            current_span.set_status(Status(StatusCode.ERROR))
            return error_response

    def handle_request(self, message: RequestFormat) -> ResponseFormat:
        """Process a request and generate a response"""

        # Analyze the request
        analysis = self.analyze_request(message)
        return ResponseFormat(**analysis)