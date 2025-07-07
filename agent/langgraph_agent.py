from typing import Dict, List, Any, Optional, TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from opentelemetry import trace, context
from openinference.semconv.trace import SpanAttributes
from openinference.instrumentation import using_prompt_template
from agent.constants import PROJECT_NAME
from agent.arxiv_client import SimpleResearchAgent
from agent.knowledge_graph import KnowledgeGraphManager
from agent.prompts import Prompts
import asyncio
import contextvars
import logging
import os
from dotenv import load_dotenv
import json

load_dotenv()
logger = logging.getLogger("langgraph_agent")

class AgentState(TypedDict):
    """State for the LangGraph agent"""
    messages: Annotated[List[BaseMessage], add_messages]
    intent: Optional[str]
    plan: Optional[List[str]]
    current_step: int
    research_data: Optional[Dict[str, Any]]
    knowledge_data: Optional[Dict[str, Any]]
    final_response: Optional[str]
    user_request: str
    session_id: str

class IntentDetectionNode:
    """Node for detecting user intent"""
    
    def __init__(self, llm: ChatOpenAI, prompts: Prompts):
        self.llm = llm
        self.prompts = prompts
    
    def __call__(self, state: AgentState) -> AgentState:
        """Detect intent from user request"""
        try:
            user_request = state["user_request"]
            context = state["context"]
            
            # Prepare prompt template variables
            template_variables = {
                "user_request": user_request,
                "context": context
            }
            
            # Get the template string from the prompt
            template_messages = self.prompts.intent_detection_prompt.messages
            template_string = f"{template_messages[0].prompt.template}\n\nUser: {template_messages[1].prompt.template}"
            
            with using_prompt_template(
                template=template_string,
                variables=template_variables,
                version="v1.0"
            ):
                response = self.llm.invoke(
                    self.prompts.intent_detection_prompt.format_messages(**template_variables)
                )
        
            intent = response.content.strip().lower()
            
            # Validate intent
            valid_intents = ["research", "analysis", "knowledge_query", "general"]
            if intent not in valid_intents:
                intent = "general"
            
            state["intent"] = intent
            state["messages"].append(AIMessage(content=f"Detected intent: {intent}"))
        
            logger.info(f"Detected intent: {intent} for request: {user_request}")
            
        except Exception as e:
            logger.error(f"Error in intent detection: {str(e)}")
            state["intent"] = "general"
            state["messages"].append(AIMessage(content="Error detecting intent, defaulting to general"))
        
        return state

class PlanningNode:
    """Node for creating execution plans"""
    
    def __init__(self, llm: ChatOpenAI, prompts: Prompts):
        self.llm = llm
        self.prompts = prompts
    
    def __call__(self, state: AgentState) -> AgentState:
        """Create execution plan based on intent"""
        try:
            intent = state["intent"]
            user_request = state["user_request"]
            context = ""  # Add context handling if needed
            
            # Prepare prompt template variables
            template_variables = {
                "intent": intent,
                "user_request": user_request,
                "context": context
            }
            
            # Get the template string from the prompt
            template_messages = self.prompts.planning_prompt.messages
            template_string = f"{template_messages[0].prompt.template}\n\nUser: {template_messages[1].prompt.template}"
            
            with using_prompt_template(
                template=template_string,
                variables=template_variables,
                version="v1.0"
            ):
                response = self.llm.invoke(
                    self.prompts.planning_prompt.format_messages(**template_variables)
                )
            
            # Parse the JSON response
            try:
                plan = json.loads(response.content.strip())
                if not isinstance(plan, list):
                    raise ValueError("Plan must be a list")
            except (json.JSONDecodeError, ValueError):
                # Fallback to default plans
                plan = self._get_default_plan(intent)
            
            state["plan"] = plan
            state["current_step"] = 0
            state["messages"].append(AIMessage(content=f"Created plan: {plan}"))
            
            logger.info(f"Created plan for intent {intent}: {plan}")
            
        except Exception as e:
            logger.error(f"Error in planning: {str(e)}")
            state["plan"] = self._get_default_plan(state["intent"])
            state["current_step"] = 0
            state["messages"].append(AIMessage(content="Error creating plan, using default"))
        
        return state
    
    def _get_default_plan(self, intent: str | None) -> List[str]:
        """Get default plan for each intent type"""
        defaults = {
            "research": [
                "Extract research topic",
                "Search for papers",
                "Analyze findings",
                "Synthesize results"
            ],
            "analysis": [
                "Identify target papers",
                "Retrieve content",
                "Analyze content",
                "Present insights"
            ],
            "knowledge_query": [
                "Search knowledge base",
                "Retrieve information",
                "Formulate response"
            ],
            "general": [
                "Process request",
                "Generate response"
            ]
        }
        return defaults.get(intent, ["Process request", "Generate response"])

class ResearchExecutionNode:
    """Node for executing research tasks"""
    
    def __init__(self, llm: ChatOpenAI, knowledge_graph: KnowledgeGraphManager, research_agent: SimpleResearchAgent, prompts: Prompts):
        self.llm = llm
        self.research_agent = research_agent
        self.knowledge_graph = knowledge_graph
        self.prompts = prompts
    
    async def __call__(self, state: AgentState) -> AgentState:
        """Execute research tasks"""
        try:
            current_step = state["current_step"]
            plan = state["plan"]
            user_request = state["user_request"]
            
            if current_step >= len(plan):
                return state
            
            current_action = plan[current_step]
            
            if "extract" in current_action.lower() and "topic" in current_action.lower():
                # Extract research topic
                template_variables = {"user_request": user_request}
                template_messages = self.prompts.topic_extraction_prompt.messages
                template_string = f"{template_messages[0].prompt.template}\n\nUser: {template_messages[1].prompt.template}"
                
                with using_prompt_template(
                    template=template_string,
                    variables=template_variables,
                    version="v1.0"
                ):
                    response = self.llm.invoke(
                        self.prompts.topic_extraction_prompt.format_messages(**template_variables)
                    )
                topic = response.content.strip()
                
                if not state["research_data"]:
                    state["research_data"] = {}
                state["research_data"]["topic"] = topic
                state["messages"].append(AIMessage(content=f"Extracted research topic: {topic}"))
                
            elif "search" in current_action.lower():
                # Search for papers
                topic = state["research_data"].get("topic", user_request)

                with trace.get_tracer(__name__).start_as_current_span(
                    "research_topic",
                    attributes={
                        SpanAttributes.OPENINFERENCE_SPAN_KIND: "TOOL",
                        SpanAttributes.TOOL_NAME: "research_topic"
                    }
                ) as tool_span:
                    tool_span.set_attribute(SpanAttributes.INPUT_VALUE, topic)  
                    research_result = await self.research_agent.research_topic(topic, max_papers=10)
                    tool_span.set_attribute(SpanAttributes.OUTPUT_VALUE, json.dumps({"papers_found": research_result.get("papers_found", 0)}))

                if not state["research_data"]:
                    state["research_data"] = {}
                state["research_data"]["search_results"] = research_result
                
                papers_found = research_result.get("papers_found", 0)
                state["messages"].append(AIMessage(content=f"Found {papers_found} relevant papers"))
                
            elif "analyze" in current_action.lower() and "research_data" in state and isinstance(state["research_data"], dict):
                # Analyze findings
                search_results = state["research_data"].get("search_results", {})
                papers = search_results.get("papers", [])
                
                if papers:
                    # Analyze top papers
                    analyzed_papers = []
                    for paper in papers[:3]:  # Analyze top 3 papers
                        paper_id = paper.get("id", "").replace("http://arxiv.org/abs/", "")
                        if paper_id:
                            with trace.get_tracer(__name__).start_as_current_span(
                                "analyze_paper",
                                attributes={
                                    SpanAttributes.OPENINFERENCE_SPAN_KIND: "TOOL",
                                    SpanAttributes.TOOL_NAME: "analyze_paper"
                                }
                            ) as tool_span:
                                tool_span.set_attribute(SpanAttributes.INPUT_VALUE, paper_id)
                                analysis = await self.research_agent.analyze_paper(paper_id)
                                tool_span.set_attribute(SpanAttributes.OUTPUT_VALUE, json.dumps({"analysis": analysis}))
                            if "error" not in analysis:
                                analyzed_papers.append(analysis)
                                # Add paper to knowledge graph
                                with trace.get_tracer(__name__).start_as_current_span(
                                    "add_research_paper",
                                    attributes={
                                        SpanAttributes.OPENINFERENCE_SPAN_KIND: "TOOL",
                                        SpanAttributes.TOOL_NAME: "add_research_paper"
                                    }
                                ) as tool_span:
                                    tool_span.set_attribute(SpanAttributes.INPUT_VALUE, paper_id)
                                    self.knowledge_graph.add_research_paper(analysis)
                                    tool_span.set_attribute(SpanAttributes.OUTPUT_VALUE, json.dumps({"paper_added": True}))
                    
                    state["research_data"]["analyzed_papers"] = analyzed_papers
                    state["messages"].append(AIMessage(content=f"Analyzed {len(analyzed_papers)} papers and added to knowledge graph"))
                
            elif "synthesize" in current_action.lower():
                # Synthesize results with knowledge graph integration
                research_data = state["research_data"]
                topic = research_data.get("topic", user_request)
                
                # Get existing knowledge from graph
                knowledge_summary = self.knowledge_graph.get_knowledge_summary(topic)
                state["knowledge_data"] = knowledge_summary
                
                # Synthesize with both new research and existing knowledge
                synthesis = await self._synthesize_research(research_data, user_request, knowledge_summary)
                state["research_data"]["synthesis"] = synthesis
                
                # Add synthesis insights to knowledge graph
                self.knowledge_graph.add_research_insight(
                    synthesis, 
                    topic,
                    {"research_session": user_request, "date": "2024"}
                )
                
                state["messages"].append(AIMessage(content="Synthesized research findings with existing knowledge"))
            
            state["current_step"] += 1
            
        except Exception as e:
            logger.error(f"Error in research execution: {str(e)}")
            state["messages"].append(AIMessage(content=f"Error in research step: {str(e)}"))
            state["current_step"] += 1
        
        return state
    
    async def _synthesize_research(self, research_data: Dict, user_request: str, knowledge_summary: Dict = None) -> str:
        """Synthesize research findings"""
        try:
            topic = research_data.get("topic", "")
            search_results = research_data.get("search_results", {})
            analyzed_papers = research_data.get("analyzed_papers", [])
            
            synthesis_prompt = f"""
            Based on the research conducted on "{topic}", synthesize the key findings:
            
            Papers found: {search_results.get('papers_found', 0)}
            Papers analyzed: {len(analyzed_papers)}
            
            Key papers analyzed:
            """
            
            for paper in analyzed_papers:
                synthesis_prompt += f"\n- {paper.get('title', 'Unknown title')}"
                synthesis_prompt += f"\n  Authors: {', '.join(paper.get('authors', []))}"
                synthesis_prompt += f"\n  Abstract: {paper.get('abstract', 'No abstract')[:200]}..."
            
            synthesis_prompt += f"\n\nOriginal request: {user_request}"
            synthesis_prompt += "\n\nProvide a comprehensive synthesis of the research findings."
            
            response = self.llm.invoke([HumanMessage(content=synthesis_prompt)])
            return response.content
            
        except Exception as e:
            logger.error(f"Error synthesizing research: {str(e)}")
            return f"Error synthesizing research: {str(e)}"

class KnowledgeQueryNode:
    """Node for querying existing knowledge"""
    
    def __init__(self, llm: ChatOpenAI, knowledge_graph: KnowledgeGraphManager, prompts: Prompts):
        self.llm = llm
        self.knowledge_graph = knowledge_graph
        self.prompts = prompts
    
    async def __call__(self, state: AgentState) -> AgentState:
        """Execute knowledge query tasks"""
        try:
            current_step = state["current_step"]
            plan = state["plan"]
            user_request = state["user_request"]
            
            if current_step >= len(plan):
                return state
            
            current_action = plan[current_step]
            
            if "search" in current_action.lower() and "knowledge" in current_action.lower():
                # Search existing knowledge
                with trace.get_tracer(__name__).start_as_current_span(
                    "search_knowledge",
                    attributes={
                        SpanAttributes.OPENINFERENCE_SPAN_KIND: "TOOL",
                        SpanAttributes.TOOL_NAME: "search_knowledge"
                    }
                ) as tool_span:
                    tool_span.set_attribute(SpanAttributes.INPUT_VALUE, user_request)
                    knowledge_results = self.knowledge_graph.search_knowledge(user_request, limit=10)
                    tool_span.set_attribute(SpanAttributes.OUTPUT_VALUE, json.dumps({"knowledge_results": knowledge_results}))
                
                if not state["knowledge_data"]:
                    state["knowledge_data"] = {}
                state["knowledge_data"]["search_results"] = knowledge_results
                
                state["messages"].append(AIMessage(content=f"Found {len(knowledge_results)} relevant knowledge items"))
                
            elif "retrieve" in current_action.lower():
                # Retrieve specific information
                topic = user_request  # Could be enhanced with topic extraction
                knowledge_summary = self.knowledge_graph.get_knowledge_summary(topic)
                
                if not state["knowledge_data"]:
                    state["knowledge_data"] = {}
                state["knowledge_data"]["summary"] = knowledge_summary
                
                state["messages"].append(AIMessage(content="Retrieved comprehensive knowledge summary"))
                
            elif "formulate" in current_action.lower():
                # Formulate response based on knowledge
                knowledge_data = state["knowledge_data"]
                response = await self._formulate_knowledge_response(knowledge_data, user_request)
                state["knowledge_data"]["response"] = response
                state["messages"].append(AIMessage(content="Formulated response from knowledge base"))
            
            state["current_step"] += 1
            
        except Exception as e:
            logger.error(f"Error in knowledge query: {str(e)}")
            state["messages"].append(AIMessage(content=f"Error in knowledge query: {str(e)}"))
            state["current_step"] += 1
        
        return state
    
    async def _formulate_knowledge_response(self, knowledge_data: Dict, user_request: str) -> str:
        """Formulate response based on knowledge data"""
        try:
            template_variables = {
                "user_request": user_request,
                "knowledge_data": json.dumps(knowledge_data, indent=2)
            }
            template_messages = self.prompts.knowledge_response_prompt.messages
            template_string = f"{template_messages[0].prompt.template}\n\nUser: {template_messages[1].prompt.template}"
            
            with using_prompt_template(
                template=template_string,
                variables=template_variables,
                version="v1.0"
            ):
                response = self.llm.invoke(
                    self.prompts.knowledge_response_prompt.format_messages(
                        user_request=user_request, 
                        knowledge_data=json.dumps(knowledge_data, indent=2)
                    )
                )
            return response.content
            
        except Exception as e:
            logger.error(f"Error formulating knowledge response: {str(e)}")
            return f"Error formulating response: {str(e)}"

class ResponseGenerationNode:
    """Node for generating final responses"""
    
    def __init__(self, llm: ChatOpenAI, prompts: Prompts):
        self.llm = llm
        self.prompts = prompts
    
    def __call__(self, state: AgentState) -> AgentState:
        """Generate final response"""
        try:
            user_request = state["user_request"]
            research_data = state.get("research_data", {})
            research_data_str = json.dumps(research_data, indent=2) if research_data else "No research data available"
            
            template_variables = {
                "user_request": user_request,
                "research_data": research_data_str
            }
            template_messages = self.prompts.response_generation_prompt.messages
            template_string = f"{template_messages[0].prompt.template}\n\nUser: {template_messages[1].prompt.template}"
            
            with using_prompt_template(
                template=template_string,
                variables=template_variables,
                version="v1.0"
            ):
                response = self.llm.invoke(
                    self.prompts.response_generation_prompt.format_messages(
                        user_request=user_request,
                        research_data=research_data_str
                    )
                )
            
            state["final_response"] = response.content
            state["messages"].append(AIMessage(content=response.content))
            
        except Exception as e:
            logger.error(f"Error in response generation: {str(e)}")
            state["final_response"] = "I apologize, but I encountered an error while generating the response."
            state["messages"].append(AIMessage(content=state["final_response"]))
        
        return state

class LangGraphResearchAgent:
    """Main LangGraph-based research agent"""
    
    def __init__(self, tracer=None):
        self.llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            temperature=float(os.getenv("OPENAI_TEMPERATURE", 0.1))
        )
        
        # Create shared tracer instance - use provided or get global tracer
        self.tracer = tracer if tracer else trace.get_tracer(PROJECT_NAME)
        self.knowledge_graph = KnowledgeGraphManager()
        self.research_agent = SimpleResearchAgent()
        
        # Create prompts instance
        self.prompts = Prompts()
        
        # Initialize nodes with shared knowledge graph and prompts
        self.intent_node = IntentDetectionNode(self.llm, self.prompts)
        self.planning_node = PlanningNode(self.llm, self.prompts)
        self.research_node = ResearchExecutionNode(self.llm, self.knowledge_graph, self.research_agent, self.prompts)
        self.knowledge_node = KnowledgeQueryNode(self.llm, self.knowledge_graph, self.prompts)
        self.response_node = ResponseGenerationNode(self.llm, self.prompts)
        
        # Build graph
        self.graph = self._build_graph()
        
        # Memory for conversation state
        self.memory = MemorySaver()
        
        # Compile graph with memory
        self.compiled_graph = self.graph.compile(checkpointer=self.memory)
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("intent_detection", self.intent_node)
        workflow.add_node("planning", self.planning_node)
        workflow.add_node("research_execution", self._async_research_wrapper)
        workflow.add_node("knowledge_query", self._async_knowledge_wrapper)
        workflow.add_node("response_generation", self.response_node)
        
        # Add edges
        workflow.add_edge(START, "intent_detection")
        workflow.add_edge("intent_detection", "planning")
        
        # Conditional edge to route based on intent
        workflow.add_conditional_edges(
            "planning",
            self._route_by_intent,
            {
                "research": "research_execution",
                "knowledge_query": "knowledge_query",
                "general": "response_generation"
            }
        )
        
        # Conditional edge for research execution
        workflow.add_conditional_edges(
            "research_execution",
            self._should_continue_execution,
            {
                "continue": "research_execution",
                "finish": "response_generation"
            }
        )
        
        # Conditional edge for knowledge query
        workflow.add_conditional_edges(
            "knowledge_query",
            self._should_continue_execution,
            {
                "continue": "knowledge_query",
                "finish": "response_generation"
            }
        )
        
        workflow.add_edge("response_generation", END)
        
        return workflow
    
    def _async_research_wrapper(self, state: AgentState) -> AgentState:
        """Wrapper to handle async research execution"""
        # Preserve the current OpenTelemetry context
        current_context = context.get_current()
        ctx = contextvars.copy_context()
        
        def run_with_context():
            # Attach the OpenTelemetry context to the new event loop
            token = context.attach(current_context)
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(self.research_node(state))
                finally:
                    loop.close()
            finally:
                context.detach(token)
        
        # Run in the preserved context
        return ctx.run(run_with_context)
    
    def _async_knowledge_wrapper(self, state: AgentState) -> AgentState:
        """Wrapper to handle async knowledge query execution"""
        # Preserve the current OpenTelemetry context
        current_context = context.get_current()
        ctx = contextvars.copy_context()
        
        def run_with_context():
            # Attach the OpenTelemetry context to the new event loop
            token = context.attach(current_context)
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(self.knowledge_node(state))
                finally:
                    loop.close()
            finally:
                context.detach(token)
        
        # Run in the preserved context
        return ctx.run(run_with_context)
    
    def _route_by_intent(self, state: AgentState) -> str:
        """Route to appropriate execution node based on intent"""
        intent = state.get("intent", "general")
        
        if intent == "research" or intent == "analysis":
            return "research"
        elif intent == "knowledge_query":
            return "knowledge_query"
        else:
            return "general"
    
    def _should_continue_execution(self, state: AgentState) -> str:
        """Determine if execution should continue"""
        current_step = state["current_step"]
        plan = state["plan"]
        
        if current_step >= len(plan):
            return "finish"
        
        return "continue"
    
    async def process_request(self, user_request: str, session_id: str) -> Dict[str, Any]:
        """Process a user request through the LangGraph workflow"""
        try:
            # Initialize state
            initial_state = AgentState(
                messages=[HumanMessage(content=user_request)],
                intent=None,
                plan=None,
                current_step=0,
                research_data=None,
                knowledge_data=None,
                final_response=None,
                user_request=user_request,
                session_id=session_id
            )
            
            # Configure run
            config = {"configurable": {"thread_id": session_id}}
            
            # Execute workflow
            final_state = await self.compiled_graph.ainvoke(initial_state, config=config)
            
            return {
                "response": final_state.get("final_response", "No response generated"),
                "intent": final_state.get("intent"),
                "plan": final_state.get("plan"),
                "research_data": final_state.get("research_data"),
                "messages": [msg.content for msg in final_state.get("messages", [])]
            }
            
        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            return {
                "response": "I apologize, but I encountered an error while processing your request.",
                "error": str(e)
            }