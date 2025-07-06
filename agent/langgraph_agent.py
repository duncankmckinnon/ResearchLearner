from typing import Dict, List, Any, Optional, TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from opentelemetry import trace
from openinference.semconv.trace import SpanAttributes
from agent.arxiv_client import SimpleResearchAgent
from agent.knowledge_graph import KnowledgeGraphManager
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
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at analyzing user requests and determining their intent.
            
            Analyze the user's request and determine the primary intent. Choose from these categories:
            
            1. "research" - User wants to research a topic, find papers, get academic insights
            2. "analysis" - User wants to analyze specific papers or research findings
            3. "knowledge_query" - User wants to query existing knowledge base
            4. "general" - General conversation, questions not related to research
            
            Respond with ONLY the intent category as a single word.
            
            Examples:
            - "Find papers about transformer architectures" -> research
            - "Analyze the paper arxiv:2023.12345" -> analysis
            - "What did I learn about neural networks?" -> knowledge_query
            - "Hello, how are you?" -> general
            """),
            ("human", "{user_request}")
        ])
    
    def __call__(self, state: AgentState) -> AgentState:
        """Detect intent from user request"""
        try:
            user_request = state["user_request"]
            
            response = self.llm.invoke(
                self.prompt.format_messages(user_request=user_request)
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
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at creating execution plans for different types of requests.
            
            Based on the user's request and detected intent, create a step-by-step plan.
            
            For "research" intent, typical steps might include:
            1. Extract research topic/keywords
            2. Search for relevant papers
            3. Download and analyze top papers
            4. Synthesize findings
            5. Present results
            
            For "analysis" intent, typical steps might include:
            1. Identify specific papers/documents
            2. Download or retrieve content
            3. Analyze content thoroughly
            4. Extract key insights
            5. Present analysis
            
            For "knowledge_query" intent, typical steps might include:
            1. Search knowledge base
            2. Retrieve relevant information
            3. Synthesize response
            
            For "general" intent, typical steps might include:
            1. Understand the question
            2. Formulate response
            
            Respond with a JSON list of steps as strings.
            """),
            ("human", "Intent: {intent}\nUser request: {user_request}")
        ])
    
    def __call__(self, state: AgentState) -> AgentState:
        """Create execution plan based on intent"""
        try:
            intent = state["intent"]
            user_request = state["user_request"]
            
            response = self.llm.invoke(
                self.prompt.format_messages(intent=intent, user_request=user_request)
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
    
    def _get_default_plan(self, intent: str) -> List[str]:
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
    
    def __init__(self, llm: ChatOpenAI, knowledge_graph: KnowledgeGraphManager):
        self.llm = llm
        self.research_agent = SimpleResearchAgent()
        self.knowledge_graph = knowledge_graph
        self.topic_extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", """Extract the main research topic or keywords from the user's request.
            
            Focus on the core subject matter they want to research.
            Respond with just the topic/keywords, no additional text.
            
            Examples:
            - "Find papers about transformer architectures" -> "transformer architectures"
            - "Research quantum computing applications" -> "quantum computing applications"
            - "I want to learn about neural networks" -> "neural networks"
            """),
            ("human", "{user_request}")
        ])
    
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
                response = self.llm.invoke(
                    self.topic_extraction_prompt.format_messages(user_request=user_request)
                )
                topic = response.content.strip()
                
                if not state["research_data"]:
                    state["research_data"] = {}
                state["research_data"]["topic"] = topic
                state["messages"].append(AIMessage(content=f"Extracted research topic: {topic}"))
                
            elif "search" in current_action.lower():
                # Search for papers
                topic = state["research_data"].get("topic", user_request)
                research_result = await self.research_agent.research_topic(topic, max_papers=10)
                
                if not state["research_data"]:
                    state["research_data"] = {}
                state["research_data"]["search_results"] = research_result
                
                papers_found = research_result.get("papers_found", 0)
                state["messages"].append(AIMessage(content=f"Found {papers_found} relevant papers"))
                
            elif "analyze" in current_action.lower():
                # Analyze findings
                search_results = state["research_data"].get("search_results", {})
                papers = search_results.get("papers", [])
                
                if papers:
                    # Analyze top papers
                    analyzed_papers = []
                    for paper in papers[:3]:  # Analyze top 3 papers
                        paper_id = paper.get("id", "").replace("http://arxiv.org/abs/", "")
                        if paper_id:
                            analysis = await self.research_agent.analyze_paper(paper_id)
                            if "error" not in analysis:
                                analyzed_papers.append(analysis)
                                # Add paper to knowledge graph
                                self.knowledge_graph.add_research_paper(analysis)
                    
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
    
    def __init__(self, llm: ChatOpenAI, knowledge_graph: KnowledgeGraphManager):
        self.llm = llm
        self.knowledge_graph = knowledge_graph
    
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
                knowledge_results = self.knowledge_graph.search_knowledge(user_request, limit=10)
                
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
            prompt = f"""
            Based on the following knowledge from the knowledge base, formulate a comprehensive response to the user's request.
            
            User Request: {user_request}
            
            Knowledge Data:
            {json.dumps(knowledge_data, indent=2)}
            
            Provide a helpful and informative response that addresses the user's question using the available knowledge.
            """
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content
            
        except Exception as e:
            logger.error(f"Error formulating knowledge response: {str(e)}")
            return f"Error formulating response: {str(e)}"

class ResponseGenerationNode:
    """Node for generating final responses"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful research assistant. Generate a comprehensive response based on the research data and user request.
            
            If research data is available, include:
            - Key findings from the research
            - Relevant paper citations
            - Actionable insights
            - Suggestions for further research
            
            If no research data is available, provide a helpful general response.
            
            Be conversational but informative.
            """),
            ("human", "User request: {user_request}\n\nResearch data: {research_data}\n\nGenerate a comprehensive response.")
        ])
    
    def __call__(self, state: AgentState) -> AgentState:
        """Generate final response"""
        try:
            user_request = state["user_request"]
            research_data = state.get("research_data", {})
            
            response = self.llm.invoke(
                self.prompt.format_messages(
                    user_request=user_request,
                    research_data=json.dumps(research_data, indent=2) if research_data else "No research data available"
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
    
    def __init__(self, trace_provider=None):
        self.llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            temperature=float(os.getenv("OPENAI_TEMPERATURE", 0.1))
        )
        
        # Create shared knowledge graph manager instance
        self.knowledge_graph = KnowledgeGraphManager(trace_provider=trace_provider)
        
        # Initialize nodes with shared knowledge graph
        self.intent_node = IntentDetectionNode(self.llm)
        self.planning_node = PlanningNode(self.llm)
        self.research_node = ResearchExecutionNode(self.llm, self.knowledge_graph)
        self.knowledge_node = KnowledgeQueryNode(self.llm, self.knowledge_graph)
        self.response_node = ResponseGenerationNode(self.llm)
        
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
        ctx = contextvars.copy_context()
        
        def run_with_context():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.research_node(state))
            finally:
                loop.close()
        
        # Run in the preserved context
        return ctx.run(run_with_context)
    
    def _async_knowledge_wrapper(self, state: AgentState) -> AgentState:
        """Wrapper to handle async knowledge query execution"""
        # Preserve the current OpenTelemetry context
        ctx = contextvars.copy_context()
        
        def run_with_context():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.knowledge_node(state))
            finally:
                loop.close()
        
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