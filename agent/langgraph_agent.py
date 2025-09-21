from typing import Dict, List, Any, Optional, TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from agent.arxiv_client import SimpleResearchAgent
from agent.knowledge_tools import get_knowledge_tools, get_knowledge_tool
from agent.prompts import Prompts
import asyncio
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
    context: str
    tool_instructions: Optional[str]  # Instructions for what tools to use and when
    available_tools: Optional[List[str]]  # List of available tool names
    tool_results: Optional[List[Dict[str, Any]]]  # Results from tool execution
    tools_used: Optional[List[str]]  # Track which tools have been called
    tool_call_count: int  # Track number of tool call iterations

class LangGraphResearchAgent:
    """Main LangGraph-based research agent"""
    
    def __init__(self):
        # Get knowledge tools
        self.knowledge_tools = get_knowledge_tools()
        
        # Create base LLM without tools (for intent detection)
        self.base_llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1"),
            temperature=float(os.getenv("OPENAI_TEMPERATURE", 0.1))
        )
        # Create LLM with tools bound (for tool execution)
        self.llm = self.base_llm.bind_tools(self.knowledge_tools)
        
        self.research_agent = SimpleResearchAgent()
        
        # Knowledge graph access will be through tools only
        
        # Create prompts instance
        self.prompts = Prompts()
        
        # Note: Using simplified three-node architecture instead of separate node classes
        
        # Create tool node for knowledge tools
        self.tool_node = ToolNode(self.knowledge_tools)
        
        # Build graph
        self.graph = self._build_graph()
        
        # Memory for conversation state
        self.memory = MemorySaver()
        
        # Compile graph with memory
        self.compiled_graph = self.graph.compile(checkpointer=self.memory)
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes using standard LangGraph pattern
        workflow.add_node("intent_and_setup", self._intent_and_setup_node)
        workflow.add_node("agent", self._agent_node)  # LLM with tools bound
        workflow.add_node("tools", self.tool_node)   # Standard ToolNode
        workflow.add_node("response_compilation", self._response_compilation_node)
        
        # Standard LangGraph flow
        workflow.add_edge(START, "intent_and_setup")
        workflow.add_edge("intent_and_setup", "agent")
        
        # Conditional routing based on tool calls
        workflow.add_conditional_edges(
            "agent",
            self._should_continue_to_tools,
            {
                "tools": "tools",
                "response": "response_compilation"
            }
        )
        
        # After tools, go back to agent for potential additional tool calls
        workflow.add_edge("tools", "agent")
        workflow.add_edge("response_compilation", END)
        
        return workflow
    
    def _intent_and_setup_node(self, state: AgentState) -> AgentState:
        """First node: determine the intent and setup the appropriate tools to call for that intent"""
        try:
            user_request = state["user_request"]
            context = state["context"]
            
            # Use LLM to detect intent with detailed prompting
            prompt = self.prompts.intent_detection_prompt.format_messages(
                user_request=user_request,
                context=context
            )
            if state["messages"] is []:
                state["messages"] = prompt
            else:
                state["messages"].extend(prompt)
            logger.info(f"Intent detection prompt: {prompt}")
            response = self.base_llm.invoke(
                prompt
            )
            
            # Parse the JSON response from the LLM
            try:
                import json
                intent_data = json.loads(response.content.strip())

                intent = intent_data.get("intent", "general")
                suggested_tools = intent_data.get("suggested_tools", ["search_knowledge"])
                instructions = intent_data.get("instructions", "Process the request using available tools.")

                # Validate intent
                valid_intents = ["research", "analysis", "knowledge_query", "general"]
                if intent not in valid_intents:
                    logger.warning(f"Invalid intent '{intent}' received, defaulting to 'general'")
                    intent = "general"
                    suggested_tools = ["search_knowledge"]
                    instructions = "Process the request using basic knowledge search."

            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to parse LLM intent response: {e}. Using fallback detection.")
                # Fallback to basic keyword detection
                raise e
            
            # Set up state with LLM-determined intent and tool configuration
            state["intent"] = intent
            state["available_tools"] = suggested_tools
            state["tool_instructions"] = instructions
            
            # Add system message with context
            system_content = f"""
            {instructions}

            Available tools: {', '.join(suggested_tools)}

            User request: {user_request}
            Context: {context}

            You should decide which tools to call based on the request. Call tools when you need to search, retrieve, or store information in the knowledge graph.
            """
            
            state["messages"].append(AIMessage(content=f"Intent detected: {intent}. Setting up tools: {suggested_tools}"))
            logger.info(f"Intent detected: {intent}, tools configured: {suggested_tools}")
            
            return state
            
        except Exception as e:
            logger.error(f"Error in intent and setup: {str(e)}")
            state["intent"] = "general"
            state["available_tools"] = ["search_knowledge"]
            state["tool_instructions"] = "Error occurred during setup. Using basic knowledge search."
            state["messages"].append(AIMessage(content=f"Error in setup: {str(e)}"))
            return state
    
    def _agent_node(self, state: AgentState) -> AgentState:
        """Standard LangGraph agent node - LLM with tools that can make tool calls"""
        try:
            user_request = state["user_request"]
            intent = state["intent"]
            available_tools = state.get("available_tools", [])
            tool_instructions = state.get("tool_instructions", "")
            tools_used = state.get("tools_used") or []
            tool_call_count = state.get("tool_call_count", 0)
            
            # Filter LLM to only use tools configured for this intent
            configured_tools = [tool for tool in self.knowledge_tools if tool.name in available_tools]
            focused_llm = self.llm.bind_tools(configured_tools) if configured_tools else self.llm
            
            # Use prompt from prompts class
            prompt_messages = self.prompts.agent_execution_prompt.format_messages(
                instructions=tool_instructions,
                available_tools=', '.join(available_tools),
                user_request=user_request,
                intent=intent,
                messages=state["messages"]
            )
            
            # Call LLM - it will decide which tools to call
            response = focused_llm.invoke(prompt_messages)
            
            # Track tool usage in state instead of logging
            if hasattr(response, 'tool_calls') and response.tool_calls:
                # LangChain tool calls have structure: {"name": "...", "args": {...}, "id": "...", "type": "tool_call"}
                new_tool_calls = [tc["name"] for tc in response.tool_calls]
                
                # Update state with tool usage tracking
                if tools_used is None:
                    tools_used = []
                tools_used.extend(new_tool_calls)
                state["tools_used"] = tools_used
                state["tool_call_count"] = tool_call_count + 1
                
                # Add informational message about tools being called
                state["messages"].append(AIMessage(
                    content=f"Calling tools: {', '.join(new_tool_calls)} (iteration {tool_call_count + 1})"
                ))
            else:
                # No tool calls - agent is done gathering information
                state["messages"].append(AIMessage(
                    content=f"Information gathering complete. Used tools: {', '.join(tools_used) if tools_used else 'none'}"
                ))
            
            # Add the actual LLM response to messages
            state["messages"].append(response)
            
            return state
            
        except Exception as e:
            logger.error(f"Error in agent node: {str(e)}")
            state["messages"].append(AIMessage(content=f"Error in agent processing: {str(e)}"))
            return state
    
    def _should_continue_to_tools(self, state: AgentState) -> str:
        """Determine if we should continue to tools or move to response compilation"""
        messages = state["messages"]
        if not messages:
            return "response"
        
        last_message = messages[-1]
        
        # Check if the last message has tool calls
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"
        
        return "response"
    
    def _response_compilation_node(self, state: AgentState) -> AgentState:
        """Third node: compile the response from the collected context"""
        try:
            user_request = state["user_request"]
            intent = state["intent"]
            tools_used = state.get("tools_used") or []
            tool_call_count = state.get("tool_call_count", 0)
            messages = state["messages"]
            
            # Extract tool results from ToolMessage objects in the message history
            tool_results = []
            for msg in messages:
                if isinstance(msg, ToolMessage):
                    tool_results.append({
                        "tool": getattr(msg, 'name', 'unknown_tool'),
                        "result": msg.content
                    })
            
            # Prepare research data for response generation
            research_data = {
                "intent": intent,
                "tools_used": ', '.join(tools_used) if tools_used else 'none',
                "tool_call_count": tool_call_count,
                "tool_results": tool_results
            }

            # Use the response generation prompt from prompts class
            prompt_messages = self.prompts.response_generation_prompt.format_messages(
                user_request=user_request,
                research_data=str(research_data)
            )

            # Generate final response using the structured information
            response = self.base_llm.invoke(prompt_messages)
            
            state["final_response"] = response.content
            state["messages"].append(response)
            
            logger.info(f"Response compiled for intent: {intent} using {len(tools_used)} tools across {tool_call_count} iterations")
            
            return state
            
        except Exception as e:
            logger.error(f"Error in response compilation: {str(e)}")
            error_response = "I apologize, but I encountered an error while compiling the response. Please try again."
            state["final_response"] = error_response
            state["messages"].append(AIMessage(content=error_response))
            return state
    
    async def process_request(self, user_request: str, session_id: str, context: str) -> Dict[str, Any]:
        """Process a user request through the LangGraph workflow"""
        try:
            # Initialize state
            initial_state = AgentState(
                messages=[],
                intent=None,
                plan=None,
                current_step=0,
                research_data=None,
                knowledge_data=None,
                final_response=None,
                user_request=user_request,
                session_id=session_id,
                context=context,
                tool_results=None,
                tools_used=None,
                tool_call_count=0
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