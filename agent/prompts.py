from typing import Dict, List
from langchain_core.prompts import ChatPromptTemplate

class Prompts:
    base_prompt = """
            # Role
            You are a helpful assistant that can answer questions and help with tasks.
        """

    def format_prompt(self, request: str, context: str = None) -> List[Dict]:
        return [
            {
                "role": "system",
                "content": f"{self.base_prompt}"
            },
            {
                "role": "user",
                "content": f"Customer message: {request}" + (f"\n\nContext: {context}" if context else "")
            }
        ]

    @property
    def intent_detection_prompt(self) -> ChatPromptTemplate:
        """Prompt for detecting user intent"""
        return ChatPromptTemplate.from_messages([
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
            ("human", "User request: {user_request}\n\nContext: {context}")
        ])

    @property
    def planning_prompt(self) -> ChatPromptTemplate:
        """Prompt for creating execution plans"""
        return ChatPromptTemplate.from_messages([
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
            ("human", "Intent: {intent}\n\nUser request: {user_request}\n\nContext: {context}")
        ])

    @property
    def topic_extraction_prompt(self) -> ChatPromptTemplate:
        """Prompt for extracting research topics"""
        return ChatPromptTemplate.from_messages([
            ("system", """Extract the main research topic or keywords from the user's request.
            
            Focus on the core subject matter they want to research. 
            If the subject is clear, include keywords that are related to the subject but not necessarily in the request.   
            Respond with just the topic/keywords, no additional text.
            
            Examples:
            - "Find papers about transformer architectures" -> "transformer architectures", "attention mechanism"
            - "Research quantum computing applications" -> "quantum computing applications"
            - "I want to learn about neural networks" -> "neural networks", "deep learning"
            """),
            ("human", "User request: {user_request}")
        ])

    @property
    def knowledge_response_prompt(self) -> ChatPromptTemplate:
        """Prompt for formulating knowledge-based responses"""
        return ChatPromptTemplate.from_messages([
            ("system", """Based on the following knowledge from the knowledge base, formulate a comprehensive response to the user's request.
            Provide a helpful and informative response that addresses the user's question using the available knowledge.
            """),
            ("human", "User request: {user_request}\n\nKnowledge data: {knowledge_data}")
        ])

    @property
    def response_generation_prompt(self) -> ChatPromptTemplate:
        """Prompt for generating final responses"""
        return ChatPromptTemplate.from_messages([
            ("system", """You are a helpful research assistant. Generate a comprehensive response based on the research data and user request.
            
            If research data is available, include:
            - Key findings from the research
            - Relevant paper citations
            - Actionable insights
            - Suggestions for further research
            
            If no research data is available, provide a helpful general response.
            
            Be conversational but informative.
            """),
            ("human", "User request: {user_request}\n\nResearch data: {research_data}")
        ])

