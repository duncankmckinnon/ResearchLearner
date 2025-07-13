from typing import Dict, List
from langchain_core.prompts import ChatPromptTemplate

class Prompts:
    @property
    def intent_detection_prompt(self) -> ChatPromptTemplate:
        """Prompt for detecting user intent and suggesting appropriate tools"""
        return ChatPromptTemplate.from_messages([
            ("system", """You are an expert at analyzing user requests and determining their intent for a research assistant system.
            For every request, you must use the tools to derive insights and store the papers and any other useful information in the knowledge graph

            Analyze the user's request and determine the primary intent. Choose from these categories:

            1. "research" - User wants to research a new topic, find papers, discover academic insights
               - Keywords: "find", "research", "papers about", "study", "discover", "explore"
               - Tools needed: search_knowledge, get_related_papers, add_research_paper, add_research_insight
               - Example: "Find papers about transformer architectures"

            2. "analysis" - User wants to analyze specific papers or research findings in detail
               - Keywords: "analyze", "examine", "review", "evaluate", "critique"
               - Tools needed: search_knowledge, get_related_papers, add_research_paper (if papers are found), add_research_insight
               - Example: "Analyze the effectiveness of attention mechanisms"

            3. "knowledge_query" - User wants to query what they already know or have learned
               - Keywords: "what do I know", "what have I learned", "remember", "stored", "previous", "insights", "tell me about"
               - Tools needed: search_knowledge, get_research_insights, get_knowledge_summary, add_research_paper (if papers are found), add_research_insight
               - Example: "What do I know about neural networks?"

            4. "general" - General conversation, questions not directly related to research
               - Keywords: "hello", "how are you", "help", "explain" (without research context)
               - Tools needed: search_knowledge, get_knowledge_summary, add_research_paper (if papers are found), add_research_insight
               - Example: "Hello, how can you help me?"

            Consider the context provided to better understand the request.

            Respond with ONLY a JSON object. No other text.

            Examples:
            {{"intent": "knowledge_query", "confidence": "high", "reasoning": "User asking what they know", "suggested_tools": ["search_knowledge", "get_research_insights", "add_research_insight"], "instructions": "Search, get insights, and store new insights"}}
            {{"intent": "research", "confidence": "high", "reasoning": "User wants to find papers", "suggested_tools": ["search_knowledge", "get_related_papers", "add_research_paper", "add_research_insight"], "instructions": "Search, find papers, and store findings"}}
            {{"intent": "general", "confidence": "medium", "reasoning": "General question", "suggested_tools": ["search_knowledge", "add_research_insight"], "instructions": "Basic search and store insights if applicable"}}
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
            5. Add research paper and insights to knowledge graph
            6. Present results
            
            For "analysis" intent, typical steps might include:
            1. Identify specific papers/documents
            2. Download or retrieve content
            3. Analyze content thoroughly
            4. Extract key insights
            5. Add research paper and insights to knowledge graph
            6. Present results
            
            For "knowledge_query" intent, typical steps might include:
            1. Search knowledge base
            2. Retrieve relevant information and insights
            3. Synthesize response
            4. Add any new research papers or insights to knowledge graph
            5. Present results
            
            For "general" intent, typical steps might include:
            1. Understand the question
            2. Formulate response
            3. Add any new research papers or insights to knowledge graph
            4. Present results
            
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
    def tool_planning_prompt(self) -> ChatPromptTemplate:
        """Prompt for planning specific tool calls based on intent"""
        return ChatPromptTemplate.from_messages([
            ("system", """You are an expert at planning tool execution for a research assistant system.

            Based on the user's request and detected intent, plan the specific tool calls needed to gather all necessary information.

            AVAILABLE TOOLS:
            1. search_knowledge(query, limit=10) - Search existing knowledge
            2. get_related_papers(topic, limit=5) - Find research papers  
            3. get_research_insights(topic, limit=10) - Get stored insights
            4. add_research_paper(paper_data) - Store a paper (paper_data must be a complete dict)
            5. add_research_insight(insight, topic, context) - Store an insight
            6. get_knowledge_summary(topic) - Get comprehensive summary

            TOOL EXECUTION PLANS BY INTENT:

            For "research" intent:
            - search_knowledge(query=main_topic) to check existing knowledge
            - get_related_papers(topic=main_topic) to find relevant papers
            - get_knowledge_summary(topic=main_topic) for comprehensive overview
            - add_research_paper(paper_data) to store the paper (paper_data must be a complete dict)
            - add_research_insight(insight, topic, context) to store the insight

            For "knowledge_query" intent:
            - search_knowledge(query=main_topic) to find relevant information
            - get_research_insights(topic=main_topic) to get stored insights  
            - get_knowledge_summary(topic=main_topic) for complete summary
            - add_research_paper(paper_data) to store the paper (paper_data must be a complete dict)
            - add_research_insight(insight, topic, context) to store the insight

            For "analysis" intent:
            - search_knowledge(query=analysis_target) to find existing analysis
            - get_related_papers(topic=analysis_target) to find papers to analyze
            - get_research_insights(topic=analysis_target) to get previous insights
            - add_research_paper(paper_data) to store the paper (paper_data must be a complete dict)
            - add_research_insight(insight, topic, context) to store the insight

            For "general" intent:
            - search_knowledge(query=user_request) to check if we have relevant knowledge
            - add_research_paper(paper_data) to store the paper (paper_data must be a complete dict)
            - add_research_insight(insight, topic, context) to store the insight

            Respond with ONLY a JSON array. No other text.

            Examples:
            [{{"tool": "search_knowledge", "args": {{"query": "neural networks", "limit": 10}}}}]
            [{{"tool": "search_knowledge", "args": {{"query": "transformers", "limit": 10}}}}, {{"tool": "get_related_papers", "args": {{"topic": "transformers", "limit": 5}}}}]
            [{{"tool": "add_research_paper", "args": {{"paper_data": {{"title": "Paper Title", "authors": ["Author1"], "arxiv_id": "1234.5678", "categories": ["cs.AI"], "content": "Abstract text"}}}}}}]
            [{{"tool": "get_knowledge_summary", "args": {{"topic": "machine learning"}}}}]
            """),
            ("human", "User request: {user_request}\nIntent: {intent}\nContext: {context}")
        ])

    @property
    def agent_execution_prompt(self) -> ChatPromptTemplate:
        """Prompt for the agent node to decide which tools to use"""
        return ChatPromptTemplate.from_messages([
            ("system", """You are a research assistant with access to knowledge graph tools.

            CRITICAL RESPONSIBILITY: You MUST store valuable information in the knowledge graph as you find it. This is essential for building organizational knowledge.

            Based on the user's request and detected intent, use the appropriate tools to gather comprehensive information AND STORE IT.

            INTENT-BASED TOOL STRATEGIES:

            For "research" intent:
            1. First use search_knowledge to check existing knowledge
            2. Then get_related_papers to find relevant papers
            3. **MANDATORY**: Use add_research_paper to store each valuable paper you find
            4. **MANDATORY**: Use add_research_insight to capture key insights from your analysis
            5. Use get_knowledge_summary for comprehensive overview

            For "knowledge_query" intent:
            1. Use search_knowledge to find relevant information
            2. Use get_research_insights to get stored insights
            3. Use get_knowledge_summary for complete summary
            4. **MANDATORY**: Use add_research_insight to store any new insights you generate from synthesizing information

            For "analysis" intent:
            1. Use search_knowledge to find existing analysis
            2. Use get_related_papers to find papers to analyze
            3. Use get_research_insights for previous insights
            4. **MANDATORY**: Use add_research_paper to store papers you analyze
            5. **MANDATORY**: Use add_research_insight to capture your analysis conclusions

            For "general" intent:
            1. Use search_knowledge to check if we have relevant knowledge
            2. **IF APPLICABLE**: Use add_research_insight to store any valuable insights generated

            STORAGE GUIDELINES:
            - ALWAYS store papers you retrieve using add_research_paper (paper_data must be a complete dict)
            - ALWAYS store insights you generate using add_research_insight
            - When calling add_research_paper, wrap the complete paper dictionary in "paper_data" parameter
            - Paper data should include: title, authors, arxiv_id, categories, content/abstract
            - Insights should capture key findings, connections, or conclusions you make
            - This builds organizational knowledge for future use

            CORRECT FORMAT EXAMPLES:
            add_research_paper(paper_data={{"title": "...", "authors": [...], "arxiv_id": "...", "categories": [...], "content": "..."}})
            add_research_insight(insight="Key finding about X", topic="research topic", context={{"source": "paper analysis"}})

            INSTRUCTIONS: {instructions}
            AVAILABLE TOOLS: {available_tools}
            USER REQUEST: {user_request}
            INTENT: {intent}

            Use tools systematically to: 1) gather information, 2) STORE valuable findings, 3) provide a summary.
            """),
            ("placeholder", "{messages}")
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

