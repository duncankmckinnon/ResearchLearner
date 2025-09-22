from typing import Dict, List
from langchain_core.prompts import ChatPromptTemplate

class Prompts:
    @property
    def intent_detection_prompt(self) -> ChatPromptTemplate:
        """Prompt for detecting user intent and suggesting appropriate tools"""
        return ChatPromptTemplate.from_messages([
            ("system", """Analyze the user's request and determine the primary intent. Choose from these categories:

            1. "research" - Research new topics, find papers, discover academic insights
               - Tools: search_knowledge, get_related_papers, add_research_paper, add_research_insight
               - Instructions: Search knowledge graph, find papers related to [topic], generate insights from papers, store papers and insights in knowledge graph

            2. "analysis" - Analyze specific papers or research findings in detail
               - Tools: search_knowledge, get_related_papers, add_research_paper, add_research_insight
               - Instructions: Search knowledge graph for [topic], generate insights from findings, store insights in knowledge graph

            3. "knowledge_query" - Query existing knowledge and stored insights
               - Tools: search_knowledge, get_research_insights, get_knowledge_summary
               - Instructions: Search knowledge graph for [topic], collect prior insights and papers, summarize findings

            4. "general" - General conversation or questions answerable with knowledge graph
               - Tools: search_knowledge
               - Instructions: Search knowledge graph for relevant topics, provide helpful response

            Replace [topic] with the actual topic from the user's request in the instructions.

            Respond with ONLY a JSON object containing: intent, suggested_tools, instructions.
            """),
            ("human", "User request: {user_request}\n\nContext: {context}")
        ])


    @property
    def agent_execution_prompt(self) -> ChatPromptTemplate:
        """Prompt for the agent node to decide which tools to use"""
        return ChatPromptTemplate.from_messages([
            ("system", """You are a research assistant with access to knowledge graph tools.

            CRITICAL: You MUST use the available tools to fulfill this request. Always start by calling tools.

            INSTRUCTIONS: {instructions}
            AVAILABLE TOOLS: {available_tools}
            USER REQUEST: {user_request}
            INTENT: {intent}

            STORAGE REQUIREMENTS:
            - Store ALL papers using add_research_paper(paper_data={{"title": "...", "authors": [...], "arxiv_id": "...", "categories": [...], "content": "..."}})
            - Generate MULTIPLE insights using add_research_insight (3-5 insights minimum)
            - Base insights on the collection of papers AND your prior knowledge from search results
            - CALL MULTIPLE TOOLS IN PARALLEL when possible (e.g., multiple add_research_paper calls together, multiple add_research_insight calls together)

            CORRECT TOOL CALL FORMAT:
            add_research_paper(paper_data=complete_paper_dict)
            add_research_insight(insight="...", topic="...", context={{...}})

            For "research" intent: Use search_knowledge first, then get_related_papers, then call multiple add_research_paper tools in parallel for all papers, then call multiple add_research_insight tools in parallel
            For "analysis" intent: Use search_knowledge and get_related_papers, then store papers and generate multiple insights
            For "knowledge_query" intent: Use search_knowledge and get_research_insights
            For "general" intent: Use search_knowledge to check existing knowledge

            Start by calling the first relevant tool from the available tools list.
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

