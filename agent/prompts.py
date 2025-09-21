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

            Your primary responsibility is to store valuable information as you find it using add_research_paper and add_research_insight.

            INSTRUCTIONS: {instructions}
            AVAILABLE TOOLS: {available_tools}
            USER REQUEST: {user_request}
            INTENT: {intent}

            Key requirements:
            - Store papers using add_research_paper with complete paper_data dict
            - Generate multiple insights using add_research_insight throughout your research
            - Use tools systematically to gather comprehensive information

            Decide which tools to call based on the request and continue until you have sufficient information.
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

