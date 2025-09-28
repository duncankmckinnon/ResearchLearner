from typing import Dict, List
from langchain_core.prompts import ChatPromptTemplate

class Prompts:
    @property
    def intent_detection_prompt(self) -> ChatPromptTemplate:
        """Prompt for detecting user intent and suggesting appropriate tools"""
        return ChatPromptTemplate.from_messages([
            ("system", """Analyze the user's request and determine the primary intent. Choose from these categories:

            1. Research new topics, find papers, discover academic insights
            - Example: "What are the latest research on transformer architectures?"
            - intent: "research"
            - tools: search_knowledge, get_related_papers, add_research_paper, add_research_insight
            - instructions: Search knowledge graph, find papers related to [topic], generate insights from papers, store papers and insights in knowledge graph

            2. Analyze specific papers or research findings in detail
            - Example: "What are the key findings of the paper 'Attention is all you need'?"
            - intent: "analysis"
            - tools: search_knowledge, get_related_papers, add_research_paper, add_research_insight
            - instructions: Search knowledge graph for [topic], generate insights from findings, store insights in knowledge graph

            3. Query existing knowledge and stored insights
            - Example: "What have we learned about neural networks so far?"
            - intent: "knowledge_query"
            - tools: search_knowledge, get_research_insights, get_knowledge_summary
            - instructions: Search knowledge graph for [topic], collect prior insights and papers, summarize findings

            4. General conversation or questions answerable with knowledge graph
            - Example: "Give me an interesting fact you've learned"
            - intent: "general"
            - tools: search_knowledge
            - instructions: Search knowledge graph for relevant topics, provide helpful response
 
            Replace [topic] with the actual topic from the user's request in the instructions.

            Respond with ONLY a JSON object in this exact format - no other text:
            {{"intent": "...", "suggested_tools": ["...", "..."], "instructions": "..."}}
            """),
            ("human", "User request: {user_request}\n\nContext: {context}")
        ])


    @property
    def agent_execution_prompt(self) -> ChatPromptTemplate:
        """Prompt for the agent node to decide which tools to use"""
        return ChatPromptTemplate.from_messages([
            ("system", """You are an agent with access to knowledge graph tools for research tasks. 
            Your goal is to collect the information needed to fulfill the user's request by following the instructions and only using the tools specified below.

            CRITICAL: You MUST follow the instructions and use only the suggested tools to fulfill this request. Always start by calling tools.

            USER REQUEST: {user_request}
            INTENT: {intent}
            INSTRUCTIONS: {instructions}
            TOOLS: {suggested_tools}
            
            STORAGE REQUIREMENTS:
            - If new papers are retrieved, store ALL new papers using add_research_paper(paper_data={{"title": "...", "authors": [...], "arxiv_id": "...", "categories": [...], "content": "..."}})
            - If new papers are retrieved, always generate insights from them and store them using add_research_insight (3 insights minimum)
            - Base insights on the collection of papers AND your prior knowledge from the knowledge graph
            - Call tools in parallel when possible (e.g., multiple add_research_paper calls together, multiple add_research_insight calls together)
            """),
            ("placeholder", "{messages}")
        ])

    @property
    def response_generation_prompt(self) -> ChatPromptTemplate:
        """Prompt for generating final responses"""
        return ChatPromptTemplate.from_messages([
            ("system", """You are a helpful research assistant.
            Generate a comprehensive response based on the research data and user request.
            
            If research data is available, include:
            - A general summary of the research and response to the user's request
            - Key findings from the research
            - Relevant paper citations
            - Actionable insights
            
            If no research data is available, provide a helpful general response.
            
            Be conversational but informative.
            """),
            ("human", "User request: {user_request}\n\nResearch data: {research_data}")
        ])

