from typing import Dict, List
from langchain_core.prompts import ChatPromptTemplate

# Decorator to make class methods act like properties
class classproperty:
    def __init__(self, func):
        self.func = func
        
    def __get__(self, instance, owner):
        return self.func(owner)

class Prompts:
    @classproperty
    def intent_detection_prompt(cls) -> ChatPromptTemplate:
        """Prompt for detecting user intent and suggesting appropriate tools"""
        return ChatPromptTemplate.from_messages([
            ("system", """Analyze the user's request and determine the primary intent. Choose from these categories:

            1. Research new topics, find papers, discover academic insights
            - Example: "What are the latest research on transformer architectures?"
            - intent: "research"
            - tools: search_knowledge, get_related_papers, find_stored_paper, download_and_process_paper, add_research_insight
            - instructions: Search knowledge graph, find papers related to [topic], download and process full PDF content for key papers, generate insights from papers, store papers and insights in knowledge graph

            2. Analyze a specific paper or research finding in detail
            - Example: "What are the key findings of the paper 'Attention is all you need'?"
            - intent: "analysis"
            - tools: find_stored_paper, search_knowledge, get_related_papers, download_and_process_paper, add_research_insight
            - instructions: 
            -- If a paper title is provided, find the paper in the knowledge graph. If the paper is found, use it for analysis. If the paper is NOT found, download and process the full PDF content for detailed analysis and generate insights from the paper.
            -- If a paper title is NOT provided, search the knowledge graph for relevant insights and papers to use in the analysis.
            -- If the questions is about a broader topic, search the knowledge graph for relevant insights and papers to use in the analysis.

            3. Query existing knowledge and stored insights
            - Example: "What have we learned about neural networks so far?"
            - intent: "knowledge_query"
            - tools: search_knowledge, find_stored_paper, get_research_insights, get_knowledge_summary
            - instructions: Search knowledge graph for [topic] or [paper title], collect prior insights and papers, summarize findings

            4. General conversation or questions answerable with knowledge graph
            - Example: "Give me an interesting fact you've learned"
            - intent: "general"
            - tools: search_knowledge
            - instructions: Search knowledge graph for relevant topics, provide helpful response
 
            Replace [topic] with the actual topic from the user's request in the instructions.
            Replace [paper title] with the actual paper title from the user's request in the instructions if applicable.

            Respond with ONLY a JSON object in this exact format - no other text:
            {{"intent": "...", "suggested_tools": ["...", "..."], "instructions": "..."}}
            """),
            ("human", "User request: {user_request}\n\nContext: {context}")
        ])


    @classproperty
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
            - Use find_stored_paper FIRST to check if papers are already stored with full content before downloading from arxiv
            - If papers are already stored, use the existing full content for analysis
            - Only use download_and_process_paper for papers NOT found in storage
            - Always generate insights from papers and store them using add_research_insight (3 insights minimum)
            - Base insights on the collection of related papers AND your prior knowledge from the knowledge graph
            - Call tools in parallel when possible (e.g., multiple find_stored_paper calls together, multiple download_and_process_paper calls together, multiple add_research_insight calls together)
            """),
            ("placeholder", "{messages}")
        ])

    @classproperty
    def response_generation_prompt(cls) -> ChatPromptTemplate:
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

