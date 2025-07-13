from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

class RequestFormat(BaseModel):
    conversation_hash: str = Field(description="The conversation hash associated with the request")
    request_timestamp: Optional[str] = Field(default=datetime.now().isoformat(), description="The timestamp of the request")
    customer_message: str = Field(description="The message of the request")


class ResponseFormat(BaseModel):
    response: str = Field(description="The response to the request")
    intent: Optional[str] = Field(default=None, description="The detected intent of the request")
    plan: Optional[List[str]] = Field(default=None, description="The execution plan created for the request")
    research_data: Optional[Dict[str, Any]] = Field(default=None, description="Research data if applicable")


class KnowledgeSearchRequest(BaseModel):
    query: str = Field(description="Search query for the knowledge store")
    limit: Optional[int] = Field(default=10, description="Maximum number of results to return")


class KnowledgeSearchResult(BaseModel):
    content: str = Field(description="The content of the knowledge item")
    metadata: Dict[str, Any] = Field(description="Metadata associated with the knowledge item")
    relevance_score: float = Field(description="Relevance score of the result")
    id: str = Field(description="Unique identifier of the knowledge item")


class ResearchPaper(BaseModel):
    title: str = Field(description="Title of the research paper")
    authors: List[str] = Field(description="List of authors")
    arxiv_id: str = Field(description="ArXiv ID of the paper")
    categories: List[str] = Field(description="Research categories")
    relevance_score: float = Field(description="Relevance score")
    content: str = Field(description="Paper content/abstract")
    source: str = Field(description="Source of the paper (knowledge_graph or arxiv_search)")


class ResearchInsight(BaseModel):
    insight: str = Field(description="The research insight content")
    topic: str = Field(description="Topic of the insight")
    context: Dict[str, Any] = Field(description="Context information")
    relevance_score: float = Field(description="Relevance score")
    added_date: str = Field(description="Date when the insight was added")


class KnowledgeSummary(BaseModel):
    topic: str = Field(description="Topic of the knowledge summary")
    related_papers: List[ResearchPaper] = Field(description="Related research papers")
    research_insights: List[ResearchInsight] = Field(description="Research insights")
    general_knowledge: List[KnowledgeSearchResult] = Field(description="General knowledge items")
    total_papers: int = Field(description="Total number of related papers")
    total_insights: int = Field(description="Total number of insights")
    total_knowledge_items: int = Field(description="Total number of general knowledge items")