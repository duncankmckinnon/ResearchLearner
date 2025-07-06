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