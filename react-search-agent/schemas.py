from typing import List

from pydantic import BaseModel, Field


class Source(BaseModel):
    """Schema for a source used by agent"""

    url: str = Field(description="The URL to the source")


class AgentResponse(BaseModel):
    """Schema for a response from an agent with answers and sources"""

    answer: str = Field(description="The answer to the question")
    sources: List[Source] = Field(
        default_factory=list, description="The source to answer"
    )
