"""
LLM-powered qualitative analysis of SEC 10-K filings.
"""

from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


class Finding(BaseModel):
    """A qualitative finding supported by evidence from a 10-K."""

    title: str = Field(
        ...,
        description="Short title of the finding",
    )
    explanation: str = Field(
        ...,
        description="Explanation based only on the 10-K filing",
    )
    evidence: str = Field(
        ...,
        description="Short passage or specific fact supporting the finding",
    )


class TenKAnalysis(BaseModel):
    """Structured qualitative analysis extracted from a 10-K."""

    business_summary: str
    major_risks: list[Finding]
    business_trends: list[Finding]
    management_outlook: str
    red_flags: list[Finding]