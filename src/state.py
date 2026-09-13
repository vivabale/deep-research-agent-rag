"""State management for the Deep Research Agent."""

from typing import Annotated, List, Dict, Optional, Literal   # Literal限制值
from pydantic import BaseModel, Field
from langgraph.graph import MessagesState
from langchain_core.messages import BaseMessage


class SearchQuery(BaseModel):
    """一次搜索任务.
        讓AI知道爲什麽做"""
    query: str = Field(description="The search query text")
    purpose: str = Field(description="Why this query is being made")
    completed: bool = Field(default=False)


class SearchResult(BaseModel):
    """A search result with content."""
    query: str = Field(description="The original query")
    title: str = Field(description="Result title")
    url: str = Field(description="Result URL")
    snippet: str = Field(description="Result snippet/summary")
    content: Optional[str] = Field(default=None, description="Full scraped content if available")
    source_type: str = Field(default="web")
    metadata: Dict = Field(default_factory=dict)


class ReportSection(BaseModel):
    """A section of the research report."""
    title: str = Field(description="Section title")
    content: str = Field(description="Section content in markdown")
    sources: List[str] = Field(default_factory=list, description="Source URLs used")


class ResearchPlan(BaseModel):
    """Research plan with queries and outline."""
    topic: str = Field(description="The research topic")
    objectives: List[str] = Field(description="Research objectives")
    search_queries: List[SearchQuery] = Field(description="Search queries to execute")
    report_outline: List[str] = Field(description="Outline of report sections")


class ResearchState(BaseModel):
    """State for the research workflow."""

    
    # User input
    research_topic: str = Field(description="The topic to research")
    
    # 计划阶段 (Planner 产生)
    #這個階段調用了 ResearchPlan ---> ResearchPlan裏面繼續調用 SearchQuery
    plan: Optional[ResearchPlan] = Field(default=None, description="Research plan")
    
    # 搜索阶段 (Searcher 产生)
    search_results: List[SearchResult] = Field(
        default_factory=list,
        description="All search results collected"
    )
    
    # 综合阶段 (Synthesizer 产生)
    key_findings: List[str] = Field(
        default_factory=list,
        description="Key findings extracted from search results"
    )
    
    # 报告阶段 (Writer 产生)
    report_sections: List[ReportSection] = Field(
        default_factory=list,
        description="Generated report sections"
    )
    
    final_report: Optional[str] = Field(
        default=None,
        description="Complete final report in markdown"
    )
    
    # 流程控制
    current_stage: Literal[
        "planning", "searching", "synthesizing", "reporting", "complete"
    ] = Field(default="planning")
    
    error: Optional[str] = Field(default=None, description="Error message if any")
    
    # 元数据
    iterations: int = Field(default=0, description="Number of iterations")
    
    # Quality and metrics
    quality_score: Optional[Dict] = Field(default=None, description="Report quality metrics")
    credibility_scores: List[Dict] = Field(default_factory=list, description="Source credibility scores")
    rag_used: bool = Field(default=False, description="Whether local RAG was used")
    rag_sources: List[str] = Field(default_factory=list, description="Local RAG sources used")
    rag_stats: Dict = Field(default_factory=dict, description="RAG usage statistics")
    
    # LLM tracking
    llm_calls: int = Field(default=0, description="Total number of LLM API calls")
    total_input_tokens: int = Field(default=0, description="Total input tokens used")
    total_output_tokens: int = Field(default=0, description="Total output tokens generated")
    llm_call_details: List[Dict] = Field(default_factory=list, description="Details of each LLM call")
    
    class Config:
        arbitrary_types_allowed = True

