"""Agent nodes for the research workflow with dependency injection."""

import asyncio
from typing import List, Optional, Dict, Any, Protocol
import logging
import time
import json
import re
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.language_models import BaseChatModel
from langchain.agents import create_agent

from src.state import ResearchState, ResearchPlan, SearchQuery, ReportSection, SearchResult
from src.utils.tools import get_research_tools
from src.config import config
from src.utils.credibility import CredibilityScorer
from src.utils.citations import CitationFormatter
from src.llm_tracker import estimate_tokens
from src.exceptions import PlanningError, SearchError, SynthesisError, ReportGenerationError
from src.rag.retriever import KnowledgeBaseRetriever
from src.prompts import (
    PLANNER_SYSTEM_PROMPT, PLANNER_USER_TEMPLATE,
    SEARCHER_SYSTEM_PROMPT, SEARCHER_USER_TEMPLATE,
    SYNTHESIZER_SYSTEM_PROMPT, SYNTHESIZER_USER_TEMPLATE,
    WRITER_SYSTEM_PROMPT, WRITER_USER_TEMPLATE
)
from src.callbacks import (
    emit_planning_start, emit_planning_complete,
    emit_search_start, emit_search_results, 
    emit_extraction_start, emit_extraction_complete,
    emit_synthesis_start, emit_synthesis_progress, emit_synthesis_complete,
    emit_writing_start, emit_writing_section, emit_writing_complete,
    emit_error
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# LLM Factory
# =============================================================================

def get_llm(
    temperature: float = 0.7, 
    model_override: Optional[str] = None,
    provider_override: Optional[str] = None
) -> BaseChatModel:
    """Get LLM instance based on configuration.
    
    Args:
        temperature: Temperature for the LLM
        model_override: Optional model name to override config.model_name
        provider_override: Optional provider to override config.model_provider
        
    Returns:
        LLM instance (ChatOllama, ChatGoogleGenerativeAI, or ChatOpenAI)
    """
    model_name = model_override or config.model_name
    provider = provider_override or config.model_provider
    
    if provider == "ollama":
        logger.info(f"Using Ollama model: {model_name}")
        return ChatOllama(
            model=model_name,
            base_url=config.ollama_base_url,
            temperature=temperature,
            num_ctx=8192,
        )
    elif provider == "llamacpp":
        logger.info(f"Using llama.cpp server model: {model_name}")
        return ChatOpenAI(
            model=model_name,
            base_url=f"{config.llamacpp_base_url}/v1",
            api_key="not-needed",
            temperature=temperature
        )
    elif provider == "deepseek":
        logger.info(f"Using DeepSeek model: {model_name}")
        return ChatOpenAI(
            model=model_name,
            base_url=f"{config.deepseek_base_url}/v1",
            api_key=config.deepseek_api_key,
            temperature=temperature
        )
    else:  # gemini
        logger.info(f"Using Gemini model: {model_name}")
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=config.google_api_key,
            temperature=temperature
        )


# =============================================================================
# Research Planner Agent
# =============================================================================

class ResearchPlanner:
    """Autonomous agent responsible for planning research strategy."""
    
    def __init__(self, llm: Optional[BaseChatModel] = None, max_retries: int = 3):
        self.llm = llm or get_llm(temperature=0.7)
        self.max_retries = max_retries

    """
    异步函数：返回协程对象，需要 await   方法中調用的異步方法
        Python 程序默认是从上到下逐行执行的，遇到耗时操作（比如调用大模型API、请求网页、查数据库）时，
    整个程序会卡在那里干等结果返回，这期间CPU虽然闲着但没法去做别的事情，这叫同步阻塞。
    async def 就是用来打破这种阻塞的——它定义了一个异步函数，当函数内部遇到 await 等待的操作时，
    会自动让出CPU控制权，让程序先去处理其他任务，等等待的结果返回后再回来继续执行。
    这样一来，程序在等待耗时操作期间不会白白闲着，可以并发处理多个任务，整体效率大幅提升。
    所以，async def 的本质是让函数具备“等待时不阻塞、能切换去做别的事”的能力。
    """
    async def plan(self, state: ResearchState) -> Dict[str, Any]:
        """根据topic 制定研究计划。
        
        Returns dict with updates that LangGraph will merge into state.
        """
        logger.info(f"Planning research for: {state.research_topic}")

        # 通知前端：「规划阶段开始了」
        await emit_planning_start(state.research_topic)

        # format() "把模板里的空填上具体数字/文字"。
        system_prompt = PLANNER_SYSTEM_PROMPT.format(
            max_queries=config.max_search_queries,
            max_sections=config.max_report_sections
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", PLANNER_USER_TEMPLATE)
        ])
        
        for attempt in range(self.max_retries):
            try:
                # 记录开始时间
                start_time = time.time()
                chain = prompt | self.llm | JsonOutputParser()
                
                input_text = f"{state.research_topic} {config.max_search_queries} {config.max_report_sections}"
                input_tokens = estimate_tokens(input_text)

                result = await chain.ainvoke({
                    "topic": state.research_topic,
                    "max_queries": config.max_search_queries,
                    "max_sections": config.max_report_sections
                })
                
                duration = time.time() - start_time
                output_tokens = estimate_tokens(str(result))
                
                call_detail = {
                    'agent': 'ResearchPlanner',
                    'operation': 'plan',
                    'model': config.model_name,
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'duration': round(duration, 2),
                    'attempt': attempt + 1
                }

                # 检查返回格式
                if not all(key in result for key in ["topic", "objectives", "search_queries", "report_outline"]):
                    raise PlanningError("Invalid plan structure returned")

                # 检查query 是否存在,后面Searcher 需要 state.plan.search_queries
                if not result["search_queries"]:
                    raise PlanningError("No search queries generated")

                # 傳給state裏面的ResearchPlan ---> 研究计划
                plan = ResearchPlan(
                    topic=result["topic"],
                    objectives=result["objectives"][:5],
                    search_queries=[
                        # 轉成 state裏面的 SearchQuery
                        # 這是一個列表推导式
                        SearchQuery(query=sq["query"], purpose=sq["purpose"])
                        for sq in result["search_queries"][:config.max_search_queries]
                    ],
                    report_outline=result["report_outline"][:config.max_report_sections]
                )
                
                logger.info(f"Created plan with {len(plan.search_queries)} queries (max: {config.max_search_queries})")
                logger.info(f"Report outline has {len(plan.report_outline)} sections (max: {config.max_report_sections})")

                # 发送planning 完成事件
                await emit_planning_complete(len(plan.search_queries), len(plan.report_outline))
                
                return {
                    "plan": plan,
                    "current_stage": "searching",
                    "iterations": state.iterations + 1,
                    "llm_calls": state.llm_calls + 1,
                    "total_input_tokens": state.total_input_tokens + input_tokens,
                    "total_output_tokens": state.total_output_tokens + output_tokens,
                    "llm_call_details": state.llm_call_details + [call_detail]
                }
                
            except Exception as e:
                logger.warning(f"Planning attempt {attempt + 1} failed: {str(e)}")
                if attempt == self.max_retries - 1:
                    logger.error(f"Planning failed after {self.max_retries} attempts")
                    await emit_error(f"Planning failed: {str(e)}")
                    return {
                        "error": f"Planning failed: {str(e)}",
                        "iterations": state.iterations + 1
                    }
                else:
                    await asyncio.sleep(2 ** attempt)
        
        return {
            "error": "Planning failed: Maximum retries exceeded",
            "iterations": state.iterations + 1
        }


# =============================================================================
# Research Searcher Agent
# =============================================================================

class ResearchSearcher:
    """负责执行研究搜索."""
    
    def __init__(
        self, 
        llm: Optional[BaseChatModel] = None,
        credibility_scorer: Optional[CredibilityScorer] = None,
        kb_retriever: Optional[KnowledgeBaseRetriever] = None,
        max_retries: int = 3
    ):
        self.llm = llm or get_llm(temperature=0.3)
        self.tools = get_research_tools(agent_type="search")
        self.credibility_scorer = credibility_scorer or CredibilityScorer()
        self.kb_retriever = kb_retriever or KnowledgeBaseRetriever()
        self.max_retries = max_retries
        
    async def search(self, state: ResearchState) -> Dict[str, Any]:
        """Autonomously execute research searches using tools.
        
        Returns dict with search results that LangGraph will merge into state.
        """
        if not state.plan:
            await emit_error("No research plan available")
            return {"error": "No research plan available"}

        # 记录搜索任务数
        logger.info(f"Autonomous agent researching: {len(state.plan.search_queries)} planned queries")
        
        total_queries = len(state.plan.search_queries)
        # enumerate 函数，用来同时获取索引和元素
        # i 表示当前是第几个   query  表示当前这个 SearchQuery 对象
        for i, query in enumerate(state.plan.search_queries, 1):
            await emit_search_start(query.query, i, total_queries)
        
        max_searches = config.max_search_queries
        max_results_per_search = config.max_search_results_per_query
        expected_total_results = max_searches * max_results_per_search

        # 生成 Searcher 的 system prompt
        # format() 就是"把模板里的空填上具体数字/文字"
        system_prompt = SEARCHER_SYSTEM_PROMPT.format(
            max_searches=max_searches,
            max_results_per_search=max_results_per_search,
            expected_total_results=expected_total_results
        )
        
        agent_graph = create_agent(
            self.llm,
            self.tools,
            system_prompt=system_prompt
        )
        
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()

                # 拿到ResearchPlanner生成的 plan 中的 objectives 轉化為字符串  大模型需要字符串而不是列表
                # f"- {obj}"的作用只是让每个objective 前面带-
                objectives_text = "\n".join(f"- {obj}" for obj in state.plan.objectives)
                queries_text = "\n".join(
                    f"- {q.query} (Purpose: {q.purpose})" 
                    for q in state.plan.search_queries
                )

                # 用戶的輸入模板
                input_message = SEARCHER_USER_TEMPLATE.format(
                    topic=state.research_topic,
                    objectives=objectives_text,
                    queries=queries_text,
                    min_sources=expected_total_results
                )
                
                input_tokens = estimate_tokens(input_message)

                # invoke 是同步调用（会阻塞等待），ainvoke 是异步调用（不阻塞，可以同时做其他事）
                result = await agent_graph.ainvoke({
                    "messages": [{"role": "user", "content": input_message}]
                })
                
                duration = time.time() - start_time

                # .get(key, 默认值) = 有就拿，没有就给默认值，绝不报错
                messages = result.get('messages', [])
                # 為了估算output tokens 拿出 messages的最後一條
                output_text = ""
                if messages:
                    output_text = str(messages[-1].content if hasattr(messages[-1], 'content') else str(messages[-1]))
                output_tokens = estimate_tokens(output_text)

                # 提取搜索结果，把所有 messages 传给 _extract_results_from_messages()
                search_results = self._extract_results_from_messages(messages)

                # 统计信息
                logger.info(f"Autonomous agent collected {len(search_results)} results")
                
                total_extracted_chars = sum(
                    len(r.content) if r.content else 0 
                    for r in search_results
                )
                extracted_count = sum(1 for r in search_results if r.content)

                # 通知「正文提取完成」
                await emit_extraction_complete(extracted_count, total_extracted_chars)

                # 检查是否有结果
                if not search_results:
                    await emit_error("Agent did not collect any search results")
                    raise SearchError("Agent did not collect any search results")

                # 可信度打分
                # 得到一個按照得分從高到低排序的列表，列表裏存的是字典，字典包含 result 和 sorce
                scored_results = self.credibility_scorer.score_search_results(search_results)
                
                filtered_scored = [
                    item for item in scored_results
                    if item['credibility']['score'] >= config.min_credibility_score
                ]

                credibility_scores = [item['credibility'] for item in filtered_scored]
                sorted_results = [item['result'] for item in filtered_scored]

                # RAG fallback: only trigger when high-quality web evidence is insufficient
                rag_used = False
                rag_sources = []
                rag_stats = {
                    "used": False,
                    "count": 0,
                    "sources": [],
                    "scores": []
                }

                high_quality_results = [
                    item for item in scored_results
                    if item['credibility']['score'] >= 60
                ]

                need_rag = len(high_quality_results) < 2

                if need_rag:
                    try:
                        rag_results = await self.kb_retriever.search(
                            state.research_topic
                        )

                        if rag_results:
                            # Keep web and local evidence in one ranking pool
                            for result in rag_results:
                                sorted_results.append(result)
                                credibility_scores.append(
                                    result.metadata.get("credibility", {})
                                )

                            sorted_results = sorted(
                                sorted_results,
                                key=lambda r: r.metadata.get("score", 0),
                                reverse=True
                            )

                            rag_used = True
                            rag_sources = [
                                r.metadata.get('source', r.url)
                                for r in rag_results
                            ]
                            rag_stats = {
                                "used": True,
                                "count": len(rag_results),
                                "sources": rag_sources,
                                "scores": [
                                    r.metadata.get("score", 0)
                                    for r in rag_results
                                ]
                            }
                            logger.info(
                                f"Added {len(rag_results)} local RAG results"
                            )
                    except Exception as rag_error:
                        logger.warning(f"RAG fallback failed: {rag_error}")

                # 记录过滤结果
                logger.info(f"Filtered {len(search_results)} -> {len(sorted_results)} results (min_credibility={config.min_credibility_score})")

                # 修改狀態， 在state.py代碼中SearchQuery默認字段completed為False
                # 此時已經搜索完畢，所以修改狀態為True   但是這裏沒有return "plan": state.plan
                for q in state.plan.search_queries:
                    q.completed = True
                
                call_detail = {
                    'agent': 'ResearchSearcher',
                    'operation': 'autonomous_search',
                    'model': config.model_name,
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'duration': round(duration, 2),
                    'results_count': len(sorted_results),
                    'original_results_count': len(search_results),
                    'min_credibility_score': config.min_credibility_score,
                    'attempt': attempt + 1
                }
                
                return {
                    "search_results": sorted_results,           # 经过可信度评分、过滤、排序后的SearchResult 列表
                    "credibility_scores": credibility_scores,
                    "rag_used": rag_used,
                    "rag_sources": rag_sources,
                    "rag_stats": rag_stats,
                    "current_stage": "synthesizing",            # 表明下一阶段是synthesizing
                    "iterations": state.iterations + 1,
                    "llm_calls": state.llm_calls + 1,
                    "total_input_tokens": state.total_input_tokens + input_tokens,
                    "total_output_tokens": state.total_output_tokens + output_tokens,
                    "llm_call_details": state.llm_call_details + [call_detail]
                }
                
            except Exception as e:
                logger.warning(f"Search attempt {attempt + 1} failed: {str(e)}")
                if attempt == self.max_retries - 1:
                    logger.error(f"Search failed after {self.max_retries} attempts")
                    await emit_error(f"Search failed: {str(e)}")
                    return {
                        "error": f"Search failed: {str(e)}",
                        "iterations": state.iterations + 1
                    }
                else:
                    await asyncio.sleep(2 ** attempt)
        
        return {
            "error": "Search failed: Maximum retries exceeded",
            "iterations": state.iterations + 1
        }
    
    def _extract_results_from_messages(self, messages: list) -> List[SearchResult]:
        """Extract search results from agent messages."""
        search_results = []
        
        for msg in messages:
            # 从 LLM 返回的消息里，把「网络搜索结果」扒出来，转换成 SearchResult 对象
            if hasattr(msg, 'name') and msg.name == 'web_search':
                try:
                    content = msg.content
                    # 工具返回的数据可能是 字符串形式的 JSON 或者是  Python 对象
                    if isinstance(content, str):
                        tool_results = json.loads(content) # ← 字符串 → 解析成 Python 对象
                    else:
                        tool_results = content             # ← 已经是对象，直接用
                    
                    if isinstance(tool_results, list):
                        for item in tool_results:
                            if isinstance(item, dict):
                                search_results.append(SearchResult(
                                    query=item.get('query', ''),
                                    title=item.get('title', ''),
                                    url=item.get('url', ''),
                                    snippet=item.get('snippet', ''),
                                    content=None     # 此時 content 的結果是 None
                                ))
                except Exception as e:
                    logger.warning(f"Error parsing tool result: {e}")

            # 消息列表里找到 extract_webpage_content 工具的结果，把它填充到之前搜索结果的 content 字段里
            if hasattr(msg, 'name') and msg.name == 'extract_webpage_content':
                try:
                    content = msg.content
                    if search_results and content:
                        for sr in reversed(search_results): # ← 从后往前遍历
                            if not sr.content:              # ← 找到第一个 content 为空的
                                sr.content = content        # ← 把内容填进去
                                break
                except Exception as e:
                    logger.warning(f"Error updating content: {e}")
        
        return search_results


# =============================================================================
# Research Synthesizer Agent
# =============================================================================

class ResearchSynthesizer:
    """Autonomous agent responsible for synthesizing research findings."""
    
    def __init__(self, llm: Optional[BaseChatModel] = None, max_retries: int = 3):
        self.llm = llm or get_llm(temperature=0.3, model_override=config.summarization_model)
        self.tools = get_research_tools(agent_type="synthesis")
        self.max_retries = max_retries
        
    async def synthesize(self, state: ResearchState) -> Dict[str, Any]:
        """Autonomously synthesize key findings using tools and reasoning.
        
        Returns dict with key findings that LangGraph will merge into state.
        """
        logger.info(f"Synthesizing findings from {len(state.search_results)} results")
        
        if not state.search_results:
            await emit_error("No search results to synthesize")
            return {"error": "No search results to synthesize"}

        # 通知 synthesis 开始事件
        await emit_synthesis_start(len(state.search_results))
        
        agent_graph = create_agent(
            self.llm,
            self.tools,
            system_prompt=SYNTHESIZER_SYSTEM_PROMPT
        )

        # 设置最大使用结果数，一次 synthesis 最多處理 20 個搜索結果
        max_results = 20
        
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()

                # 根据retry 次数调整使用多少结果
                current_max = max(5, max_results - (attempt * 5))

                # 截取要给Agent 的搜索结果
                results_to_use = state.search_results[:current_max]
                credibility_scores_to_use = state.credibility_scores[:current_max] if state.credibility_scores else []

                # 傳給私有化方法   目的：格式化搜索资料
                results_text = self._format_results_text(results_to_use, credibility_scores_to_use)
                
                input_message = SYNTHESIZER_USER_TEMPLATE.format(
                    topic=state.research_topic,
                    results=results_text
                )
                
                input_tokens = estimate_tokens(input_message)
                
                result = await agent_graph.ainvoke({
                    "messages": [{"role": "user", "content": input_message}]
                })
                
                duration = time.time() - start_time

                # 有則返回内容，沒有返回空列表
                messages = result.get('messages', [])
                # 估算token
                output_text = ""
                if messages:
                    last_msg = messages[-1]
                    output_text = str(last_msg.content if hasattr(last_msg, 'content') else str(last_msg))
                output_tokens = estimate_tokens(output_text)

                # 保存一次 LLM 调用记录
                call_detail = {
                    'agent': 'ResearchSynthesizer',
                    'operation': 'autonomous_synthesis',
                    'model': config.summarization_model,
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'duration': round(duration, 2),
                    'attempt': attempt + 1
                }
                
                key_findings = self._extract_findings(output_text, state.search_results)

                # 記錄數量
                logger.info(f"Extracted {len(key_findings)} key findings")

                # 表明已經完成synthesize
                await emit_synthesis_complete(len(key_findings))
                
                return {
                    "key_findings": key_findings,
                    "current_stage": "reporting",
                    "iterations": state.iterations + 1,
                    "llm_calls": state.llm_calls + 1,
                    "total_input_tokens": state.total_input_tokens + input_tokens,
                    "total_output_tokens": state.total_output_tokens + output_tokens,
                    "llm_call_details": state.llm_call_details + [call_detail]
                }
                
            except Exception as e:
                logger.warning(f"Synthesis attempt {attempt + 1} failed: {str(e)}")
                if attempt == self.max_retries - 1:
                    logger.error(f"Synthesis failed after {self.max_retries} attempts")
                    await emit_error(f"Synthesis failed: {str(e)}")
                    return {
                        "error": f"Synthesis failed: {str(e)}",
                        "iterations": state.iterations + 1
                    }
                else:
                    await asyncio.sleep(2 ** attempt)
        
        return {
            "error": "Synthesis failed: Maximum retries exceeded",
            "iterations": state.iterations + 1
        }
    
    def _format_results_text(self, results: list, credibility_scores: list) -> str:
        """Format search results with credibility information."""

        # 长度不匹配 → 只显示搜索结果，不加置信度分数
        if len(results) != len(credibility_scores):
            return "\n\n".join([
                f"[{i+1}] {r.title}\nURL: {r.url}\nSnippet: {r.snippet}\n" +
                (f"Content: {r.content[:300]}..." if r.content else "")
                for i, r in enumerate(results)
            ])

        # 长度匹配 → 同时显示搜索结果 + 置信度分数
        return "\n\n".join([
            f"[{i+1}] {r.title}\n"
            f"URL: {r.url}\n"
            f"Credibility: {cred.get('level', 'unknown').upper()} (Score: {cred.get('score', 'N/A')}/100) - {', '.join(cred.get('factors', []))}\n"
            f"Snippet: {r.snippet}\n" +
            (f"Content: {r.content[:300]}..." if r.content else "")
            for i, (r, cred) in enumerate(zip(results, credibility_scores))
        ])
    
    def _extract_findings(self, output_text: str, search_results: list) -> List[str]:
        """Extract key findings from synthesis output.
        这个函数把 LLM 生成的「一大段文字」里，提取出「关键发现」列表，作为 Writer 写报告的依据。"""

        # 先尝试从JSON 数组提取   在LLM 输出的文字里找：[ ... ] 形式的内容
        json_match = re.search(r'\[(.*?)\]', output_text, re.DOTALL)
        
        key_findings = []
        if json_match:
            try:
                # 把匹配到的字符串轉換為python對象
                findings = json.loads(json_match.group(0))
                if isinstance(findings, list): # 如果匹配的是列表
                    key_findings = [str(f) for f in findings] # 确保每个元素是字符串
                else:
                    key_findings = [str(findings)]
            except json.JSONDecodeError:
                pass

        # JSON 没找到，按行解析
        if not key_findings:
            lines = output_text.split('\n')
            for line in lines:
                # 清理符号
                line = line.strip().lstrip('-').lstrip('*').lstrip('>').strip()
                line = re.sub(r'^\d+\.\s*', '', line)
                if len(line) > 30 and not line.startswith('[') and not line.startswith(']'):
                    key_findings.append(line)
            # 限制15條
            key_findings = key_findings[:15]

        # 两种方法都失败
        if not key_findings and search_results:
            logger.warning("Agent produced no findings, creating basic ones from results")
            # 直接从搜索结果生成
            key_findings = [
                f"{r.title}: {r.snippet[:100]}..."
                for r in search_results[:10]
                if r.snippet
            ]
        
        return key_findings


# =============================================================================
# Report Writer Agent
# =============================================================================

class ReportWriter:
    """Autonomous agent responsible for writing research reports."""
    
    def __init__(
        self, 
        llm: Optional[BaseChatModel] = None,
        citation_formatter: Optional[CitationFormatter] = None,
        citation_style: str = 'apa',
        max_retries: int = 3
    ):
        self.llm = llm or get_llm(temperature=0.7)
        self.tools = get_research_tools(agent_type="writing")
        self.max_retries = max_retries
        self.citation_style = citation_style    # 引用格式
        self.citation_formatter = citation_formatter or CitationFormatter()
        
    async def write_report(self, state: ResearchState) -> Dict[str, Any]:
        """Write the final research report with validation and retry.
        
        Returns dict with report data that LangGraph will merge into state.
        """
        logger.info("Writing final report")
        
        if not state.plan or not state.key_findings:
            await emit_error("Insufficient data for report generation")
            return {"error": "Insufficient data for report generation"}

        # 发出「开始写作」事件
        await emit_writing_start(len(state.plan.report_outline))
        
        report_llm_calls = 0
        report_input_tokens = 0
        report_output_tokens = 0
        report_call_details = []
        
        for attempt in range(self.max_retries):
            try:
                report_sections = []    # 創建section 最終存在state的ReportSection
                total_sections = len(state.plan.report_outline) # 拿到section 总数

                # 表示 Plan 不只是控制搜索，也控制最终报告结构
                for section_idx, section_title in enumerate(state.plan.report_outline, 1):
                    # 每开始写一个section，都发事件
                    await emit_writing_section(section_title, section_idx, total_sections)
                    
                    section, section_tokens = await self._write_section(
                        state.research_topic,
                        section_title,
                        state.key_findings,
                        state.search_results
                    )
                    if section:
                        report_sections.append(section)
                        if section_tokens:
                            report_llm_calls += 1
                            report_input_tokens += section_tokens['input_tokens']
                            report_output_tokens += section_tokens['output_tokens']
                            report_call_details.append(section_tokens)

                # 如果 report_sections 是空的（没有任何章节），就抛出异常
                if not report_sections:
                    raise ReportGenerationError("No report sections generated")
                
                temp_state = ResearchState(
                    research_topic=state.research_topic,
                    plan=state.plan,
                    report_sections=report_sections,
                    search_results=state.search_results
                )
                
                final_report = self._compile_report(temp_state)
                
                if state.search_results:
                    final_report = self.citation_formatter.update_report_citations(
                        final_report,
                        style=self.citation_style,
                        search_results=state.search_results
                    )
                
                if state.credibility_scores:
                    high_cred_sources = [
                        i+1 for i, score in enumerate(state.credibility_scores)
                        if score.get('level') == 'high'
                    ]
                    if high_cred_sources:
                        final_report += f"\n\n---\n\n**Note:** {len(high_cred_sources)} high-credibility sources were prioritized in this research."
                
                if len(final_report) < 500:
                    raise ReportGenerationError("Report too short - insufficient content")
                
                logger.info(f"Report generation complete: {len(final_report)} chars")
                
                await emit_writing_complete(len(final_report))
                
                return {
                    "report_sections": report_sections,
                    "final_report": final_report,
                    "current_stage": "complete",
                    "iterations": state.iterations + 1,
                    "llm_calls": state.llm_calls + report_llm_calls,
                    "total_input_tokens": state.total_input_tokens + report_input_tokens,
                    "total_output_tokens": state.total_output_tokens + report_output_tokens,
                    "llm_call_details": state.llm_call_details + report_call_details
                }
                
            except Exception as e:
                logger.warning(f"Report attempt {attempt + 1} failed: {str(e)}")
                if attempt == self.max_retries - 1:
                    logger.error(f"Report generation failed after {self.max_retries} attempts")
                    await emit_error(f"Report generation failed: {str(e)}")
                    return {
                        "error": f"Report writing failed: {str(e)}",
                        "iterations": state.iterations + 1
                    }
                else:
                    await asyncio.sleep(2 ** attempt)
        
        return {
            "error": "Report generation failed: Maximum retries exceeded",
            "iterations": state.iterations + 1
        }
    
    async def _write_section(
        self,
        topic: str,             # 研究主题
        section_title: str,     # 來自Planner 规划好的章节名称
        findings: List[str],    # 來自Synthesizer 核心发现
        search_results: List    # 來自Searcher 原始资料来源
    ) -> tuple:
        """Write a single report section."""
        logger.info(f"Writing section: {section_title}")

        # 生成 Writer 的 system prompt
        system_prompt = WRITER_SYSTEM_PROMPT.format(min_words=config.min_section_words)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])
        
        try:
            start_time = time.time()
            
            sources_context = ""
            if search_results:
                sources_context = "\nAvailable Sources for Citation:\n" + "\n".join(
                    f"[{i+1}] {r.title} ({r.url})"
                    for i, r in enumerate(search_results[:15])
                )

            # 生成输入消息
            input_message = WRITER_USER_TEMPLATE.format(
                topic=topic,
                section_title=section_title,
                min_words=config.min_section_words,
                findings=chr(10).join(f"- {f}" for f in findings),
                sources_context=sources_context
            )

            input_tokens = estimate_tokens(input_message)

            # 調用大模型 --- ReportWriter 不是 Agent
            chain = prompt | self.llm | StrOutputParser()
            content = await chain.ainvoke({"input": input_message})

            # 確保是content字符串
            if not isinstance(content, str):
                content = str(content)
            
            duration = time.time() - start_time
            output_tokens = estimate_tokens(content)

            # 保存一次 LLM 調用信息
            call_detail = {
                'agent': 'ReportWriter',
                'operation': f'write_section_{section_title[:30]}',
                'model': config.model_name,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'duration': round(duration, 2)
            }

            # 檢查生成內容是否有效
            if not content or len(content.strip()) < 50:
                logger.warning(f"Section '{section_title}' generated insufficient content: {len(content)} chars")
                if findings:
                    logger.info(f"Creating fallback content for section '{section_title}'")
                    content = f"\n\n{chr(10).join(findings[:3])}\n\n"
                else:
                    logger.error(f"Cannot create section '{section_title}' - no content and no findings")
                    return None, None

            # 提取引用 URL
            citations = re.findall(r'\[(\d+)\]', content) # 找到content裏面的 [num] ，返回列表
            source_urls = []
            # set(citations) 去重，避免重复添加同一个 URL
            for cite_num in set(citations):
                idx = int(cite_num) - 1 # 减 1 ---> Python 列表索引从 0 开始,但是citations的引用文獻是從1開始
                # 確保引用存在
                if 0 <= idx < len(search_results):
                    source_urls.append(search_results[idx].url)

            # 創建 state 裏面的 ReportSection
            section = ReportSection(
                title=section_title,
                content=content,
                sources=source_urls
            )
            
            logger.info(f"Successfully wrote section '{section_title}': {len(content)} chars")
            return section, call_detail
            
        except Exception as e:
            logger.error(f"Error writing section '{section_title}': {str(e)}")
            return None, None
    
    def _compile_report(self, state: ResearchState) -> str:
        """把报告的所有章节（report_sections）和参考资料拼接成一份完整的 Markdown 格式报告。"""

        # 双重保险： 防止 search_results 不存在或为 None
        search_results = getattr(state, 'search_results', []) or []
        report_sections = getattr(state, 'report_sections', []) or []

        # 统计一共引用了多少个不同的来源
        unique_sources = set() # 一個集合，可去重
        # 从 search_results 收集
        for result in search_results:
            if hasattr(result, 'url') and result.url:
                unique_sources.add(result.url)

        # 从 report_sections 收集
        for section in report_sections:
            if hasattr(section, 'sources'):
                unique_sources.update(section.sources)
        
        source_count = len(unique_sources) if unique_sources else len(search_results)

        # 构建报告头部
        report_parts = [
            f"# {state.research_topic}\n",
            f"**Deep Research Report**\n",
            f"\n## Executive Summary\n",
            f"This report provides a comprehensive analysis of {state.research_topic}. ",
            f"The research was conducted across **{source_count} sources** ",
            f"and synthesized into **{len(report_sections)} key sections**.\n",
            f"\n## Research Objectives\n"
        ]
        
        if state.plan and hasattr(state.plan, 'objectives'):
            for i, obj in enumerate(state.plan.objectives, 1):
                report_parts.append(f"{i}. {obj}\n")
        
        report_parts.append("\n---\n")

        # 添加章节内容
        has_references_section = False
        for section in report_sections:
            content = section.content.strip()
            
            if "## References" in content or section.title.lower() == "references":
                has_references_section = True
            
            if content.startswith(f"## {section.title}"):
                report_parts.append(f"\n{content}\n\n")
            else:
                report_parts.append(f"\n## {section.title}\n\n")
                report_parts.append(content)
                report_parts.append("\n")

        # 生成参考文献
        if not has_references_section:
            report_parts.append("\n---\n\n## References\n\n")
        
        source_info = []
        seen_urls = set()

        # 从 search_results 收集
        for result in search_results:
            if hasattr(result, 'url') and result.url and result.url not in seen_urls:
                seen_urls.add(result.url)
                title = getattr(result, 'title', '')
                source_info.append((result.url, title))

        # 从 report_sections 收集
        for section in report_sections:
            if hasattr(section, 'sources'):
                for url in section.sources:
                    if url not in seen_urls:
                        seen_urls.add(url)
                        source_info.append((url, ''))
        
        if not has_references_section:
            if source_info:
                for i, (url, title) in enumerate(source_info[:30], 1):
                    citation = self.citation_formatter.format_apa(url, title)
                    report_parts.append(f"{i}. {citation}\n")
            else:
                report_parts.append("*No sources were available for this research.*\n")
        
        return "".join(report_parts)
