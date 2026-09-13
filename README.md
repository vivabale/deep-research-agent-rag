# Deep Research Agent (with RAG Extension)

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2.57+-green.svg)](https://github.com/langchain-ai/langgraph)

A production-ready multi-agent autonomous research system built with LangGraph and LangChain. Four specialized agents collaborate to conduct comprehensive research on any topic, generating detailed citation-backed reports with credibility scoring and quality metrics.

**This fork adds a local RAG (Retrieval-Augmented Generation) knowledge base retrieval module**, enabling the research agent to combine web search with a private document corpus.

**Supports:** Local models (Ollama, llama.cpp) and Cloud APIs (Google Gemini, OpenAI, DeepSeek)

---

## Table of Contents

- [Demo](#demo)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Key Components](#key-components)
- [RAG Extension](#rag-extension)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [License](#license)

---

## Demo

https://github.com/user-attachments/assets/df8404c6-7423-4a49-864a-bd4d59885c1b

*Watch the full demo video to see the Deep Research Agent in action, showcasing the multi-agent workflow, real-time progress updates, and comprehensive report generation.*

---

## Features

### Core Capabilities

| Feature | Description |
|---------|-------------|
| **Multi-Agent Architecture** | Four specialized autonomous agents orchestrated by LangGraph's StateGraph |
| **Autonomous Research** | Search agent dynamically decides queries, sources, and extraction depth |
| **Credibility Scoring** | Automatic source evaluation (0-100) based on domain authority |
| **Quality Validation** | Section-level validation with retry logic and exponential backoff |
| **Multi-Format Export** | Reports in Markdown, HTML, and plain text |
| **LLM Usage Tracking** | Real-time monitoring of API calls, tokens, and costs |
| **Research Caching** | 7-day TTL file-based caching with MD5 topic hashing |
| **Web Interface** | Interactive Chainlit UI with real-time progress |

### Production-Ready Features

| Feature | Description |
|---------|-------------|
| **Circuit Breaker** | Automatic failure detection and recovery for external services |
| **Connection Pooling** | HTTP/2 with persistent connections via httpx |
| **Checkpointing** | Workflow state persistence for crash recovery |
| **Typed Exceptions** | Domain-specific error handling for better debugging |
| **Dependency Injection** | Testable agent architecture with injectable LLMs |
| **Search Provider Abstraction** | Extensible search backend (DuckDuckGo, with easy addition of others) |

### RAG Extension (Added in This Fork)

| Feature | Description |
|---------|-------------|
| **Local Knowledge Base** | Retrieve from a private document corpus (`knowledge_base/`) |
| **Multi-Format Loading** | Supports `.txt`, `.md`, `.pdf`, `.docx` |
| **Qwen Embeddings** | Vector embeddings via Alibaba Cloud DashScope `text-embedding-v4` |
| **Adaptive Splitting** | FAQ splitter + sentence-level recursive splitter |
| **Automatic Fallback** | Uses RAG when web search results are insufficient |
| **Unified State** | RAG results merged into `search_results` with `source_type='local_rag'` |

---

## Architecture

### High-Level Flow

![Deep Research Agent Flow Diagram](assets/flow.png)

### Agent Responsibilities

#### ResearchPlanner
- Analyzes research topics and generates 3-5 SMART objectives
- Creates targeted search queries covering different aspects
- Designs report outline with up to 8 sections
- Uses structured JSON output for reliability

#### ResearchSearcher (Autonomous Agent)
- LangChain-powered autonomous agent using `create_agent()`
- Dynamically decides which queries to execute
- Uses `web_search` and `extract_webpage_content` tools
- **Falls back to local RAG retrieval when web evidence is insufficient**
- All sources scored for credibility and filtered (default threshold: 40)
- Circuit breaker protection against service failures

#### ResearchSynthesizer
- Analyzes aggregated results with credibility awareness
- Prioritizes HIGH-credibility sources (score ≥70)
- Resolves contradictions using credibility hierarchy
- Progressive truncation handles token limits

#### ReportWriter
- Generates structured sections with academic tone
- Adds proper citations (APA, MLA, Chicago, IEEE)
- Validates section quality with retry on failure
- Compiles final markdown with references

---

## Installation

### Prerequisites

- Python 3.11+
- pip or uv package manager
- One of:
  - [Ollama](https://ollama.com/) (local models)
  - [llama.cpp](https://github.com/ggerganov/llama.cpp) (local models, maximum performance)
  - [Google Gemini API](https://makersuite.google.com/app/apikey) (cloud)
  - [OpenAI API](https://platform.openai.com/api-keys) (cloud)
  - [DeepSeek API](https://platform.deepseek.com/) (cloud)

### Quick Start

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/v9_work.git
cd v9_work

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install additional RAG dependencies
pip install pypdf python-docx openai

# Run
python main.py
```

### Using Ollama (Recommended for Local)

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model
ollama pull qwen2.5:7b

# Configure .env
MODEL_PROVIDER=ollama
MODEL_NAME=qwen2.5:7b
SUMMARIZATION_MODEL=qwen2.5:7b
```

### Using llama.cpp (Maximum Performance)

```bash
# Download GGUF model
huggingface-cli download Qwen/Qwen2.5-7B-Instruct-GGUF \
  qwen2.5-7b-instruct-q4_k_m.gguf --local-dir ./models

# Start server with tool calling
./llama-server -m ./models/qwen2.5-7b-instruct-q4_k_m.gguf \
  --host 0.0.0.0 --port 8080 -ngl 35 --ctx-size 4096 --jinja

# Configure .env
MODEL_PROVIDER=llamacpp
MODEL_NAME=qwen2.5-7b-instruct-q4_k_m
LLAMACPP_BASE_URL=http://localhost:8080
```

### Using Cloud APIs

```bash
# Gemini
MODEL_PROVIDER=gemini
GEMINI_API_KEY=your_api_key_here
MODEL_NAME=gemini-2.5-flash

# OpenAI
MODEL_PROVIDER=openai
OPENAI_API_KEY=your_api_key_here
MODEL_NAME=gpt-4o-mini

# DeepSeek
MODEL_PROVIDER=deepseek
DEEPSEEK_API_KEY=your_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com
MODEL_NAME=deepseek-chat
```

---

## Usage

### Command Line

```bash
# Interactive mode
python main.py

# Direct topic
python main.py "Impact of quantum computing on cryptography"
```

### Web Interface

```bash
chainlit run app.py --host 127.0.0.1 --port 8000
```

Features:
- Real-time progress with stage indicators
- Quality metrics and LLM usage statistics
- Multiple format downloads (MD, HTML, TXT)
- Research history tracking

### Building the RAG Index (New)

```bash
# 1. Put documents into knowledge_base/
#    Supported: .txt .md .pdf .docx
# 2. Build the index
python -m src.rag.build_index

# Expected output:
# INFO:src.rag.build_index:documents=6 chunks=1754
# 1754
```

### Programmatic API

```python
import asyncio
from src.graph import run_research

async def research():
    # Basic usage
    state = await run_research(
        topic="Your research topic",
        verbose=True,
        use_cache=True
    )
    
    # Access results
    print(state["final_report"])
    print(f"Sources: {len(state['search_results'])}")
    print(f"Findings: {len(state['key_findings'])}")
    print(f"Tokens: {state['total_input_tokens'] + state['total_output_tokens']:,}")
    print(f"RAG used: {state.get('rag_used', False)}")
    print(f"RAG sources: {state.get('rag_sources', [])}")

asyncio.run(research())
```

### With Persistence (Crash Recovery)

```python
from src.graph import run_research_with_persistence, resume_research

# Run with SQLite persistence
state = await run_research_with_persistence(
    topic="Your topic",
    thread_id="my-research-001"
)

# Resume interrupted workflow
state = await resume_research(thread_id="my-research-001")
```

---

## Configuration

### Environment Variables

```bash
# =============================================================================
# MODEL PROVIDER (required)
# =============================================================================
MODEL_PROVIDER=deepseek            # Options: ollama, llamacpp, gemini, openai, deepseek

# =============================================================================
# PROVIDER-SPECIFIC SETTINGS
# =============================================================================

# DeepSeek
DEEPSEEK_API_KEY=your_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com
MODEL_NAME=deepseek-chat
SUMMARIZATION_MODEL=deepseek-chat

# Ollama
MODEL_NAME=qwen2.5:7b
SUMMARIZATION_MODEL=qwen2.5:7b
OLLAMA_BASE_URL=http://localhost:11434

# llama.cpp
MODEL_NAME=qwen2.5-7b-instruct-q4_k_m
LLAMACPP_BASE_URL=http://localhost:8080

# Gemini
GEMINI_API_KEY=your_api_key_here
MODEL_NAME=gemini-2.5-flash
SUMMARIZATION_MODEL=gemini-2.5-flash

# OpenAI
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.openai.com  # Optional
MODEL_NAME=gpt-4o-mini
SUMMARIZATION_MODEL=gpt-4o-mini

# =============================================================================
# SEARCH SETTINGS (optional)
# =============================================================================
MAX_SEARCH_QUERIES=3               # Number of search queries
MAX_SEARCH_RESULTS_PER_QUERY=3     # Results per query
MIN_CREDIBILITY_SCORE=40           # Filter threshold (0-100)

# =============================================================================
# REPORT SETTINGS (optional)
# =============================================================================
MAX_REPORT_SECTIONS=8              # Maximum sections in report
CITATION_STYLE=apa                 # Options: apa, mla, chicago, ieee

# =============================================================================
# EMBEDDING SETTINGS (RAG extension)
# =============================================================================
DASHSCOPE_API_KEY=your_dashscope_api_key
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
EMBEDDING_MODEL=text-embedding-v4
EMBEDDING_DIMENSIONS=1024

# =============================================================================
# RAG SETTINGS
# =============================================================================
KNOWLEDGE_BASE_DIR=knowledge_base
RAG_INDEX_PATH=storage/rag_index.json
RAG_MIN_SCORE=0.35
RAG_TOP_K=5
```

### Model Provider Comparison

| Provider | Cost | Privacy | Speed | Setup |
|----------|------|---------|-------|-------|
| **Ollama** | Free | Local | Fast | Easy |
| **llama.cpp** | Free | Local | Fastest | Manual |
| **Gemini** | Free tier | Cloud | Fast | API key |
| **OpenAI** | Pay-per-use | Cloud | Fast | API key |
| **DeepSeek** | Pay-per-use | Cloud | Fast | API key |

---

## Project Structure

```
v9_work/
├── src/
│   ├── __init__.py           # Package initialization
│   ├── config.py             # Configuration management (Pydantic)
│   ├── state.py              # State models (ResearchState, etc.)
│   ├── agents.py             # Agent implementations with DI
│   ├── graph.py              # LangGraph workflow + checkpointing
│   ├── callbacks.py          # Progress callback system
│   ├── llm_tracker.py        # Token and cost tracking
│   ├── exceptions.py         # Typed domain exceptions
│   │
│   ├── prompts/              # Extracted prompt templates
│   │   ├── __init__.py
│   │   ├── planner.py        # Planning prompts
│   │   ├── searcher.py       # Search prompts
│   │   ├── synthesizer.py    # Synthesis prompts
│   │   └── writer.py         # Writing prompts
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── tools.py          # LangChain @tool functions
│   │   ├── web_utils.py      # httpx client, circuit breaker, search providers
│   │   ├── cache.py          # Research caching (7-day TTL)
│   │   ├── credibility.py    # Source credibility scoring
│   │   ├── citations.py      # Citation formatting
│   │   ├── exports.py        # Multi-format export
│   │   └── history.py        # Research history
│   │
│   └── rag/                  # NEW: Local RAG module
│       ├── __init__.py
│       ├── build_index.py    # Offline index builder
│       ├── loader.py         # Directory loader (via factory)
│       ├── splitter.py       # Splitter selector
│       ├── embeddings.py     # Qwen text-embedding-v4 wrapper
│       ├── vector_store.py   # JSON vector store + cosine similarity
│       ├── retriever.py      # Async retrieval wrapper
│       ├── schemas.py        # Document / DocumentChunk
│       ├── loaders/          # Multi-format loaders
│       │   ├── base.py
│       │   ├── factory.py
│       │   ├── text_loader.py
│       │   ├── pdf_loader.py
│       │   └── docx_loader.py
│       └── splitters/        # Splitting strategies
│           ├── base.py
│           ├── faq_splitter.py
│           ├── recursive_splitter.py
│           └── selector.py
│
├── knowledge_base/           # NEW: Your documents go here
├── storage/                  # NEW: RAG index output
├── outputs/                  # Generated reports
├── .cache/
│   ├── research/             # Cached results
│   ├── checkpoints/          # Workflow checkpoints (SQLite)
│   └── research_history.json
│
├── assets/                   # Documentation assets
├── main.py                   # CLI entry point
├── app.py                    # Chainlit web interface
├── requirements.txt          # Dependencies
├── pyproject.toml            # Project metadata
├── LICENSE                   # Original MIT License
├── LICENSE-RAG               # NEW: MIT for RAG extension
└── README.md
```

---

## Key Components

### Exception Hierarchy

```python
DeepResearchError
├── ConfigurationError
├── PlanningError
├── SearchError
│   └── RateLimitError
├── ContentExtractionError
├── SynthesisError
├── ReportGenerationError
├── CircuitOpenError
└── LLMError
```

### Credibility Scoring

Sources are scored (0-100) based on:

| Factor | Points |
|--------|--------|
| Trusted domain (.edu, .gov, academic) | +30 |
| HTTPS enabled | +5 |
| Academic/research path | +10 |
| Suspicious TLD (.xyz, .tk) | -20 |
| No HTTPS | -10 |

Default filter threshold: 40 (configurable via `MIN_CREDIBILITY_SCORE`).

RAG results from the local knowledge base are assigned a fixed score of 85 (`high` credibility) with the factor `verified_local_knowledge_base`.

### Circuit Breaker States

```
CLOSED ──► (5 failures) ──► OPEN ──► (30s timeout) ──► HALF_OPEN ──► (success) ──► CLOSED
                              │                            │
                              └──────── (failure) ◄────────┘
```

---

## RAG Extension

### How It Works

```
knowledge_base/   →   loader   →   splitter   →   embedding   →   vector_store
  .txt / .md          (multi-      (FAQ /        (Qwen           (JSON)
  .pdf / .docx        format)      recursive)    text-embedding-v4)
                                                                    │
                                                                    ▼
                                                     storage/rag_index.json
                                                                    │
                                                                    ▼
                                            KnowledgeBaseRetriever
                                                                    │
                                                                    ▼
                                        ResearchSearcher (RAG fallback)
                                                                    │
                                                                    ▼
                                      web results + local RAG sources
```

### When RAG Is Triggered

RAG fallback is activated when:

- Web search returns fewer than 2 high-credibility results (score ≥ 60), OR
- No web results are collected at all

When triggered, up to `RAG_TOP_K` chunks are retrieved from the local knowledge base and merged into the result pool, then re-ranked by similarity score.

### Supported Formats

| Format | Loader | Notes |
|--------|--------|-------|
| `.txt` | `TextLoader` | UTF-8, errors ignored |
| `.md`  | `TextLoader` | Same as txt |
| `.pdf` | `PDFLoader` | Via `pypdf`, page-by-page extraction |
| `.docx`| `DocxLoader` | Via `python-docx`, paragraph extraction |

### Validation

The RAG pipeline has been validated with:

- **1754 indexed chunks** across 6 documents
- TXT and PDF knowledge documents
- Semantic retrieval testing (top scores 0.78–0.83 for relevant queries)

---

## API Reference

### Core Functions

```python
# Main research function
async def run_research(
    topic: str,
    verbose: bool = True,
    use_cache: bool = True,
    use_checkpoints: bool = True,
    thread_id: Optional[str] = None
) -> Dict[str, Any]

# With SQLite persistence
async def run_research_with_persistence(
    topic: str,
    verbose: bool = True,
    use_cache: bool = True,
    thread_id: Optional[str] = None
) -> Dict[str, Any]

# Resume interrupted workflow
async def resume_research(
    thread_id: str,
    additional_input: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]

# Check workflow state
async def get_workflow_state(thread_id: str) -> Optional[Dict[str, Any]]

# List saved threads
def list_research_threads() -> List[str]
```

### Response Structure

```python
{
    "research_topic": str,
    "plan": ResearchPlan,
    "search_results": List[SearchResult],
    "credibility_scores": List[Dict],
    "key_findings": List[str],
    "report_sections": List[ReportSection],
    "final_report": str,
    "current_stage": str,
    "error": Optional[str],
    "iterations": int,
    "llm_calls": int,
    "total_input_tokens": int,
    "total_output_tokens": int,
    "llm_call_details": List[Dict],
    "rag_used": bool,              # NEW
    "rag_sources": List[str],      # NEW
    "rag_stats": Dict              # NEW
}
```

---

## Output Format

Reports follow this structure:

```markdown
# [Research Topic]

**Deep Research Report**

## Executive Summary
[Overview with source count and section count]

## Research Objectives
1. [Objective 1]
2. [Objective 2]
...

---

## [Section 1 Title]
[Content with inline citations [1], [2]]

## [Section 2 Title]
[Content with inline citations [3], [4]]

---

## References
1. [Formatted citation - APA/MLA/Chicago/IEEE]
2. [Formatted citation]
...

---

**Note:** X high-credibility sources were prioritized.
```

---

## Development

### Running Tests

```bash
pytest tests/ -v
```

### Adding a New Search Provider

```python
# src/utils/web_utils.py
class GoogleSearchProvider(SearchProvider):
    @property
    def name(self) -> str:
        return "google"
    
    async def search(self, query: str, max_results: int) -> List[SearchResult]:
        # Implementation
        pass

# Register in WebSearchTool
tool = WebSearchTool(providers=[
    DuckDuckGoProvider(),
    GoogleSearchProvider()  # Fallback
])
```

### Customizing Prompts

Edit files in `src/prompts/`:
- `planner.py` - Research planning strategy
- `searcher.py` - Search agent instructions
- `synthesizer.py` - Synthesis methodology
- `writer.py` - Report writing style

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

This project follows the **MIT License** of the original project.

- **Original project code**: see [LICENSE](LICENSE) — Copyright (c) 2025 Deep Research Agent
- **Newly added RAG code** (`src/rag/`): see [LICENSE-RAG](LICENSE-RAG) — Copyright (c) 2025 [Your Name]

Original attribution and license are preserved as required by the MIT License.

---

## Acknowledgments

Built with:
- [LangGraph](https://github.com/langchain-ai/langgraph) - Workflow orchestration
- [LangChain](https://github.com/langchain-ai/langchain) - LLM framework
- [Chainlit](https://github.com/Chainlit/chainlit) - Web interface
- [httpx](https://www.python-httpx.org/) - Async HTTP client
- [DuckDuckGo](https://duckduckgo.com/) - Web search

RAG extension built with:
- [pypdf](https://pypdf.readthedocs.io/) - PDF parsing
- [python-docx](https://python-docx.readthedocs.io/) - DOCX parsing
- [Alibaba Cloud DashScope](https://dashscope.aliyun.com/) - `text-embedding-v4` embeddings

Supports:
- [Ollama](https://ollama.com/) & [llama.cpp](https://github.com/ggerganov/llama.cpp) - Local models
- [Google Gemini](https://ai.google.dev/), [OpenAI](https://openai.com/), [DeepSeek](https://platform.deepseek.com/) - Cloud APIs

### Original Project

This fork is based on [tarun7r/deep-research-agent](https://github.com/tarun7r/deep-research-agent).
Thanks to the original author for the open-source contribution.

RAG extension by [Your Name].

---

## Contact

- **Original Author**: [tarun7r](https://github.com/tarun7r)
  - LinkedIn: [Tarun Sai Goddu](https://www.linkedin.com/in/tarunsaigoddu/)
  - Hugging Face: [tarun7r](https://huggingface.co/tarun7r)
  - Email: tarunsaiaa@gmail.com
- **Fork Author**: [Your Name](https://github.com/YOUR_USERNAME) — RAG extension