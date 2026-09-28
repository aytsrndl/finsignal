# FinSignal

A hybrid multi-agent system for stock analysis. Specialized code and ML models handle the numbers; LLMs handle interpretation and synthesis.

> **Status: in active development.** The SEC Filing Agent's quantitative layer is complete and validated. Its LLM layer is in progress. The other four agents are planned. See [Status](#status).

## Core idea

Code and ML models do what they're good at: speed, precision, math. LLMs do what they're good at: reasoning, interpretation, synthesis. Each agent computes exact numbers with code, then hands structured JSON to an LLM (or, for the final agent, to a decision-making LLM). The LLM only ever reasons about numbers it didn't invent.

## Architecture

Five agents. Four specialists feed structured JSON into a Portfolio Manager that produces the final BUY / HOLD / SELL decision.

| Agent | Question it answers | Compute engine | LLM role | Status |
|---|---|---|---|---|
| **SEC Filing** | What is the company, fundamentally? | Python (EDGAR / XBRL) | Risk analysis from 10-K text | Quant layer done, LLM layer in progress |
| **Sentiment** | What is the market feeling? | FinBERT | Summarize mood, flag divergences | Planned |
| **Technical** | What is price action saying? | pandas / ta | Interpret signal convergence | Planned |
| **Risk Manager** | How much could this hurt the portfolio? | NumPy (VaR, correlation) | Write risk memo | Planned |
| **Portfolio Manager** | What do we do? | none | Pure LLM synthesis | Planned |

Agents communicate through structured JSON rather than natural language, which keeps data flow testable, loggable, and reproducible.

## Status

**SEC Filing Agent, quantitative layer (done):**
- Ticker to CIK lookup, filing list, and filing URL construction against SEC EDGAR
- Company Facts (XBRL) extraction, cleaned into deduplicated annual series
- Seven metrics: revenue growth, net income growth, net margin, debt-to-equity, debt-to-assets, current ratio, ROE
- Tag-robust resolvers for revenue and net income, and total liabilities derived from the accounting identity (`Assets - Equity`) instead of a tag lookup
- Structured JSON output via `build_sec_metrics(ticker)`
- Validated on AAPL, MSFT, WMT, and GOOGL

**SEC Filing Agent, qualitative layer (in progress):**
- 10-K fetching and HTML-to-text extraction (done, validated on AAPL and WMT)
- Pydantic schema for structured LLM output (scaffolded in `agents/sec_filing_agent.py`)
- LLM analysis call (next)

## Example output

`build_sec_metrics("AAPL")` returns each metric's most recent value along with the fiscal year it comes from (rounded here for readability):

```json
{
  "ticker": "AAPL",
  "entity_name": "Apple Inc.",
  "revenue_growth":    {"value": 0.064, "year": "2025"},
  "net_income_growth": {"value": 0.195, "year": "2025"},
  "net_margin":        {"value": 0.269, "year": "2025"},
  "debt_to_equity":    {"value": 3.87,  "year": "2025"},
  "debt_to_assets":    {"value": 0.79,  "year": "2025"},
  "current_ratio":     {"value": 0.89,  "year": "2025"},
  "roe":               {"value": 1.52,  "year": "2025"}
}
```

Metrics carry their own year because different series can end in different years for the same company. A metric that can't be computed is returned as `null` rather than raising.

## Design decisions

The reasoning behind the non-obvious choices lives in [DECISIONS.md](DECISIONS.md). A few highlights:

- **XBRL over HTML scraping for financials.** The SEC's Company Facts API returns pre-extracted numbers, which is far more robust than parsing filing tables.
- **Resolvers and derivation over hardcoded tags.** Tag choice varies by company. Where an accounting identity exists, it is used. Where it doesn't (e.g. revenue), a priority list of candidate tags is tried in order.
- **Recent years only for flow metrics.** Revenue and net income are restricted to 2021+ because older filings mix quarterly and annual figures under the same tags.
- **Prompt-steering over section parsing for 10-Ks.** Instead of fragile code that locates Risk Factors and MD&A, the full text goes to the LLM with a section-targeted, structured prompt.

## Setup

Requires Python 3.12. Commands are for Windows PowerShell.

```powershell
git clone https://github.com/aytsrndl/finsignal.git
cd finsignal
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

**SEC User-Agent (required).** The SEC requires requests to identify the requester. Edit `USER_AGENT` in `tools/edgar_parser.py` to your own name and email.

**LLM API key.** Create a `.env` file in the project root (it is gitignored) containing your provider key, for example:

```
OPENAI_API_KEY=your-key-here
```

Never commit this file.

## Usage

```python
from tools.edgar_parser import build_sec_metrics, get_cik_from_ticker, get_latest_10k_html, html_to_text

# Quantitative metrics as structured JSON
metrics = build_sec_metrics("AAPL")

# 10-K text ready for LLM analysis
text = html_to_text(get_latest_10k_html(get_cik_from_ticker("AAPL")))
```

## Project structure

```
finsignal/
├── agents/
│   └── sec_filing_agent.py   # LLM analysis layer (in progress)
├── tools/
│   └── edgar_parser.py       # EDGAR retrieval, metrics, 10-K text extraction
├── DECISIONS.md              # Design-decision log with reasoning
└── requirements.txt
```

Planned additions: the remaining four agents, an orchestrator, a SQLite portfolio store, a Streamlit frontend, and automated tests.

## Known limitations

- Revenue and net income coverage starts at 2021 by design. Full historical reconciliation is deferred.
- Revenue and net income use tag-fallback lists, so a company with an unusual tag may return `null` for that metric until its tag is added.
- ROE uses year-end equity rather than average equity.
- Extracted 10-K text still contains some harmless XBRL header noise.
- Validated on a handful of large US companies. Banks and other unusual balance sheets are untested.
- Manual sanity checks only. Automated tests are planned.
- In production, the parsing layer would likely be replaced by a commercial data vendor. It was built here to understand the problem directly.

## Roadmap

1. Finish the SEC Filing Agent's LLM analysis and package it behind a single ticker-in, JSON-out interface
2. Technical Agent (yfinance and indicators)
3. Sentiment Agent (FinBERT)
4. Risk Manager Agent
5. Portfolio Manager Agent and orchestration
6. Streamlit frontend and demo

## Disclaimer

FinSignal is an educational project. Nothing it outputs is investment advice.

## Author

Aytunc Sarandal
