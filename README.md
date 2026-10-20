# FinSignal

A hybrid multi-agent system for stock analysis. Specialized code and ML models handle the numbers; LLMs handle interpretation and synthesis.

> **Status: in active development.** The SEC Filing Agent has a metrics-to-LLM pipeline with offline tests. Live LLM validation is pending. The Technical Agent computes daily signals; Sentiment, Risk, and Portfolio Manager agents are planned. See [Status](#status).

## Core idea

Code and ML models do what they're good at: speed, precision, math. LLMs do what they're good at: reasoning, interpretation, synthesis. Each agent computes exact numbers with code, then hands structured JSON to an LLM (or, for the final agent, to a decision-making LLM). The LLM only ever reasons about numbers it didn't invent.

## Architecture

Five agents. Four specialists feed structured JSON into a Portfolio Manager that produces the final BUY / HOLD / SELL decision.

| Agent | Question it answers | Compute engine | LLM role | Status |
|---|---|---|---|---|
| **SEC Filing** | What is the company, fundamentally? | Python (EDGAR / XBRL) | Risk analysis from 10-K text | Pipeline implemented; live LLM validation pending |
| **Sentiment** | What is the market feeling? | FinBERT | Summarize mood, flag divergences | Planned |
| **Technical** | What is price action saying? | Python / yfinance | Optional future interpretation | Deterministic indicators and JSON CLI implemented; cloud live fetch blocked |
| **Risk Manager** | How much could this hurt the portfolio? | NumPy (VaR, correlation) | Write risk memo | Planned |
| **Portfolio Manager** | What do we do? | none | Pure LLM synthesis | Planned |

Agents communicate through structured JSON rather than natural language, which keeps data flow testable, loggable, and reproducible.

## Status

**SEC Filing Agent, quantitative layer (done):**
- Ticker to CIK lookup, filing list, and filing URL construction against SEC EDGAR
- Company Facts (XBRL) extraction, cleaned into deduplicated annual series
- Seven metrics: revenue growth, net income growth, net margin, liabilities-to-equity, liabilities-to-assets, current ratio, ROE
- Tag-robust resolvers for revenue and net income, and total liabilities derived from the accounting identity (`Assets - Equity`) instead of a tag lookup
- Structured JSON output via `build_sec_metrics(ticker)`
- Validated on AAPL, MSFT, WMT, and GOOGL

**SEC Filing Agent, qualitative layer (implemented, live LLM validation pending):**
- 10-K fetching and HTML-to-text extraction (done, validated on AAPL and WMT)
- Structured OpenAI analysis using the `TenKAnalysis` Pydantic schema
- Filing source metadata, per-metric fiscal years, and explicit missing-data warnings
- Source-passage citations and failure handling for incomplete/refused responses
- Callable agent and JSON CLI, covered with mocked provider responses

## Example output

`build_sec_metrics("AAPL")` returns each metric's most recent value along with the fiscal year it comes from (rounded here for readability):

```json
{
  "ticker": "AAPL",
  "entity_name": "Apple Inc.",
  "revenue_growth":    {"value": 0.064, "year": "2025"},
  "net_income_growth": {"value": 0.195, "year": "2025"},
  "net_margin":        {"value": 0.269, "year": "2025"},
  "liabilities_to_equity":    {"value": 3.87,  "year": "2025"},
  "liabilities_to_assets":    {"value": 0.79,  "year": "2025"},
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

**SEC User-Agent (required for live requests).** The SEC requires requests to identify the requester. Set `SEC_USER_AGENT` in your environment or in a gitignored `.env` file in the project root:

```dotenv
SEC_USER_AGENT="Your Name your-email@example.com"
```

The parser can be imported without making network requests. Live requests fail with a configuration error if `SEC_USER_AGENT` is missing and use a 30-second timeout.

**LLM API key (required for qualitative analysis).** Add your OpenAI key to the gitignored `.env` file, for example:

```
OPENAI_API_KEY=your-key-here
FINSIGNAL_LLM_MODEL=gpt-4.1-mini
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
│   └── sec_filing_agent.py   # SEC report pipeline and JSON CLI
├── tools/
│   └── edgar_parser.py       # EDGAR retrieval, metrics, 10-K text extraction
├── DECISIONS.md              # Design-decision log with reasoning
└── requirements.txt
```

Planned additions: the remaining four agents, an orchestrator, a SQLite portfolio store, and a Streamlit frontend.

## Tests

Run the offline suite from the repository root with your virtual environment active:

```sh
python -m unittest discover -s tests -v
```

The suite uses synthetic company-facts fixtures and mocked HTTP responses. It covers the seven metrics, annual-series handling, tag fallbacks, missing values, zero denominators, request configuration, filing retrieval, HTML extraction, and qualitative schemas. It requires no SEC access or API credentials. Live SEC access remains a separate integration check.

To run the Walmart filing demo with SEC access configured:

```sh
python -m tools.edgar_parser
```

## Known limitations

- Revenue and net income coverage starts at 2021 by design. Full historical reconciliation is deferred.
- Revenue and net income use tag-fallback lists, so a company with an unusual tag may return `null` for that metric until its tag is added.
- ROE uses year-end equity rather than average equity.
- Extracted 10-K text still contains some harmless XBRL header noise.
- Validated on a handful of large US companies. Banks and other unusual balance sheets are untested.
- Offline automated tests use synthetic data; broad validation against live filings remains limited.
- In production, the parsing layer would likely be replaced by a commercial data vendor. It was built here to understand the problem directly.

## Roadmap

1. Validate the SEC Filing Agent against live SEC data and OpenAI responses
2. Validate Technical Agent live data access; indicators and offline CLI implemented
3. Sentiment Agent (FinBERT)
4. Risk Manager Agent
5. Portfolio Manager Agent and orchestration
6. Streamlit frontend and demo

## Disclaimer

FinSignal is an educational project. Nothing it outputs is investment advice.


## SEC analysis CLI

With `SEC_USER_AGENT` and `OPENAI_API_KEY` configured, run from the repository root:

```sh
python -m agents.sec_filing_agent AAPL
# Optional model override:
python -m agents.sec_filing_agent AAPL --model gpt-4.1-mini
```

The CLI writes a JSON report to stdout, or a short error to stderr with a nonzero
exit status. Requests use the configured model (`FINSIGNAL_LLM_MODEL`, default
`gpt-4.1-mini`); your account must have access to a model supporting Responses API
structured outputs. Live runs incur provider usage charges. No API key is needed
for the offline suite.

You can also call the agent directly:

```python
from agents.sec_filing_agent import analyze_sec_filing

report = analyze_sec_filing("AAPL")
print(report.model_dump_json(indent=2))
```

Reports include seven metrics, their fiscal years, the latest 10-K's date and URL,
a retrieval timestamp, qualitative analysis, model name, and warnings. Company
Facts can contain data newer than the selected annual filing; this is not a
historical point-in-time analysis. Missing metrics remain null.

The full extracted filing goes to the model, with no silent truncation. Filings
above 800,000 characters are rejected before an LLM call; Python callers can
explicitly adjust `max_filing_chars` for their selected model. This character
limit is only a size guard, not a token-count or context-window guarantee.
The filing is indexed into passages of at most 1,200 characters without dropping
or rewriting any extracted text. The LLM cites passage IDs; Python inserts those
exact source passages into each finding's `evidence` field. The public report
schema is unchanged, though evidence may now include a longer passage for context.
This guarantees that evidence text comes from the filing, but does not establish
that the passage supports the model's interpretation. Review the evidence alongside
the finding.

Unknown passage IDs trigger at most one corrective LLM request, with the source
passages and validation failures. This retry incurs additional provider usage.
Every citation is rechecked; a successful correction is noted in report warnings.
Provider/network errors and unresolved unknown IDs fail the report. No investment
decision is produced by this agent.

## Technical Agent

Install the updated dependencies in your active Python 3.12 virtual environment:

```sh
python -m pip install -r requirements.txt
python -m agents.technical_agent AAPL
```

No LLM or API key is required. The initial interpretation is generated from
calculated signals; LLM commentary can be added later without changing the math.
The CLI writes JSON to stdout or an error to stderr with exit status 1.

The shared price adapter requests two years of daily OHLCV from Yahoo Finance
through `yfinance`. Symbols such as `BRK.B` map to Yahoo's `BRK-B` while reports
retain the input symbol. Network access may require `query1.finance.yahoo.com`,
`query2.finance.yahoo.com`, and `fc.yahoo.com`; follow any provider redirects
when diagnosing your environment's policy. The cloud live check failed with a
proxy CONNECT 403, so live price retrieval has not been verified here.

Reports include source, currency, retrieval time, latest bar date, history length,
raw and adjusted latest close, warnings, and the following indicators:

| Indicator | Convention |
|---|---|
| RSI(14) | Wilder smoothing; needs 15 closes; flat=50, gains only=100, losses only=0 |
| SMA(50), SMA(200) | Arithmetic average of adjusted closes |
| MACD(12,26,9) | SMA-seeded EMAs; signal and histogram available after 34 closes |
| Relative volume | Latest volume divided by the preceding 20 sessions' average, excluding the latest session |
| Trailing range | Minimum adjusted low and maximum adjusted high over 20 sessions |

Price indicators use split- and dividend-adjusted closes. Daily highs/lows are
scaled by that day's adjusted-close/close ratio for the trailing range. These
levels are on the adjusted basis, not necessarily historical executable prices.
Volume uses provider-reported units. The current exchange-local date is always
excluded, even after market close, to avoid partial-session contamination. It
becomes eligible on the following day. Missing, nonfinite, duplicate, unsorted,
or inconsistent OHLCV inputs fail validation rather than being filled silently.
Short histories produce null indicators and explicit warnings. Dates more than
seven calendar days old are flagged as potentially stale. There is no exchange
calendar reconciliation, persistent analysis cache, or cross-provider verification.

Moving-average and MACD crosses describe events between the previous and latest
bars. Being above a moving average alone is not reported as a new crossover.
Trend is up only when adjusted price is above both averages and SMA50 is above
SMA200; the reverse is a downtrend, other complete cases are mixed. RSI thresholds
are 70/30. These rules describe historical data and do not issue BUY/HOLD/SELL
recommendations or calibrated forecasts. Trailing extremes are not proven
support/resistance levels.

For offline development, use the synthetic, steadily rising fixture:

```sh
python -m agents.technical_agent TEST --prices-csv tests/fixtures/prices.csv --currency USD
python -m unittest discover -s tests -v
```

CSV input requires `date,open,high,low,close,adjusted_close,volume`, in ascending
unique date order. Adjustments, symbol identity, and session completeness of a CSV
are the caller's responsibility. The fixture is synthetic and intentionally old,
so a stale-data warning is expected.

The price history can be fetched once and shared with future agents:

```python
from tools.price_data import fetch_price_history
from agents.technical_agent import analyze_technical

history = fetch_price_history("AAPL")
technical_report = analyze_technical("AAPL", history=history)
# A future risk agent can consume the same history without downloading it again.
```

### SEC ratio naming correction

The SEC output keys and Pydantic fields formerly named `debt_to_equity` and
`debt_to_assets` are now `liabilities_to_equity` and `liabilities_to_assets`.
The corresponding parser functions are now `calculate_liabilities_to_equity`
and `calculate_liabilities_to_assets`. Calculations are unchanged: the numerator
is total liabilities (assets minus equity), not interest-bearing debt. This is an
intentional breaking schema correction; update consumers of the old names.
