# Design Decisions

A running log of non-obvious engineering and architectural choices made while
building FinSignal, with the reasoning behind each. The goal is that future-me
(or a collaborator) can see *why* a choice was made, not just *what* the code does.

Format: each entry states the decision, the reasoning, and the alternative that
was rejected. Newest entries at the top.

---

## SEC Filing Agent — Qualitative Layer (planned)

### Use prompt-steering with full text, not code-based section parsing
**Decision:** For the qualitative analysis of 10-K filings, strip the document
to plain text and feed it to the LLM with a structured, section-targeted prompt
(naming Item 1A Risk Factors and Item 7 MD&A, asking for specific JSON outputs).
Do NOT write code to parse out individual labeled sections.

**Reasoning:** Code-based section extraction is the most fragile part of the
agent — 10-K HTML formatting varies wildly per company, so locating section
boundaries reliably is hard and brittle. Modern LLMs have large context windows
(200K+ tokens) that comfortably fit a full 10-K (~100-150K tokens) and already
understand 10-K structure from training, so they can locate and focus on the
relevant sections themselves when instructed. This leverages the LLM's strengths
to avoid the code's weaknesses. A structured prompt (fill specific slots: top
risks, MD&A themes, red flags) forces engagement with the target sections AND
produces schema-ready JSON output.

**Known limitation (accepted for v1):** Prompt-steering *focuses* attention but
does not *isolate* — the full document is still in context, so some signal
dilution ("lost in the middle") and full-token cost remain. True code extraction
would be more precise and cheaper per call. Accepted because: (a) cost is
negligible at our usage scale, (b) the simplicity gain is large, (c) we can add
crude keyword-based section-grabbing later as a refinement if analysis quality
disappoints.

**Rejected alternative:** Parsing exact section boundaries with BeautifulSoup +
heading detection. Too fragile for the precision gain, given large context
windows make it unnecessary for v1.

**Note:** The one transformation still needed is HTML -> plain text (strip tags,
~2 lines of BeautifulSoup). Feeding raw HTML would waste tokens on markup and
add noise; feed stripped text instead.

---

## SEC Filing Agent — Data Layer (`tools/edgar_parser.py`)

### Restrict revenue analysis to recent years (2021+) for v1
**Decision:** `get_revenue` filters out fiscal years before 2021 by default
(`min_year=2021`).

**Reasoning:** Apple's (and most companies') revenue is tagged under multiple
XBRL concepts that changed over time, and individual tags contain mistagged
sub-component figures in older filings. This produced nonsense growth numbers
(e.g. -73% then +527%). The ASC 606 accounting standard (~2018) standardized
revenue reporting, so post-2021 data is consistent. Slicing to recent years
gives *correct* data and unblocks building the rest of the agent, rather than
stalling on full historical reconciliation.

**Rejected alternative:** Building a complete historical tag-reconciliation
engine now. Too much data-janitorial effort for v1; the priority is learning
the pipeline and getting a working end-to-end agent. Documented as a TODO to
revisit during a later hardening pass.

### Build revenue extraction as a resolver, not a hardcoded tag
**Decision:** `get_revenue` tries a priority list of candidate tags
(`REVENUE_TAGS`) in order and uses the first that returns data, rather than
hardcoding a single tag.

**Reasoning:** XBRL tag choice varies *per company* — Apple, Microsoft, and a
bank may each tag revenue differently. A hardcoded tag breaks on the second
company. The resolver encodes general knowledge about how revenue is tagged,
not specific knowledge about one company. Validated by testing on both Apple
and Microsoft without code changes — both returned clean data.

**Rejected alternative:** Hardcoding Apple's tag. Would not generalize; the
whole point of the agent is to analyze arbitrary companies.

### Key annual values by period-end date year, not the `fy` field
**Decision:** In `get_annual_values`, the fiscal year is derived from the
period-end date (`date[:4]`) rather than the XBRL `fy` field.

**Reasoning:** The `fy` field was consistently one year ahead of the actual
period-end date, because XBRL sometimes tags a *comparative* prior-year figure
with the *filing's* fiscal year. The period-end date is the trustworthy "as-of"
signal. (Note: this overturned an earlier instinct to use `fy` for robustness —
the actual data revealed `fy` carried its own ambiguity.)

**Rejected alternative:** Using `fy` directly. Produced year/date misalignment
in the output.

### Deduplicate with a dict keyed by year, last-write-wins
**Decision:** Duplicate dates are collapsed by keying a dict on the year and
letting later assignments overwrite earlier ones (no "if not present" guard).

**Reasoning:** The same figure is reported across multiple filings, creating
duplicate dates. A dict allows one value per key, so duplicates collapse
automatically. Last-wins is preferred because later entries in the data tend to
come from newer filings with more up-to-date (possibly restated) figures.

**Rejected alternative:** First-wins (guarding with `if year not in dict`).
Would keep the oldest reported version of each figure instead of the freshest.

### Use EAFP (try/except) over LBYL (check-first) for nested access
**Decision:** Functions that navigate deep into the facts structure
(`get_concept_values`, `get_revenue`) wrap the access in try/except rather than
checking each key exists first.

**Reasoning:** A single existence check often can't guard every step of a
multi-level access (`facts["facts"]["us-gaap"][concept]["units"]["USD"]` can
fail at several points). One try/except catches a failure anywhere along the
chain and converts it to a clear error message. This is the Pythonic default.

**Rejected alternative:** Checking `if concept in ...` before accessing. Only
guards the first key, leaving deeper keys (e.g. `"USD"`) unguarded.

---

## Architecture & Project Setup

### Layered, single-responsibility functions
**Decision:** Each function does one thing and consumes the previous one's
output: extract → clean → compute → resolve.

**Reasoning:** Small functions that pass clean data forward are testable in
isolation and easy to debug. When a bug appeared in `calculate_growth`, it could
be tested independently because it didn't tangle with fetching or cleaning.

### Annual (10-K) data as the workhorse for v1 math
**Decision:** Nearly all quantitative metrics are computed from the annual
series; quarterly (10-Q) data is fetched but mostly set aside.

**Reasoning:** Flow metrics like revenue need consistent year-long windows to be
comparable. Annual 10-K data is audited, authoritative, and gives clean
year-over-year comparisons. Quarterly data adds value for recency and momentum
signals but introduces windowing and seasonality complexity — deferred to v2.

### Constants at module level, logic in functions
**Decision:** Configuration like `USER_AGENT`, `TICKERS_URL`, and `REVENUE_TAGS`
lives at the top of the file, not inside functions.

**Reasoning:** Tunable configuration belongs where it's findable, not buried in
logic. Adding a revenue tag shouldn't require digging inside a function body.

### Develop in VS Code, not Colab
**Decision:** The project is built as a proper software project in VS Code with
Git, a virtual environment, and a multi-file structure — not in notebooks.

**Reasoning:** FinSignal is a multi-file system with cross-imports, a database,
a Streamlit frontend, and tests. Notebooks fight all of that. Notebooks remain
useful for throwaway exploration (e.g. the revenue tag diagnostic) but the
product itself lives in `.py` files under version control.

---

## Lessons / Recurring Themes

- **Indentation is control flow in Python.** A misplaced `return` doesn't
  crash — it silently returns wrong results. Caught the `calculate_growth`
  one-item-list bug with print markers that made loop progress visible.
- **Make the invisible visible when debugging.** Both the growth bug and the
  revenue bug were caught by printing intermediate state, not by reading code.
- **Sanity-check numbers against reality.** Domain knowledge (Apple's revenue is
  ~$390B and steady) caught a bug the code couldn't. The math being auditable is
  good, but the *inputs* to the math need validation too.
- **Test generalization, not just correctness.** The Microsoft test was the real
  proof of the resolver. Getting something working on one company says nothing
  about whether it generalizes.
- **Real data is messy, and that's where the engineering is.** The hardest part
  of the SEC agent is normalizing inconsistent XBRL into clean, comparable
  numbers — not the agent logic itself.
