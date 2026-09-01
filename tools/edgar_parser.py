"""
EDGAR Parser - tools for fetching and parsing EDGAR data
"""

import requests
import json
from bs4 import BeautifulSoup
import re
from openai import OpenAI
from pydantic import BaseModel, Field

# SEC requires a header for request identification
# Format: "Your Name your-email@domain.com"
USER_AGENT = "Aytunc Sarandal aytsrndl@gmail.com"

# SEC's public ticker-to-CIK mapping file
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

#Example of revenue tags to check in order of priority
REVENUE_TAGS = [
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "Revenues",
]

NET_INCOME_TAGS = [
    "NetIncomeLoss",
    "ProfitLoss",            # occasional alternative
]

def get_cik_from_ticker(ticker):
    """
    Fetch the 10-digit zero-padded CIK for a given ticker symbol
    from SEC's ticker mapping file.

    Args:
    ticker: Stock ticker symbol (e.g., "AAPL")

    Returns:
    10-digit zero-padded CIK as a string (e.g., "0000320193")

    Raises:
    ValueError: If the ticker is not found in the SEC mapping.
    """
    response = requests.get(TICKERS_URL, headers={"User-Agent": USER_AGENT})
    response.raise_for_status()
    tickers_data = response.json()

    # The data is a dict of dicts, keyed by index, each with 'ticker' and 'cik_str'
    for entry in tickers_data.values():
        if entry['ticker'].lower() == ticker.lower():
            return str(entry['cik_str']).zfill(10) # Ensure CIK is zero-padded to 10 digits

    raise ValueError(f"Ticker '{ticker}' not found in SEC mapping.")

def get_recent_filings(cik, form_types = ("10-K", "10-Q"), limit=10):
    """
    Fetch recent filings for a given CIK and filter by form types.

    Args:
    cik: 10-digit zero-padded CIK as a string (e.g., "0000320193")
    form_types: Tuple of form types to filter (default: ("10-K", "10-Q"))
    count: Number of recent filings to return (default: 10)

    Returns:
    List of dictionaries with keys: 'form_type', 'filing_date', 'accession_number', 'primary_document'.
    """
    # SEC's EDGAR API endpoint for company filings
    EDGAR_API_URL = f"https://data.sec.gov/submissions/CIK{cik}.json"

    response = requests.get(EDGAR_API_URL, headers={"User-Agent": USER_AGENT})
    response.raise_for_status()
    data = response.json()


    recent = data["filings"]["recent"]
    forms = recent["form"]
    
    filings = []
    for i in range(len(forms)):
        if recent["form"][i] in form_types:
            filings.append({
            "form_type":    recent["form"][i],
            "filing_date":  recent["filingDate"][i],
            "accession_number": recent["accessionNumber"][i],
            "primary_document": recent["primaryDocument"][i],
            })
        if len(filings) >= limit:
            break
    return filings

    
def build_filing_url(cik, accession_number, primary_document):
        """
        Construct the URL to access the filing document on SEC's EDGAR system.

        Args:
        cik: 10-digit zero-padded CIK as a string (e.g., "0000320193")
        accession_number: Accession number of the filing (e.g., "0000320193-23-000010")
        primary_document: Primary document name (e.g., "a10-k20221231.htm")

        Returns:
        URL string to access the filing document.
        """
        base_url = "https://www.sec.gov/Archives/edgar/data"
        cik_no_padding = cik.lstrip("0")  # Remove leading zeros for URL
        return f"{base_url}/{cik_no_padding}/{accession_number.replace('-', '')}/{primary_document}"

def get_company_facts(cik):
    """
    Fetch the company facts data for a given CIK.

    Args:
    cik: 10-digit zero-padded CIK as a string (e.g., "0000320193")

    Returns:
    Dictionary containing company facts data.
    """
    FACTS_API_URL = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    response = requests.get(FACTS_API_URL, headers = {"User-Agent": USER_AGENT})
    response.raise_for_status()
    data = response.json()
    return data

def get_concept_values(facts, concept):
    """
    Extract a clean time-series value for a specific financial concept from the company facts data.

    Args:
    facts: Dictionary containing company facts data (from get_company_facts)
    concept: The financial concept to extract (e.g., "Assets", "Liabilities")

    Returns:
    List of dicts with 'date' and 'value' keys.

    Raises:
    KeyError: If the concept is not found in the facts data.
    """
    try:
        concept_data = facts["facts"]["us-gaap"][concept]["units"]["USD"]
    except KeyError:
        raise KeyError(f"Concept '{concept}' not found in company facts data.")
    
    values = []
    for entry in concept_data:
        values.append({
            "date": entry["end"],
            "value": entry["val"],
            "fy": entry.get("fy"),       # fiscal year
            "fp": entry.get("fp"),       # fiscal period ("FY", "Q1", etc.)
            "form": entry.get("form"),
        })
    return values

def get_annual_values(facts, concept):
    """
    Extract clean, deduplicated annual values for a specific financial concept from the company facts data.

    Keeps only annual figures (from 10K filings, full-year period),
    removes duplicate years and sorts chronologically,

    Args:
    facts: Dictionary containing company facts data (from get_company_facts)
    concept: The financial concept to extract (e.g., "Assets", "Liabilities")

    Returns:
    List of dicts with 'year', 'date', and 'value' keys, sorted by year ascending.
    """
    concept_data = get_concept_values(facts, concept)
    annual_values = {}
    for entry in concept_data:
        if entry.get("fp") == "FY" and entry.get("form") == "10-K":  # Only consider full-year data
            year = entry["date"][:4]  # Extract year from date string
            if year not in annual_values:
                annual_values[year] = {
                    "year": year,
                    "date": entry["date"],
                    "value": entry["value"]
                }
    result = sorted(annual_values.values(), key=lambda x: x["year"])
    return result

def calculate_growth(annual_values):
    """
    Calculate year-over-year growth rates for a list of annual values.

    Args:
    annual_values: List of dicts with 'year', 'date', and 'value' keys, sorted by year ascending.

    Returns:
    List of dicts with 'year', 'date', 'value', and 'growth' keys, where 'growth' is the YoY growth rate.
    """
    growth_data = []
    previous_value = None
    for entry in annual_values:
        current_value = entry["value"]
        growth = None
        if previous_value is not None and previous_value != 0:
            growth = (current_value - previous_value) / previous_value
        growth_data.append({
            "year": entry["year"],
            "date": entry["date"],
            "value": current_value,
            "growth": growth
        })
        previous_value = current_value
    return growth_data


def get_revenue(facts, min_year = 2021):
    """ 
    Resolve a companies annual revenue using a priority list of XBRL tags. Restricted to recent years (default: 2021 and later).

    Tries each tag in REVENUE_TAGS in order, returning the first one that has data for the requested years.

    Args:
    facts: Dictionary containing company facts data (from get_company_facts)
    min_year: Minimum year to consider for revenue data (default: 2021)

    Returns:
    List of dicts with 'year', 'date', and 'value' keys, sorted by year ascending.

    Raises:
    KeyError:If none of the candidate tags are found.
    """
    for tag in REVENUE_TAGS:
        try:
            annual_values = get_annual_values(facts, tag)
        except KeyError:
            continue   # this tag isn't present; try the next one

        filtered_values = [v for v in annual_values if int(v["year"]) >= min_year]
        if filtered_values:
           return filtered_values

    raise KeyError(f"No revenue data found for any of the candidate tags: {REVENUE_TAGS}")

def calculate_ratio(facts, numerator_concept, denominator_concept, min_year=None):
    """
    Calculate a financial ratio for all years where both numerator and
    denominator are available.

    Args:
        facts: Dictionary containing company facts data (from get_company_facts)
        numerator_concept: The us-gaap concept for the numerator.
        denominator_concept: The us-gaap concept for the denominator.
        min_year: If set, only include years >= this value (default None = all years).

    Returns:
        List of dicts with 'year' and 'ratio', sorted by year ascending.
        Years where the denominator is zero are skipped.
    """
    numerator_values = get_annual_values(facts, numerator_concept)
    denominator_values = get_annual_values(facts, denominator_concept)

    denominator_by_year = {d["year"]: d["value"] for d in denominator_values}
    ratios = []
    for n in numerator_values:
        year = n["year"]
        if min_year is not None and int(year) < min_year:   # NEW: year filter
            continue
        if year not in denominator_by_year:
            continue
        denominator_value = denominator_by_year[year]
        if denominator_value == 0:
            continue
        ratio = n["value"] / denominator_value
        ratios.append({"year": year, "ratio": ratio})

    return sorted(ratios, key=lambda x: x["year"])

def get_net_income(facts, min_year = 2021):
    """
    Get annual net income (or loss) values for a company, using a priority list of XBRL tags.

    Tries each tag in NET_INCOME_TAGS in order, returning the first one that has data for the requested years.

    Args:
    facts: Dictionary containing company facts data (from get_company_facts)
    min_year: Minimum year to consider for net income data (default: 2021)

    Returns:
    List of dicts with 'year', 'date', and 'value' keys, sorted by year ascending.

    Raises:
    KeyError:If none of the candidate tags are found.
    """
    for tag in NET_INCOME_TAGS:
        try:
            annual_values = get_annual_values(facts, tag)
        except KeyError:
            continue   # this tag isn't present; try the next one

        filtered_values = [v for v in annual_values if int(v["year"]) >= min_year]
        if filtered_values:
           return filtered_values

    raise KeyError(f"No net income data found for any of the candidate tags: {NET_INCOME_TAGS}")

def calculate_net_margin(facts):
    """
    Calculate net margin (net income / revenue) for all years where both are available.

    Uses get_revenue and get_net_income to resolve the appropriate tags for each.

    Returns:
        List of dicts with 'year' and 'net_margin', sorted by year ascending.
    """
    revenue_data = get_revenue(facts)
    net_income_data = get_net_income(facts)

    revenue_by_year = {r["year"]: r["value"] for r in revenue_data}
    net_margin = []
    for ni in net_income_data:
        year = ni["year"]
        if year not in revenue_by_year:
            continue
        revenue_value = revenue_by_year[year]
        if revenue_value == 0:
            continue
        margin = ni["value"] / revenue_value
        net_margin.append({"year": year, "net_margin": margin})

    return sorted(net_margin, key=lambda x: x["year"])

def get_total_liabilities(facts):
    """
    Derive total liabilities per year from the accounting identity:
    Liabilities = Assets - StockholdersEquity.

    More robust than a 'Liabilities' tag lookup, since not all companies
    report total liabilities under that concept, but nearly all report
    Assets and StockholdersEquity.

    Args:
        facts: The full facts dict from get_company_facts().

    Returns:
        List of dicts with 'year', 'date', 'value', sorted by year ascending.
    """
    assets = get_annual_values(facts, "Assets")
    equity = get_annual_values(facts, "StockholdersEquity")

    equity_by_year = {e["year"]: e["value"] for e in equity}
    liabilities = []
    for a in assets:
        year = a["year"]
        if year not in equity_by_year:
            continue
        liab_value = a["value"] - equity_by_year[year]
        liabilities.append({
            "year": year,
            "date": a["date"],
            "value": liab_value,
        })

    return sorted(liabilities, key=lambda x: x["year"])

def _ratio_from_series(numerator_series, denominator_series, min_year=None):
    """
    Divide two pre-computed annual series, aligned by year. Internal helper
    for ratios whose inputs are derived/resolved rather than raw tags.

    Args:
        numerator_series: list of {"year", "value", ...}
        denominator_series: list of {"year", "value", ...}
        min_year: if set, only include years >= this value.

    Returns:
        List of dicts with 'year' and 'ratio', sorted by year ascending.
    """
    denom_by_year = {d["year"]: d["value"] for d in denominator_series}
    ratios = []
    for n in numerator_series:
        year = n["year"]
        if min_year is not None and int(year) < min_year:
            continue
        if year not in denom_by_year:
            continue
        denom_value = denom_by_year[year]
        if denom_value == 0:
            continue
        ratios.append({"year": year, "ratio": n["value"] / denom_value})

    return sorted(ratios, key=lambda x: x["year"])

def calculate_debt_to_equity(facts):
    """
    Calculate the Debt-to-Equity ratio for all years available.

    """
    liabilities = get_total_liabilities(facts)
    equity = get_annual_values(facts, "StockholdersEquity")
    return _ratio_from_series(liabilities, equity)

def calculate_current_ratio(facts):
    """
    Calculate the Current Ratio (current assets / current liabilities) for all years available.

    """
    return calculate_ratio(facts, "AssetsCurrent", "LiabilitiesCurrent")

def calculate_debt_to_assets(facts):
    """
    Calculate the Debt-to-Assets (total liabilities / total assets) ratio for all years available.

    """
    liabilities = get_total_liabilities(facts)
    assets = get_annual_values(facts, "Assets")
    return _ratio_from_series(liabilities, assets)

def calculate_roe(facts):
    """Return on equity (net income / stockholders' equity) per year."""
    return calculate_ratio(facts, "NetIncomeLoss", "StockholdersEquity", min_year=2021)

def _latest(series, value_key):
    """
    Return the most recent {value, year} from a time series of annual data. 
    or {value: None, year: None} if the series is empty.

    Args:
        series: List of dicts with 'year' and value_key.
        value_key: The key in the dict to extract the value from.
    """
    if not series:
        return {"value": None, "year": None}
    latest = series[-1]
    return {"value": latest[value_key], "year": latest["year"]}

def build_sec_metrics(ticker):
    """
    Assemble all quanitative metrics for a given ticker into a single dictionary.
    
    Each metric reports it's most recent value along with the fiscal year that value is from.
    Metrics that can't be computed are recorded as None.

    Args:
        ticker: Stock ticker symbol (e.g., "AAPL")

    Returns:
        Dictionary of metrics with keys:
            - revenue
            - net_income
            - net_margin
            - debt_to_equity
            - current_ratio
            - debt_to_assets
            - roe
    """
    cik = get_cik_from_ticker(ticker)
    facts = get_company_facts(cik)

    def safe(fn, value_key, *args):
        """Run a metric function, return latest {value, year}, or None on failure."""
        try:
            return _latest(fn(*args), value_key)
        except Exception:
            return {"value": None, "year": None}
    
    return {
        "ticker": ticker,
        "entity_name": facts.get("entityName"),
        "revenue_growth":    safe(lambda f: calculate_growth(get_revenue(f)), "growth", facts),
        "net_income_growth": safe(lambda f: calculate_growth(get_net_income(f)), "growth", facts),
        "net_margin":        safe(calculate_net_margin, "net_margin", facts),
        "debt_to_equity":    safe(calculate_debt_to_equity, "ratio", facts),
        "debt_to_assets":    safe(calculate_debt_to_assets, "ratio", facts),
        "current_ratio":     safe(calculate_current_ratio, "ratio", facts),
        "roe":               safe(calculate_roe, "ratio", facts),
    }

def get_latest_10k_html(cik):
    """
    Fetch the most recent 10-K filing for a given CIK and return the HTML content of the primary document.

    Args:
        cik: 10-digit zero-padded CIK as a string (e.g., "0000320193")

    Returns:
        The raw HTML content of the most recent 10-K filing's primary document, or (None, None) if no 10-K filings are found.
    """
    filings = get_recent_filings(cik, form_types=("10-K",))
    if not filings:
        raise ValueError(f"No 10-K filings found for CIK {cik}")
    latest = filings[0]
    url = build_filing_url(cik, latest["accession_number"], latest["primary_document"])
    response = requests.get(url, headers={"User-Agent": USER_AGENT})
    response.raise_for_status()
    return response.text

def html_to_text(html):
    """
    Convert raw HTML content to clean text using BeautifulSoup.

    Args:
        html: Raw HTML string.

    Returns:
        Clean text extracted from the HTML.
    """
    soup = BeautifulSoup(html, "html.parser")
    
    for element in soup(["script", "style"]):
        element.decompose()
    text = soup.get_text(separator=" ")
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"\b\w+:\S+", " ", text)
    text = " ".join(text.split())

    return text


html = get_latest_10k_html(get_cik_from_ticker("WMT"))
print(f"Raw HTML: {len(html):,} characters")

text = html_to_text(html)
print(f"Stripped text: {len(text):,} characters")
print(f"\nFirst 500 chars of text:\n{text[:500]}")


