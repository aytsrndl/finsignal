"""
EDGAR Parser - tools for fetching and parsing EDGAR data
"""

import requests

# SEC requires a header for request identification
# Format: "Your Name your-email@domain.com"
USER_AGENT = "Aytunc Sarandal aytsrndl@gmail.com"

# SEC's public ticker-to-CIK mapping file
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

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
    List of dictionaries with keys: 'form_type', 'filing_date', 'filing_url'
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
      


if __name__ == "__main__":
    test_tickers = ["AAPL", "MSFT", "GOOGL"]
    for t in test_tickers:
        cik = get_cik_from_ticker(t)
        print(f"{t} -> CIK: {cik}")
    
    print()

    apple_cik = get_cik_from_ticker("AAPL")

    filings = get_recent_filings(apple_cik)
    print(f"Recent 10k/10Q filings for AAPL (CIK {apple_cik}):")
    for f in filings:
        url = build_filing_url(apple_cik, f["accession_number"], f["primary_document"])
        print(f"{f['form_type']} filed on {f['filing_date']} - {url}")
    
    print()

    facts = get_company_facts(apple_cik)
    print(f"Entity: {facts['entityName']}")
    print(f"Number of us-gaap concepts: {len(facts['facts']['us-gaap'])}")

    assets_values = get_concept_values(facts, "Assets")
    print(f"Assets time series for AAPL:")
    for v in assets_values:
        print(f"Date: {v['date']}, Value: {v['value']}")

    annual_assets = get_annual_values(facts, "Assets")
    print(f"Annual Assets for AAPL:")
    for v in annual_assets:
        print(f"Year: {v['year']}, Date: {v['date']}, Value: {v['value']}")

    growth_assets = calculate_growth(annual_assets)
    print("\n=== GROWTH TEST START ===")
    for v in growth_assets:
        if v["growth"] is None:
            print(f"  {v['year']}: ${v['value']:,}  (baseline)")
        else:
            print(f"  {v['year']}: ${v['value']:,}  ({v['growth']:+.1%})")
    print("=== GROWTH TEST END ===")
