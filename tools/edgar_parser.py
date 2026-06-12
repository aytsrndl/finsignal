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

if __name__ == "__main__":
    test_ticker = "AAPL"
    cik = get_cik_from_ticker(test_ticker)
    print(f"CIK for {test_ticker}: {cik}")

    print()

    filings = get_recent_filings(cik)
    print(f"Recent 10K/10Q filings for {test_ticker} (CIK: {cik}):")
    for f in filings:
        print(f"- {f['form_type']} filed on {f['filing_date']} (Accession: {f['accession_number']}, Document: {f['primary_document']})")