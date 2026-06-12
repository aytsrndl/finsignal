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
    """
    response = requests.get(TICKERS_URL, headers={"User-Agent": USER_AGENT})
    response.raise_for_status()
    tickers_data = response.json()

    # The data is a dict of dicts, keyed by index, each with 'ticker' and 'cik_str'
    for entry in tickers_data.values():
        if entry['ticker'].lower() == ticker.lower():
            return str(entry['cik_str']).zfill(10) # Ensure CIK is zero-padded to 10 digits

    raise ValueError(f"Ticker '{ticker}' not found in SEC mapping.")

if __name__ == "__main__":
    test_tickers = ["AAPL", "MSFT", "GOOGL", "NVDA"]
    for t in test_tickers:
        cik = get_cik_from_ticker(t)
        print(f"{t}: {cik}")