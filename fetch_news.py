# ==============================================================================
# SETUP INSTRUCTIONS
# ==============================================================================
# 1. Ensure you have Python 3.8 or higher installed.
# 2. It is highly recommended to use a virtual environment:
#    python -m venv venv
#    source venv/bin/activate  # On Windows use: venv\Scripts\activate
# 3. Install the required dependencies by running the following command in your terminal:
#
#    pip install newspaper4k lxml_html_clean duckduckgo-search jupyter_client gnews yfinance pandas
#
# ==============================================================================

# --- Cell 1: Imports and Environment Setup ---
import shutil
import os
import yfinance as yf
import pandas as pd
import json
import time
import warnings
import logging
from datetime import datetime, timedelta
from contextlib import redirect_stderr
from ddgs import DDGS
from gnews import GNews
from newspaper import Article

# Suppress warnings and logs for clean terminal output
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
logging.getLogger('yfinance').setLevel(logging.CRITICAL)


# --- Cell 2: Directory Management ---
# NOTE: The original Kaggle paths have been updated for local/GitHub usage.
# If you are pulling in an existing dataset, place it in the 'input_dir'.

input_dir = "./input_data"
working_dir = "./working_data"

# Create working directory if it doesn't exist
os.makedirs(working_dir, exist_ok=True)

for file_name in ["nifty50_train_data.jsonl", "completed_tickers.txt"]:
    src = os.path.join(input_dir, file_name)
    dst = os.path.join(working_dir, file_name)
    if os.path.exists(src):
        shutil.copy(src, dst)
        print(f"Copied {file_name} to working directory.")
    else:
        print(f"Could not find {file_name} in Input. It will be created during runtime.")


# --- Cell 3: Configuration ---
NIFTY50_MAP = {
    "ADANIENT.NS": "Adani Enterprises",
    "ADANIPORTS.NS": "Adani Ports",
    "APOLLOHOSP.NS": "Apollo Hospitals",
    "ASIANPAINT.NS": "Asian Paints",
    "AXISBANK.NS": "Axis Bank",
    "BAJAJ-AUTO.NS": "Bajaj Auto",
    "BAJFINANCE.NS": "Bajaj Finance",
    "BAJAJFINSV.NS": "Bajaj Finserv",
    "BEL.NS": "Bharat Electronics",
    "BHARTIARTL.NS": "Bharti Airtel",
    "BPCL.NS": "Bharat Petroleum",
    "BRITANNIA.NS": "Britannia Industries",
    "CIPLA.NS": "Cipla",
    "COALINDIA.NS": "Coal India",
    "DIVISLAB.NS": "Divi's Laboratories",
    "DRREDDY.NS": "Dr. Reddy's Laboratories",
    "EICHERMOT.NS": "Eicher Motors",
    "GRASIM.NS": "Grasim Industries",
    "HCLTECH.NS": "HCL Technologies",
    "HDFCBANK.NS": "HDFC Bank",
    "HDFCLIFE.NS": "HDFC Life",
    "HEROMOTOCO.NS": "Hero MotoCorp",
    "HINDALCO.NS": "Hindalco Industries",
    "HINDUNILVR.NS": "Hindustan Unilever",
    "ICICIBANK.NS": "ICICI Bank",
    "INDUSINDBK.NS": "IndusInd Bank",
    "INFY.NS": "Infosys",
    "ITC.NS": "ITC Limited",
    "JSWSTEEL.NS": "JSW Steel",
    "KOTAKBANK.NS": "Kotak Mahindra Bank",
    "LT.NS": "Larsen and Toubro",
    "M&M.NS": "Mahindra and Mahindra",
    "MARUTI.NS": "Maruti Suzuki",
    "NESTLEIND.NS": "Nestle India",
    "NTPC.NS": "NTPC Limited",
    "ONGC.NS": "ONGC",
    "POWERGRID.NS": "Power Grid Corporation",
    "RELIANCE.NS": "Reliance Industries",
    "SBILIFE.NS": "SBI Life Insurance",
    "SBIN.NS": "State Bank of India",
    "SHRIRAMFIN.NS": "Shriram Finance",
    "SUNPHARMA.NS": "Sun Pharma",
    "TATACONSUM.NS": "Tata Consumer Products",
    "TATAMOTORS.NS": "Tata Motors",
    "TATASTEEL.NS": "Tata Steel",
    "TCS.NS": "Tata Consultancy Services",
    "TECHM.NS": "Tech Mahindra",
    "TITAN.NS": "Titan Company",
    "ULTRACEMCO.NS": "UltraTech Cement",
    "WIPRO.NS": "Wipro"
}

QUERIES = [
    # 1. Core Financials
    "earnings", "revenue", "profit", "loss", "guidance", "quarter results", "dividend",
    # 2. Analyst Actions
    "target price", "upgrade", "downgrade",
    # 3. Corporate Actions & Strategy
    "acquisition", "merger", "partnership", "expansion", "contract won",
    # 4. Management & Regulatory Shocks
    "resignation", "probe", "penalty", "lawsuit", "SEBI"
]


# --- Cell 4: Helper Functions ---
def _normalize_to_date(value):
    try:
        ts = pd.Timestamp(value)
        if pd.isna(ts): return None
        return ts.date()
    except: return None

def _download_close_series(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
    if data.empty or "Close" not in data: return pd.Series(dtype="float64")
    close = data["Close"]
    if isinstance(close, pd.DataFrame): close = close.iloc[:, 0]
    return close.dropna().copy().set_axis(pd.to_datetime(close.dropna().index).normalize())

def calculate_3_day_alpha(ticker, date):
    """Calculates 3-trading-day forward alpha vs ^NSEI."""
    base_date = _normalize_to_date(date)
    if not base_date: return None

    start_date = (base_date - timedelta(days=10)).isoformat()
    end_date = (base_date + timedelta(days=35)).isoformat()

    try:
        stock_close = _download_close_series(ticker, start_date, end_date)
        bench_close = _download_close_series("^NSEI", start_date, end_date)
    except: return None

    if stock_close.empty or bench_close.empty: return None

    cutoff = pd.Timestamp(base_date)
    stock_fwd = stock_close[stock_close.index > cutoff]
    bench_fwd = bench_close[bench_close.index > cutoff]
    common_dates = stock_fwd.index.intersection(bench_fwd.index).sort_values()

    if len(common_dates) < 3: return None

    s_start, s_end = stock_fwd.loc[common_dates[:3]].iloc[0], stock_fwd.loc[common_dates[:3]].iloc[-1]
    b_start, b_end = bench_fwd.loc[common_dates[:3]].iloc[0], bench_fwd.loc[common_dates[:3]].iloc[-1]

    if s_start <= 0 or b_start <= 0: return None
    return float(((s_end - s_start) / s_start) - ((b_end - b_start) / b_start))


# --- Cell 5: Main Scraping Logic ---
def fetch_and_parse_news(ticker, company_name):
    """Uses Google News with a Bulletproof Fallback to RSS Summaries."""
    records = []
    seen_urls = set()
    
    print(f"\n--- Scraping data for {company_name} ({ticker}) ---")
    
    # Initialize GNews: Locks search to India (IN) and English (en)
    google_news = GNews(language='en', country='IN', max_results=50)    
    for q in QUERIES:
        search_term = f'"{company_name}" {q}'
        print(f"Searching: {search_term}")
        
        try:
            results = google_news.get_news(search_term)
        except Exception as e:
            print(f"  [!] GNews Error: {e}")
            time.sleep(5)
            continue
            
        if not results:
            print("  [-] No results found.")
            continue
            
        print(f"  [*] Found {len(results)} raw articles. Processing...")
        saved_count = 0
        
        for r in results:
            url = r.get("url")
            pub_date = _normalize_to_date(r.get("published date"))
            title = r.get("title", "")
            description = r.get("description", "")
            
            # Handle the publisher dictionary safely
            publisher_info = r.get("publisher")
            publisher = publisher_info.get("title", "Unknown Source") if isinstance(publisher_info, dict) else str(publisher_info)
            
            if not url or url in seen_urls: continue
            seen_urls.add(url)
            
            if not pub_date:
                continue
            
            # 1. Check Alpha
            alpha_val = calculate_3_day_alpha(ticker, pub_date)
            if alpha_val is None:
                continue
            
            true_alpha = 1 if alpha_val > 0 else 0
            
            # 2. Establish our Fallback Baseline (We ALWAYS have this data)
            # This ensures we never lose a valid labeled data point due to paywalls
            text_to_save = f"{title} | {publisher} | {description}"
            
            # 3. Try to get the full article, but don't panic if it fails
            try:
                article = google_news.get_full_article(url)
                if article and article.text:
                    full_text = article.text.strip()
                    # Only upgrade to full text if it's actually substantial (>300 chars)
                    if len(full_text) > 300:
                        text_to_save = full_text
            except Exception:
                pass # Silently fallback to our baseline summary text
                
            # 4. Save the record
            if len(text_to_save) > 50:
                records.append({
                    "ticker": ticker,
                    "company_name": company_name,
                    "published_date": pub_date.isoformat(),
                    "news_text": text_to_save[:2500], 
                    "true_alpha": true_alpha
                })
                saved_count += 1
                
                # Print whether we used the full text or the fallback summary
                source_type = "FULL TEXT" if len(text_to_save) > 300 else "SUMMARY"
                print(f"  [+] SAVED ({source_type}): {pub_date.isoformat()} (Alpha: {true_alpha}) | {publisher}")
        
        if saved_count == 0:
            print("  [-] All articles for this query were skipped (Likely due to date falling on weekend/holiday).")
            
        time.sleep(2) 
            
    return records


# --- Cell 6: Execution with Checkpointing ---
if __name__ == "__main__":
    # Pointing to the new working directory
    filename = os.path.join(working_dir, "nifty50_train_data.jsonl")
    checkpoint_file = os.path.join(working_dir, "completed_tickers.txt")
    
    # 1. Read the checkpoint file to see who is already done
    completed_tickers = set()
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            completed_tickers = set(line.strip() for line in f)
            
    if completed_tickers:
        print(f"[*] Found checkpoint file! Skipping {len(completed_tickers)} already completed companies...")
    
    # 2. Iterate through NIFTY 50
    for ticker, name in NIFTY50_MAP.items():
        # Skip if already done
        if ticker in completed_tickers:
            print(f"Skipping {name} ({ticker}) - Already scraped.")
            continue
            
        # Run the scraper
        stock_data = fetch_and_parse_news(ticker, name)
        
        # Save the scraped articles
        if stock_data:
            with open(filename, "a", encoding="utf-8") as f:
                for record in stock_data:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"--- Data Appended: {len(stock_data)} articles for {name} ---")
        else:
            print(f"--- No valid articles found for {name} ---")
            
        # 3. Update the checkpoint file AFTER successfully finishing the company
        with open(checkpoint_file, "a", encoding="utf-8") as f:
            f.write(ticker + "\n")
            
        print(f"[*] Checkpoint saved for {ticker}.")
        
    print("\n✅ All 50 NIFTY 50 companies have been successfully scraped!")
