import yfinance as yf
import pandas as pd
import json
from datetime import datetime

def scrape_financial_data(ticker_symbol, period="1mo"):
    """
    Scrapes stock history and news using yfinance.
    Saves the output to a text format optimized for LLM reading.
    """
    print(f"📉 Fetching data for: {ticker_symbol}...")
    
    # Initialize the Ticker
    stock = yf.Ticker(ticker_symbol)
    
    # --- PART A: Stock Price History ---
    # Get historical market data
    history = stock.history(period=period)
    
    if history.empty:
        return f"ERROR: No data found for {ticker_symbol}. Check if the ticker is correct (e.g., use 'RELIANCE.NS' for Indian stocks)."
    
    # We only need the last 5-7 days for a quick context
    recent_history = history.tail(7)[['Close', 'Volume', 'High', 'Low']]
    recent_data_str = recent_history.to_string()
    
    # --- PART B: Recent News ---
    # yfinance returns a list of dictionaries for news
    news_list = []
    try:
        # Get top 3 latest news items
        raw_news = stock.news[:3] 
        for item in raw_news:
            title = item.get('title', 'No Title')
            # Extract publisher if available
            publisher = item.get('publisher', 'Unknown Source')
            # Format: Title (Source)
            news_list.append(f"- {title} (Source: {publisher})")
            
        news_str = "\n".join(news_list) if news_list else "No recent news found."
    except Exception as e:
        news_str = f"Could not fetch news: {e}"

    # --- PART C: Format for the LLM ---
    # This structure helps the Llama model understand the data later
    formatted_output = (
        f"*** FINANCIAL DATA REPORT: {ticker_symbol} ***\n"
        f"Date Generated: {datetime.now().strftime('%Y-%m-%d')}\n\n"
        f"### 📊 MARKET DATA (Last 7 Days)\n"
        f"{recent_data_str}\n\n"
        f"### 📰 LATEST NEWS HEADLINES\n"
        f"{news_str}\n"
        f"--------------------------------------------------\n"
    )
    
    return formatted_output

# ==========================================
# USAGE EXAMPLE
# ==========================================

# 1. Define your Tickers
# For Indian stocks, add '.NS' (e.g., RELIANCE.NS, TATAMOTORS.NS)
# For US stocks, just the ticker (e.g., NVDA, AAPL)
tickers = ["RELIANCE.NS", "ZOMATO.NS"] 

# 2. Run Scraping and Save to File
all_data = ""
for t in tickers:
    data = scrape_financial_data(t)
    print(data) # Print to screen to verify
    all_data += data + "\n\n"

# 3. Save to a file (This stays in your Kaggle Output)
filename = "scraped_finance_data.txt"
with open(filename, "w", encoding="utf-8") as f:
    f.write(all_data)

print(f"\n✅ SUCCESS! Data saved to '{filename}'. You can download this file or load it in the next step.")