import requests
from bs4 import BeautifulSoup

def fetch_latest_crypto_news(query: str = "Bitcoin") -> str:
    """
    Scrapes a generic search or news aggregator for the latest headlines.
    For demonstration purposes, we will scrape CoinTelegraph or CryptoNews using basic BS4.
    """
    try:
        # Note: Depending on the site, structure might change. This is a generic example using CoinGecko news or similar RSS feeds which are easier.
        url = f"https://news.google.com/rss/search?q={query}+crypto&hl=en-US&gl=US&ceid=US:en"
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.content, features="xml")
        
        items = soup.findAll('item')
        news_text = []
        for item in items[:5]: # Top 5 recent news
            news_text.append(item.title.text)
            
        if not news_text:
             return "No recent news found."
             
        return "\n".join(news_text)
    except Exception as e:
        print(f"Error fetching news: {e}")
        return "Failed to fetch market news."
