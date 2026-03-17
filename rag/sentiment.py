import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from .news_scraper import fetch_latest_crypto_news

def get_market_sentiment_score() -> float:
    """
    Fetches news and uses an LLM to evaluate the sentiment as a float between -1.0 (Bearish) and 1.0 (Bullish).
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Warning: OPENAI_API_KEY not set. Returning neutral sentiment.")
        return 0.0
        
    try:
        # Fetch News
        news = fetch_latest_crypto_news("Bitcoin OR Ethereum")
        
        # Setup LLM
        llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
        
        prompt = PromptTemplate(
            input_variables=["news"],
            template="""You are an expert crypto quantitative analyst.
Analyze the following recent news headlines:
{news}

Determine the overall market sentiment score.
Respond ONLY with a single float value between -1.0 (Extreme Bearish) and 1.0 (Extreme Bullish). Do not include any other text."""
        )
        
        chain = prompt | llm
        result = chain.invoke({"news": news})
        
        # Parse result
        score_text = result.content.strip()
        try:
            score = float(score_text)
            score = max(-1.0, min(1.0, score)) # Clamp
            return score
        except ValueError:
            print(f"LLM returned invalid score format: {score_text}")
            return 0.0
            
    except Exception as e:
        print(f"Error calculating sentiment: {e}")
        return 0.0

