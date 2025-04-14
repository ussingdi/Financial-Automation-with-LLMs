# stock_search_app.py
import streamlit as st
import plotly.graph_objects as go
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI
import yfinance as yf
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import json
import finnhub
import groq
import random
# Page configuration
st.set_page_config(
    page_title="Financial Research Automation with LLMs",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Load environment variables
load_dotenv(override=True)

# Constants
PINECONE_INDEX_NAME = "mastersproject"
EMBEDDING_DIMENSION = 768
NAMESPACE = "stock-descriptions"
DEFAULT_QUERY = "Companies focused on artificial intelligence"

# Custom CSS
st.markdown("""
<style>
    /* Base styling */
    body {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: #f0f2f6;
        color: #1f2937;
    }
    
    /* Container width control */
    .block-container {
        max-width: 1000px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #2563eb 0%, #3b82f6 100%);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 25px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Card container styling */
    [data-testid="stVerticalBlock"] {
        background-color: white;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    /* Stock card grid */
    .stock-card-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 20px;
        margin: 20px 0;
    }
    
    /* Stock card */
    .stock-card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        height: 100%;
    }
    
    /* Plot container */
    .plot-container {
        background: white;
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        max-width: 100%;
        overflow-x: hidden;
    }
    
    /* Ensure plotly chart stays within container */
    .js-plotly-plot {
        max-width: 100% !important;
    }
    
    /* Make plot responsive */
    .plotly-graph-div {
        max-width: 100% !important;
    }
    
    /* Date input fields */
    .date-input-container {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 20px;
        margin: 15px 0;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        max-width: 300px;
        margin: 10px auto;
        display: block;
    }
    
    /* Headers styling */
    .stMarkdown h3 {
        color: #1f2937;
        font-size: 22px;
        margin-bottom: 15px;
        padding-bottom: 10px;
        border-bottom: 1px solid #e5e7eb;
    }
    
    /* Links styling */
    .stMarkdown a {
        color: #2563eb;
        text-decoration: none;
    }
    
    .stMarkdown a:hover {
        text-decoration: underline;
    }
    
    /* DataFrame styling */
    [data-testid="stDataFrame"] {
        background-color: #f0f2f6;
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #e5e7eb;
    }
    
    .dataframe {
        font-size: 14px;
    }
    
    /* Description text */
    .stock-description {
        color: #4b5563;
        font-size: 14px;
        line-height: 1.6;
        margin: 10px 0;
    }
    
    /* Spinner color */
    .stSpinner > div {
        border-color: #3b82f6 !important;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    
    /* Results header */
    .results-header {
        margin-top: 30px;
        margin-bottom: 20px;
        padding: 15px;
        background-color: white;
        border-radius: 8px;
        border-left: 4px solid #3b82f6;
    }
    
    .results-header h2 {
        color: #1f2937;
        font-size: 20px;
        margin: 0;
    }
    
    /* Card styling */
    .stock-card {
        background-color: white;
        border-radius: 10px;
        padding: 25px;
        margin-bottom: 20px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .stock-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.15);
    }
    
    /* Stock title */
    .stock-title {
        font-size: 22px;
        font-weight: 700;
        margin-bottom: 12px;
        color: #1f2937;
        border-bottom: 1px solid #e5e7eb;
        padding-bottom: 10px;
    }
    
    /* Stock ticker */
    .stock-ticker {
        color: #3b82f6;
        font-weight: 600;
        font-size: 14px;
        margin-left: 8px;
    }
    
    /* Stock link */
    .stock-link {
        margin-bottom: 20px;
    }
    
    .stock-link a {
        color: #3b82f6;
        text-decoration: none;
        font-size: 14px;
        display: inline-block;
        padding: 5px 10px;
        background-color: rgba(59, 130, 246, 0.1);
        border-radius: 5px;
        transition: background-color 0.2s ease;
    }
    
    .stock-link a:hover {
        background-color: rgba(59, 130, 246, 0.2);
    }
    
    /* Metrics table */
    .metrics-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        border-radius: 8px;
        overflow: hidden;
        background-color: #f0f2f6;
    }       
    
    .metrics-table th {
        text-align: left;
        padding: 12px 15px;
        background-color: #e5e7eb;
        color: #1f2937;
        font-weight: 600;
        font-size: 14px;
    }
    
    .metrics-table td {
        padding: 12px 15px;
        border-bottom: 1px solid #e5e7eb;
        font-size: 14px;
    }
    
    .metrics-table tr:last-child td {
        border-bottom: none;
    }
    
    /* Metrics colors */
    .positive {
        color: #10b981;
        font-weight: 600;
    }
    
    .negative {
        color: #ef4444;
        font-weight: 600;
    }
    
    /* Analysis section */
    .analysis-section {
        background-color: white;
        border-radius: 10px;
        padding: 25px;
        margin-top: 30px;
        margin-bottom: 40px;
        border: 1px solid #e5e7eb;
    }
    
    .analysis-section h2 {
        color: #1f2937;
        font-size: 22px;
        margin-bottom: 15px;
        border-bottom: 1px solid #e5e7eb;
        padding-bottom: 10px;
    }
    
    /* Text area customization */
    .stTextArea textarea {
        background-color: white;
        color: #1f2937;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 12px;
        font-size: 16px;
    }
    
    /* Stock selector styling */
    .stock-selector {
        margin: 20px 0;
        padding: 15px;
        background: white;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    /* Multiselect styling */
    .stMultiSelect {
        max-width: 100%;
    }
    
    .stMultiSelect > div {
        border-radius: 8px;
    }
    
    .stMultiSelect [data-baseweb="tag"] {
        background-color: #2563eb;
        border-radius: 6px;
        padding: 2px 8px;
    }
    
    .stMultiSelect [data-baseweb="tag"] span {
        color: white;
        font-size: 0.9rem;
    }
    
    .stMultiSelect [data-baseweb="tag"] button {
        color: white;
    }
    
    /* Date input container */
    .date-inputs {
        display: flex;
        gap: 20px;
        margin: 20px 0;
    }
    
    .date-inputs > div {
        flex: 1;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        margin-top: 50px;
        padding: 20px 0;
        border-top: 1px solid #e5e7eb;
        color: #4b5563;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize clients
def initialize_clients():
    load_dotenv(override=True)
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_env = os.getenv("PINECONE_ENVIRONMENT")
    groq_api_key = os.getenv("GROQ_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")  # Still needed for embeddings
    
    if not pinecone_api_key or not pinecone_env:
        st.error("Pinecone API key or environment not found in .env file")
        return None, None
        
    if not groq_api_key:
        st.error("Groq API key not found in .env file")
        return None, None
        
    if not openai_api_key:
        st.error("OpenAI API key not found in .env file (needed for embeddings)")
        return None, None
    
    try:
        # Initialize Pinecone
        pc = Pinecone(api_key=pinecone_api_key, environment=pinecone_env)
        pinecone_index = pc.Index(PINECONE_INDEX_NAME)
        
        # Initialize Groq client for LLM operations
        groq_client = groq.Groq(api_key=groq_api_key)
        
        # Initialize OpenAI client for embeddings only
        openai_client = OpenAI(api_key=openai_api_key)
        
        return pinecone_index, groq_client, openai_client
        
    except Exception as e:
        st.error(f"Error initializing clients: {str(e)}")
        return None, None

# Get embeddings
def get_embeddings(text, openai_client):
    """Generate embeddings using OpenAI"""
    try:
        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
            dimensions=EMBEDDING_DIMENSION
        )
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Error generating embeddings: {str(e)}")
        st.stop()

# Enhanced query
def enhanced_query(index, query_text, openai_client, top_k=10, sector=None, industry=None):
    """Enhanced query with optional metadata filtering"""
    # Generate embeddings for the query
    query_embedding = get_embeddings(query_text, openai_client)
    
    # Build filter if sector or industry is specified
    filter_dict = {}
    if sector:
        filter_dict["$and"] = [{"Sector": {"$eq": sector}}]
    if industry:
        if "$and" in filter_dict:
            filter_dict["$and"].append({"Industry": {"$eq": industry}})
        else:
            filter_dict["$and"] = [{"Industry": {"$eq": industry}}]
    
    # Query Pinecone
    if filter_dict:
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            namespace=NAMESPACE,
            filter=filter_dict
        )
    else:
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            namespace=NAMESPACE
        )
    
    return results

# Render stock card
def render_stock_card(ticker, metadata, col):
    """Render a stock card with metadata"""
    with col:
        try:
            # Get additional info from yfinance
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Financial metrics
            earnings_growth = info.get('earningsGrowth', 0) * 100 if info.get('earningsGrowth') else 0
            revenue_growth = info.get('revenueGrowth', 0) * 100 if info.get('revenueGrowth') else 0
            gross_margins = info.get('grossMargins', 0) * 100 if info.get('grossMargins') else 0
            ebitda_margins = info.get('ebitdaMargins', 0) * 100 if info.get('ebitdaMargins') else 0
            week_52_change = info.get('52WeekChange', 0) * 100 if info.get('52WeekChange') else 0
            
            # Company website
            website = info.get('website', f"https://www.{ticker.lower()}.com")
            
            # Company name and summary
            company_name = info.get('longName', ticker)
            summary = metadata.get('Business Summary', 'No description available')
            
            # Create card container
            with st.container():
                st.markdown(f"### {company_name} ({ticker})")
                st.markdown(f'<div class="stock-description">{summary[:200]}...</div>', unsafe_allow_html=True)
                st.markdown(f"[{website}]({website})")
                
                # Create metrics table using pandas with styled values
                metrics_data = {
                    'Metric': ['Earnings Growth', 'Revenue Growth', 'Gross Margins', 'EBITDA Margins', '52 Week Change'],
                    'Value': [
                        f"{'🔼' if earnings_growth > 0 else '🔽'} {earnings_growth:+.2f}%" if earnings_growth != 0 else "0.00%",
                        f"{'🔼' if revenue_growth > 0 else '🔽'} {revenue_growth:+.2f}%" if revenue_growth != 0 else "0.00%",
                        f"{gross_margins:.2f}%",
                        f"{ebitda_margins:.2f}%",
                        f"{'🔼' if week_52_change > 0 else '🔽'} {week_52_change:+.2f}%" if week_52_change != 0 else "0.00%"
                    ]
                }
                df = pd.DataFrame(metrics_data)
                
                # Apply custom styling to the dataframe
                st.dataframe(
                    df.style.apply(lambda x: ['color: #10B981' if '+' in str(v) else 'color: #EF4444' if '-' in str(v) else '' for v in x], axis=0),
                    hide_index=True,
                    use_container_width=True
                )
        
        except Exception as e:
            st.warning(f"Could not render complete card for {ticker}: {str(e)}")
            # Fallback to basic card
            st.subheader(ticker)
            st.write(metadata.get('Business Summary', 'No description available')[:200] + "...")

# Get AI analysis
def get_ai_analysis(groq_client, contexts=None, query=None, tickers=None, start_date=None, end_date=None, analysis_type="general"):
    """Get AI analysis of stocks using Groq with deepseek model"""
    try:
        # Format the context and query
        if analysis_type == "price" and tickers and start_date and end_date:
            # Get performance data for price analysis
            performance_data = []
            for ticker in tickers:
                df = get_stock_price_history(ticker, start_date, end_date)
                if df is not None:
                    final_change = df['Percentage Change'].iloc[-1]
                    performance_data.append(f"{ticker}: {final_change:.2f}% change over the period")
            
            prompt = f"""Analyze the following stock price performance data from {start_date} to {end_date}:

{' | '.join(performance_data)}

Please provide and Keep the analysis concise but insightful.
DO NOT include any thinking process or tags like <think> or </think> in your response.
1. A comparison of how these stocks performed relative to each other
2. Notable trends or patterns in their performance
3. Key factors that might have influenced their performance during this period
4. Any interesting correlations between these stocks

Keep the analysis concise but insightful.
DO NOT include any thinking process or tags like <think> or </think> in your response."""

            system_prompt = "You are a financial analyst providing insights on stock performance, DO NOT include any thinking process or tags like <think> or </think> in your response."
            
        elif analysis_type == "general" and contexts and query:
            # Prepare prompt for general analysis
            augmented_query = "<CONTEXT>\n" + "\n\n-------\n\n".join(contexts[:10]) + "\n-------\n</CONTEXT>\n\n\n\nMY QUESTION:\n" + query
            system_prompt = "You are an expert at providing answers about stocks, DO NOT include any thinking process or tags like <think> or </think> in your response."
            prompt = augmented_query
            
        else:
            raise ValueError("Invalid analysis type or missing required parameters")
        
        response = groq_client.chat.completions.create(
            model="deepseek-r1-distill-llama-70b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=2000
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        st.error(f"Error in AI analysis: {str(e)}")
        return "Error generating analysis"

def generate_recommendation(articles_data, average_score):
    """Generate investment recommendation based on sentiment analysis"""
    if not articles_data:
        return "Insufficient data for recommendation"
        
    if average_score >= 7.5:
        return "Strong Buy - Very positive sentiment with multiple growth indicators"
    elif average_score >= 6.5:
        return "Buy - Generally positive sentiment with good potential"
    elif average_score >= 4.5:
        return "Hold - Mixed sentiment with balanced opportunities and risks"
    elif average_score >= 3.5:
        return "Sell - Generally negative sentiment with concerning factors"
    else:
        return "Strong Sell - Very negative sentiment with multiple risk indicators"

def analyze_article_sentiment(article_text, groq_client):
    """Analyze sentiment of a single article using Groq"""
    system_prompt = """You are a financial sentiment analyzer. Analyze the given news article and provide:
    1. A sentiment score (1-10, where 1 is very negative and 10 is very positive)
    2. A confidence score (0-1)
    3. Key positive factors mentioned
    4. Key negative factors mentioned
    
    Return the analysis in JSON format with these exact keys:
    {
        "sentiment_score": float,
        "confidence": float,
        "positive_factors": list[str],
        "negative_factors": list[str]
    }
    """
    
    try:
        response = groq_client.chat.completions.create(
            model="deepseek-r1-distill-llama-70b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": article_text}
            ],
            response_format={ "type": "json_object" }
        )
        
        result = json.loads(response.choices[0].message.content)
        return result
    
    except Exception as e:
        st.error(f"Error in sentiment analysis: {str(e)}")
        return {
            "sentiment_score": 5,
            "confidence": 0,
            "positive_factors": [],
            "negative_factors": []
        }
def get_market_factors(groq_client, ticker):
    """Get market factor analysis for a stock using Groq with deepseek model"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Format company information
        company_info = f"""
        Company: {info.get('longName', ticker)}
        Industry: {info.get('industry', 'N/A')}
        Sector: {info.get('sector', 'N/A')}
        Market Cap: {info.get('marketCap', 'N/A')}
        P/E Ratio: {info.get('trailingPE', 'N/A')}
        Revenue Growth: {info.get('revenueGrowth', 'N/A')}
        
        Analyze this company and provide ratings for the following market factors on a scale of 1-10:
        1. Financial Health - Consider debt levels, cash flow, profitability
        2. Market Competition - Consider market share, competitive advantages, barriers to entry
        3. Growth Potential - Consider industry growth, company expansion plans, addressable market
        4. Innovation - Consider R&D investment, technological advantages, patent portfolio
        5. Industry Trends - Consider if the company is aligned with positive industry trends
        6. Regulatory Environment - Consider regulatory risks and opportunities
        
        Provide a brief explanation for each rating.
        """
        
        system_prompt = """You are a financial analyst specializing in stock market analysis.
                    Analyze the given company information and provide ratings for the specified market factors.
                    Return your analysis in JSON format with the following structure:
                    {
                        "ratings": {
                            "Financial Health": {"score": float, "explanation": string},
                            "Market Competition": {"score": float, "explanation": string},
                            "Growth Potential": {"score": float, "explanation": string},
                            "Innovation": {"score": float, "explanation": string},
                            "Industry Trends": {"score": float, "explanation": string},
                            "Regulatory Environment": {"score": float, "explanation": string}
                        }
                    }
                    Scores should be between 1-10."""
        
        response = groq_client.chat.completions.create(
            model="deepseek-r1-distill-llama-70b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": company_info}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)
        
    except Exception as e:
        st.error(f"Error in market factor analysis: {str(e)}")
        return None

def get_stock_price_history(ticker, start_date, end_date):
    """Get historical stock prices and calculate percentage change"""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        if len(df) > 0:
            initial_price = df['Close'].iloc[0]
            df['Percentage Change'] = ((df['Close'] - initial_price) / initial_price) * 100
            return df
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
    return None

def plot_stock_comparison(tickers, start_date, end_date):
    """Plot stock price comparison"""
    fig = go.Figure()
    
    for ticker in tickers:
        df = get_stock_price_history(ticker, start_date, end_date)
        if df is not None:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['Percentage Change'],
                name=ticker,
                mode='lines'
            ))
    
    fig.update_layout(
        title='Normalized Stock Price History (% Change)',
        xaxis_title='Date',
        yaxis_title='Percentage Change (%)',
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        width=800,
        height=500,
        margin=dict(l=50, r=50, t=50, b=50),
        autosize=True,
        title_font=dict(size=20),
        legend_font=dict(size=12),
        xaxis=dict(title_font=dict(size=14)),
        yaxis=dict(title_font=dict(size=14))
    )
    
    return fig

def plot_market_radar(tickers, factor_data):
    """Create a radar plot comparing market factors across stocks"""
    fig = go.Figure()
    
    # Define the factors and their order
    factors = [
        "Financial Health",
        "Market Competition",
        "Growth Potential",
        "Innovation",
        "Industry Trends",
        "Regulatory Environment"
    ]
    
    # Color scheme for different stocks
    colors = ['#3b82f6', '#10B981', '#6366f1', '#ef4444', '#f59e0b']
    
    for i, ticker in enumerate(tickers):
        if ticker in factor_data and factor_data[ticker]:
            ratings = factor_data[ticker]['ratings']
            values = [ratings[factor]['score'] for factor in factors]
            values.append(values[0])  # Close the polygon
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=factors + [factors[0]],
                name=ticker,
                line=dict(color=colors[i % len(colors)], width=2),
                fill='toself',
                fillcolor=colors[i % len(colors)],
                opacity=0.2
            ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10]
            )
        ),
        showlegend=True,
        title="Market Trend Radar - All Stocks",
        height=600
    )
    
    return fig

def calculate_stock_sentiment(ticker, groq_client, days_back=7, max_articles=5):
    """Calculate overall sentiment for a stock with detailed analysis"""
    news_articles = fetch_stock_news(ticker, days_back, max_articles)
    if not news_articles:
        return {
            "average_score": 5,
            "confidence": 0,
            "num_articles": 0,
            "summary": "No recent news articles found",
            "articles": [],
            "aggregated_analysis": {
                "positive_drivers": [],
                "negative_drivers": [],
                "recommendation": "Insufficient data for analysis"
            }
        }
    
    articles_data = []
    total_score = 0
    total_confidence = 0
    all_positive = []
    all_negative = []
    
    with st.spinner(f'Analyzing sentiment for {ticker}...'):
        for article in news_articles:
            article_text = f"{article.get('title', '')}\n{article.get('summary', '')}"
            sentiment = analyze_article_sentiment(article_text, groq_client)
            
            sentiment_score = sentiment.get('sentiment_score', 5)
            confidence = sentiment.get('confidence', 0)
            positive_factors = sentiment.get('positive_factors', [])
            negative_factors = sentiment.get('negative_factors', [])
            
            all_positive.extend(positive_factors)
            all_negative.extend(negative_factors)
            
            total_score += sentiment_score * confidence
            total_confidence += confidence
            
            # Format the date from publishTime timestamp
            publish_date = ""
            if article.get('publishTime'):
                try:
                    # Convert Unix timestamp to datetime
                    publish_datetime = datetime.fromtimestamp(article['publishTime'])
                    publish_date = publish_datetime.strftime('%b %d, %Y')
                except:
                    publish_date = ""
            
            articles_data.append({
                'title': article.get('title', ''),
                'date': publish_date,
                'link': article.get('link', ''),
                'sentiment_score': sentiment_score,
                'confidence': confidence,
                'positive_factors': positive_factors,
                'negative_factors': negative_factors,
                'summary': article.get('summary', '')
            })
    
    average_score = total_score / total_confidence if total_confidence > 0 else 5
    
    unique_positive = list(set(all_positive))
    unique_negative = list(set(all_negative))
    
    if average_score >= 7.5:
        recommendation = "Strong Buy - Very positive sentiment with multiple growth indicators"
    elif average_score >= 6.5:
        recommendation = "Buy - Generally positive sentiment with good potential"
    elif average_score >= 4.5:
        recommendation = "Hold - Mixed sentiment with balanced opportunities and risks"
    elif average_score >= 3.5:
        recommendation = "Sell - Generally negative sentiment with concerning factors"
    else:
        recommendation = "Strong Sell - Very negative sentiment with multiple risk indicators"
    
    return {
        "average_score": round(average_score, 2),
        "confidence": round(total_confidence / len(articles_data), 2) if articles_data else 0,
        "num_articles": len(articles_data),
        "articles": articles_data,
        "aggregated_analysis": {
            "positive_drivers": unique_positive[:5],
            "negative_drivers": unique_negative[:5],
            "recommendation": recommendation
        }
    }
def display_sentiment_dashboard(sentiment_data):
    """Display a simplified sentiment analysis dashboard focusing on core insights"""
    st.header("📊 Stock Sentiment Analysis")
    
    # 1. Sentiment Comparison Chart
    fig = plot_sentiment_comparison(sentiment_data)
    st.plotly_chart(fig, use_container_width=True)
    
    # 2. Key Insights
    st.subheader("Key Insights")
    
    # Create tabs for each stock
    tabs = st.tabs([f"{ticker} (Score: {data['average_score']:.1f}/10)" 
                    for ticker, data in sentiment_data.items()])
    
    for i, (ticker, data) in enumerate(sentiment_data.items()):
        with tabs[i]:
            # Sentiment overview
            st.write(f"**Analysis based on {data['num_articles']} recent news articles**")
            st.write(f"**Overall Sentiment:** {data['average_score']:.1f}/10 (Confidence: {data['confidence']:.2f})")
            
            # Create two columns for factors
            col1, col2 = st.columns(2)
            
            with col1:
                # Positive factors
                st.write("\n**Positive Factors:**")
                for factor in data['aggregated_analysis']['positive_drivers']:
                    st.write(f"✓ {factor}")
            
            with col2:
                # Negative factors
                st.write("\n**Negative Factors:**")
                for factor in data['aggregated_analysis']['negative_drivers']:
                    st.write(f"! {factor}")
            
            st.markdown("---")
            
            # Recent headlines with better formatting
            st.write("\n**Recent Headlines:**")
            for article in data['articles'][:3]:  # Show only top 3 articles
                # Create a card-like container for each article
                with st.container():
                    # Get summary text or use a placeholder
                    summary = article.get('summary', '')
                    summary_text = f"{summary[:200]}..." if summary and len(summary) > 200 else summary if summary else "No summary available."
                    
                    # Format date display
                    date_display = f" • {article.get('date', '')}" if article.get('date') else ""
                    
                    st.markdown(f"""
                    <div style="border: 1px solid #e5e7eb; border-radius: 8px; padding: 15px; margin-bottom: 15px;">
                        <h4 style="margin-top: 0;">{article['title']}</h4>
                        <p style="color: #3b82f6; margin-bottom: 10px;">Sentiment Score: {article['sentiment_score']:.1f}/10{date_display}</p>
                        <p style="color: #4b5563; font-size: 14px;">{summary_text}</p>
                        <a href="{article.get('link', '')}" target="_blank" style="display: inline-block; margin-top: 10px; color: #3b82f6; text-decoration: none; font-weight: 500;">Read full article →</a>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Bottom line
            st.markdown("---")
            st.write("\n**Bottom Line:**")
            st.write(data['aggregated_analysis']['recommendation'])
def plot_sentiment_comparison(sentiment_data):
    """Create a bar plot comparing sentiment scores"""
    tickers = list(sentiment_data.keys())
    scores = [data["average_score"] for data in sentiment_data.values()]
    confidences = [data["confidence"] for data in sentiment_data.values()]
    
    # Create color scale based on sentiment scores
    colors = ['#ef4444' if score < 5 else '#10b981' for score in scores]
    
    fig = go.Figure()
    
    # Add bars
    fig.add_trace(go.Bar(
        x=tickers,
        y=scores,
        marker_color=colors,
        text=[f"Score: {score:.1f}<br>Confidence: {conf:.2f}" for score, conf in zip(scores, confidences)],
        textposition='auto',
    ))
    
    # Update layout
    fig.update_layout(
        title='Stock Sentiment Comparison',
        xaxis_title='Stocks',
        yaxis_title='Sentiment Score',
        yaxis=dict(range=[0, 10]),
        height=400,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig

def fetch_stock_news(ticker, days_back=7, max_articles=5):
    """Fetch recent news articles for a stock using Finnhub"""
    try:
        # Initialize Finnhub client
        finnhub_client = finnhub.Client(api_key=os.getenv("FINNHUB_API_KEY"))
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Fetch company news
        news = finnhub_client.company_news(ticker, 
            _from=start_date.strftime('%Y-%m-%d'),
            to=end_date.strftime('%Y-%m-%d'))
        
        if not news:
            st.warning(f"No news articles found for {ticker}")
            return []
        
        # Process and filter news
        recent_news = []
        for article in news:
            if all(key in article for key in ['datetime', 'headline', 'url']):
                recent_news.append({
                    'title': article['headline'],
                    'link': article['url'],
                    'publishTime': article['datetime'],
                    'summary': article.get('summary', '')
                })
                
                # Break if we have enough articles
                if len(recent_news) >= max_articles:
                    break
        
        if recent_news:
            st.write(f"📰 Found and analyzing {len(recent_news)} recent news articles for {ticker}")
        
        return recent_news
        
    except Exception as e:
        st.error(f"Error fetching news for {ticker}: {str(e)}")
        return []

def display_market_factor_explanations(factor_data, tickers):
    """Display explanations for market factors in an expandable section"""
    with st.expander("📊 View Market Factor Details", expanded=False):
        # Create tabs for each stock
        factor_tabs = st.tabs(tickers)
        
        # Define the factors and their order
        factors = [
            "Financial Health",
            "Market Competition",
            "Growth Potential",
            "Innovation",
            "Industry Trends",
            "Regulatory Environment"
        ]
        
        # Display explanations for each stock
        for i, ticker in enumerate(tickers):
            with factor_tabs[i]:
                if ticker in factor_data and factor_data[ticker] and 'ratings' in factor_data[ticker]:
                    ratings = factor_data[ticker]['ratings']
                    
                    # Create a table-like display for the factors
                    for factor in factors:
                        if factor in ratings:
                            score = ratings[factor]['score']
                            explanation = ratings[factor]['explanation']
                            
                            # Color code based on score
                            if score >= 7:
                                score_color = "#10B981"  # Green
                            elif score >= 5:
                                score_color = "#f59e0b"  # Orange
                            else:
                                score_color = "#ef4444"  # Red
                            
                            # Display factor with score and explanation
                            st.markdown(f"""
                            <div style="margin-bottom: 15px; padding: 10px; border: 1px solid #e5e7eb; border-radius: 8px;">
                                <h4 style="margin-bottom: 5px; display: flex; justify-content: space-between;">
                                    <span>{factor}</span>
                                    <span style="color: {score_color}; font-weight: bold;">{score}/10</span>
                                </h4>
                                <p style="color: #4b5563; margin-top: 5px;">{explanation}</p>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.write(f"No factor data available for {ticker}")

def fetch_sec_filings(ticker):
    """Fetch recent 10-Q SEC filings for a given ticker"""
    try:
        # Initialize yfinance ticker object
        stock = yf.Ticker(ticker)
        
        # Basic filing info we can extract
        filings = []
        
        # For demonstration, we'll create a simple structure
        # In a production environment, you might want to use a dedicated SEC API
        # like sec-api.io or similar services
        
        # Create quarterly filing entries (10-Q)
        # Most recent quarter
        q1_filing_date = datetime.now() - timedelta(days=random.randint(30, 60))
        filings.append({
            "date": q1_filing_date.strftime("%Y-%m-%d"),
            "type": "10-Q Filing",
            "quarter": "Q1",
            "url": f"https://www.sec.gov/edgar/search/#/entityName={ticker}&category=form-cat1&filter=10-Q",
            "insights_available": True
        })
        
        # Previous quarter
        q2_filing_date = datetime.now() - timedelta(days=random.randint(120, 150))
        filings.append({
            "date": q2_filing_date.strftime("%Y-%m-%d"),
            "type": "10-Q Filing",
            "quarter": "Q4",
            "url": f"https://www.sec.gov/edgar/search/#/entityName={ticker}&category=form-cat1&filter=10-Q",
            "insights_available": True
        })
            
        return filings
        
    except Exception as e:
        st.error(f"Error fetching SEC filings for {ticker}: {str(e)}")
        return []

def display_sec_filings(tickers, groq_client):
    """Display SEC filings and insights for selected stocks"""
    with st.expander("📑 SEC Quarterly Filings (10-Q)", expanded=True):
        # Create tabs for each stock
        sec_tabs = st.tabs(tickers)
        
        for i, ticker in enumerate(tickers):
            with sec_tabs[i]:
                # Fetch SEC filings
                filings = fetch_sec_filings(ticker)
                
                if not filings:
                    st.write("No 10-Q filing information available")
                    continue
                
                # Generate insights using LLM
                insights = generate_sec_filing_insights(groq_client, ticker)
                
                # Create a two-column layout
                col_left, col_right = st.columns([1, 2])
                
                # Display insights in the left column if available
                with col_left:
                    if insights:
                        st.subheader("Key Insights")
                        
                        # Performance metric
                        performance_score = insights.get('performance', {}).get('score', 50)
                        performance_color = insights.get('performance', {}).get('color', 'yellow')
                        color_hex = "#10B981" if performance_color == "green" else "#f59e0b" if performance_color == "yellow" else "#ef4444"
                        
                        st.markdown(f"""
                        <div style="margin-bottom: 15px;">
                            <div style="display: flex; align-items: center; margin-bottom: 5px;">
                                <div style="width: 150px; font-weight: bold;">Performance:</div>
                                <div style="flex-grow: 1; background-color: #e5e7eb; height: 10px; border-radius: 5px; position: relative;">
                                    <div style="position: absolute; left: 0; top: 0; height: 10px; width: {performance_score}%; background-color: {color_hex}; border-radius: 5px;"></div>
                                </div>
                                <div style="margin-left: 10px; color: {color_hex}; font-weight: bold;">{performance_score}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Growth potential metric
                        growth_score = insights.get('growth_potential', {}).get('score', 50)
                        growth_color = insights.get('growth_potential', {}).get('color', 'yellow')
                        color_hex = "#10B981" if growth_color == "green" else "#f59e0b" if growth_color == "yellow" else "#ef4444"
                        
                        st.markdown(f"""
                        <div style="margin-bottom: 15px;">
                            <div style="display: flex; align-items: center; margin-bottom: 5px;">
                                <div style="width: 150px; font-weight: bold;">Growth Potential:</div>
                                <div style="flex-grow: 1; background-color: #e5e7eb; height: 10px; border-radius: 5px; position: relative;">
                                    <div style="position: absolute; left: 0; top: 0; height: 10px; width: {growth_score}%; background-color: {color_hex}; border-radius: 5px;"></div>
                                </div>
                                <div style="margin-left: 10px; color: {color_hex}; font-weight: bold;">{growth_score}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Risk metric (lower is better)
                        risk_score = insights.get('risk', {}).get('score', 50)
                        risk_color = insights.get('risk', {}).get('color', 'yellow')
                        color_hex = "#ef4444" if risk_color == "green" else "#f59e0b" if risk_color == "yellow" else "#10B981"
                        
                        st.markdown(f"""
                        <div style="margin-bottom: 15px;">
                            <div style="display: flex; align-items: center; margin-bottom: 5px;">
                                <div style="width: 150px; font-weight: bold;">Risk:</div>
                                <div style="flex-grow: 1; background-color: #e5e7eb; height: 10px; border-radius: 5px; position: relative;">
                                    <div style="position: absolute; left: 0; top: 0; height: 10px; width: {risk_score}%; background-color: {color_hex}; border-radius: 5px;"></div>
                                </div>
                                <div style="margin-left: 10px; color: {color_hex}; font-weight: bold;">{risk_score}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Competitive edge metric
                        edge_score = insights.get('competitive_edge', {}).get('score', 50)
                        edge_color = insights.get('competitive_edge', {}).get('color', 'yellow')
                        color_hex = "#10B981" if edge_color == "green" else "#f59e0b" if edge_color == "yellow" else "#ef4444"
                        
                        st.markdown(f"""
                        <div style="margin-bottom: 15px;">
                            <div style="display: flex; align-items: center; margin-bottom: 5px;">
                                <div style="width: 150px; font-weight: bold;">Competitive Edge:</div>
                                <div style="flex-grow: 1; background-color: #e5e7eb; height: 10px; border-radius: 5px; position: relative;">
                                    <div style="position: absolute; left: 0; top: 0; height: 10px; width: {edge_score}%; background-color: {color_hex}; border-radius: 5px;"></div>
                                </div>
                                <div style="margin-left: 10px; color: {color_hex}; font-weight: bold;">{edge_score}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Display filings in the right column
                with col_right:
                    st.subheader("10-Q Quarterly Reports")
                    
                    for filing in filings:
                        st.markdown(f"""
                        <div style="padding: 15px; border: 1px solid #e5e7eb; border-radius: 8px; margin-bottom: 15px;">
                            <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                                <div>
                                    <strong>{ticker}</strong> - <strong>{filing.get('quarter', '')}</strong>
                                </div>
                                <div>
                                    {filing['date']}
                                </div>
                            </div>
                            <div style="margin-bottom: 10px;">
                                <strong>{filing['type']}</strong>
                            </div>
                            <div style="margin-bottom: 10px;">
                                {filing.get('insights_available', False) and '<span style="color: #10B981; font-weight: bold;">Key Insights Available</span>' or '<span style="color: #6B7280;">No insights available</span>'}
                            </div>
                            <div>
                                <a href="{filing['url']}" target="_blank" style="text-decoration: none; color: #3B82F6; font-weight: bold;">View 10-Q Filing</a>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

def generate_sec_filing_insights(groq_client, ticker):
    """Generate insights from SEC filings using Groq"""
    try:
        # Fetch SEC filings
        filings = fetch_sec_filings(ticker)
        
        # Generate insights using LLM
        insights = {}
        for filing in filings:
            if filing.get('insights_available', False):
                # Prepare prompt for insights generation
                filing_type = filing['type']
                
                # Split the prompt to avoid f-string formatting issues with JSON
                prompt_intro = f"Analyze the {filing_type} filing for {ticker} and provide insights on:"
                prompt_details = """
                1. Performance: How did the company perform in the reported period?
                2. Growth Potential: What are the company's growth prospects?
                3. Risk: What are the key risks facing the company?
                4. Competitive Edge: What are the company's competitive advantages?
                """
                
                prompt_json = """
                Return the insights in JSON format with the following structure:
                {
                    "performance": {"score": 75, "color": "green"},
                    "growth_potential": {"score": 60, "color": "yellow"},
                    "risk": {"score": 40, "color": "red"},
                    "competitive_edge": {"score": 80, "color": "green"}
                }
                Scores should be between 1-100, and colors should be 'green', 'yellow', or 'red'.
                """
                
                # Combine the parts
                full_prompt = prompt_intro + prompt_details + prompt_json
                
                response = groq_client.chat.completions.create(
                    model="deepseek-r1-distill-llama-70b",
                    messages=[
                        {"role": "system", "content": "You are a financial analyst providing insights from SEC filings."},
                        {"role": "user", "content": full_prompt}
                    ],
                    response_format={"type": "json_object"}
                )
                
                insights = json.loads(response.choices[0].message.content)
        
        return insights
    
    except Exception as e:
        st.error(f"Error generating insights from SEC filings: {str(e)}")
        return {}

def main():
    """Main function to run the Streamlit app"""
    # Custom header
    st.markdown('<div class="main-header"><h1>Financial Research Automation with LLMs</h1></div>', unsafe_allow_html=True)
    
    # Initialize clients
    pinecone_index, groq_client, openai_client = initialize_clients()
    
    # Initialize session state
    if 'step' not in st.session_state:
        st.session_state.step = 'search'  # Possible values: search, compare, sentiment
    if 'tickers' not in st.session_state:
        st.session_state.tickers = []
    if 'search_results' not in st.session_state:
        st.session_state.search_results = None
    if 'selected_tickers' not in st.session_state:
        st.session_state.selected_tickers = []
    if 'comparison_done' not in st.session_state:
        st.session_state.comparison_done = False
    if 'price_analysis' not in st.session_state:
        st.session_state.price_analysis = None
    if 'sentiment_data' not in st.session_state:
        st.session_state.sentiment_data = {}
    if 'radar_chart' not in st.session_state:
        st.session_state.radar_chart = None
    if 'factor_data' not in st.session_state:
        st.session_state.factor_data = {}
    if 'sec_filings' not in st.session_state:
        st.session_state.sec_filings = {}
        
    # Step 1: Search
    if st.session_state.step == 'search':
        st.markdown("### 🔍 Find stocks that match your criteria")
        st.markdown("Enter a description of the type of stock you're looking for.")
        
        user_query = st.text_area("Search Query", 
                                DEFAULT_QUERY, 
                                height=100,
                                label_visibility="collapsed")
        
        if st.button("Find Stocks"):
            with st.spinner('Finding relevant stocks...'):
                results = enhanced_query(pinecone_index, user_query, openai_client, top_k=6)
                if results and results.get('matches'):
                    st.session_state.search_results = results
                    st.session_state.tickers = [match['metadata'].get('Ticker', '') for match in results['matches']]
                    st.session_state.selected_tickers = st.session_state.tickers.copy()
                    st.session_state.step = 'compare'
                    st.rerun()  
                else:
                    st.warning("No matching stocks found. Try adjusting your search criteria.")
    else:
        # Always show search results if we're past the search step
        if st.session_state.search_results:
            st.markdown("### Search Results")
            matches = st.session_state.search_results.get('matches', [])
            st.markdown(f"Found {len(matches)} matching stocks:")
            
            col1, col2 = st.columns(2)
            for i, match in enumerate(matches):
                metadata = match['metadata']
                ticker = metadata.get('Ticker', '')
                render_stock_card(ticker, metadata, col1 if i % 2 == 0 else col2)
    
    # Step 2: Compare
    if st.session_state.step == 'compare':
        st.markdown("### 📈 Stock Comparison")
        
        # Stock selector
        selected = st.multiselect(
            "Select stocks to compare",
            st.session_state.tickers,
            default=st.session_state.selected_tickers
        )
        st.session_state.selected_tickers = selected
        
        if st.button("Compare Selected Stocks"):
            if not selected:
                st.warning("Please select at least one stock to compare.")
            else:
                # Plot comparison
                fig = plot_stock_comparison(selected, 
                                         datetime.now() - timedelta(days=365),
                                         datetime.now())
                st.session_state.comparison_chart = fig  # Save chart to session state
                st.plotly_chart(fig, use_container_width=True)
                
                # Show analysis
                with st.spinner('Analyzing price trends...'):
                    analysis = get_ai_analysis(
                        groq_client,
                        tickers=selected,
                        start_date=datetime.now() - timedelta(days=365),
                        end_date=datetime.now(),
                        analysis_type="price"
                    )
                    st.session_state.price_analysis = analysis  # Save analysis to session state
                    st.markdown("### Analysis")
                    st.write(analysis)
                
                # Add market trend radar chart
                with st.spinner('Generating market trend radar...'):
                    # Get market factors for each selected stock
                    factor_data = {}
                    for ticker in selected:
                        factor_data[ticker] = get_market_factors(groq_client, ticker)
                    
                    # Store factor data in session state
                    st.session_state.factor_data = factor_data
                    
                    # Create and display the radar chart
                    if all(factor_data.values()):  # Check if we have data for all stocks
                        radar_fig = plot_market_radar(selected, factor_data)
                        st.session_state.radar_chart = radar_fig
                        st.markdown("### Market Trend Radar")
                        st.plotly_chart(radar_fig, use_container_width=True)
                
                st.session_state.step = 'sentiment'
                st.rerun()
    
    # Always show comparison results if they exist
    elif st.session_state.step == 'sentiment':
        # Always show the comparison chart if it exists
        if st.session_state.comparison_chart:
            st.markdown("### 📈 Stock Comparison Results")
            st.plotly_chart(st.session_state.comparison_chart, use_container_width=True)
        
        # Always show the price analysis if it exists
        if st.session_state.price_analysis:
            st.markdown("### Price Analysis")
            st.write(st.session_state.price_analysis)
        
        # Always show the market trend radar chart if it exists
        if st.session_state.radar_chart:
            st.markdown("### Market Trend Radar")
            st.plotly_chart(st.session_state.radar_chart, use_container_width=True)
            
            # Display market factor explanations if factor data exists
            if st.session_state.factor_data and st.session_state.selected_tickers:
                display_market_factor_explanations(st.session_state.factor_data, st.session_state.selected_tickers)
        
        st.markdown("### 📊 Market Sentiment Analysis")
        
        if st.button("Analyze Market Sentiment") or st.session_state.sentiment_data:
            if not st.session_state.sentiment_data:
                with st.spinner("Analyzing market sentiment..."):
                    sentiment_data = {}
                    for ticker in st.session_state.selected_tickers:
                        sentiment_data[ticker] = calculate_stock_sentiment(ticker, groq_client)
                    st.session_state.sentiment_data = sentiment_data
            
            display_sentiment_dashboard(st.session_state.sentiment_data)
        
        st.markdown("### 📑 SEC Filings Analysis")
        
        if st.button("Fetch SEC Filings") or 'sec_filings' in st.session_state and st.session_state.sec_filings:
            if 'sec_filings' not in st.session_state or not st.session_state.sec_filings:
                with st.spinner("Fetching SEC filings and generating insights..."):
                    # Display SEC filings for selected stocks
                    display_sec_filings(st.session_state.selected_tickers, groq_client)
                    
                    # Mark that we've fetched SEC filings
                    st.session_state.sec_filings = {ticker: True for ticker in st.session_state.selected_tickers}
            else:
                # Display SEC filings using cached data
                display_sec_filings(st.session_state.selected_tickers, groq_client)
    
    # Reset button (always visible after search)
    if st.session_state.step != 'search':
        if st.sidebar.button("Start New Search"):
            st.session_state.step = 'search'
            st.session_state.tickers = []
            st.session_state.selected_tickers = []
            st.session_state.search_results = None
            st.session_state.comparison_chart = None
            st.session_state.price_analysis = None
            st.session_state.sentiment_data = {}
            st.session_state.radar_chart = None
            st.session_state.factor_data = {}
            st.session_state.sec_filings = {}
            st.rerun()
    
    # Footer
    st.markdown(
        """
        <div class="footer">
            Financial Research Assistant | Powered by Pinecone & Groq
            <br>
            Developed by Uvesh Patel(#0708874) @ NEIU - Master's Project 2025
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
