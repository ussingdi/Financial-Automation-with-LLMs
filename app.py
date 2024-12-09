import streamlit as st
from pinecone import Pinecone
import os
import yfinance as yf
from datetime import datetime, timedelta
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from groq import Groq

st.set_page_config(
    page_title="Stock Analysis AI | Uvesh",
    page_icon="📈",
    layout="wide"
)


# Constants
HUGGINGFACE_MODEL = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_QUERY = "Give me Best Electric cars stock to invest?"
PINECONE_INDEX_NAME = "stocks33"

# CSS Styles
def load_css():
    """Load external CSS"""
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def initialize_clients():
    """Initialize Pinecone and Groq clients"""
    load_dotenv(override=True)
    
    groq_api_key = os.getenv("GROQ_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_env = os.getenv("PINECONE_ENVIRONMENT")
    
    # Initialize Groq with explicit API key
    try:
        groq_client = Groq(
            api_key=groq_api_key
        )
    except Exception as e:
        st.error(f"❌ Failed to initialize Groq: {str(e)}")
        st.stop()
    
    # Initialize Pinecone with explicit API key
    try:
        pc = Pinecone(
            api_key=pinecone_api_key,
            environment=pinecone_env
        )
        pinecone_index = pc.Index(PINECONE_INDEX_NAME)
        return groq_client, pinecone_index
    except Exception as e:
        st.error(f"❌ Failed to initialize Pinecone: {str(e)}")
        st.stop()

def get_embeddings(text, model_name=HUGGINGFACE_MODEL):
    """Generate embeddings for the given text"""
    model = SentenceTransformer(model_name)
    return model.encode(text)

def render_stock_card(ticker, info, col):
    """Render a single stock card"""
    with col:
        with st.container():
            earnings_growth = info.get('earningsGrowth', 0) * 100
            revenue_growth = info.get('revenueGrowth', 0) * 100
            gross_margins = info.get('grossMargins', 0) * 100
            ebitda_margins = info.get('ebitdaMargins', 0) * 100
            week_52_change = info.get('52WeekChange', 0) * 100
            
            st.markdown(
                f"""
                # {ticker}
                {info.get('longBusinessSummary', '')[:200]}...
                [https://www.{ticker.lower()}.com]({info.get('website', '#')})
                
                | Metric | Value |
                |--------|-------|
                |Earnings Growth|<span class="{'negative' if earnings_growth < 0 else 'positive'}">{earnings_growth:.2f}%</span>|
                |Revenue Growth|<span class="{'negative' if revenue_growth < 0 else 'positive'}">{revenue_growth:.2f}%</span>|
                |Gross Margins|{gross_margins:.2f}%|
                |EBITDA Margins|{ebitda_margins:.2f}%|
                |52 Week Change|<span class="{'negative' if week_52_change < 0 else 'positive'}">{week_52_change:.2f}%</span>|
                """, 
                unsafe_allow_html=True
            )

def get_ai_analysis(client, contexts, query):
    """Get AI analysis of the stocks"""
    augmented_query = "<CONTEXT>\n" + "\n\n-------\n\n".join(contexts[:10]) + "\n-------\n</CONTEXT>\n\n\n\nMY QUESTION:\n" + query
    system_prompt = """You are an expert at providing answers about stocks. Please answer my question provided."""
    
    completion = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": augmented_query}
        ]
    )
    return completion.choices[0].message.content

def main():
    load_css()
    st.title("Stock Analysis with AI")
    
    # Initialize clients
    groq_client, pinecone_index = initialize_clients()
    
    # Get user input
    user_query = st.text_area("Enter Description of what kind of stock you are looking for?", DEFAULT_QUERY)
    
    if st.button("Find Stocks"):
        try:
            with st.spinner('Finding relevant stocks...'):
                # Get embeddings and search AFTER button click
                raw_query_embedding = get_embeddings(user_query)
                top_matches = pinecone_index.query(
                    vector=raw_query_embedding.tolist(), 
                    top_k=10, 
                    include_metadata=True, 
                    namespace="stock-descriptions"
                )
                
                contexts = [item['metadata']['text'] for item in top_matches['matches']]
                tickers = [item['metadata']['Ticker'] for item in top_matches['matches']]
                
                # Display stock cards
                col1, col2 = st.columns(2)
                for i, ticker in enumerate(tickers):
                    stock = yf.Ticker(ticker)
                    render_stock_card(ticker, stock.info, col1 if i % 2 == 0 else col2)
                
                # Get and display AI analysis
                analysis = get_ai_analysis(groq_client, contexts, user_query)
                
            st.success('Found matching stocks!')
            st.write(analysis)
            
            # Add footer
            st.markdown(
                """
                <div class="footer">
                    Made by Uvesh @ Head Starter
                </div>
                """,
                unsafe_allow_html=True
            )
                
        except Exception as e:
            st.error(f"Error analyzing stocks: {str(e)}")
    st.markdown(
                """
                <div class="footer">
                    Made by Uvesh @ Head Starter
                </div>
                """,
                unsafe_allow_html=True
            )

if __name__ == "__main__":
    main()

