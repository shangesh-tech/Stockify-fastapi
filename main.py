from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
import yfinance as yf
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from cachetools import TTLCache
import logging

# Load environment variables
load_dotenv()

# FastAPI instance
app = FastAPI()

# Allow CORS for specific origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://stockify-pink.vercel.app"],  
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Define request body models
class StockRequest(BaseModel):
    symbol: str
    time_period: str  # e.g., "1mo", "1y", "5y"

class InvestmentRequest(BaseModel):
    experience: str  # Investment experience (e.g., "Beginner", "Medium", "Advanced")
    risk: str  # Risk tolerance (e.g., "Low Risk", "Conservative", "High Risk")
    investment_type: str  # "SIP" or "Lumpsum"
    amount: float  # Amount to invest

# Setup API keys and models
gemini_api_key = os.getenv("GEMINI_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

if not gemini_api_key:
    raise ValueError("No Gemini API key found. Please set GEMINI_API_KEY in .env file")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable is required.")

# Initialize AI models
gemini = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", 
    google_api_key=gemini_api_key
)

groq_model = ChatGroq(
    model="mixtral-8x7b-32768",
    temperature=0,
    max_tokens=1000,
)

# Set up TTL cache
cache = TTLCache(maxsize=100, ttl=600)

# Comprehensive Analysis Prompt
comprehensive_analysis_prompt = """
You are an expert financial analyst specializing in the Indian stock market. 
Provide a holistic analysis of {symbol} based on the following comprehensive data:

FUNDAMENTAL DATA:
{financial_data}

TECHNICAL INDICATORS:
{technical_data}

Comprehensive Analysis Requirements:
1. Detailed Financial Health Assessment
   - Valuation metrics interpretation
   - Financial stability evaluation
   - Comparative industry performance

2. Technical Market Analysis
   - Current market trend (bullish/bearish)
   - Moving averages interpretation
   - Momentum and trend indicators

3. Investment Recommendations
   - Short-term trading strategy
   - Long-term investment potential
   - Risk assessment
   - Potential entry and exit points

4. Market Outlook
   - Sector-specific trends
   - Macroeconomic factors
   - Potential catalysts or challenges

Ensure the analysis is comprehensive, actionable, and provides clear insights for investors in the Indian market context.
"""

# Investment combinations
investment_combinations = {
    "Beginner": {
        "Low": {
            "gold": 30,
            "nifty_50": 50,
            "blue_chips": 20,
            "mid_cap": 0,
            "small_cap": 0,
            "total_percentage": 100
        },
        "Conservative": {
            "gold": 15,
            "nifty_50": 35,
            "blue_chips": 25,
            "mid_cap": 20,
            "small_cap": 5,
            "total_percentage": 100
        },
        "High": {
            "gold": 5,
            "nifty_50": 15,
            "blue_chips": 25,
            "mid_cap": 35,
            "small_cap": 20,
            "total_percentage": 100
        }
    },
    "Intermediate": {
        "Low": {
            "gold": 25,
            "nifty_50": 75,
            "blue_chips": 0,
            "mid_cap": 0,
            "small_cap": 0,
            "total_percentage": 100
        },
        "Conservative": {
            "gold": 15,
            "nifty_50": 40,
            "blue_chips": 30,
            "mid_cap": 10,
            "small_cap": 5,
            "total_percentage": 100
        },
        "High": {
            "gold": 5,
            "nifty_50": 10,
            "blue_chips": 20,
            "mid_cap": 35,
            "small_cap": 30,
            "total_percentage": 100
        }
    },
    "Advanced": {
        "Low": {
            "gold": 10,
            "nifty_50": 90,
            "blue_chips": 0,
            "mid_cap": 0,
            "small_cap": 0,
            "total_percentage": 100
        },
        "Conservative": {
            "gold": 5,
            "nifty_50": 40,
            "blue_chips": 30,
            "mid_cap": 15,
            "small_cap": 10,
            "total_percentage": 100
        },
        "High": {
            "gold": 3,
            "nifty_50": 12,
            "blue_chips": 20,
            "mid_cap": 35,
            "small_cap": 30,
            "total_percentage": 100
        }
    }
}

# Stock analysis functions
def fetch_comprehensive_stock_data(symbol: str, time_period: str):
    try:
        stock = yf.Ticker(symbol)
        
        period_mapping = {
            "1mo": ("1mo", "1d"),
            "1y": ("1y", "1mo"),
            "5y": ("5y", "1mo")
        }
        
        if time_period not in period_mapping:
            raise ValueError("Invalid time period")
        
        period, interval = period_mapping[time_period]
        
        historical_data = stock.history(period=period, interval=interval)
        
        if historical_data.empty:
            alternative_periods = [
                ("max", "1mo"),
                ("5y", "1d"),
                ("2y", "1mo")
            ]
            
            for alt_period, alt_interval in alternative_periods:
                historical_data = stock.history(period=alt_period, interval=alt_interval)
                if not historical_data.empty:
                    break
            
            if historical_data.empty:
                raise ValueError("No stock data found after multiple attempts")
        
        financial_info = {
            "Symbol": symbol,
            "Current_Price": historical_data['Close'].iloc[-1],
            "PE_Ratio": stock.info.get('trailingPE', 'N/A'),
            "Forward_PE": stock.info.get('forwardPE', 'N/A'),
            "Market_Cap": stock.info.get('marketCap', 'N/A'),
            "Dividend_Yield": stock.info.get('dividendYield', 'N/A'),
            "EPS": stock.info.get('trailingEps', 'N/A'),
            "Revenue": stock.info.get('totalRevenue', 'N/A'),
            "Debt_to_Equity": stock.info.get('debtToEquity', 'N/A'),
            "Return_on_Equity": stock.info.get('returnOnEquity', 'N/A'),
            "Industry": stock.info.get('industry', 'N/A'),
            "Sector": stock.info.get('sector', 'N/A')
        }
        
        technical_indicators = calculate_technical_indicators(historical_data, min_periods=10)
        
        return {
            "historical_data": historical_data,
            "fundamental_data": financial_info,
            "technical_indicators": technical_indicators
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Comprehensive data fetch error: {str(e)}")

def calculate_technical_indicators(data, min_periods=10):
    try:
        if len(data) < min_periods:
            print(f"Warning: Only {len(data)} data points available. Using minimal analysis.")
        
        close_prices = data['Close']

        indicators = {
            "Current_Price": close_prices.iloc[-1],
            "SMA_50": safe_rolling_mean(close_prices, min(50, len(close_prices))),
            "SMA_200": safe_rolling_mean(close_prices, min(200, len(close_prices))),
            "EMA_50": safe_exponential_mean(close_prices, min(50, len(close_prices))),
            "EMA_200": safe_exponential_mean(close_prices, min(200, len(close_prices))),
            "RSI": calculate_rsi(close_prices, periods=min(14, len(close_prices))),
            "Price_Change_Percentage": calculate_price_change_percentage(close_prices)
        }
        
        if len(close_prices) >= 26:
            macd_line, signal_line = calculate_macd(close_prices)
            indicators.update({
                "MACD_Line": macd_line,
                "MACD_Signal": signal_line
            })
        else:
            indicators.update({
                "MACD_Line": "Insufficient data",
                "MACD_Signal": "Insufficient data"
            })
        
        return indicators
    except Exception as e:
        raise ValueError(f"Technical indicator calculation error: {str(e)}")

def safe_rolling_mean(series, window):
    return series.rolling(window=window, min_periods=1).mean().iloc[-1]

def safe_exponential_mean(series, span):
    return series.ewm(span=span, adjust=False).mean().iloc[-1]

def calculate_rsi(price_series, periods=14):
    delta = price_series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods, min_periods=1).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs)).iloc[-1]
    return round(rsi, 2)

def calculate_macd(price_series, fast_period=12, slow_period=26, signal_period=9):
    ema_fast = price_series.ewm(span=fast_period, adjust=False).mean()
    ema_slow = price_series.ewm(span=slow_period, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    return round(macd_line.iloc[-1], 2), round(signal_line.iloc[-1], 2)

def calculate_price_change_percentage(price_series):
    return round(((price_series.iloc[-1] - price_series.iloc[0]) / price_series.iloc[0]) * 100, 2)

def perform_comprehensive_analysis(stock_data, symbol):
    try:
        fundamental_data = str(stock_data['fundamental_data'])
        technical_data = str(stock_data['technical_indicators'])
        
        prompt_template = PromptTemplate(
            input_variables=["financial_data", "technical_data", "symbol"], 
            template=comprehensive_analysis_prompt
        )
        
        chain = LLMChain(prompt=prompt_template, llm=gemini)
        
        analysis = chain.run(
            financial_data=fundamental_data, 
            technical_data=technical_data,
            symbol=symbol
        )
        
        return analysis
    except Exception as e:
        print(f"Comprehensive Analysis Error: {e}")
        return f"Unable to generate comprehensive analysis: {str(e)}"

# Investment portfolio functions
def calculate_portfolio(experience, risk, amount):
    portfolio = investment_combinations.get(experience, {}).get(risk)
    if not portfolio:
        raise HTTPException(status_code=400, detail="Invalid experience or risk type")
    
    allocation = {
        "gold": {
            "amount": amount * (portfolio["gold"] / 100),
            "percentage": portfolio["gold"]
        },
        "nifty_50": {
            "amount": amount * (portfolio["nifty_50"] / 100),
            "percentage": portfolio["nifty_50"]
        },
        "blue_chips": {
            "amount": amount * (portfolio["blue_chips"] / 100),
            "percentage": portfolio["blue_chips"]
        },
        "mid_cap": {
            "amount": amount * (portfolio["mid_cap"] / 100),
            "percentage": portfolio["mid_cap"]
        },
        "small_cap": {
            "amount": amount * (portfolio["small_cap"] / 100),
            "percentage": portfolio["small_cap"]
        },
    }
    
    portfolio_summary = {
        "experience": experience,
        "risk": risk,
        "gold_investment": allocation["gold"],
        "nifty_50_investment": allocation["nifty_50"],
        "blue_chips_investment": allocation["blue_chips"],
        "mid_cap_investment": allocation["mid_cap"],
        "small_cap_investment": allocation["small_cap"],
    }

    return portfolio_summary

def generate_portfolio_summary(experience, risk, allocation):
    cache_key = f"{experience}_{risk}_{str(allocation)}"
    
    if cache.get(cache_key):
        logging.info("Cache hit for portfolio summary.")
        return cache[cache_key]
    
    prompt = f"""
    Based on the following investment experience '{experience}' and risk tolerance '{risk}', 
    the portfolio allocation is as follows:
    - Gold: {allocation['gold_investment']['amount']} ({allocation['gold_investment']['percentage']}%)
    - Nifty 50: {allocation['nifty_50_investment']['amount']} ({allocation['nifty_50_investment']['percentage']}%)
    - Blue Chips: {allocation['blue_chips_investment']['amount']} ({allocation['blue_chips_investment']['percentage']}%)
    - Mid Cap: {allocation['mid_cap_investment']['amount']} ({allocation['mid_cap_investment']['percentage']}%)
    - Small Cap: {allocation['small_cap_investment']['amount']} ({allocation['small_cap_investment']['percentage']}%)
    
    Provide a short summary of this portfolio and pros/cons based on these investments in JSON format.
    """

    messages = [
        ("system", "You are an investment advisor. Based on the given portfolio allocation, provide pros and cons."),
        ("human", prompt)
    ]
    
    ai_msg = groq_model.invoke(messages)
    response = ai_msg.content

    cache[cache_key] = response

    return response

# API Routes
@app.get("/")
async def root():
    return {"message": "Welcome to the AI Financial Analysis Server!"}

# API Routes (continued)
@app.post("/get-stock-analysis")
async def get_stock_analysis(stock: StockRequest):
    try:
        stock_data = fetch_comprehensive_stock_data(stock.symbol, stock.time_period)
        
        analysis = perform_comprehensive_analysis(stock_data, stock.symbol)
        
        return {
            "symbol": stock.symbol,
            "time_period": stock.time_period,
            "fundamental_data": stock_data['fundamental_data'],
            "technical_indicators": stock_data['technical_indicators'],
            "comprehensive_analysis": analysis
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get-investment-portfolio")
async def get_investment_portfolio(request: InvestmentRequest):
    try:
        # Calculate the portfolio allocation
        allocation = calculate_portfolio(
            request.experience,
            request.risk,
            request.amount
        )
        
        # Generate portfolio summary using Groq
        portfolio_summary = generate_portfolio_summary(request.experience, request.risk, allocation)
        
        return {
            "portfolio_allocation": allocation,
            "portfolio_summary": portfolio_summary
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Running FastAPI server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
