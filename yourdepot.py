import depot as dt
#import Consors as con
import yfinance as yf #Index(['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits'], dtype='str')
#import pytradegate as tg #Euro stocks by ISIN: https://github.com/cloasdata/pytradegate
#import requests as rq
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
isProd = False
show_stocks = True

# Configure Streamlit page
st.set_page_config(
    page_title="Depot Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed" #expanded"
)
stock_dict = {
            'ASML':'ASME.DE', 'Heidelberg Mat':'HEI.DE', 'Ver Bio':'VBK.DE',
            'Orkla': 'OKL.SG', 'Infineon':'IFX.DE', 'Sanofi':'SAN.PA',
            'Umicore':'NVJP.DE', 'Leonardo':'LDO.MI',
            'Hello Fresh': 'HFG.F', 'Google': 'GOOGL',
            'Vestas':'VWSB.DE', 'Nordex': 'NDX1.DE', 'Siemens Energy':'ENR.F',
            'Amazon': 'AMZN', 'BYD China': 'BYDDF', 'Tesla': 'TSLA',
            'NVIDIA': 'NVDA', 'Meta': 'META', 'Netflix': 'NFLX',
            'AMD': 'AMD', 'Intel': 'INTC', 'Apple': 'AAPL',
            'Ottobock': 'OBCK.DE', 'USA 2028 3.63%':'US91282CHE49.SG'
            }
stocks={} # inverse to search for stock symbol
for key, value in stock_dict.items():
    stocks[value]=key

depot = dt.Depot("depot.csv", "order.csv", stock_dict)
currency = depot.currency

def show(stock):
    st.metric(
        label=f"📈 {stock.name}",
        value=f"{stock.changeRel*100:+.1f}%",
        delta=f"{stock.changeAbs:.0f} {currency}"
        )

class StockAnalyzer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        
    def fetch_stock_data(self, symbol, period="1y"):
        """Fetch stock data with error handling"""
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {str(e)}")
            data = None
        try:
            info = stock.info
            esg  = stock.sustainability #Environment, Social and Governance(ESG)
            return data, info
        except Exception as e:
            st.error(f"Error fetching {symbol} sustainability: {str(e)}")
            return data, None
    
    def calculate_technical_indicators(self, data):
        """Calculate comprehensive technical indicators using pure pandas/numpy"""
        df = data.copy()
        
        # Simple Moving Averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        
        # Exponential Moving Averages
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
        df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Price-based indicators
        df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close'] * 100
        df['Price_Change'] = df['Close'] - df['Open']
        df['Price_Change_Pct'] = (df['Close'] - df['Open']) / df['Open'] * 100
        
        # Volatility (Average True Range approximation)
        df['High_Low'] = df['High'] - df['Low']
        df['High_Close'] = np.abs(df['High'] - df['Close'].shift())
        df['Low_Close'] = np.abs(df['Low'] - df['Close'].shift())
        df['True_Range'] = df[['High_Low', 'High_Close', 'Low_Close']].max(axis=1)
        df['ATR'] = df['True_Range'].rolling(window=14).mean()
        
        # Stochastic Oscillator
        low_14 = df['Low'].rolling(window=14).min()
        high_14 = df['High'].rolling(window=14).max()
        df['Stoch_K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
        df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
        
        return df
    
    def prepare_ml_features(self, data):
        """Prepare features for machine learning"""
        df = data.copy()
        
        # Returns and momentum
        df['Returns'] = df['Close'].pct_change()
        df['Returns_5d'] = df['Close'].pct_change(5)
        df['Returns_10d'] = df['Close'].pct_change(10)
        
        # Lag features
        for lag in [1, 2, 3, 5, 10]:
            df[f'Close_lag_{lag}'] = df['Close'].shift(lag)
            df[f'Volume_lag_{lag}'] = df['Volume'].shift(lag)
            df[f'Returns_lag_{lag}'] = df['Returns'].shift(lag)
        
        # Rolling statistics
        for window in [5, 10, 20, 50]:
            df[f'Close_mean_{window}'] = df['Close'].rolling(window).mean()
            df[f'Close_std_{window}'] = df['Close'].rolling(window).std()
            df[f'Volume_mean_{window}'] = df['Volume'].rolling(window).mean()
            df[f'High_mean_{window}'] = df['High'].rolling(window).mean()
            df[f'Low_mean_{window}'] = df['Low'].rolling(window).mean()
        
        # Price position relative to moving averages
        df['Price_vs_SMA20'] = (df['Close'] - df['SMA_20']) / df['SMA_20'] * 100
        df['Price_vs_SMA50'] = (df['Close'] - df['SMA_50']) / df['SMA_50'] * 100
        
        # Volatility features
        df['Price_volatility_10d'] = df['Returns'].rolling(10).std()
        df['Price_volatility_20d'] = df['Returns'].rolling(20).std()
        return df
    
    def train_prediction_model(self, data):
        """Train ML model for price prediction"""
        df = self.prepare_ml_features(data)
        df = df.dropna()
        
        if len(df) < 100:  # Need minimum data
            return None
        
        # Features for prediction (excluding target-related columns)
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits', 
                       'Returns', 'Returns_5d', 'Returns_10d']
        feature_cols = [col for col in df.columns if not any(exc in col for exc in exclude_cols)]
        feature_cols = [col for col in feature_cols if 'lag' in col or 'mean' in col or 
                       'std' in col or col in ['RSI', 'MACD', 'Price_vs_SMA20', 'Price_vs_SMA50', 
                                              'Price_volatility_10d', 'Price_volatility_20d', 'ATR']]
        
        if len(feature_cols) < 5:
            return None
        
        X = df[feature_cols].ffill(limit_area='inside')
        y = df['Close'].shift(-1)  # Predict next day's close
        
        # Remove last row (no target) and any remaining NaN
        X = X[:-1]
        y = y[:-1]
        
        # Remove any remaining NaN values
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        if len(X) < 50:
            return None
        
        # Split and train
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.model.fit(X_train_scaled, y_train)
        
        # Calculate accuracy
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        return {
            'train_score': train_score,
            'test_score': test_score,
            'feature_importance': dict(zip(feature_cols, self.model.feature_importances_)),
            'last_features': X.iloc[-1:],
            'feature_cols': feature_cols
        }
    
    def predict_next_price(self, model_info):
        """Predict next trading day price"""
        if model_info is None:
            return None
        
        last_features_scaled = self.scaler.transform(model_info['last_features'])
        prediction = self.model.predict(last_features_scaled)[0]
        return prediction
    
    def generate_market_analysis(self, data, info, symbol):
        """Generate AI-powered market analysis"""
        latest = data.iloc[-1]
        prev = data.iloc[-2]
        name = f" ({stocks[symbol]})" if symbol in stocks else ""
        # Price movement
        price_change = latest['Close'] - prev['Close']
        price_change_pct = (price_change / prev['Close']) * 100
        
        # Technical analysis
        rsi = latest.get('RSI', 50)
        sma_20 = latest.get('SMA_20', latest['Close'])
        sma_50 = latest.get('SMA_50', latest['Close'])
        bb_upper = latest.get('BB_upper', latest['Close'])
        bb_lower = latest.get('BB_lower', latest['Close'])
        
        # Volume analysis
        avg_volume = data['Volume'].rolling(20).mean().iloc[-1]
        volume_ratio = latest['Volume'] / avg_volume if avg_volume > 0 else 1
        
        # MACD analysis
        macd = latest.get('MACD', 0)
        macd_signal = latest.get('MACD_signal', 0)
        
        # Generate analysis
        analysis = []
        
        # Price trend
        if price_change_pct > 3:
            analysis.append(f"🚀 {symbol}{name} shows exceptional bullish momentum with a {price_change_pct:.2f}% surge")
        elif price_change_pct > 1:
            analysis.append(f"🟢 {symbol}{name} demonstrates strong upward movement (+{price_change_pct:.2f}%)")
        elif price_change_pct > 0:
            analysis.append(f"🟡 {symbol}{name} shows modest gains (+{price_change_pct:.2f}%)")
        elif price_change_pct > -1:
            analysis.append(f"🟡 {symbol}{name} experiences slight decline ({price_change_pct:.2f}%)")
        elif price_change_pct > -3:
            analysis.append(f"🔴 {symbol}{name} shows moderate bearish pressure ({price_change_pct:.2f}%)")
        else:
            analysis.append(f"🔻 {symbol}{name} faces significant selling pressure ({price_change_pct:.2f}%)")
        
        # RSI analysis
        if rsi > 80:
            analysis.append(f"🚨 RSI at {rsi:.1f} indicates severely overbought conditions - potential reversal ahead")
        elif rsi > 70:
            analysis.append(f"⚠️ RSI at {rsi:.1f} shows overbought territory - exercise caution")
        elif rsi < 20:
            analysis.append(f"🛒 RSI at {rsi:.1f} signals severely oversold - strong buying opportunity")
        elif rsi < 30:
            analysis.append(f"💡 RSI at {rsi:.1f} suggests oversold conditions - potential buying opportunity")
        elif 40 <= rsi <= 60:
            analysis.append(f"⚖️ RSI at {rsi:.1f} indicates balanced momentum")
        else:
            analysis.append(f"📊 RSI at {rsi:.1f} shows {('bullish' if rsi > 50 else 'bearish')} bias")
        
        # Moving average analysis
        if latest['Close'] > sma_20 > sma_50:
            analysis.append("📈 Strong bullish alignment - price above both 20 and 50-day MAs")
        elif latest['Close'] < sma_20 < sma_50:
            analysis.append("📉 Bearish trend confirmed - price below key moving averages")
        elif latest['Close'] > sma_20 and sma_20 < sma_50:
            analysis.append("🔄 Mixed signals - short-term bullish but longer-term bearish")
        else:
            analysis.append("➡️ Consolidation phase - awaiting directional breakout")
        
        # Bollinger Bands analysis
        if latest['Close'] > bb_upper:
            analysis.append("📊 Price trading above upper Bollinger Band - potential overbought")
        elif latest['Close'] < bb_lower:
            analysis.append("📊 Price near lower Bollinger Band - potential oversold bounce")
        
        # MACD analysis
        if macd > macd_signal and macd > 0:
            analysis.append("⚡ MACD shows strong bullish momentum")
        elif macd < macd_signal and macd < 0:
            analysis.append("⚡ MACD indicates bearish momentum")
        elif macd > macd_signal:
            analysis.append("⚡ MACD bullish crossover - momentum improving")
        else:
            analysis.append("⚡ MACD bearish crossover - momentum weakening")
        
        # Volume analysis
        if volume_ratio > 2:
            analysis.append("🔥 Exceptional volume surge confirms strong conviction")
        elif volume_ratio > 1.5:
            analysis.append("📊 High volume validates price movement")
        elif volume_ratio < 0.5:
            analysis.append("📊 Below-average volume suggests weak conviction")
        else:
            analysis.append("📊 Normal volume levels")
        
        # Market cap context
        market_cap = info.get('marketCap', 0)
        if market_cap:
            if market_cap > 200e9:  # > 200B
                analysis.append("🏢 Large-cap stability with lower volatility expected")
            elif market_cap > 10e9:  # > 10B
                analysis.append("🏢 Mid-cap stock with balanced growth-stability profile")
            else:
                analysis.append("🏢 Small-cap stock with higher growth potential and volatility")
        
        return analysis

def create_advanced_chart(data, symbol):
    """Create advanced candlestick chart with technical indicators"""
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(f'{symbol} Price Action & Moving Averages', 'Volume', 'MACD', 'RSI & Stochastic'),
        row_heights=[0.5, 0.15, 0.2, 0.15]
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price',
            increasing_line_color='#00ff88',
            decreasing_line_color='#ff4444'
        ),
        row=1, col=1
    )
    
    # Moving averages
    colors = ['#ff9500', '#007aff', '#5856d6']
    mas = [('SMA_20', 'SMA 20'), ('SMA_50', 'SMA 50'), ('SMA_200', 'SMA 200')]
    
    for i, (ma_col, ma_name) in enumerate(mas):
        if ma_col in data.columns and not data[ma_col].isna().all():
            fig.add_trace(
                go.Scatter(x=data.index, y=data[ma_col], 
                          line=dict(color=colors[i], width=1.5), name=ma_name),
                row=1, col=1
            )
    
    # Bollinger Bands
    if all(col in data.columns for col in ['BB_upper', 'BB_lower']):
        fig.add_trace(
            go.Scatter(x=data.index, y=data['BB_upper'], 
                      line=dict(color='rgba(128,128,128,0.5)', width=1), name='BB Upper',
                      showlegend=False),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=data.index, y=data['BB_lower'], 
                      line=dict(color='rgba(128,128,128,0.5)', width=1), name='BB Lower',
                      fill='tonexty', fillcolor='rgba(128,128,128,0.1)',
                      showlegend=False),
            row=1, col=1
        )
    
    # Volume
    volume_colors = ['#00ff88' if data['Close'].iloc[i] >= data['Open'].iloc[i] else '#ff4444' 
                    for i in range(len(data))]
    fig.add_trace(
        go.Bar(x=data.index, y=data['Volume'], 
              marker_color=volume_colors, name='Volume', opacity=0.7),
        row=2, col=1
    )
    
    if 'Volume_SMA' in data.columns:
        fig.add_trace(
            go.Scatter(x=data.index, y=data['Volume_SMA'], 
                      line=dict(color='white', width=1), name='Vol SMA'),
            row=2, col=1
        )
    
    # MACD
    if all(col in data.columns for col in ['MACD', 'MACD_signal', 'MACD_histogram']):
        fig.add_trace(
            go.Scatter(x=data.index, y=data['MACD'], 
                      line=dict(color='#007aff', width=2), name='MACD'),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=data.index, y=data['MACD_signal'], 
                      line=dict(color='#ff9500', width=2), name='Signal'),
            row=3, col=1
        )
        
        histogram_colors = ['#00ff88' if val >= 0 else '#ff4444' for val in data['MACD_histogram']]
        fig.add_trace(
            go.Bar(x=data.index, y=data['MACD_histogram'], 
                  marker_color=histogram_colors, name='Histogram', opacity=0.6),
            row=3, col=1
        )
    
    # RSI and Stochastic
    if 'RSI' in data.columns:
        fig.add_trace(
            go.Scatter(x=data.index, y=data['RSI'], 
                      line=dict(color='#af52de', width=2), name='RSI'),
            row=4, col=1
        )
        # RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.7, row=4, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.7, row=4, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.5, row=4, col=1)
    
    if 'Stoch_K' in data.columns:
        fig.add_trace(
            go.Scatter(x=data.index, y=data['Stoch_K'], 
                      line=dict(color='#ffcc00', width=1.5), name='Stoch %K'),
            row=4, col=1
        )
        fig.add_trace(
            go.Scatter(x=data.index, y=data['Stoch_D'], 
                      line=dict(color='#ff6600', width=1.5), name='Stoch %D'),
            row=4, col=1
        )
    
    fig.update_layout(
        title=f'{symbol} - Technical Analysis Dashboard',
        xaxis_rangeslider_visible=False,
        height=900,
        showlegend=True,
        template='plotly_dark',
        font=dict(size=10)
    )
    
    # Remove x-axis labels from all but bottom subplot
    for i in range(1, 4):
        fig.update_xaxes(showticklabels=False, row=i, col=1)
    
    return fig

def create_performance_metrics(data, symbol):
    """Create performance metrics visualization"""
    # Calculate returns
    data['Daily_Returns'] = data['Close'].pct_change()
    data['Cumulative_Returns'] = (1 + data['Daily_Returns']).cumprod() - 1
    
    # Performance metrics
    total_return = data['Cumulative_Returns'].iloc[-1] * 100
    volatility = data['Daily_Returns'].std() * np.sqrt(252) * 100  # Annualized
    sharpe_ratio = (data['Daily_Returns'].mean() * 252) / (data['Daily_Returns'].std() * np.sqrt(252))
    
    max_drawdown = ((data['Close'] / data['Close'].expanding().max()) - 1).min() * 100
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Return", f"{total_return:.1f}%")
    with col2:
        st.metric("Volatility (Ann.)", f"{volatility:.1f}%")
    with col3:
        st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
    with col4:
        st.metric("Max Drawdown", f"{max_drawdown:.1f}%")
    
    # Cumulative returns chart
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['Cumulative_Returns'] * 100,
            mode='lines',
            name='Cumulative Returns',
            line=dict(color='#00ff88', width=2)
        )
    )
    
    fig.update_layout(
        title=f'{symbol} Cumulative Returns (%)',
        xaxis_title='Date',
        yaxis_title='Cumulative Return (%)',
        template='plotly_dark',
        height=400
    )
    
    st.plotly_chart(fig, width="stretch")

# Streamlit App
def main():  
    depot_title = "💰My Depot"
    default_title = "Dashboard" 
    # Sidebar
    st.sidebar.header("📊 Dashboard Controls")
    st.sidebar.markdown("---")
    
    # Stock selection with popular choices
    popular_stocks = stock_dict
    stock_choice = st.sidebar.selectbox(
        "🏢 Select Stock:",
        options=list(popular_stocks.keys()) + ['Custom'],
        index=4
    )
    
    if stock_choice == 'Custom':
        symbol = st.sidebar.text_input("Enter Stock Symbol:", value="VWSYF", ma1x_chars=10).upper()
    else:
        symbol = popular_stocks[stock_choice]
    
    # Time period
    period = st.sidebar.selectbox(
        "📅 Analysis Period:",
        options=['8h', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y'],
        index=5
    )
    
    st.sidebar.markdown("---")
    
    # Change mode from depot to stock analysis
    st.sidebar.subheader("🔧 Mode")
    show_depot      = st.sidebar.checkbox("💰 Depot or Stock", value=True)
    # Analysis options
    st.sidebar.subheader("🔧 Options for stock mode")
    show_prediction = st.sidebar.checkbox("🔮 ML Price Prediction", value=False)
    show_technical = st.sidebar.checkbox("📈 Technical Charts", value=isProd)
    show_performance = st.sidebar.checkbox("📊 Performance Metrics", value=False)
    show_analysis = st.sidebar.checkbox("🧠 AI Market Analysis", value=isProd)
    show_company = st.sidebar.checkbox("⚖ Company Info", value=False)
    st.sidebar.markdown("---")

    #title of main page
    if 'stock_choice' in locals():
        if show_depot:
            st.title(depot_title)
        else:
            st.title(stock_choice) #popular_stocks.keys[stock_choice])
    else:
        st.title(default_title)

    
    if st.sidebar.button("🔄 Refresh Data", type="primary"):
        st.cache_data.clear()
        st.rerun()
    
    # Initialize analyzer
    analyzer = StockAnalyzer()
    
    # Fetch and display data
    with st.spinner(f"📡 Fetching live data for {symbol}..."):
        data, info = analyzer.fetch_stock_data(symbol, period)
    
    if data is None or data.empty:
        st.error(f"❌ Could not fetch data for {symbol}. Please verify the symbol and try again.")
        st.info("💡 Try popular symbols like AAPL, GOOGL, etc.")
        return
    
    # Calculate technical indicators
    if not show_depot:
        with st.spinner("⚙️ Calculating technical indicators..."):
            data = analyzer.calculate_technical_indicators(data)
    
    # Main dashboard header
    st.markdown("---")
    # Personal Depot
    if show_depot:
        h5=0
        h15=0
        h20=0
        for s in depot.stocks:#st.text(f"Processing {s.name} at {now['Close'].iloc[-1]:.2f} with {now['Close'].size} datasets.")
            s.fetch(period="1mo") # is None for wrong symbol
            try: #fails for constant bonds without s.data
                wd = s.data['Close'].size #workingdays differs on marketplace
                latest =  s.data['Close'].iloc[-1]     #yesterday
                if wd > 1: #2 datasets at least
                    prev =  s.data['Close'].iloc[-2]       #previous day
                    v7   =  s.data['Close'].iloc[-6]       #5 workingdays
                    v30  =  s.data['Close'].iloc[0]        
                else:                    
                    prev = latest #if new day did not start
            except Exception as e:
                latest = s.initial #if missing data for symbol.
                t = "Wrong stock symbol or missing data: {}"
            h15  += s.amount *(latest - s.average(15)) # working days
            h5   += s.amount *(latest - s.average(5)) 
        depot.update()  # show 0.0%, if wrong stock symbol in depot.csv
        #st.text(s.name + f": {s.changeAbs:.2f}€")
        #st.text(depot.show() + str(len(depot.stocks))+" stocks in depot.")
        depot.show() # only in console

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric(
                label="💰 overall, past",
                value=f"{depot.changeAbs/1000:.1f} k{currency}",
                delta=f"{depot.changeRel*100:+.1f}%"
                )
        with col2:
            st.metric(
                label="💰 current depot",
                value=f"{depot.total/1000:.0f} k{currency}",
                delta=f"{depot.changeAbs:.0f} ({depot.changeRel*100:+.1f}%)"                 )
        with col3:
            st.metric(
                label="💰 3 week average",
                value=f"{h15/1000:.1f} k{currency}",
                delta=f"{h15/depot.total*100:+.1f}%" #load info from class depot
                )
        with col4:
            st.metric(
                label="💰 1 week average",
                value=f"{h5:.0f}{currency}",
                delta=f"{h5/depot.total*100:+.1f}%" #Load info from class Depot
                )

    if show_stocks and len(depot.stocks) > 0 and show_depot:
        col1, col2, col3, col4, col5 = st.columns(5) # only oldest stocks
        stock = depot.stocks[0]
        with col1:
            show(stock)
        if len(depot.stocks) > 1:
            stock = depot.stocks[1]
            with col2:
                show(stock)
        if len(depot.stocks) > 2:
            stock = depot.stocks[2]
            with col3:
                show(stock)
        if len(depot.stocks) > 3:
            stock = depot.stocks[3]
            with col4:
                show(stock)
        if len(depot.stocks) > 4:
            stock = depot.stocks[4]
            with col5:
                show(stock)
    
    # Key metrics row.
    if not show_depot:
        col1, col2, col3, col4, col5 = st.columns(5)
        latest_price = data['Close'].iloc[-1]
        prev_price = data['Close'].iloc[-2]
        price_change = latest_price - prev_price
        price_change_pct = (price_change / prev_price) * 100
        
        with col1:
            st.metric(
                label="💰 Current Price",
                value=f"${latest_price:.2f}",
                delta=f"{price_change:.2f} ({price_change_pct:+.2f}%)"
            )
        
        with col2:
            volume = data['Volume'].iloc[-1]
            avg_volume = data['Volume'].rolling(20).mean().iloc[-1]
            volume_change = ((volume - avg_volume) / avg_volume) * 100 if avg_volume > 0 else 0
            st.metric(
                label="📊 Volume",
                value=f"{volume:,.0f}",
                delta=f"{volume_change:+.1f}% vs 20d avg"
            )
        
        with col3:
            if 'RSI' in data.columns and not pd.isna(data['RSI'].iloc[-1]):
                rsi = data['RSI'].iloc[-1]
                rsi_status = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
                st.metric(
                    label="⚡ RSI (14)",
                    value=f"{rsi:.1f}",
                    delta=rsi_status
                )
            else:
                st.metric(label="⚡ RSI (14)", value="N/A")
        
        with col4:
            if 'SMA_20' in data.columns and not pd.isna(data['SMA_20'].iloc[-1]):
                sma_20 = data['SMA_20'].iloc[-1]
                sma_distance = ((latest_price - sma_20) / sma_20) * 100
                st.metric(
                    label="📈 vs 20d-average",
                    value=f"{sma_distance:+.1f}%",
                    delta="Above" if sma_distance > 0 else "Below"
                )
            else:
                st.metric(label="📈 vs SMA 20", value="N/A")
        
        with col5:
            market_cap = info.get('marketCap', 0)
            if market_cap:
                if market_cap > 1e12:
                    cap_display = f"${market_cap/1e12:.2f}T"
                elif market_cap > 1e9:
                    cap_display = f"${market_cap/1e9:.1f}B"
                else:
                    cap_display = f"${market_cap/1e6:.0f}M"
                st.metric(label="🏢 Market Cap", value=cap_display)
            else:
                st.metric(label="🏢 Market Cap", value="N/A")
        
            #st.markdown("---")#with indent the line starts at text below indicators

    # Advanced Chart
    if show_technical and not show_depot:
        st.subheader("📈 Advanced Technical Analysis")
        with st.spinner("Creating advanced charts..."):
            chart = create_advanced_chart(data, symbol)
            st.plotly_chart(chart, width="stretch")
    
    # Performance Metrics
    if show_performance and not show_depot:
        st.subheader("📊 Performance Analysis")
        create_performance_metrics(data, symbol)
    
    # ML Prediction
    if show_prediction and not show_depot:
        st.subheader("🔮 Machine Learning Price Prediction")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            with st.spinner("🤖 Training AI prediction model..."):
                model_info = analyzer.train_prediction_model(data)
            
            if model_info:
                prediction = analyzer.predict_next_price(model_info)
                current_price = data['Close'].iloc[-1]
                predicted_change = ((prediction - current_price) / current_price) * 100
                
                st.success("✅ Model trained successfully!")
                
                pred_col1, pred_col2 = st.columns(2)
                with pred_col1:
                    st.metric(
                        label="🎯 Next Day Prediction",
                        value=f"${prediction:.2f}",
                        delta=f"{predicted_change:+.2f}%"
                    )
                
                with pred_col2:
                    confidence = model_info['test_score']
                    confidence_level = "High" if confidence > 0.8 else "Medium" if confidence > 0.6 else "Low"
                    st.metric(
                        label="🎲 Model Confidence",
                        value=f"{confidence:.1%}",
                        delta=confidence_level
                    )
                
                # Model performance
                st.info(f"📈 **Training Accuracy:** {model_info['train_score']:.1%} | **Test Accuracy:** {model_info['test_score']:.1%}")
            else:
                st.warning("⚠️ Insufficient data for reliable ML prediction. Need more historical data.")
        
        with col2:
            if model_info:
                # Feature importance
                importance_df = pd.DataFrame(
                    list(model_info['feature_importance'].items()),
                    columns=['Feature', 'Importance']
                ).sort_values('Importance', ascending=False).head(10)
                
                fig_importance = px.bar(
                    importance_df, 
                    x='Importance', 
                    y='Feature',
                    orientation='h',
                    title="🔍 Top 10 Most Important Features",
                    template='plotly_dark'
                )
                fig_importance.update_layout(height=400)
                st.plotly_chart(fig_importance, width="stretch")
    
    # AI Market Analysis
    if show_analysis and not show_depot:
        st.subheader("🧠 AI-Powered Market Analysis")
        
        with st.spinner("🤖 Generating intelligent market insights..."):
            analysis = analyzer.generate_market_analysis(data, info, symbol)
        
        # Display analysis in an attractive format
        for i, insight in enumerate(analysis):
            if i == 0:  # First insight (price movement) gets special treatment
                if "🚀" in insight or "🟢" in insight:
                    st.success(insight)
                elif "🔴" in insight or "🔻" in insight:
                    st.error(insight)
                else:
                    st.warning(insight)
            else:
                st.info(insight)
    
    # Additional Analysis Tabs 
    if show_company and not show_depot:
        st.markdown("---")
        tab1, tab2, tab3 = st.tabs(["📋 Company Info", "📊 Raw Data", "🔧 Technical Indicators"])
    
        with tab1:
            if info:
                col1, col2 = st.columns(2)
            
                with col1:
                    st.write("### 🏢 Company Details")
                    company_info = {
                        "Company Name": info.get('longName', 'N/A'),
                        "Sector": info.get('sector', 'N/A'),
                        "Industry": info.get('industry', 'N/A'),
                        "Country": info.get('country', 'N/A'),
                        "Website": info.get('website', 'N/A'),
                        "Employees": f"{info.get('fullTimeEmployees', 'N/A'):,}" if info.get('fullTimeEmployees') else 'N/A'
                    }
                
                    for key, value in company_info.items():
                        st.write(f"**{key}:** {value}")
            
                with col2:
                    st.write("### 📈 Financial Metrics")
                    financial_info = {
                        "P/E Ratio": f"{info.get('trailingPE', 'N/A'):.2f}" if info.get('trailingPE') else 'N/A',
                        "Forward P/E": f"{info.get('forwardPE', 'N/A'):.2f}" if info.get('forwardPE') else 'N/A',
                        "PEG Ratio": f"{info.get('pegRatio', 'N/A'):.2f}" if info.get('pegRatio') else 'N/A',
                        "Price to Book": f"{info.get('priceToBook', 'N/A'):.2f}" if info.get('priceToBook') else 'N/A',
                        "Dividend Yield": f"{info.get('dividendYield', 0):.2f}%" if info.get('dividendYield') else 'N/A',
                        "Beta": f"{info.get('beta', 'N/A'):.2f}" if info.get('beta') else 'N/A',
                        "52W High": f"${info.get('fiftyTwoWeekHigh', 'N/A'):.2f}" if info.get('fiftyTwoWeekHigh') else 'N/A',
                        "52W Low": f"${info.get('fiftyTwoWeekLow', 'N/A'):.2f}" if info.get('fiftyTwoWeekLow') else 'N/A'
                    }
                
                    for key, value in financial_info.items():
                        st.write(f"**{key}:** {value}")
            else:
                st.warning("Company information not available")
    
        with tab2:
            st.write("### 📊 Recent Price Data")
            display_data = data[['Open', 'High', 'Low', 'Close', 'Volume']].tail(20)
            display_data.index = display_data.index.strftime('%Y-%m-%d')
            st.dataframe(display_data, width="stretch")
        
            # Download option
            csv = display_data.to_csv()
            st.download_button(
                label="📥 Download Data as CSV",
                data=csv,
                file_name=f'{symbol}_stock_data.csv',
                mime='text/csv'
            )
    
        with tab3:
            st.write("### 🔧 Technical Indicators (Last 10 Days)")
        
            tech_columns = ['Close', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 'MACD_signal', 'BB_upper', 'BB_lower', 'ATR']
            available_columns = [col for col in tech_columns if col in data.columns]
        
            if available_columns:
                tech_data = data[available_columns].tail(10)
                tech_data.index = tech_data.index.strftime('%Y-%m-%d')
                st.dataframe(tech_data.round(3), width="stretch")
            else:
                st.warning("Technical indicators not available")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 20px;'>
            <p>🚀 <strong>Depot Dashboard</strong> - Dividends, coupons and depot growth and technical analysis</p>
            <p><em>⚠️ This is for educational purposes only. Not financial advice.</em></p>
            <p>Based on AI Stock Dashboard by <a href='https://erikthiart.com' target='_blank'>Erik Thiart</a>, extended by W20Physik, 📊 Powered by <a href='https://plotly.com' target='_blank'>Plotly</a> and <a href='https://streamlit.io' target='_blank'>Streamlit</a></p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
