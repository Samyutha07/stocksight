import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
import numpy as np


st.set_page_config(page_title="StockSight", page_icon="📈", layout="wide")

st.title("📈 StockSight")
st.write("Analyse any stock with real data and ML predictions")

ticker = st.sidebar.text_input("Enter stock ticker", value="AAPL")
period = st.sidebar.selectbox("Time period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"])

st.sidebar.markdown("---")
load = st.sidebar.button("Load Stock Data")

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

if load:
    st.subheader(f"Showing data for: {ticker.upper()}")

    data = yf.download(ticker, period=period, progress=False)
    data = data.droplevel(1, axis=1)
    ticker_obj = yf.Ticker(ticker)
    info = ticker_obj.info

    if data.empty:
        st.error("Could not find the ticker. Try AAPL, TSLA, GOOGL.")
    else:
        st.success(f"Loaded {len(data)} days of data!")
        with st.expander("📋 View Raw Data"):
            display_data = data.tail(10).copy()
            display_data.index = display_data.index.strftime("%Y-%m-%d")
            st.dataframe(display_data)

        # Key Stats
        price_change = data['Close'].iloc[-1] - data['Close'].iloc[-2]
        pct_change = (price_change / data['Close'].iloc[-2]) * 100

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Current Price", f"${data['Close'].iloc[-1]:.2f}", f"{pct_change:.2f}%")
        col2.metric("Market Cap", f"${info.get('marketCap', 0)/1e9:.2f}B")
        col3.metric("52W High", f"${info.get('fiftyTwoWeekHigh', 'N/A')}")
        col4.metric("52W Low", f"${info.get('fiftyTwoWeekLow', 'N/A')}")

        # Company Info
        with st.expander("Company Info"):
            st.write(info.get("longBusinessSummary", "No description available."))
            st.write(f"**Sector:** {info.get('sector', 'N/A')}")
            st.write(f"**Industry:** {info.get('industry', 'N/A')}")
            st.write(f"**Website:** {info.get('website', 'N/A')}")

        # Price Chart
        st.subheader("Price Chart")

        data["MA20"] = data["Close"].rolling(window=20).mean()
        data["MA50"] = data["Close"].rolling(window=50).mean()

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            vertical_spacing=0.05,
                            row_heights=[0.7, 0.3])

        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name="Price"
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=data.index,
            y=data["MA20"],
            name="20-day MA",
            line=dict(color="cyan", width=1.5)
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=data.index,
            y=data["MA50"],
            name="50-day MA",
            line=dict(color="yellow", width=1.5)
        ), row=1, col=1)

        fig.add_trace(go.Bar(
            x=data.index,
            y=data["Volume"],
            name="Volume",
            marker_color="rgba(100, 149, 237, 0.5)"
        ), row=2, col=1)

        fig.update_layout(
            title=f"{ticker.upper()} Price History",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            template="plotly_dark",
            height=500,
            xaxis_rangeslider_visible=False
        )

        st.plotly_chart(fig, width="stretch")

        st.subheader("RSI - Relative Strength Index")

        data["RSI"] = compute_rsi(data["Close"])

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=data.index, y=data["RSI"], name="RSI", line=dict(color="violet")))
        fig3.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
        fig3.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
        fig3.update_layout(template="plotly_dark", height=300, yaxis=dict(range=[0, 100]))

        st.plotly_chart(fig3, use_container_width=True)

        # ML Price Prediction
        st.subheader("ML Price Prediction (Linear Regression)")

        close_prices = data["Close"].values
        X = np.arange(len(close_prices)).reshape(-1, 1)
        y = close_prices

        model = LinearRegression()
        model.fit(X, y)

        future_days = 30
        future_X = np.arange(len(close_prices), len(close_prices) + future_days).reshape(-1, 1)
        predicted = model.predict(future_X)

        last_date = data.index[-1]
        future_dates = pd.date_range(start=last_date, periods=future_days + 1, freq="B")[1:]

        fig2 = go.Figure()

        fig2.add_trace(go.Scatter(
            x=data.index, y=data["Close"],
            name="Actual Price", line=dict(color="cyan")
        ))

        fig2.add_trace(go.Scatter(
            x=future_dates, y=predicted,
            name="Predicted", line=dict(color="orange", dash="dash")
        ))

        fig2.update_layout(
            title=f"{ticker.upper()} Price Prediction — Next 30 Days",
            template="plotly_dark",
            height=400
        )

        st.plotly_chart(fig2, width="stretch")