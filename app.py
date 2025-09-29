import streamlit as st
import pandas as pd
from vnstock3 import Vnstock
import plotly.graph_objects as go

st.title("📊 Quant Trading Demo - VNStock3")

# Nhập mã cổ phiếu
symbol = st.text_input("Nhập mã cổ phiếu (VD: VNM, FPT, SSI):", "VNM")

# Lấy dữ liệu từ VNStock3
vn = Vnstock()
stock = vn.stock(symbol, "HOSE")

df = stock.quote.history(
    start="2024-01-01", 
    end="2024-12-31", 
    interval="1D"
)

st.subheader(f"Dữ liệu giá {symbol}")
st.dataframe(df.tail())

# Vẽ biểu đồ nến
fig = go.Figure(data=[go.Candlestick(
    x=df['time'],
    open=df['open'],
    high=df['high'],
    low=df['low'],
    close=df['close'],
    name="OHLC"
)])
fig.update_layout(xaxis_rangeslider_visible=False, title=f"Biểu đồ nến {symbol}")
st.plotly_chart(fig)

# Tính SMA
df["SMA20"] = df["close"].rolling(20).mean()
df["SMA50"] = df["close"].rolling(50).mean()

# Vẽ chiến lược SMA Crossover
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=df["time"], y=df["close"], mode="lines", name="Close"))
fig2.add_trace(go.Scatter(x=df["time"], y=df["SMA20"], mode="lines", name="SMA20"))
fig2.add_trace(go.Scatter(x=df["time"], y=df["SMA50"], mode="lines", name="SMA50"))
fig2.update_layout(title="Chiến lược SMA Crossover")
st.plotly_chart(fig2)

# Sinh tín hiệu mua/bán
df["Signal"] = 0
df.loc[(df["SMA20"] > df["SMA50"]) & (df["SMA20"].shift(1) <= df["SMA50"].shift(1)), "Signal"] = 1
df.loc[(df["SMA20"] < df["SMA50"]) & (df["SMA20"].shift(1) >= df["SMA50"].shift(1)), "Signal"] = -1

st.subheader("📌 Tín hiệu giao dịch")
st.dataframe(df[["time", "close", "SMA20", "SMA50", "Signal"]].tail(20))
