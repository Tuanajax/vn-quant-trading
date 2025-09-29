import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from vnstock3 import Vnstock
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="VN Quant Trading Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize vnstock
@st.cache_resource
def get_stock_data():
    return Vnstock()

stock = get_stock_data()

# Sidebar
st.sidebar.title("🎯 VN Quant Platform")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Chọn tính năng:",
    ["📊 Trading Signals", "💼 Portfolio Optimization", "🌐 Market Overview"]
)

# Helper functions
@st.cache_data(ttl=3600)
def load_stock_data(symbol, start_date, end_date):
    """Load stock data from vnstock"""
    try:
        df = stock.stock(symbol=symbol, source='VCI').quote.history(
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d')
        )
        return df
    except Exception as e:
        st.error(f"Lỗi khi tải dữ liệu {symbol}: {str(e)}")
        return None

def calculate_rsi(data, period=14):
    """Calculate RSI indicator"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calculate MACD indicator"""
    ema_fast = data.ewm(span=fast, adjust=False).mean()
    ema_slow = data.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_bollinger_bands(data, period=20, std_dev=2):
    """Calculate Bollinger Bands"""
    sma = data.rolling(window=period).mean()
    std = data.rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return upper_band, sma, lower_band

def generate_signals(df):
    """Generate buy/sell signals based on technical indicators"""
    signals = {}
    close = df['close']
    
    # RSI Signal
    rsi = calculate_rsi(close)
    latest_rsi = rsi.iloc[-1]
    if latest_rsi < 30:
        signals['RSI'] = {'signal': 'BUY', 'value': latest_rsi, 'strength': 'Strong'}
    elif latest_rsi > 70:
        signals['RSI'] = {'signal': 'SELL', 'value': latest_rsi, 'strength': 'Strong'}
    else:
        signals['RSI'] = {'signal': 'HOLD', 'value': latest_rsi, 'strength': 'Neutral'}
    
    # MACD Signal
    macd, signal_line, _ = calculate_macd(close)
    if macd.iloc[-1] > signal_line.iloc[-1] and macd.iloc[-2] <= signal_line.iloc[-2]:
        signals['MACD'] = {'signal': 'BUY', 'strength': 'Medium'}
    elif macd.iloc[-1] < signal_line.iloc[-1] and macd.iloc[-2] >= signal_line.iloc[-2]:
        signals['MACD'] = {'signal': 'SELL', 'strength': 'Medium'}
    else:
        signals['MACD'] = {'signal': 'HOLD', 'strength': 'Neutral'}
    
    # Moving Average Signal
    ma20 = close.rolling(window=20).mean()
    ma50 = close.rolling(window=50).mean()
    if ma20.iloc[-1] > ma50.iloc[-1] and close.iloc[-1] > ma20.iloc[-1]:
        signals['MA'] = {'signal': 'BUY', 'strength': 'Medium'}
    elif ma20.iloc[-1] < ma50.iloc[-1] and close.iloc[-1] < ma20.iloc[-1]:
        signals['MA'] = {'signal': 'SELL', 'strength': 'Medium'}
    else:
        signals['MA'] = {'signal': 'HOLD', 'strength': 'Neutral'}
    
    # Bollinger Bands Signal
    upper, middle, lower = calculate_bollinger_bands(close)
    if close.iloc[-1] < lower.iloc[-1]:
        signals['BB'] = {'signal': 'BUY', 'strength': 'Medium'}
    elif close.iloc[-1] > upper.iloc[-1]:
        signals['BB'] = {'signal': 'SELL', 'strength': 'Medium'}
    else:
        signals['BB'] = {'signal': 'HOLD', 'strength': 'Neutral'}
    
    return signals

def calculate_portfolio_metrics(returns, weights):
    """Calculate portfolio return and risk"""
    portfolio_return = np.sum(returns.mean() * weights) * 252
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    sharpe_ratio = portfolio_return / portfolio_std if portfolio_std > 0 else 0
    return portfolio_return, portfolio_std, sharpe_ratio

# PAGE 1: Trading Signals
if page == "📊 Trading Signals":
    st.title("📊 Tín hiệu Mua/Bán")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        symbol = st.text_input("Nhập mã cổ phiếu (VD: VCB, VHM, HPG):", "VCB").upper()
    
    with col2:
        days = st.selectbox("Khoảng thời gian:", [30, 60, 90, 180, 365], index=2)
    
    if st.button("🔍 Phân tích", type="primary"):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        with st.spinner(f"Đang tải dữ liệu {symbol}..."):
            df = load_stock_data(symbol, start_date, end_date)
        
        if df is not None and not df.empty:
            # Display current price
            current_price = df['close'].iloc[-1]
            prev_price = df['close'].iloc[-2]
            price_change = current_price - prev_price
            price_change_pct = (price_change / prev_price) * 100
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Giá hiện tại", f"{current_price:,.0f}", f"{price_change_pct:+.2f}%")
            with col2:
                st.metric("Cao nhất", f"{df['high'].max():,.0f}")
            with col3:
                st.metric("Thấp nhất", f"{df['low'].min():,.0f}")
            with col4:
                st.metric("Volume TB", f"{df['volume'].mean()/1000000:.2f}M")
            
            # Generate signals
            signals = generate_signals(df)
            
            # Display signals
            st.subheader("🎯 Tín hiệu giao dịch")
            
            signal_cols = st.columns(4)
            for idx, (indicator, signal_data) in enumerate(signals.items()):
                with signal_cols[idx]:
                    signal = signal_data['signal']
                    color = "🟢" if signal == "BUY" else "🔴" if signal == "SELL" else "🟡"
                    st.markdown(f"**{indicator}**")
                    st.markdown(f"{color} **{signal}**")
                    st.caption(signal_data['strength'])
            
            # Calculate overall signal
            buy_count = sum(1 for s in signals.values() if s['signal'] == 'BUY')
            sell_count = sum(1 for s in signals.values() if s['signal'] == 'SELL')
            
            st.markdown("---")
            if buy_count > sell_count:
                st.success(f"### 🟢 TÍN HIỆU TỔNG HỢP: MUA ({buy_count}/4 indicators)")
            elif sell_count > buy_count:
                st.error(f"### 🔴 TÍN HIỆU TỔNG HỢP: BÁN ({sell_count}/4 indicators)")
            else:
                st.info(f"### 🟡 TÍN HIỆU TỔNG HỢP: GIỮ")
            
            # Price chart with indicators
            st.subheader("📈 Biểu đồ giá và chỉ báo")
            
            fig = go.Figure()
            
            # Candlestick
            fig.add_trace(go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Price'
            ))
            
            # Moving averages
            ma20 = df['close'].rolling(window=20).mean()
            ma50 = df['close'].rolling(window=50).mean()
            
            fig.add_trace(go.Scatter(x=df.index, y=ma20, name='MA20', line=dict(color='orange', width=1)))
            fig.add_trace(go.Scatter(x=df.index, y=ma50, name='MA50', line=dict(color='blue', width=1)))
            
            # Bollinger Bands
            upper, middle, lower = calculate_bollinger_bands(df['close'])
            fig.add_trace(go.Scatter(x=df.index, y=upper, name='BB Upper', line=dict(color='gray', width=1, dash='dash')))
            fig.add_trace(go.Scatter(x=df.index, y=lower, name='BB Lower', line=dict(color='gray', width=1, dash='dash')))
            
            fig.update_layout(
                title=f'{symbol} - Biểu đồ nến Nhật',
                yaxis_title='Giá (VND)',
                xaxis_title='Ngày',
                height=500,
                xaxis_rangeslider_visible=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Technical indicators charts
            col1, col2 = st.columns(2)
            
            with col1:
                # RSI chart
                rsi = calculate_rsi(df['close'])
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(x=df.index, y=rsi, name='RSI', line=dict(color='purple')))
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
                fig_rsi.update_layout(title='RSI (14)', height=300)
                st.plotly_chart(fig_rsi, use_container_width=True)
            
            with col2:
                # MACD chart
                macd, signal_line, histogram = calculate_macd(df['close'])
                fig_macd = go.Figure()
                fig_macd.add_trace(go.Scatter(x=df.index, y=macd, name='MACD', line=dict(color='blue')))
                fig_macd.add_trace(go.Scatter(x=df.index, y=signal_line, name='Signal', line=dict(color='orange')))
                fig_macd.add_trace(go.Bar(x=df.index, y=histogram, name='Histogram'))
                fig_macd.update_layout(title='MACD', height=300)
                st.plotly_chart(fig_macd, use_container_width=True)

# PAGE 2: Portfolio Optimization
elif page == "💼 Portfolio Optimization":
    st.title("💼 Tối ưu hóa Danh mục Đầu tư")
    
    st.markdown("""
    Nhập danh sách mã cổ phiếu để tối ưu hóa danh mục theo phương pháp Modern Portfolio Theory.
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        symbols_input = st.text_input(
            "Nhập mã cổ phiếu (cách nhau bởi dấu phẩy):",
            "VCB,VHM,HPG,VNM,TCB"
        )
        symbols = [s.strip().upper() for s in symbols_input.split(',')]
    
    with col2:
        investment = st.number_input("Vốn đầu tư (triệu VND):", min_value=1, value=100)
        days_portfolio = st.selectbox("Dữ liệu lịch sử:", [180, 365, 730], index=1)
    
    optimization_method = st.radio(
        "Phương pháp tối ưu:",
        ["Max Sharpe Ratio", "Min Volatility", "Equal Weight"],
        horizontal=True
    )
    
    if st.button("🎯 Tối ưu hóa", type="primary"):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_portfolio)
        
        with st.spinner("Đang tải dữ liệu và tối ưu hóa..."):
            # Load data for all symbols
            all_data = {}
            for sym in symbols:
                df = load_stock_data(sym, start_date, end_date)
                if df is not None and not df.empty:
                    all_data[sym] = df['close']
            
            if len(all_data) < 2:
                st.error("Cần ít nhất 2 mã cổ phiếu có dữ liệu hợp lệ!")
            else:
                # Create price dataframe
                prices_df = pd.DataFrame(all_data)
                returns = prices_df.pct_change().dropna()
                
                # Calculate optimal weights
                n_assets = len(prices_df.columns)
                
                if optimization_method == "Equal Weight":
                    weights = np.array([1/n_assets] * n_assets)
                else:
                    # Monte Carlo simulation for optimization
                    num_portfolios = 10000
                    results = np.zeros((3, num_portfolios))
                    weights_record = []
                    
                    for i in range(num_portfolios):
                        w = np.random.random(n_assets)
                        w = w / np.sum(w)
                        weights_record.append(w)
                        
                        port_return, port_std, sharpe = calculate_portfolio_metrics(returns, w)
                        results[0,i] = port_return
                        results[1,i] = port_std
                        results[2,i] = sharpe
                    
                    if optimization_method == "Max Sharpe Ratio":
                        max_sharpe_idx = np.argmax(results[2])
                        weights = weights_record[max_sharpe_idx]
                    else:  # Min Volatility
                        min_vol_idx = np.argmin(results[1])
                        weights = weights_record[min_vol_idx]
                
                # Calculate portfolio metrics
                port_return, port_std, sharpe = calculate_portfolio_metrics(returns, weights)
                
                # Display results
                st.success("✅ Tối ưu hóa thành công!")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Lợi nhuận kỳ vọng", f"{port_return*100:.2f}%")
                with col2:
                    st.metric("Rủi ro (Volatility)", f"{port_std*100:.2f}%")
                with col3:
                    st.metric("Sharpe Ratio", f"{sharpe:.2f}")
                with col4:
                    st.metric("Số cổ phiếu", len(symbols))
                
                # Allocation table
                st.subheader("📊 Phân bổ Danh mục Tối ưu")
                
                allocation_df = pd.DataFrame({
                    'Mã CK': prices_df.columns,
                    'Tỷ trọng (%)': weights * 100,
                    'Vốn (triệu)': weights * investment,
                    'Giá hiện tại': [prices_df[col].iloc[-1] for col in prices_df.columns],
                })
                allocation_df['Số CP'] = (allocation_df['Vốn (triệu)'] * 1000000 / allocation_df['Giá hiện tại']).astype(int)
                allocation_df = allocation_df.sort_values('Tỷ trọng (%)', ascending=False)
                
                st.dataframe(allocation_df.style.format({
                    'Tỷ trọng (%)': '{:.2f}',
                    'Vốn (triệu)': '{:.2f}',
                    'Giá hiện tại': '{:,.0f}',
                    'Số CP': '{:,}'
                }), use_container_width=True)
                
                # Visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    # Pie chart
                    fig_pie = px.pie(
                        allocation_df,
                        values='Tỷ trọng (%)',
                        names='Mã CK',
                        title='Phân bổ Danh mục'
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col2:
                    # Risk-Return scatter
                    if optimization_method != "Equal Weight":
                        fig_scatter = go.Figure()
                        fig_scatter.add_trace(go.Scatter(
                            x=results[1] * 100,
                            y=results[0] * 100,
                            mode='markers',
                            marker=dict(
                                size=5,
                                color=results[2],
                                colorscale='Viridis',
                                showscale=True,
                                colorbar=dict(title="Sharpe Ratio")
                            ),
                            text=[f'Sharpe: {s:.2f}' for s in results[2]],
                            name='Portfolios'
                        ))
                        fig_scatter.add_trace(go.Scatter(
                            x=[port_std * 100],
                            y=[port_return * 100],
                            mode='markers',
                            marker=dict(size=15, color='red', symbol='star'),
                            name='Optimal Portfolio'
                        ))
                        fig_scatter.update_layout(
                            title='Efficient Frontier',
                            xaxis_title='Risk (Volatility %)',
                            yaxis_title='Return (%)',
                            height=400
                        )
                        st.plotly_chart(fig_scatter, use_container_width=True)
                
                # Correlation matrix
                st.subheader("🔗 Ma trận Tương quan")
                corr_matrix = returns.corr()
                fig_corr = px.imshow(
                    corr_matrix,
                    text_auto='.2f',
                    color_continuous_scale='RdBu_r',
                    aspect='auto',
                    title='Ma trận tương quan giữa các cổ phiếu'
                )
                st.plotly_chart(fig_corr, use_container_width=True)

# PAGE 3: Market Overview
elif page == "🌐 Market Overview":
    st.title("🌐 Tổng quan Thị trường")
    
    st.info("📌 Tính năng đang được phát triển. Sẽ bao gồm: VN-Index, Top gainers/losers, Market breadth, Sector performance")
    
    # Market indices placeholder
    st.subheader("📊 Chỉ số Thị trường")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("VN-Index", "1,245.67", "+2.3%")
    with col2:
        st.metric("HNX-Index", "234.56", "-0.5%")
    with col3:
        st.metric("UPCOM-Index", "89.12", "+1.2%")
    
    st.markdown("---")
    st.markdown("""
    ### 🚀 Tính năng sắp ra mắt:
    - 📈 Top cổ phiếu tăng/giảm mạnh
    - 💰 Thanh khoản thị trường
    - 🏭 Hiệu suất theo ngành
    - 📰 Tin tức thị trường
    - 🔥 Cổ phiếu hot nhất
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📚 Hướng dẫn:
- **Trading Signals**: Phân tích kỹ thuật và tín hiệu mua/bán
- **Portfolio Optimization**: Tối ưu danh mục theo MPT
- **Market Overview**: Tổng quan thị trường VN

### ⚠️ Lưu ý:
Đây chỉ là công cụ hỗ trợ phân tích. 
Không phải lời khuyên đầu tư.
""")
