# peak_streamlit_app.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from sklearn.metrics import classification_report

st.set_page_config(page_title="BIST/XU100 Directional Trading", layout="wide")
st.title("📈 Directional Trading & Regime Detection")

# =====================================================
# 1. USER INPUT
# =====================================================
ticker = st.text_input("Enter Stock/Index Ticker (e.g., XU100.IS)", value="XU100.IS")
years = st.number_input("Number of Years to Download (max 5)", min_value=1, max_value=5, value=3)
transaction_cost = st.number_input("Transaction Cost per Trade (%)", min_value=0.0, max_value=1.0, value=0.0)/100

# =====================================================
# 2. DOWNLOAD DATA
# =====================================================
@st.cache_data
def download_stock_data(ticker, years):
    end = pd.Timestamp.today()
    start = end - pd.DateOffset(years=years)
    data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if data.empty:
        return None
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data = data[['Open','High','Low','Close']].dropna()
    return data

data = download_stock_data(ticker, years)

if data is None or len(data) < 30:
    st.error("❌ Not enough data to run the analysis. Please try a different ticker or longer period.")
    st.stop()

st.success(f"✓ Downloaded {len(data)} trading days: {data.index[0].date()} → {data.index[-1].date()}")

# =====================================================
# 3. VISUAL PEAK/TROUGH DETECTION
# =====================================================
def detect_peaks_troughs(prices, smoothing_sigma=3, min_distance=10, prominence_factor=0.5):
    prices_smooth = gaussian_filter1d(prices.values, sigma=smoothing_sigma)
    volatility = prices.pct_change().std()
    avg_price = prices.mean()
    prominence_threshold = avg_price * volatility * prominence_factor
    peaks, _ = find_peaks(prices_smooth, distance=min_distance, prominence=prominence_threshold)
    troughs, _ = find_peaks(-prices_smooth, distance=min_distance, prominence=prominence_threshold)
    return prices_smooth, peaks, troughs

prices = data['Close']
prices_smooth, peaks, troughs = detect_peaks_troughs(prices)

st.subheader("📌 Detected Peaks & Troughs")
st.write(f"Peaks (SHORT signals): {len(peaks)}")
st.write(f"Troughs (LONG signals): {len(troughs)}")

# =====================================================
# 4. BACKTESTING FUNCTION
# =====================================================
def backtest_directional(prices, peaks, troughs, transaction_cost=0.0):
    n = len(prices)
    returns = np.zeros(n)
    positions = np.zeros(n)
    trades = []

    all_signals = [(idx, -1, 'PEAK', 'SHORT') for idx in peaks] + [(idx, 1, 'TROUGH', 'LONG') for idx in troughs]
    all_signals.sort(key=lambda x: x[0])

    current_position = 0
    for signal_idx, position_type, signal_name, action in all_signals:
        signal_price = prices.iloc[signal_idx]
        signal_date = prices.index[signal_idx]
        if position_type != current_position:
            # Close existing
            if current_position != 0:
                entry_price, entry_date, entry_signal = trade_entry
                price_change = (signal_price - entry_price)/entry_price
                exit_return = (price_change - transaction_cost) if current_position==1 else (-price_change - transaction_cost)
                trades.append({
                    'Entry_Date': entry_date,
                    'Entry_Signal': entry_signal,
                    'Entry_Price': entry_price,
                    'Exit_Date': signal_date,
                    'Exit_Signal': signal_name,
                    'Exit_Price': signal_price,
                    'Position': 'LONG' if current_position==1 else 'SHORT',
                    'Return_%': exit_return*100,
                    'Holding_Days': (signal_date-entry_date).days
                })
            # Enter new
            current_position = position_type
            trade_entry = (signal_price, signal_date, signal_name)
        positions[signal_idx:] = current_position

    # Calculate returns
    for i in range(1,n):
        if positions[i-1]!=0:
            price_return = (prices.iloc[i]-prices.iloc[i-1])/prices.iloc[i-1]
            returns[i] = price_return if positions[i-1]==1 else -price_return

    cum_returns = np.cumsum(returns)
    cum_log_returns = np.cumsum(np.log1p(returns))
    total_return = np.exp(cum_log_returns[-1])-1
    num_years = len(prices)/252
    annualized_return = (1+total_return)**(1/num_years)-1
    sharpe_ratio_val = np.mean(returns)/ (np.std(returns)+1e-10) * np.sqrt(252)
    cumulative_wealth = np.exp(cum_log_returns)
    running_max = np.maximum.accumulate(cumulative_wealth)
    max_drawdown = np.min((cumulative_wealth - running_max)/running_max)
    return {'returns': returns, 'cum_returns': cum_returns, 'cum_log_returns': cum_log_returns,
            'positions': positions, 'trades': trades,
            'metrics': {'total_return': total_return,
                        'annualized_return': annualized_return,
                        'sharpe_ratio': sharpe_ratio_val,
                        'max_drawdown': max_drawdown,
                        'num_trades': len(trades)}}

backtest = backtest_directional(prices, peaks, troughs, transaction_cost)

st.subheader("📊 Backtest Metrics")
st.write(backtest['metrics'])

# =====================================================
# 5. FEATURE ENGINEERING FOR RANDOM FOREST
# =====================================================
df_rf = data.copy()
df_rf['Return'] = df_rf['Close'].pct_change()
df_rf['CumRet_3']  = df_rf['Return'].rolling(3).sum()
df_rf['CumRet_4']  = df_rf['Return'].rolling(4).sum()
df_rf['CumRet_5']  = df_rf['Return'].rolling(5).sum()
df_rf['CumRet_14'] = df_rf['Return'].rolling(10).sum()
delta = df_rf['Close'].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
avg_gain = gain.rolling(10).mean()
avg_loss = loss.rolling(10).mean()
rs = avg_gain / (avg_loss + 1e-10)
df_rf['RSI_14'] = 100 - (100/(1+rs))
df_rf['High_5'] = df_rf['High'].rolling(4).max()
df_rf['Low_5'] = df_rf['Low'].rolling(4).min()
df_rf['MA_4'] = df_rf['Close'].rolling(4).mean()
df_rf['MA_9'] = df_rf['Close'].rolling(9).mean()

# Build simple regime from peaks/troughs
regime = np.zeros(len(df_rf))
regime[peaks] = -1
regime[troughs] = 1
df_rf['Regime'] = regime
df_rf = df_rf.dropna()

FEATURES = ['CumRet_3','CumRet_4','CumRet_5','High_5','Low_5','RSI_14','MA_4','MA_9']
TARGET = 'Regime'

X = df_rf[FEATURES].values
y = df_rf[TARGET].values
prices_rf = df_rf['Close'].values

# 80/20 split
split = int(0.8*len(df_rf))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
prices_train, prices_test = prices_rf[:split], prices_rf[split:]

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# =====================================================
# 6. RANDOM FOREST WITH CLASS WEIGHT CHECK
# =====================================================
unique_classes = np.unique(y_train)
rf_trained = False
if len(unique_classes) < 2:
    st.warning("⚠️ Not enough class variation to train Random Forest. Skipping RF training.")
else:
    class_weights_array = class_weight.compute_class_weight('balanced', classes=unique_classes, y=y_train)
    class_weights_dict = dict(enumerate(class_weights_array))
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=5,
        min_samples_leaf=5,
        class_weight=class_weights_dict,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_trained = True
    st.success("✅ Random Forest trained successfully.")

# =====================================================
# 7. LATEST SIGNAL PREDICTION
# =====================================================
latest_row = df_rf.iloc[[-1]]
latest_date = latest_row.index[0]
latest_close = latest_row['Close'].values[0]
st.subheader("📌 Latest Observation")
st.write(f"Date: {latest_date.date()}, Close Price: {latest_close:.2f}")

if rf_trained:
    X_latest = scaler.transform(latest_row[FEATURES].values)
    prob_latest = rf.predict_proba(X_latest)[:,1][0]
    # Quick threshold
    position_latest = 'HOLD'
    if prob_latest >= 0.6:
        position_latest = 'LONG'
    elif prob_latest <= 0.4:
        position_latest = 'SHORT'
    st.write(f"Predicted probability of LONG: {prob_latest:.4f}")
    st.write(f"Recommended Position: {position_latest}")
else:
    st.info("Random Forest not trained — cannot predict latest position.")

# =====================================================
# 8. PLOTS
# =====================================================
st.subheader("📉 Price & Signals")
fig, ax = plt.subplots(figsize=(14,5))
ax.plot(prices.index, prices.values, label='Actual Price', color='blue')
ax.plot(prices.index, prices_smooth, label='Smoothed Price', color='darkblue', linewidth=2)
ax.scatter(prices.index[peaks], prices.iloc[peaks], color='red', marker='v', s=100, label='PEAK - SHORT')
ax.scatter(prices.index[troughs], prices.iloc[troughs], color='green', marker='^', s=100, label='TROUGH - LONG')
ax.set_title(f'{ticker} - Peaks & Troughs')
ax.set_xlabel('Date'); ax.set_ylabel('Price')
ax.legend(); ax.grid(True)
st.pyplot(fig)

st.subheader("📈 Cumulative Returns")
fig2, ax2 = plt.subplots(figsize=(14,5))
cum_log_returns = backtest['cum_log_returns']
positions_plot = backtest['positions']
ax2.plot(prices.index, cum_log_returns, label='Strategy')
ax2.fill_between(prices.index, 0, cum_log_returns, where=positions_plot==1, color='green', alpha=0.2, label='LONG')
ax2.fill_between(prices.index, 0, cum_log_returns, where=positions_plot==-1, color='red', alpha=0.2, label='SHORT')
ax2.set_title("Cumulative Log Returns")
ax2.set_xlabel('Date'); ax2.set_ylabel('Cumulative Return')
ax2.legend(); ax2.grid(True)
st.pyplot(fig2)
