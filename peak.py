# streamlit_app.py
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
import datetime
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Directional Trading App", layout="wide")

st.title("📈 Directional Trading & Regime Detection App")
st.markdown("""
This app downloads stock/index data, detects **peaks/troughs**, backtests a directional strategy,
and builds a **Random Forest regime model** to predict LONG/SHORT/HOLD signals.
""")

# =====================================================
# User Inputs
# =====================================================
ticker = st.text_input("Enter Ticker (e.g., XU100.IS):", value="XU100.IS")

years = st.selectbox("Number of years of historical data:", options=[1, 2, 3, 4, 5], index=2)

end_date = datetime.date.today()
start_date = end_date - datetime.timedelta(days=years*365)

transaction_cost = st.number_input("Transaction cost per trade (%)", min_value=0.0, max_value=5.0, value=0.0, step=0.1)/100

# =====================================================
# Download Data
# =====================================================
@st.cache_data
def download_stock_data(ticker, start, end):
    try:
        data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
        if data.empty:
            return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data = data[['Open', 'High', 'Low', 'Close']].dropna()
        data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
        data = data.dropna(subset=['Close'])
        return data
    except:
        return None

data_load_state = st.text("Downloading data...")
data = download_stock_data(ticker, start_date, end_date)
data_load_state.text("Data downloaded ✅")

if data is None or data.empty:
    st.error("Failed to download data. Check ticker symbol.")
    st.stop()

st.write(f"Data from {data.index[0].date()} to {data.index[-1].date()} ({len(data)} trading days)")
st.line_chart(data['Close'])

# =====================================================
# Peak/Trough Detection
# =====================================================
def detect_peaks_and_troughs(prices, smoothing_sigma=3, min_distance=10, prominence_factor=0.5):
    prices_smooth = gaussian_filter1d(prices.values, sigma=smoothing_sigma)
    volatility = prices.pct_change().std()
    avg_price = prices.mean()
    prominence_threshold = avg_price * volatility * prominence_factor

    peaks, _ = find_peaks(prices_smooth, distance=min_distance, prominence=prominence_threshold)
    troughs, _ = find_peaks(-prices_smooth, distance=min_distance, prominence=prominence_threshold)
    return prices_smooth, peaks, troughs

prices_smooth, peaks, troughs = detect_peaks_and_troughs(data['Close'])

# Plot Peaks/Troughs
fig, ax = plt.subplots(figsize=(12,5))
ax.plot(data.index, data['Close'], label='Close', color='blue')
ax.plot(data.index, prices_smooth, label='Smoothed', color='darkblue', linewidth=2)
ax.scatter(data.index[peaks], data['Close'].iloc[peaks], color='red', marker='v', s=100, label='PEAK (SHORT)')
ax.scatter(data.index[troughs], data['Close'].iloc[troughs], color='green', marker='^', s=100, label='TROUGH (LONG)')
ax.set_title(f"{ticker} - Peak/Trough Detection")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.legend()
st.pyplot(fig)

# =====================================================
# Backtesting
# =====================================================
def backtest_directional_strategy(prices, peaks, troughs, transaction_cost=0.001):
    n = len(prices)
    returns = np.zeros(n)
    positions = np.zeros(n)
    trades = []

    all_signals = [(idx, -1, 'PEAK') for idx in peaks] + [(idx, 1, 'TROUGH') for idx in troughs]
    all_signals.sort(key=lambda x: x[0])

    current_position = 0
    entry_price = 0
    entry_date = None

    for idx, pos_type, sig_name in all_signals:
        sig_date = prices.index[idx]
        sig_price = prices.iloc[idx]

        if pos_type != current_position:
            if current_position != 0:
                price_change = (sig_price - entry_price)/entry_price
                exit_return = price_change - transaction_cost if current_position==1 else -price_change - transaction_cost
                trades.append({'Entry_Date': entry_date, 'Exit_Date': sig_date,
                               'Position': 'LONG' if current_position==1 else 'SHORT',
                               'Return_%': exit_return*100})
            current_position = pos_type
            entry_price = sig_price
            entry_date = sig_date

        positions[idx:] = current_position

    for i in range(1,n):
        if positions[i-1]!=0:
            price_return = (prices.iloc[i]-prices.iloc[i-1])/prices.iloc[i-1]
            returns[i] = price_return if positions[i-1]==1 else -price_return

    cum_returns = np.cumsum(returns)
    cum_log_returns = np.cumsum(np.log1p(returns))
    total_return = np.exp(cum_log_returns[-1])-1
    num_years = len(prices)/252
    annualized_return = (1+total_return)**(1/num_years)-1
    sharpe = np.mean(returns)/ (np.std(returns)+1e-10)*np.sqrt(252)
    cumulative_wealth = np.exp(cum_log_returns)
    max_dd = np.min((cumulative_wealth - np.maximum.accumulate(cumulative_wealth))/np.maximum.accumulate(cumulative_wealth))
    return {
        'returns': returns, 'cumulative_returns': cum_returns,
        'cumulative_log_returns': cum_log_returns,
        'positions': positions, 'trades': trades,
        'metrics': {'total_return': total_return, 'annualized_return': annualized_return, 'sharpe': sharpe, 'max_dd': max_dd}
    }

backtest_result = backtest_directional_strategy(data['Close'], peaks, troughs, transaction_cost)

st.subheader("Backtest Performance Metrics")
metrics = backtest_result['metrics']
st.write(f"Total Return: {metrics['total_return']*100:.2f}%")
st.write(f"Annualized Return: {metrics['annualized_return']*100:.2f}%")
st.write(f"Sharpe Ratio: {metrics['sharpe']:.2f}")
st.write(f"Max Drawdown: {metrics['max_dd']*100:.2f}%")
st.write(f"Number of Trades: {len(backtest_result['trades'])}")

# =====================================================
# Random Forest Regime Model
# =====================================================
st.subheader("Random Forest Regime Model")

# Simple regime based on peaks/troughs
regime = np.zeros(len(data))
regime[troughs] = 1
regime[peaks] = -1

df_rf = data.copy()
df_rf['Return'] = df_rf['Close'].pct_change()
df_rf['CumRet_3'] = df_rf['Return'].rolling(3).sum()
df_rf['CumRet_4'] = df_rf['Return'].rolling(4).sum()
df_rf['CumRet_5'] = df_rf['Return'].rolling(5).sum()
df_rf['RSI_14'] = 100 - 100/(1 + df_rf['Return'].rolling(10).mean()/(df_rf['Return'].rolling(10).std()+1e-10))
df_rf['High_5'] = df_rf['High'].rolling(4).max()
df_rf['Low_5'] = df_rf['Low'].rolling(4).min()
df_rf['MA_4'] = df_rf['Close'].rolling(4).mean()
df_rf['MA_9'] = df_rf['Close'].rolling(9).mean()
df_rf['Regime'] = regime
df_rf = df_rf.dropna()
FEATURES = ['CumRet_3','CumRet_4','CumRet_5','High_5','Low_5','RSI_14','MA_4','MA_9']
TARGET = 'Regime'

X = df_rf[FEATURES].values
y = df_rf[TARGET].values
split = int(0.8*len(df_rf))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
cw = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
cw_dict = dict(enumerate(cw))

rf = RandomForestClassifier(n_estimators=500, max_depth=5, min_samples_leaf=5,
                            class_weight=cw_dict, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# Threshold optimization
def build_positions(prob, upper, lower):
    pos = np.zeros(len(prob))
    for i,p in enumerate(prob):
        if p>=upper: pos[i]=1
        elif p<=lower: pos[i]=-1
        else: pos[i]=pos[i-1] if i>0 else -1
    return pos

rf_prob_train = rf.predict_proba(X_train)[:,1]
returns_train = pd.Series(data['Close'].iloc[:split]).pct_change().fillna(0).values
upper_grid = np.arange(0.55,0.91,0.07)
lower_grid = np.arange(0.1,0.46,0.07)
best_sharpe = -np.inf
best_upper,best_lower=0.6,0.4
for u in upper_grid:
    for l in lower_grid:
        if l>=u: continue
        temp_pos = build_positions(rf_prob_train, u,l)
        temp_strat = temp_pos[:-1]*returns_train[1:]
        temp_sh = np.mean(temp_strat)/ (np.std(temp_strat)+1e-10)*np.sqrt(252)
        if temp_sh>best_sharpe:
            best_sharpe = temp_sh
            best_upper,best_lower = u,l

positions_train = build_positions(rf_prob_train, best_upper,best_lower)

# Latest prediction
latest_row = df_rf.iloc[[-1]]
latest_date = latest_row.index[0].date()
latest_close = latest_row['Close'].values[0]
X_latest = scaler.transform(latest_row[FEATURES].values)
prob_latest = rf.predict_proba(X_latest)[:,1][0]
position_latest = 'LONG' if prob_latest>=best_upper else 'SHORT' if prob_latest<=best_lower else 'HOLD'

st.subheader("Latest Observation & Prediction")
st.write(f"Date: {latest_date}")
st.write(f"Close Price: {latest_close:.2f}")
st.write(f"Predicted Probability of LONG: {prob_latest:.4f}")
st.write(f"Recommended Position: **{position_latest}**")
