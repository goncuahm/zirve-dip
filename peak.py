# =====================================================
# STREAMLIT APP: DIRECTIONAL TRADING + RANDOM FOREST REGIME
# =====================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils import class_weight
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="BIST 100 Directional Trading", layout="wide")

# =====================================================
# 1. CONFIGURATION
# =====================================================
TICKER = st.text_input("Ticker", value="XU100.IS")
START = st.date_input("Start Date", value=pd.to_datetime("2023-01-01"))
END = st.date_input("End Date", value=pd.to_datetime("2026-02-28"))
TRANSACTION_COST = st.number_input("Transaction Cost per Trade", value=0.0, step=0.001, format="%.3f")

st.title(f"Directional Trading on {TICKER}")
st.write(f"Period: {START} to {END}, Transaction Cost: {TRANSACTION_COST*100:.2f}% per trade")

# =====================================================
# 2. DOWNLOAD DATA
# =====================================================
@st.cache_data
def download_stock_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if data.empty:
        return None
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data = data[['Open', 'High', 'Low', 'Close']].dropna()
    data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
    data = data.dropna(subset=['Close'])
    return data

data = download_stock_data(TICKER, START, END)

if data is None:
    st.error(f"No data found for {TICKER}. Exiting.")
    st.stop()

st.success(f"Downloaded {len(data)} trading days from {data.index[0].date()} to {data.index[-1].date()}")

close_prices = data['Close']

# =====================================================
# 3. PEAK / TROUGH DETECTION
# =====================================================
st.subheader("Peak/Trough Detection")

def detect_peaks_and_troughs(prices, smoothing_sigma=3, min_distance=10, prominence_factor=0.5):
    prices_smooth = gaussian_filter1d(prices.values, sigma=smoothing_sigma)
    volatility = prices.pct_change().std()
    avg_price = prices.mean()
    prominence_threshold = avg_price * volatility * prominence_factor

    peaks, peak_props = find_peaks(prices_smooth, distance=min_distance, prominence=prominence_threshold)
    troughs, trough_props = find_peaks(-prices_smooth, distance=min_distance, prominence=prominence_threshold)

    return {
        'prices_smooth': prices_smooth,
        'peaks': peaks,
        'troughs': troughs,
        'peak_properties': peak_props,
        'trough_properties': trough_props
    }

detection = detect_peaks_and_troughs(close_prices)

signal_events = []

for idx in detection['peaks']:
    signal_events.append({
        'Date': close_prices.index[idx],
        'Type': 'PEAK',
        'Signal': 'SHORT',
        'Price': close_prices.iloc[idx]
    })

for idx in detection['troughs']:
    signal_events.append({
        'Date': close_prices.index[idx],
        'Type': 'TROUGH',
        'Signal': 'LONG',
        'Price': close_prices.iloc[idx]
    })

signal_events.sort(key=lambda x: x['Date'])
signals_df = pd.DataFrame(signal_events)
st.write(signals_df.head(10))

# =====================================================
# 4. BACKTEST STRATEGY
# =====================================================
st.subheader("Backtesting")

def backtest_directional_strategy(prices, peaks, troughs, transaction_cost=0.001):
    n = len(prices)
    returns = np.zeros(n)
    positions = np.zeros(n)
    trades = []

    all_signals = [(i, -1, 'PEAK') for i in peaks] + [(i, 1, 'TROUGH') for i in troughs]
    all_signals.sort(key=lambda x: x[0])

    current_position = 0
    entry_price = 0
    entry_date = None
    entry_signal = None

    for signal_idx, pos_type, sig_name in all_signals:
        signal_date = prices.index[signal_idx]
        signal_price = prices.iloc[signal_idx]
        if pos_type != current_position:
            if current_position != 0:
                price_change = (signal_price - entry_price) / entry_price
                exit_return = price_change - transaction_cost if current_position==1 else -price_change - transaction_cost
                trades.append({
                    'Entry_Date': entry_date,
                    'Entry_Signal': entry_signal,
                    'Entry_Price': entry_price,
                    'Exit_Date': signal_date,
                    'Exit_Signal': sig_name,
                    'Exit_Price': signal_price,
                    'Position': 'LONG' if current_position==1 else 'SHORT',
                    'Return_%': exit_return*100,
                    'Holding_Days': (signal_date-entry_date).days
                })
            current_position = pos_type
            entry_price = signal_price
            entry_date = signal_date
            entry_signal = sig_name
        positions[signal_idx:] = current_position

    for i in range(1,n):
        if positions[i-1]==1:
            returns[i] = (prices.iloc[i]-prices.iloc[i-1])/prices.iloc[i-1]
        elif positions[i-1]==-1:
            returns[i] = -(prices.iloc[i]-prices.iloc[i-1])/prices.iloc[i-1]

    cum_log_returns = np.cumsum(np.log1p(returns))
    cumulative_wealth = np.exp(cum_log_returns)
    running_max = np.maximum.accumulate(cumulative_wealth)
    drawdown = (cumulative_wealth - running_max)/running_max

    total_return = cumulative_wealth[-1]-1
    num_years = len(prices)/252
    annualized_return = (1+total_return)**(1/num_years)-1
    sharpe_ratio = np.mean(returns)/ (np.std(returns)+1e-10) * np.sqrt(252)
    max_drawdown = np.min(drawdown)
    winning_trades = [t for t in trades if t['Return_%']>0]
    losing_trades  = [t for t in trades if t['Return_%']<=0]
    win_rate = len(winning_trades)/len(trades) if len(trades)>0 else 0
    total_wins = sum([t['Return_%'] for t in winning_trades])
    total_losses = abs(sum([t['Return_%'] for t in losing_trades]))
    profit_factor = total_wins/total_losses if total_losses>0 else np.inf
    buy_hold_return = (prices.iloc[-1]-prices.iloc[0])/prices.iloc[0]
    outperformance = total_return - buy_hold_return

    return {
        'returns': returns,
        'cumulative_log_returns': cum_log_returns,
        'positions': positions,
        'trades': trades,
        'metrics': {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'num_trades': len(trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'buy_hold_return': buy_hold_return,
            'outperformance': outperformance
        }
    }

backtest_result = backtest_directional_strategy(
    prices=close_prices,
    peaks=detection['peaks'],
    troughs=detection['troughs'],
    transaction_cost=TRANSACTION_COST
)

metrics = backtest_result['metrics']
st.write("### Performance Metrics")
st.json(metrics)

# =====================================================
# 5. PLOTS (Updated with Regression Channel)
# =====================================================
st.subheader("Price & Signals with Regression Channel")
fig, axes = plt.subplots(2,1,figsize=(16,10))

# --- Linear Regression + 2-Sigma Channel ---
from sklearn.linear_model import LinearRegression

# Prepare x as numeric index
x_numeric = np.arange(len(close_prices)).reshape(-1,1)
y = close_prices.values
reg = LinearRegression().fit(x_numeric, y)
y_fit = reg.predict(x_numeric)

# Compute residuals and 2-sigma
residuals = y - y_fit
sigma = np.std(residuals)
upper_band = y_fit + 2*sigma
lower_band = y_fit - 2*sigma

# Price & Smoothed
axes[0].plot(close_prices.index, close_prices.values, label='Actual Price')
axes[0].plot(close_prices.index, detection['prices_smooth'], label='Smoothed Price', linewidth=2)
axes[0].scatter(close_prices.index[detection['peaks']], close_prices.iloc[detection['peaks']], 
                color='red', marker='v', s=80, label='PEAK')
axes[0].scatter(close_prices.index[detection['troughs']], close_prices.iloc[detection['troughs']], 
                color='green', marker='^', s=80, label='TROUGH')

# Regression line and 2-sigma channel
axes[0].plot(close_prices.index, y_fit, color='blue', linestyle='--', linewidth=2, label='Regression Line')
axes[0].fill_between(close_prices.index, lower_band, upper_band, color='blue', alpha=0.15, label='±2 Sigma')

axes[0].legend(); axes[0].grid(True)

# Cumulative Log Returns
axes[1].plot(close_prices.index, backtest_result['cumulative_log_returns'], label='Strategy')
axes[1].fill_between(close_prices.index, 0, backtest_result['cumulative_log_returns'],
                     where=backtest_result['positions']==1, color='green', alpha=0.2)
axes[1].fill_between(close_prices.index, 0, backtest_result['cumulative_log_returns'],
                     where=backtest_result['positions']==-1, color='red', alpha=0.2)
axes[1].axhline(0, color='black', linestyle='--')
axes[1].legend(); axes[1].grid(True)

st.pyplot(fig)


# =====================================================
# 6. RANDOM FOREST REGIME MODEL
# =====================================================
st.subheader("Random Forest Regime Prediction")

# Feature Engineering
df_rf = data.copy()
df_rf['Return'] = df_rf['Close'].pct_change()
df_rf['CumRet_3'] = df_rf['Return'].rolling(3).sum()
df_rf['CumRet_4'] = df_rf['Return'].rolling(4).sum()
df_rf['CumRet_5'] = df_rf['Return'].rolling(5).sum()
df_rf['CumRet_14'] = df_rf['Return'].rolling(14).sum()
delta = df_rf['Close'].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
avg_gain = gain.rolling(9).mean()
avg_loss = loss.rolling(9).mean()
rs = avg_gain/(avg_loss+1e-10)
df_rf['RSI_14'] = 100-(100/(1+rs))
df_rf['High_4'] = df_rf['High'].rolling(4).max()
df_rf['Low_4'] = df_rf['Low'].rolling(4).min()
df_rf['High_14'] = df_rf['High'].rolling(20).max() -  df_rf['High'].rolling(2).max()
df_rf['Low_14'] = df_rf['Low'].rolling(20).min() - df_rf['Low'].rolling(2).min()
df_rf = df_rf.dropna()

FEATURES = ['CumRet_3','CumRet_4','CumRet_5','High_4','Low_4','RSI_14' ,'High_14','Low_14']
TARGET = 'Regime'

# Dummy Regime if missing
if 'Regime' not in df_rf.columns:
    df_rf['Regime'] = (df_rf['Return']>0).astype(int)

X = df_rf[FEATURES].values
y = df_rf[TARGET].values

split = int(0.8*len(df_rf))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Check unique classes
unique_classes = np.unique(y_train)
rf_trained = False
if len(unique_classes) < 2:
    st.warning("Not enough class variation in training data to train RF.")
else:
    class_weights_array = class_weight.compute_class_weight('balanced', classes=unique_classes, y=y_train)
    class_weights_dict = dict(zip(unique_classes, class_weights_array))
    rf = RandomForestClassifier(n_estimators=500, max_depth=5, min_samples_leaf=5,
                                class_weight=class_weights_dict, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_trained = True
    st.success("Random Forest trained successfully.")


# =====================================================
# 7. CONFUSION MATRICES & CLASSIFICATION REPORTS
# =====================================================
if rf_trained:
    st.subheader("Confusion Matrices & Classification Reports")

    y_pred_train = rf.predict(X_train)
    y_pred_test = rf.predict(X_test)

    # --- Confusion Matrices ---
    fig, ax = plt.subplots(1,2,figsize=(14,5))
    disp_train = ConfusionMatrixDisplay(confusion_matrix(y_train, y_pred_train))
    disp_train.plot(ax=ax[0], colorbar=False)
    ax[0].set_title("In-Sample Confusion Matrix")

    disp_test = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_test))
    disp_test.plot(ax=ax[1], colorbar=False)
    ax[1].set_title("Out-of-Sample Confusion Matrix")
    st.pyplot(fig)

    # --- Classification Reports ---
    report_train = classification_report(y_train, y_pred_train, output_dict=True)
    report_test = classification_report(y_test, y_pred_test, output_dict=True)

    st.write("### In-Sample Classification Report")
    st.dataframe(pd.DataFrame(report_train).transpose().round(3))

    st.write("### Out-of-Sample Classification Report")
    st.dataframe(pd.DataFrame(report_test).transpose().round(3))

    # --- Latest Prediction ---
    latest_row = df_rf.iloc[[-1]]
    latest_date = latest_row.index[0]
    latest_close = latest_row['Close'].values[0]
    X_latest = scaler.transform(latest_row[FEATURES].values)
    prob_latest = rf.predict_proba(X_latest)[:,1][0]
    position_latest = 'HOLD'
    if prob_latest >= 0.6: position_latest='LONG'
    elif prob_latest <=0.4: position_latest='SHORT'

    st.write(f"**Latest Date:** {latest_date.date()}, **Close Price:** {latest_close:.2f}")
    st.write(f"Predicted Probability of LONG: {prob_latest:.4f}")
    st.write(f"Recommended Position: {position_latest}")





# # =====================================================
# # STREAMLIT APP: DIRECTIONAL TRADING + RANDOM FOREST REGIME
# # =====================================================

# import streamlit as st
# import yfinance as yf
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import find_peaks
# from scipy.ndimage import gaussian_filter1d
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import classification_report
# from sklearn.utils import class_weight
# import warnings
# warnings.filterwarnings('ignore')

# st.set_page_config(page_title="BIST 100 Directional Trading", layout="wide")

# # =====================================================
# # 1. CONFIGURATION
# # =====================================================
# TICKER = st.text_input("Ticker", value="XU100.IS")
# START = st.date_input("Start Date", value=pd.to_datetime("2023-01-01"))
# END = st.date_input("End Date", value=pd.to_datetime("2026-02-28"))
# TRANSACTION_COST = st.number_input("Transaction Cost per Trade", value=0.0, step=0.001, format="%.3f")

# st.title(f"Directional Trading on {TICKER}")
# st.write(f"Period: {START} to {END}, Transaction Cost: {TRANSACTION_COST*100:.2f}% per trade")

# # =====================================================
# # 2. DOWNLOAD DATA
# # =====================================================
# @st.cache_data
# def download_stock_data(ticker, start, end):
#     data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
#     if data.empty:
#         return None
#     if isinstance(data.columns, pd.MultiIndex):
#         data.columns = data.columns.get_level_values(0)
#     data = data[['Open', 'High', 'Low', 'Close']].dropna()
#     data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
#     data = data.dropna(subset=['Close'])
#     return data

# data = download_stock_data(TICKER, START, END)

# if data is None:
#     st.error(f"No data found for {TICKER}. Exiting.")
#     st.stop()

# st.success(f"Downloaded {len(data)} trading days from {data.index[0].date()} to {data.index[-1].date()}")

# close_prices = data['Close']

# # =====================================================
# # 3. PEAK / TROUGH DETECTION
# # =====================================================
# st.subheader("Peak/Trough Detection")

# def detect_peaks_and_troughs(prices, smoothing_sigma=3, min_distance=10, prominence_factor=0.5):
#     prices_smooth = gaussian_filter1d(prices.values, sigma=smoothing_sigma)
#     volatility = prices.pct_change().std()
#     avg_price = prices.mean()
#     prominence_threshold = avg_price * volatility * prominence_factor

#     peaks, peak_props = find_peaks(prices_smooth, distance=min_distance, prominence=prominence_threshold)
#     troughs, trough_props = find_peaks(-prices_smooth, distance=min_distance, prominence=prominence_threshold)

#     return {
#         'prices_smooth': prices_smooth,
#         'peaks': peaks,
#         'troughs': troughs,
#         'peak_properties': peak_props,
#         'trough_properties': trough_props
#     }

# detection = detect_peaks_and_troughs(close_prices)

# signal_events = []

# for idx in detection['peaks']:
#     signal_events.append({
#         'Date': close_prices.index[idx],
#         'Type': 'PEAK',
#         'Signal': 'SHORT',
#         'Price': close_prices.iloc[idx]
#     })

# for idx in detection['troughs']:
#     signal_events.append({
#         'Date': close_prices.index[idx],
#         'Type': 'TROUGH',
#         'Signal': 'LONG',
#         'Price': close_prices.iloc[idx]
#     })

# signal_events.sort(key=lambda x: x['Date'])
# signals_df = pd.DataFrame(signal_events)
# st.write(signals_df.head(10))

# # =====================================================
# # 4. BACKTEST STRATEGY
# # =====================================================
# st.subheader("Backtesting")

# def backtest_directional_strategy(prices, peaks, troughs, transaction_cost=0.001):
#     n = len(prices)
#     returns = np.zeros(n)
#     positions = np.zeros(n)
#     trades = []

#     all_signals = [(i, -1, 'PEAK') for i in peaks] + [(i, 1, 'TROUGH') for i in troughs]
#     all_signals.sort(key=lambda x: x[0])

#     current_position = 0
#     entry_price = 0
#     entry_date = None
#     entry_signal = None

#     for signal_idx, pos_type, sig_name in all_signals:
#         signal_date = prices.index[signal_idx]
#         signal_price = prices.iloc[signal_idx]
#         if pos_type != current_position:
#             if current_position != 0:
#                 price_change = (signal_price - entry_price) / entry_price
#                 exit_return = price_change - transaction_cost if current_position==1 else -price_change - transaction_cost
#                 trades.append({
#                     'Entry_Date': entry_date,
#                     'Entry_Signal': entry_signal,
#                     'Entry_Price': entry_price,
#                     'Exit_Date': signal_date,
#                     'Exit_Signal': sig_name,
#                     'Exit_Price': signal_price,
#                     'Position': 'LONG' if current_position==1 else 'SHORT',
#                     'Return_%': exit_return*100,
#                     'Holding_Days': (signal_date-entry_date).days
#                 })
#             current_position = pos_type
#             entry_price = signal_price
#             entry_date = signal_date
#             entry_signal = sig_name
#         positions[signal_idx:] = current_position

#     for i in range(1,n):
#         if positions[i-1]==1:
#             returns[i] = (prices.iloc[i]-prices.iloc[i-1])/prices.iloc[i-1]
#         elif positions[i-1]==-1:
#             returns[i] = -(prices.iloc[i]-prices.iloc[i-1])/prices.iloc[i-1]

#     cum_log_returns = np.cumsum(np.log1p(returns))
#     cumulative_wealth = np.exp(cum_log_returns)
#     running_max = np.maximum.accumulate(cumulative_wealth)
#     drawdown = (cumulative_wealth - running_max)/running_max

#     total_return = cumulative_wealth[-1]-1
#     num_years = len(prices)/252
#     annualized_return = (1+total_return)**(1/num_years)-1
#     sharpe_ratio = np.mean(returns)/ (np.std(returns)+1e-10) * np.sqrt(252)
#     max_drawdown = np.min(drawdown)
#     winning_trades = [t for t in trades if t['Return_%']>0]
#     losing_trades  = [t for t in trades if t['Return_%']<=0]
#     win_rate = len(winning_trades)/len(trades) if len(trades)>0 else 0
#     total_wins = sum([t['Return_%'] for t in winning_trades])
#     total_losses = abs(sum([t['Return_%'] for t in losing_trades]))
#     profit_factor = total_wins/total_losses if total_losses>0 else np.inf
#     buy_hold_return = (prices.iloc[-1]-prices.iloc[0])/prices.iloc[0]
#     outperformance = total_return - buy_hold_return

#     return {
#         'returns': returns,
#         'cumulative_log_returns': cum_log_returns,
#         'positions': positions,
#         'trades': trades,
#         'metrics': {
#             'total_return': total_return,
#             'annualized_return': annualized_return,
#             'sharpe_ratio': sharpe_ratio,
#             'max_drawdown': max_drawdown,
#             'num_trades': len(trades),
#             'win_rate': win_rate,
#             'profit_factor': profit_factor,
#             'buy_hold_return': buy_hold_return,
#             'outperformance': outperformance
#         }
#     }

# backtest_result = backtest_directional_strategy(
#     prices=close_prices,
#     peaks=detection['peaks'],
#     troughs=detection['troughs'],
#     transaction_cost=TRANSACTION_COST
# )

# metrics = backtest_result['metrics']
# st.write("### Performance Metrics")
# st.json(metrics)

# # =====================================================
# # 5. PLOTS
# # =====================================================
# st.subheader("Price & Signals")
# fig, axes = plt.subplots(2,1,figsize=(16,10))

# # Price & Smoothed
# axes[0].plot(close_prices.index, close_prices.values, label='Actual Price')
# axes[0].plot(close_prices.index, detection['prices_smooth'], label='Smoothed Price', linewidth=2)
# axes[0].scatter(close_prices.index[detection['peaks']], close_prices.iloc[detection['peaks']], color='red', marker='v', s=80, label='PEAK')
# axes[0].scatter(close_prices.index[detection['troughs']], close_prices.iloc[detection['troughs']], color='green', marker='^', s=80, label='TROUGH')
# axes[0].legend(); axes[0].grid(True)

# # Cumulative Log Returns
# axes[1].plot(close_prices.index, backtest_result['cumulative_log_returns'], label='Strategy')
# axes[1].fill_between(close_prices.index, 0, backtest_result['cumulative_log_returns'],
#                      where=backtest_result['positions']==1, color='green', alpha=0.2)
# axes[1].fill_between(close_prices.index, 0, backtest_result['cumulative_log_returns'],
#                      where=backtest_result['positions']==-1, color='red', alpha=0.2)
# axes[1].axhline(0, color='black', linestyle='--')
# axes[1].legend(); axes[1].grid(True)

# st.pyplot(fig)

# # =====================================================
# # 6. RANDOM FOREST REGIME MODEL
# # =====================================================
# st.subheader("Random Forest Regime Prediction")

# # Feature Engineering
# df_rf = data.copy()
# df_rf['Return'] = df_rf['Close'].pct_change()
# df_rf['CumRet_3'] = df_rf['Return'].rolling(3).sum()
# df_rf['CumRet_4'] = df_rf['Return'].rolling(4).sum()
# df_rf['CumRet_5'] = df_rf['Return'].rolling(5).sum()
# df_rf['CumRet_14'] = df_rf['Return'].rolling(14).sum()
# delta = df_rf['Close'].diff()
# gain = delta.clip(lower=0)
# loss = -delta.clip(upper=0)
# avg_gain = gain.rolling(14).mean()
# avg_loss = loss.rolling(14).mean()
# rs = avg_gain/(avg_loss+1e-10)
# df_rf['RSI_14'] = 100-(100/(1+rs))
# df_rf['High_5'] = df_rf['High'].rolling(5).max()
# df_rf['Low_5'] = df_rf['Low'].rolling(5).min()
# df_rf = df_rf.dropna()

# FEATURES = ['CumRet_3','CumRet_4','CumRet_5','High_5','Low_5','RSI_14']
# TARGET = 'Regime'

# # Dummy Regime if missing
# if 'Regime' not in df_rf.columns:
#     df_rf['Regime'] = (df_rf['Return']>0).astype(int)

# X = df_rf[FEATURES].values
# y = df_rf[TARGET].values

# split = int(0.8*len(df_rf))
# X_train, X_test = X[:split], X[split:]
# y_train, y_test = y[:split], y[split:]

# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# # Check unique classes
# unique_classes = np.unique(y_train)
# rf_trained = False
# if len(unique_classes) < 2:
#     st.warning("Not enough class variation in training data to train RF.")
# else:
#     class_weights_array = class_weight.compute_class_weight('balanced', classes=unique_classes, y=y_train)
#     class_weights_dict = dict(zip(unique_classes, class_weights_array))
#     rf = RandomForestClassifier(n_estimators=500, max_depth=5, min_samples_leaf=5,
#                                 class_weight=class_weights_dict, random_state=42, n_jobs=-1)
#     rf.fit(X_train, y_train)
#     rf_trained = True
#     st.success("Random Forest trained successfully.")

# # Latest prediction
# latest_row = df_rf.iloc[[-1]]
# latest_date = latest_row.index[0]
# latest_close = latest_row['Close'].values[0]

# st.write(f"**Latest Date:** {latest_date.date()}, **Close Price:** {latest_close:.2f}")

# if rf_trained:
#     X_latest = scaler.transform(latest_row[FEATURES].values)
#     prob_latest = rf.predict_proba(X_latest)[:,1][0]
#     position_latest = 'HOLD'
#     if prob_latest >= 0.6: position_latest='LONG'
#     elif prob_latest <=0.4: position_latest='SHORT'
#     st.write(f"Predicted Probability of LONG: {prob_latest:.4f}")
#     st.write(f"Recommended Position: {position_latest}")
# else:
#     st.info("Random Forest not trained — cannot predict latest position.")
