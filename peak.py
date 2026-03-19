# =====================================================
# STREAMLIT APP: ML DIRECTIONAL TRADING
# Long/Short strategy with TP/SL optimization
# =====================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils import class_weight
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ML Directional Trading",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# DARK THEME COLOURS (matches previous code palette)
# ─────────────────────────────────────────────────────────────
BG     = '#080818'
GOLD   = '#C8A84B'
TEAL   = '#00D4B4'
WHITE  = '#E8E8F4'
GREY   = '#2A2A4A'
GREEN  = '#00C853'
RED    = '#FF1744'
ORANGE = '#FF8844'
PURPLE = '#CC44FF'
BLUE   = '#4488FF'

plt.rcParams.update({
    'figure.facecolor': BG,
    'axes.facecolor':   '#0D0D28',
    'axes.edgecolor':   GREY,
    'text.color':       WHITE,
    'axes.labelcolor':  WHITE,
    'xtick.color':      WHITE,
    'ytick.color':      WHITE,
    'grid.color':       GREY,
    'grid.alpha':       0.35,
})

def style_ax(ax):
    ax.set_facecolor('#0D0D28')
    for sp in ax.spines.values():
        sp.set_color(GREY)
    ax.tick_params(colors=WHITE, labelsize=8)
    ax.grid(True, color=GREY, alpha=0.35, lw=0.5)

# ─────────────────────────────────────────────────────────────
# SIDEBAR — CONFIGURATION
# ─────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Configuration")

TICKER           = st.sidebar.text_input("Ticker", value="THYAO.IS")
START            = st.sidebar.date_input("Start Date",  value=pd.to_datetime("2022-01-01"))
END              = st.sidebar.date_input("End Date",    value=pd.to_datetime("2026-02-28"))
TRANSACTION_COST = st.sidebar.number_input("Transaction Cost (per trade)",
                                            value=0.0, step=0.001, format="%.3f")

st.sidebar.markdown("---")
st.sidebar.subheader("Stop-Loss")
LONG_SL_PCT  = st.sidebar.slider("LONG Stop-Loss %",   1, 30, 10) / 100
SHORT_SL_PCT = st.sidebar.slider("SHORT Stop-Loss %",  1, 30, 10) / 100

st.sidebar.markdown("---")
st.sidebar.subheader("Strategy Side")
SHORT_SIDE = st.sidebar.checkbox("Enable SHORT side", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Detection Parameters")
SMOOTHING_SIGMA   = st.sidebar.slider("Smoothing Sigma",    1, 10, 3)
MIN_DISTANCE      = st.sidebar.slider("Min Distance (days)",5, 40, 10)
PROMINENCE_FACTOR = st.sidebar.slider("Prominence Factor",  0.1, 2.0, 0.5)

st.sidebar.markdown("---")
st.sidebar.subheader("Model")
TRAIN_FRAC  = st.sidebar.slider("Train Fraction", 0.5, 0.95, 0.80)
RF_WEIGHT   = st.sidebar.slider("RF Weight (vs GB)", 0.0, 1.0, 0.6)
UPPER_THRESH = st.sidebar.slider("LONG  prob threshold", 0.5, 0.9, 0.6)
LOWER_THRESH = st.sidebar.slider("SHORT prob threshold", 0.1, 0.5, 0.4)

GB_WEIGHT = 1.0 - RF_WEIGHT

# ─────────────────────────────────────────────────────────────
# TITLE
# ─────────────────────────────────────────────────────────────
st.title(f"🤖 ML Directional Trading — {TICKER}")
mode_str = "LONG + SHORT" if SHORT_SIDE else "LONG ONLY"
st.caption(f"Mode: **{mode_str}**  |  SL: LONG {LONG_SL_PCT*100:.0f}%  SHORT {SHORT_SL_PCT*100:.0f}%"
           f"  |  {START} → {END}")

# ─────────────────────────────────────────────────────────────
# 1. DOWNLOAD DATA
# ─────────────────────────────────────────────────────────────
@st.cache_data
def download_stock_data(ticker, start, end):
    data = yf.download(ticker, start=str(start), end=str(end),
                       progress=False, auto_adjust=False)
    if data.empty:
        return None
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data = data[['Open', 'High', 'Low', 'Close']].dropna()
    data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
    return data.dropna(subset=['Close'])

with st.spinner("Downloading data..."):
    data = download_stock_data(TICKER, START, END)

if data is None or data.empty:
    st.error(f"No data found for {TICKER}.")
    st.stop()

st.success(f"✓ {len(data)} trading days  |  "
           f"{data.index[0].date()} → {data.index[-1].date()}  |  "
           f"Price: {float(data['Close'].min()):.2f} → {float(data['Close'].max()):.2f}")

close_prices = data['Close']

# ─────────────────────────────────────────────────────────────
# 2. PEAK / TROUGH DETECTION
# ─────────────────────────────────────────────────────────────
@st.cache_data
def detect_peaks_and_troughs(prices_values, prices_index, sigma, min_dist, prom_factor):
    prices = pd.Series(prices_values, index=prices_index)
    smooth = gaussian_filter1d(prices.values, sigma=sigma)
    vol    = prices.pct_change().std()
    prom_thresh = prices.mean() * vol * prom_factor
    peaks,   _ = find_peaks( smooth, distance=min_dist, prominence=prom_thresh)
    troughs, _ = find_peaks(-smooth, distance=min_dist, prominence=prom_thresh)
    return smooth, peaks, troughs

smooth, peaks, troughs = detect_peaks_and_troughs(
    close_prices.values, close_prices.index,
    SMOOTHING_SIGMA, MIN_DISTANCE, PROMINENCE_FACTOR)

# ─────────────────────────────────────────────────────────────
# 3. REGIME LABELS  (flip BEFORE labelling — correct version)
# ─────────────────────────────────────────────────────────────
n          = len(data)
dates      = data.index
peaks_set   = set(peaks)
troughs_set = set(troughs)

regime        = np.zeros(n, dtype=int)
current_state = 1
for i in range(n):
    if current_state == 1 and i in peaks_set:
        current_state = 0
    elif current_state == 0 and i in troughs_set:
        current_state = 1
    regime[i] = current_state

regime_series = pd.Series(regime, index=dates, name='Regime')

# ─────────────────────────────────────────────────────────────
# 4. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────
df = data.copy()
df['Return']    = df['Close'].pct_change()
df['CumRet_3']  = df['Return'].rolling(3).sum()
df['CumRet_4']  = df['Return'].rolling(4).sum()
df['CumRet_5']  = df['Return'].rolling(5).sum()
df['CumRet_14'] = df['Return'].rolling(14).sum()

delta    = df['Close'].diff()
gain     = delta.clip(lower=0)
loss     = -delta.clip(upper=0)
avg_gain = gain.rolling(9).mean()
avg_loss = loss.rolling(9).mean()
rs       = avg_gain / (avg_loss + 1e-10)
df['RSI_14'] = 100 - (100 / (1 + rs))
df['High_5']  = df['High'].rolling(4).max()
df['Low_5']   = df['Low'].rolling(4).min()
df['High_14'] = df['High'].rolling(14).max() - df['High'].rolling(2).max()
df['Low_14']  = df['Low'].rolling(14).min()  - df['Low'].rolling(2).min()
df['Regime']  = regime_series
df            = df.dropna()

FEATURES = ['CumRet_3', 'CumRet_4', 'CumRet_5',
            'High_5', 'Low_5', 'RSI_14', 'High_14', 'Low_14']

X         = df[FEATURES].values
y         = df['Regime'].values
prices_ml = df['Close'].values
highs_ml  = df['High'].values
lows_ml   = df['Low'].values
dates_ml  = df.index

# ─────────────────────────────────────────────────────────────
# 5. TRAIN / TEST SPLIT & MODELS
# ─────────────────────────────────────────────────────────────
split        = int(TRAIN_FRAC * len(df))
X_train      = X[:split];          X_test       = X[split:]
y_train      = y[:split];          y_test       = y[split:]
prices_train = prices_ml[:split];  prices_test  = prices_ml[split:]
highs_train  = highs_ml[:split];   highs_test   = highs_ml[split:]
lows_train   = lows_ml[:split];    lows_test    = lows_ml[split:]
dates_train  = dates_ml[:split];   dates_test   = dates_ml[split:]

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

with st.spinner("Training ensemble model (RF + GB)..."):
    rf = RandomForestClassifier(
        n_estimators=500, max_depth=5, min_samples_leaf=10,
        class_weight='balanced', random_state=42, n_jobs=-1)
    gb = GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.05,
        max_depth=3, subsample=0.85, random_state=42)
    rf.fit(X_train_sc, y_train)
    gb.fit(X_train_sc, y_train)

def ensemble_proba(X_sc):
    return RF_WEIGHT * rf.predict_proba(X_sc)[:, 1] + GB_WEIGHT * gb.predict_proba(X_sc)[:, 1]

prob_train = ensemble_proba(X_train_sc)
prob_test  = ensemble_proba(X_test_sc)

def build_signal(prob, upper=UPPER_THRESH, lower=LOWER_THRESH):
    sig = np.zeros(len(prob), dtype=int)
    for i, p in enumerate(prob):
        if   p >= upper: sig[i] = 1
        elif p <= lower: sig[i] = 0
        else:            sig[i] = sig[i-1] if i > 0 else 0
    return sig

raw_signal_train = build_signal(prob_train)
raw_signal_test  = build_signal(prob_test)

st.success("✓ Ensemble model trained  (RF + GB)")

# ─────────────────────────────────────────────────────────────
# 6. STRATEGY ENGINE
# ─────────────────────────────────────────────────────────────
def run_strategy(prices, highs, lows, raw_signal, all_dates,
                 long_tp, long_sl, short_tp, short_sl,
                 short_side=True, tx_cost=TRANSACTION_COST):
    """
    LONG  entry: signal flips 0->1  (or any bar cur=1 when flat after TP/SL)
    SHORT entry: signal flips 1->0  (only when short_side=True)
    TP/SL checked intraday via High/Low — take priority over signal flip.
    After TP/SL, re-enter LONG as soon as signal=1 (no flip required).
    After TP/SL short, re-enter short only on next 1->0 flip.
    """
    n       = len(prices)
    pnl     = np.zeros(n)
    pos_arr = np.zeros(n, dtype=int)
    trades  = []

    in_trade    = False
    side        = 0
    entry_price = 0.0
    entry_bar   = 0

    for i in range(n):
        cur_signal  = int(raw_signal[i])
        prev_signal = int(raw_signal[i - 1]) if i > 0 else 1 - cur_signal
        flipped     = (cur_signal != prev_signal)

        if not in_trade:
            if cur_signal == 1:
                # Enter LONG on any bar signal=1 while flat
                side        = 1
                entry_price = prices[i]
                entry_bar   = i
                in_trade    = True
            elif short_side and flipped:
                # Enter SHORT only on a genuine 1->0 flip
                side        = -1
                entry_price = prices[i]
                entry_bar   = i
                in_trade    = True
            pos_arr[i] = side if in_trade else 0
            if not in_trade:
                continue

        pos_arr[i] = side

        if side == 1:
            tp_price = entry_price * (1.0 + long_tp)
            sl_price = entry_price * (1.0 - long_sl)
            tp_hit   = highs[i] >= tp_price
            sl_hit   = lows[i]  <= sl_price

            if tp_hit or sl_hit:
                exit_p  = tp_price if tp_hit else sl_price
                reason  = 'TP' if tp_hit else 'SL'
                tpnl    = (exit_p - entry_price) / entry_price - tx_cost
                pnl[i]  = tpnl
                trades.append(_tr(entry_bar, i, 'LONG', entry_price,
                                  exit_p, reason, tpnl, all_dates))
                in_trade = False

            elif flipped:
                exit_p  = prices[i]
                tpnl    = (exit_p - entry_price) / entry_price - tx_cost
                pnl[i] += tpnl
                trades.append(_tr(entry_bar, i, 'LONG', entry_price,
                                  exit_p, 'SIGNAL', tpnl, all_dates))
                if short_side:
                    side        = -1
                    entry_price = prices[i]
                    entry_bar   = i
                    in_trade    = True
                    pos_arr[i]  = -1
                else:
                    in_trade   = False
                    pos_arr[i] = 0

        else:  # SHORT
            tp_price = entry_price * (1.0 - short_tp)
            sl_price = entry_price * (1.0 + short_sl)
            tp_hit   = lows[i]  <= tp_price
            sl_hit   = highs[i] >= sl_price

            if tp_hit or sl_hit:
                exit_p  = tp_price if tp_hit else sl_price
                reason  = 'TP' if tp_hit else 'SL'
                tpnl    = (entry_price - exit_p) / entry_price - tx_cost
                pnl[i]  = tpnl
                trades.append(_tr(entry_bar, i, 'SHORT', entry_price,
                                  exit_p, reason, tpnl, all_dates))
                in_trade = False

            elif flipped:
                exit_p  = prices[i]
                tpnl    = (entry_price - exit_p) / entry_price - tx_cost
                pnl[i] += tpnl
                trades.append(_tr(entry_bar, i, 'SHORT', entry_price,
                                  exit_p, 'SIGNAL', tpnl, all_dates))
                side        = 1
                entry_price = prices[i]
                entry_bar   = i
                in_trade    = True
                pos_arr[i]  = 1

    # Close open trade
    if in_trade:
        exit_p = prices[-1]
        tpnl   = ((exit_p - entry_price) / entry_price if side == 1
                  else (entry_price - exit_p) / entry_price) - tx_cost
        pnl[-1] += tpnl
        trades.append(_tr(entry_bar, n - 1,
                          'LONG' if side == 1 else 'SHORT',
                          entry_price, exit_p, 'OPEN', tpnl, all_dates))

    cum_pnl = np.cumsum(pnl)
    return cum_pnl, pnl, pos_arr, trades


def _tr(eb, xb, side, ep, xp, reason, tpnl, all_dates):
    return {
        'Entry_Bar'   : eb,
        'Exit_Bar'    : xb,
        'Side'        : side,
        'Entry_Date'  : all_dates[eb],
        'Exit_Date'   : all_dates[min(xb, len(all_dates) - 1)],
        'Entry_Price' : ep,
        'Exit_Price'  : xp,
        'Exit_Reason' : reason,
        'PnL_pct'     : tpnl * 100,
        'Holding_Days': xb - eb,
        'Status'      : 'OPEN' if reason == 'OPEN' else 'CLOSED',
    }


def metrics_from(pnl, cum_pnl, prices, trades):
    n_tr = len(trades)
    if n_tr == 0:
        return {'total_pnl': 0, 'sharpe': 0, 'max_dd': 0,
                'n_trades': 0, 'win_rate': 0, 'profit_factor': 0,
                'avg_win': 0, 'avg_loss': 0, 'bh': 0}
    td   = pd.DataFrame(trades)
    wins = td[td['PnL_pct'] > 0]
    loss = td[td['PnL_pct'] <= 0]
    pf   = (wins['PnL_pct'].sum() / abs(loss['PnL_pct'].sum())
            if len(loss) > 0 and abs(loss['PnL_pct'].sum()) > 0 else np.inf)
    act  = pnl[pnl != 0]
    sh   = (np.mean(act) / (np.std(act) + 1e-10) * np.sqrt(252)
            if len(act) > 1 else 0)
    rm   = np.maximum.accumulate(cum_pnl)
    mdd  = (cum_pnl - rm).min() * 100
    bh   = (prices[-1] - prices[0]) / prices[0] * 100
    return {
        'total_pnl'    : cum_pnl[-1] * 100,
        'sharpe'       : sh,
        'max_dd'       : mdd,
        'n_trades'     : n_tr,
        'win_rate'     : len(wins) / n_tr * 100,
        'profit_factor': pf,
        'avg_win'      : wins['PnL_pct'].mean() if len(wins) > 0 else 0,
        'avg_loss'     : loss['PnL_pct'].mean() if len(loss) > 0 else 0,
        'bh'           : bh,
    }

# ─────────────────────────────────────────────────────────────
# 7. OPTIMISE TP LEVELS  (1% – 20% in 1% steps)
# ─────────────────────────────────────────────────────────────
TP_RANGE = np.arange(0.01, 0.21, 0.01)   # 1% to 20%

with st.spinner("Optimising take-profit levels (1%–20%)..."):
    best_long_tp  = 0.05
    best_short_tp = 0.06
    best_score    = -np.inf

    opt_results = []
    for ltp in TP_RANGE:
        for stp in (TP_RANGE if SHORT_SIDE else [best_short_tp]):
            cum, pnl_arr, _, tds = run_strategy(
                prices_train, highs_train, lows_train,
                raw_signal_train, dates_train,
                long_tp=ltp, long_sl=LONG_SL_PCT,
                short_tp=stp, short_sl=SHORT_SL_PCT,
                short_side=SHORT_SIDE)
            score = cum[-1]   # total in-sample P&L
            opt_results.append({'long_tp': ltp, 'short_tp': stp, 'score': score})
            if score > best_score:
                best_score    = score
                best_long_tp  = ltp
                best_short_tp = stp

opt_df = pd.DataFrame(opt_results)

st.success(f"✓ Optimal LONG TP: **{best_long_tp*100:.0f}%**  |  "
           f"Optimal SHORT TP: **{best_short_tp*100:.0f}%**  "
           f"(in-sample P&L: {best_score*100:+.2f}%)")

# ─────────────────────────────────────────────────────────────
# 8. RUN FINAL STRATEGY WITH OPTIMAL TP
# ─────────────────────────────────────────────────────────────
cum_train, pnl_train, pos_train, trades_train = run_strategy(
    prices_train, highs_train, lows_train, raw_signal_train, dates_train,
    long_tp=best_long_tp, long_sl=LONG_SL_PCT,
    short_tp=best_short_tp, short_sl=SHORT_SL_PCT,
    short_side=SHORT_SIDE)

cum_test, pnl_test, pos_test, trades_test = run_strategy(
    prices_test, highs_test, lows_test, raw_signal_test, dates_test,
    long_tp=best_long_tp, long_sl=LONG_SL_PCT,
    short_tp=best_short_tp, short_sl=SHORT_SL_PCT,
    short_side=SHORT_SIDE)

m_tr = metrics_from(pnl_train, cum_train, prices_train, trades_train)
m_te = metrics_from(pnl_test,  cum_test,  prices_test,  trades_test)

# ─────────────────────────────────────────────────────────────
# 9. CLASSIFICATION METRICS
# ─────────────────────────────────────────────────────────────
with st.expander("📊 Model Classification Reports", expanded=False):
    y_pred_train = (prob_train >= 0.5).astype(int)
    y_pred_test  = (prob_test  >= 0.5).astype(int)

    col1, col2 = st.columns(2)
    with col1:
        st.write("**In-Sample**")
        fig_cm, ax = plt.subplots(figsize=(4, 3), facecolor=BG)
        style_ax(ax)
        cm = confusion_matrix(y_train, y_pred_train)
        im = ax.imshow(cm, cmap='Blues')
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(['SHORT', 'LONG']); ax.set_yticklabels(['SHORT', 'LONG'])
        for r in range(2):
            for c in range(2):
                ax.text(c, r, str(cm[r, c]), ha='center', va='center',
                        color=WHITE, fontsize=11)
        ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix (Train)', color=GOLD)
        st.pyplot(fig_cm)
        rep = classification_report(y_train, y_pred_train,
                                    target_names=['SHORT', 'LONG'], output_dict=True)
        st.dataframe(pd.DataFrame(rep).T.round(3))

    with col2:
        st.write("**Out-of-Sample**")
        fig_cm2, ax2 = plt.subplots(figsize=(4, 3), facecolor=BG)
        style_ax(ax2)
        cm2 = confusion_matrix(y_test, y_pred_test)
        ax2.imshow(cm2, cmap='Blues')
        ax2.set_xticks([0, 1]); ax2.set_yticks([0, 1])
        ax2.set_xticklabels(['SHORT', 'LONG']); ax2.set_yticklabels(['SHORT', 'LONG'])
        for r in range(2):
            for c in range(2):
                ax2.text(c, r, str(cm2[r, c]), ha='center', va='center',
                         color=WHITE, fontsize=11)
        ax2.set_xlabel('Predicted'); ax2.set_ylabel('Actual')
        ax2.set_title('Confusion Matrix (Test)', color=GOLD)
        st.pyplot(fig_cm2)
        rep2 = classification_report(y_test, y_pred_test,
                                     target_names=['SHORT', 'LONG'], output_dict=True)
        st.dataframe(pd.DataFrame(rep2).T.round(3))

# ─────────────────────────────────────────────────────────────
# 10. PERFORMANCE SUMMARY
# ─────────────────────────────────────────────────────────────
st.subheader("📈 Performance Summary")

c1, c2 = st.columns(2)

def metric_card(col, label, m):
    with col:
        st.markdown(f"#### {label}")
        r1, r2, r3 = st.columns(3)
        r1.metric("Total P&L",      f"{m['total_pnl']:+.2f}%")
        r2.metric("Buy & Hold",     f"{m['bh']:+.2f}%")
        r3.metric("Outperformance", f"{m['total_pnl']-m['bh']:+.2f}%")
        r4, r5, r6 = st.columns(3)
        r4.metric("Sharpe",         f"{m['sharpe']:.2f}")
        r5.metric("Max Drawdown",   f"{m['max_dd']:.2f}%")
        r6.metric("Win Rate",       f"{m['win_rate']:.1f}%")
        r7, r8, r9 = st.columns(3)
        r7.metric("Trades",         str(m['n_trades']))
        r8.metric("Avg Win",        f"{m['avg_win']:+.2f}%")
        r9.metric("Avg Loss",       f"{m['avg_loss']:+.2f}%")

metric_card(c1, "🔵 In-Sample  (Train)", m_tr)
metric_card(c2, "🟠 Out-of-Sample (Test)", m_te)

# ─────────────────────────────────────────────────────────────
# 11. TP OPTIMISATION HEATMAP / LINE CHART
# ─────────────────────────────────────────────────────────────
with st.expander("🔍 Take-Profit Optimisation Results", expanded=True):
    if SHORT_SIDE:
        # Heatmap: long_tp vs short_tp
        pivot = opt_df.pivot(index='long_tp', columns='short_tp', values='score') * 100
        fig_heat, ax_h = plt.subplots(figsize=(12, 6), facecolor=BG)
        style_ax(ax_h)
        im = ax_h.imshow(pivot.values, aspect='auto', cmap='RdYlGn',
                          origin='lower')
        ax_h.set_xticks(range(len(pivot.columns)))
        ax_h.set_xticklabels([f"{c*100:.0f}%" for c in pivot.columns],
                              rotation=45, fontsize=7, color=WHITE)
        ax_h.set_yticks(range(len(pivot.index)))
        ax_h.set_yticklabels([f"{r*100:.0f}%" for r in pivot.index],
                              fontsize=7, color=WHITE)
        ax_h.set_xlabel('SHORT TP %', color=WHITE)
        ax_h.set_ylabel('LONG TP %', color=WHITE)
        ax_h.set_title(f'TP Optimisation Heatmap (In-Sample P&L %)  '
                       f'— Best: LONG {best_long_tp*100:.0f}%  '
                       f'SHORT {best_short_tp*100:.0f}%',
                       color=GOLD, fontsize=10)
        plt.colorbar(im, ax=ax_h)
        # Mark best
        bi = list(pivot.index).index(round(best_long_tp, 2))
        bj = list(pivot.columns).index(round(best_short_tp, 2))
        ax_h.plot(bj, bi, '*', color=GOLD, ms=14, zorder=5)
        fig_heat.tight_layout()
        st.pyplot(fig_heat)
    else:
        # Line chart: long_tp vs P&L
        fig_line, ax_l = plt.subplots(figsize=(10, 4), facecolor=BG)
        style_ax(ax_l)
        ax_l.plot(opt_df['long_tp'] * 100, opt_df['score'] * 100,
                  color=TEAL, lw=2, marker='o', ms=5)
        ax_l.axvline(best_long_tp * 100, color=GOLD, lw=2, ls='--',
                     label=f'Optimal = {best_long_tp*100:.0f}%')
        ax_l.set_xlabel('LONG TP %', color=WHITE)
        ax_l.set_ylabel('In-Sample P&L %', color=WHITE)
        ax_l.set_title('LONG TP Optimisation (In-Sample)', color=GOLD)
        ax_l.legend(facecolor='#1A1A38', labelcolor=WHITE)
        st.pyplot(fig_line)

# ─────────────────────────────────────────────────────────────
# 12. CUMULATIVE P&L CHARTS
# ─────────────────────────────────────────────────────────────
st.subheader("📉 Cumulative P&L")

tab1, tab2, tab3 = st.tabs(["In-Sample", "Out-of-Sample", "Combined"])

def plot_cum_pnl(cum_pnl, prices, pos_arr, all_dates, trades, m, label):
    fig = plt.figure(figsize=(16, 10), facecolor=BG)
    gs  = gridspec.GridSpec(3, 1, figure=fig,
                            height_ratios=[2.5, 1.5, 1], hspace=0.08)

    # ── Top: price + positions ────────────────────────────────
    ax0 = fig.add_subplot(gs[0])
    style_ax(ax0)
    ax0.plot(all_dates, prices, color=TEAL, lw=1.4, label='Close', zorder=3)

    for i in range(len(pos_arr)):
        nxt = min(i + 1, len(all_dates) - 1)
        if pos_arr[i] == 1:
            ax0.axvspan(all_dates[i], all_dates[nxt],
                        alpha=0.12, color=GREEN, zorder=1)
        elif pos_arr[i] == -1:
            ax0.axvspan(all_dates[i], all_dates[nxt],
                        alpha=0.12, color=RED, zorder=1)

    if trades:
        td = pd.DataFrame(trades)
        lt = td[td['Side'] == 'LONG']
        st_ = td[td['Side'] == 'SHORT']
        if len(lt):
            ax0.scatter(lt['Entry_Date'], lt['Entry_Price'],
                        marker='^', s=90, color=GREEN,
                        edgecolor='white', lw=0.6, zorder=7,
                        label=f'LONG entry ({len(lt)})')
        if len(st_):
            ax0.scatter(st_['Entry_Date'], st_['Entry_Price'],
                        marker='v', s=90, color=RED,
                        edgecolor='white', lw=0.6, zorder=7,
                        label=f'SHORT entry ({len(st_)})')
        tp_ex  = td[td['Exit_Reason'] == 'TP']
        sl_ex  = td[td['Exit_Reason'] == 'SL']
        sig_ex = td[td['Exit_Reason'].isin(['SIGNAL', 'OPEN'])]
        if len(tp_ex):
            ax0.scatter(tp_ex['Exit_Date'], tp_ex['Exit_Price'],
                        marker='*', s=160, color=GOLD,
                        edgecolor='white', lw=0.4, zorder=8,
                        label=f'TP exit ({len(tp_ex)})')
        if len(sl_ex):
            ax0.scatter(sl_ex['Exit_Date'], sl_ex['Exit_Price'],
                        marker='x', s=130, color=ORANGE,
                        linewidths=2.0, zorder=8,
                        label=f'SL exit ({len(sl_ex)})')
        if len(sig_ex):
            ax0.scatter(sig_ex['Exit_Date'], sig_ex['Exit_Price'],
                        marker='P', s=80, color=PURPLE,
                        edgecolor='white', lw=0.4, zorder=8,
                        label=f'Signal/Open ({len(sig_ex)})')

    ax0.set_title(
        f'{label}  |  LONG TP={best_long_tp*100:.0f}%  SL={LONG_SL_PCT*100:.0f}%  '
        f'SHORT TP={best_short_tp*100:.0f}%  SL={SHORT_SL_PCT*100:.0f}%  '
        f'P&L={m["total_pnl"]:+.2f}%  Sharpe={m["sharpe"]:.2f}',
        color=GOLD, fontsize=10, fontweight='bold')
    ax0.set_ylabel('Price', color=WHITE)
    ax0.legend(fontsize=7.5, facecolor='#1A1A38', labelcolor=WHITE,
               loc='upper left', ncol=4)
    ax0.tick_params(labelbottom=False)

    # ── Middle: cumulative P&L ────────────────────────────────
    ax1 = fig.add_subplot(gs[1])
    style_ax(ax1)
    bh = np.concatenate([[0], np.cumsum(np.diff(prices) / prices[:-1])]) * 100
    ax1.plot(all_dates, cum_pnl * 100, color=TEAL, lw=2.0,
             label=f'Strategy  ({m["total_pnl"]:+.2f}%)', zorder=4)
    ax1.plot(all_dates, bh, color=GREY, lw=1.4, ls='--', alpha=0.8,
             label=f'Buy & Hold ({m["bh"]:+.2f}%)', zorder=3)
    ax1.axhline(0, color=WHITE, lw=0.7, ls='--', alpha=0.4)
    ax1.fill_between(all_dates, cum_pnl * 100, bh,
                     where=(cum_pnl * 100 >= bh),
                     alpha=0.15, color=GREEN, interpolate=True)
    ax1.fill_between(all_dates, cum_pnl * 100, bh,
                     where=(cum_pnl * 100 < bh),
                     alpha=0.15, color=RED, interpolate=True)
    ax1.set_ylabel('Cum. P&L (%)', color=WHITE)
    ax1.legend(fontsize=8, facecolor='#1A1A38', labelcolor=WHITE)
    ax1.tick_params(labelbottom=False)

    # ── Bottom: drawdown ──────────────────────────────────────
    ax2 = fig.add_subplot(gs[2])
    style_ax(ax2)
    rm = np.maximum.accumulate(cum_pnl)
    dd = (cum_pnl - rm) * 100
    ax2.fill_between(all_dates, dd, 0, color=RED, alpha=0.5)
    ax2.plot(all_dates, dd, color='darkred', lw=1.0)
    ax2.set_ylabel('Drawdown (%)', color=WHITE)
    ax2.set_xlabel('Date', color=WHITE)

    fig.tight_layout()
    return fig

with tab1:
    st.pyplot(plot_cum_pnl(cum_train, prices_train, pos_train,
                            dates_train, trades_train, m_tr,
                            "In-Sample (Train)"))

with tab2:
    st.pyplot(plot_cum_pnl(cum_test, prices_test, pos_test,
                            dates_test, trades_test, m_te,
                            "Out-of-Sample (Test)"))

with tab3:
    # Chained equity curve
    cum_chained = np.concatenate([cum_train, cum_test + cum_train[-1]])
    all_dates_c = np.concatenate([dates_train, dates_test])
    all_prices_c = np.concatenate([prices_train, prices_test])
    all_pos_c    = np.concatenate([pos_train, pos_test])
    all_trades_c = trades_train + trades_test

    fig_c = plt.figure(figsize=(16, 8), facecolor=BG)
    gs_c  = gridspec.GridSpec(2, 1, figure=fig_c,
                              height_ratios=[2.5, 1], hspace=0.08)
    ax_c0 = fig_c.add_subplot(gs_c[0])
    style_ax(ax_c0)
    bh_c = np.concatenate([[0], np.cumsum(np.diff(all_prices_c) / all_prices_c[:-1])]) * 100
    ax_c0.plot(all_dates_c, cum_chained * 100, color=TEAL, lw=2.0,
               label=f'Strategy  ({cum_chained[-1]*100:+.2f}%)')
    ax_c0.plot(all_dates_c, bh_c, color=GREY, lw=1.4, ls='--', alpha=0.8,
               label=f'Buy & Hold ({bh_c[-1]:+.2f}%)')
    ax_c0.axvline(dates_test[0], color=GOLD, lw=2.0, ls=':', alpha=0.9, zorder=5)
    ax_c0.text(dates_test[0], ax_c0.get_ylim()[1] if ax_c0.get_ylim()[1] != 0 else 1,
               f'  Train|Test\n  {dates_test[0].date()}',
               color=GOLD, fontsize=9, va='top', fontweight='bold')
    ax_c0.axvspan(all_dates_c[0],  dates_test[0],  alpha=0.05, color=BLUE)
    ax_c0.axvspan(dates_test[0],   all_dates_c[-1], alpha=0.05, color=PURPLE)
    ax_c0.axhline(0, color=WHITE, lw=0.7, ls='--', alpha=0.4)
    ax_c0.set_title(
        f'Combined Equity Curve  |  LONG TP={best_long_tp*100:.0f}%  '
        f'SHORT TP={best_short_tp*100:.0f}%  '
        f'Train P&L={m_tr["total_pnl"]:+.1f}%  Test P&L={m_te["total_pnl"]:+.1f}%',
        color=GOLD, fontsize=10, fontweight='bold')
    ax_c0.set_ylabel('Cumulative P&L (%)', color=WHITE)
    ax_c0.legend(fontsize=9, facecolor='#1A1A38', labelcolor=WHITE, ncol=4)
    ax_c0.tick_params(labelbottom=False)

    ax_c1 = fig_c.add_subplot(gs_c[1])
    style_ax(ax_c1)
    rm_c = np.maximum.accumulate(cum_chained)
    dd_c = (cum_chained - rm_c) * 100
    ax_c1.fill_between(all_dates_c, dd_c, 0, color=RED, alpha=0.5)
    ax_c1.plot(all_dates_c, dd_c, color='darkred', lw=1.0)
    ax_c1.axvline(dates_test[0], color=GOLD, lw=1.5, ls=':', alpha=0.8)
    ax_c1.set_ylabel('Drawdown (%)', color=WHITE)
    ax_c1.set_xlabel('Date', color=WHITE)
    fig_c.tight_layout()
    st.pyplot(fig_c)

# ─────────────────────────────────────────────────────────────
# 13. TRADE LOG
# ─────────────────────────────────────────────────────────────
st.subheader("📋 Trade Log")

tab_tr, tab_te, tab_lat = st.tabs(["In-Sample Trades",
                                    "Out-of-Sample Trades",
                                    "Latest Trade"])

def display_trade_log(trades, label):
    if not trades:
        st.info(f"No trades in {label}.")
        return
    td = pd.DataFrame(trades).copy()
    td['Entry_Date'] = pd.to_datetime(td['Entry_Date']).dt.date
    td['Exit_Date']  = pd.to_datetime(td['Exit_Date']).dt.date
    td['Entry_Price'] = td['Entry_Price'].round(2)
    td['Exit_Price']  = td['Exit_Price'].round(2)
    td['PnL_pct']     = td['PnL_pct'].round(3)
    disp = td[['Entry_Date', 'Exit_Date', 'Side', 'Entry_Price',
               'Exit_Price', 'PnL_pct', 'Holding_Days',
               'Exit_Reason', 'Status']].reset_index(drop=True)
    disp.index += 1

    def _colour(row):
        if row['Status'] == 'OPEN':
            bg = '#1a1a4a'
        elif row['PnL_pct'] > 0:
            bg = '#002200'
        else:
            bg = '#220000'
        return [f'background-color: {bg}'] * len(row)

    st.dataframe(
        disp.style.apply(_colour, axis=1)
                  .format({'PnL_pct': '{:+.3f}%',
                           'Entry_Price': '{:.2f}',
                           'Exit_Price':  '{:.2f}'}),
        use_container_width=True
    )

with tab_tr:
    display_trade_log(trades_train, "In-Sample")

with tab_te:
    display_trade_log(trades_test, "Out-of-Sample")

with tab_lat:
    all_trades = trades_train + trades_test
    if all_trades:
        last = all_trades[-1]
        status_icon = "🟡 OPEN" if last['Status'] == 'OPEN' else "🔴 CLOSED"
        side_icon   = "📈 LONG" if last['Side'] == 'LONG' else "📉 SHORT"

        st.markdown(f"### Latest Trade  —  {status_icon}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Side",        side_icon)
        c2.metric("Entry Date",  str(pd.Timestamp(last['Entry_Date']).date()))
        c3.metric("Entry Price", f"{last['Entry_Price']:.2f}")
        c4.metric("Status",      last['Status'])

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Exit Date",   str(pd.Timestamp(last['Exit_Date']).date()))
        c6.metric("Exit Price",  f"{last['Exit_Price']:.2f}")
        c7.metric("P&L",         f"{last['PnL_pct']:+.3f}%")
        c8.metric("Reason",      last['Exit_Reason'])

        c9, _, _, _ = st.columns(4)
        c9.metric("Holding Days", str(last['Holding_Days']))

        if last['Status'] == 'OPEN':
            # Show current unrealised P&L vs latest close
            cur_price = float(data['Close'].iloc[-1])
            if last['Side'] == 'LONG':
                unreal = (cur_price - last['Entry_Price']) / last['Entry_Price'] * 100
            else:
                unreal = (last['Entry_Price'] - cur_price) / last['Entry_Price'] * 100
            st.info(
                f"🔔 **Trade still OPEN** — Latest close: **{cur_price:.2f}**  "
                f"|  Unrealised P&L: **{unreal:+.2f}%**  "
                f"|  TP target: {last['Entry_Price'] * (1 + best_long_tp if last['Side'] == 'LONG' else 1 - best_short_tp):.2f}  "
                f"|  SL level: {last['Entry_Price'] * (1 - LONG_SL_PCT if last['Side'] == 'LONG' else 1 + SHORT_SL_PCT):.2f}"
            )
    else:
        st.info("No trades generated.")

# ─────────────────────────────────────────────────────────────
# 14. PEAK / TROUGH DETECTION CHART
# ─────────────────────────────────────────────────────────────
with st.expander("🔎 Peak/Trough Detection + Regime Labels", expanded=False):
    fig_pt, axes = plt.subplots(2, 1, figsize=(16, 8),
                                 facecolor=BG, gridspec_kw={'hspace': 0.05})
    for ax in axes:
        style_ax(ax)

    axes[0].plot(dates, close_prices.values, color=TEAL, lw=1.3, label='Close')
    axes[0].plot(dates, smooth, color='#8899FF', lw=2.0, alpha=0.8,
                 label='Smoothed')
    if len(peaks):
        axes[0].scatter(dates[peaks], close_prices.values[peaks],
                        marker='v', s=100, color=RED, edgecolor='white',
                        lw=0.6, zorder=6, label='Peak (SHORT label)')
    if len(troughs):
        axes[0].scatter(dates[troughs], close_prices.values[troughs],
                        marker='^', s=100, color=GREEN, edgecolor='white',
                        lw=0.6, zorder=6, label='Trough (LONG label)')
    axes[0].set_title('Price with Detected Peaks & Troughs', color=GOLD, fontsize=10)
    axes[0].set_ylabel('Price', color=WHITE)
    axes[0].legend(fontsize=8, facecolor='#1A1A38', labelcolor=WHITE)
    axes[0].tick_params(labelbottom=False)

    axes[1].fill_between(dates, regime, alpha=0.5,
                         color=GREEN, label='LONG regime (1)',
                         where=(regime == 1))
    axes[1].fill_between(dates, regime, alpha=0.5,
                         color=RED, label='SHORT regime (0)',
                         where=(regime == 0))
    axes[1].set_yticks([0, 1])
    axes[1].set_yticklabels(['SHORT', 'LONG'], color=WHITE)
    axes[1].set_title('Regime Labels (used to train ML model)', color=GOLD, fontsize=10)
    axes[1].set_ylabel('Regime', color=WHITE)
    axes[1].legend(fontsize=8, facecolor='#1A1A38', labelcolor=WHITE)

    st.pyplot(fig_pt)

# ─────────────────────────────────────────────────────────────
# 15. LATEST ML PREDICTION
# ─────────────────────────────────────────────────────────────
st.subheader("🔮 Latest ML Prediction")

latest_row    = df.iloc[[-1]]
latest_date   = latest_row.index[0]
latest_close  = float(latest_row['Close'].values[0])
X_lat         = scaler.transform(latest_row[FEATURES].values)
prob_lat      = float(ensemble_proba(X_lat)[0])

if prob_lat >= UPPER_THRESH:
    pred_label = "📈 LONG (Uptrend)"
    pred_color = "success"
elif prob_lat <= LOWER_THRESH:
    pred_label = "📉 SHORT (Downtrend)" if SHORT_SIDE else "⏸️ FLAT (Downtrend — short disabled)"
    pred_color = "error"
else:
    pred_label = "⚠️ UNCERTAIN (grey zone)"
    pred_color = "warning"

col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Latest Date",   str(latest_date.date()))
col_b.metric("Close Price",   f"{latest_close:.2f}")
col_c.metric("LONG Prob",     f"{prob_lat:.4f}")
col_d.metric("Signal",        pred_label)

if pred_color == "success":
    st.success(f"**Signal: {pred_label}**  — Consider LONG entry near {latest_close:.2f}  "
               f"|  TP target: {latest_close * (1 + best_long_tp):.2f}  "
               f"|  SL level: {latest_close * (1 - LONG_SL_PCT):.2f}")
elif pred_color == "error":
    st.error(f"**Signal: {pred_label}**  — Consider SHORT entry near {latest_close:.2f}  "
             f"|  TP target: {latest_close * (1 - best_short_tp):.2f}  "
             f"|  SL level: {latest_close * (1 + SHORT_SL_PCT):.2f}")
else:
    st.warning(f"**Signal: {pred_label}**  — Stay flat. "
               f"Probability {prob_lat:.4f} is in grey zone "
               f"({LOWER_THRESH} – {UPPER_THRESH}).")











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
# from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
# from sklearn.utils import class_weight
# import warnings
# warnings.filterwarnings('ignore')

# st.set_page_config(page_title="BIST 100 Directional Trading", layout="wide")

# # =====================================================
# # 1. CONFIGURATION
# # =====================================================
# TICKER = st.text_input("Ticker", value="XU100.IS")
# START = st.date_input("Start Date", value=pd.to_datetime("2023-01-01"))
# END = st.date_input("End Date", value=pd.to_datetime("2027-02-28"))
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
# # 5. PLOTS (with Regression Channel)
# # =====================================================
# st.subheader("Price & Signals with Regression Channel")
# fig, axes = plt.subplots(2,1,figsize=(16,10))

# from sklearn.linear_model import LinearRegression

# x_numeric = np.arange(len(close_prices)).reshape(-1,1)
# y = close_prices.values
# reg = LinearRegression().fit(x_numeric, y)
# y_fit = reg.predict(x_numeric)

# residuals = y - y_fit
# sigma = np.std(residuals)
# upper_band = y_fit + 2*sigma
# lower_band = y_fit - 2*sigma

# axes[0].plot(close_prices.index, close_prices.values, label='Actual Price')
# axes[0].plot(close_prices.index, detection['prices_smooth'], label='Smoothed Price', linewidth=2)
# axes[0].scatter(close_prices.index[detection['peaks']], close_prices.iloc[detection['peaks']],
#                 color='red', marker='v', s=80, label='PEAK')
# axes[0].scatter(close_prices.index[detection['troughs']], close_prices.iloc[detection['troughs']],
#                 color='green', marker='^', s=80, label='TROUGH')
# axes[0].plot(close_prices.index, y_fit, color='blue', linestyle='--', linewidth=2, label='Regression Line')
# axes[0].fill_between(close_prices.index, lower_band, upper_band, color='blue', alpha=0.15, label='±2 Sigma')
# axes[0].legend(); axes[0].grid(True)

# axes[1].plot(close_prices.index, backtest_result['cumulative_log_returns'], label='Strategy')
# axes[1].fill_between(close_prices.index, 0, backtest_result['cumulative_log_returns'],
#                      where=backtest_result['positions']==1, color='green', alpha=0.2)
# axes[1].fill_between(close_prices.index, 0, backtest_result['cumulative_log_returns'],
#                      where=backtest_result['positions']==-1, color='red', alpha=0.2)
# axes[1].axhline(0, color='black', linestyle='--')
# axes[1].legend(); axes[1].grid(True)

# st.pyplot(fig)


# # =====================================================
# # 6. BUILD REGIME FROM PEAK/TROUGH DETECTION
# #    (replaces the dummy next-day-return label)
# # =====================================================

# def build_regime(prices, detection):
#     """
#     State-machine regime consistent with the backtest and multi-ticker code:
#       1 = LONG  (from trough → next peak)
#       0 = SHORT (from peak  → next trough)
#     This ensures the RF is trained on the same signal used by the backtest,
#     fixing the bug where the old code ignored peak/trough detection entirely.
#     """
#     n           = len(prices)
#     peaks_set   = set(detection['peaks'])
#     troughs_set = set(detection['troughs'])
#     regime      = np.zeros(n, dtype=int)
#     current_state = 1          # start assuming LONG
#     for i in range(n):
#         regime[i] = current_state
#         if current_state == 1 and i in peaks_set:
#             current_state = 0
#         elif current_state == 0 and i in troughs_set:
#             current_state = 1
#     return pd.Series(regime, index=prices.index, name='Regime')


# # =====================================================
# # 7. RANDOM FOREST REGIME MODEL
# # =====================================================
# st.subheader("Random Forest Regime Prediction")

# # Feature Engineering
# df_rf = data.copy()
# df_rf['Return'] = df_rf['Close'].pct_change()
# df_rf['CumRet_3'] = df_rf['Return'].rolling(3).sum()
# df_rf['CumRet_4'] = df_rf['Return'].rolling(4).sum()
# df_rf['CumRet_5'] = df_rf['Return'].rolling(5).sum()
# # df_rf['CumRet_14'] = df_rf['Return'].rolling(14).sum()
# delta = df_rf['Close'].diff()
# gain = delta.clip(lower=0)
# loss = -delta.clip(upper=0)
# avg_gain = gain.rolling(14).mean()
# avg_loss = loss.rolling(14).mean()
# rs = avg_gain/(avg_loss+1e-10)
# df_rf['RSI_14'] = 100-(100/(1+rs))
# df_rf['High_4'] = df_rf['High'].rolling(4).max()
# df_rf['Low_4'] = df_rf['Low'].rolling(4).min()
# df_rf['High_14'] = df_rf['High'].rolling(14).max() - df_rf['High'].rolling(2).max()
# df_rf['Low_14'] = df_rf['Low'].rolling(14).min() - df_rf['Low'].rolling(2).min()

# # ── FIX: assign regime from peak/trough state machine ──────────────────────
# df_rf['Regime'] = build_regime(close_prices, detection).reindex(df_rf.index)
# # ───────────────────────────────────────────────────────────────────────────

# df_rf = df_rf.dropna()

# FEATURES = ['CumRet_3','CumRet_4','CumRet_5','High_4','Low_4','RSI_14','High_14','Low_14']
# TARGET = 'Regime'

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


# # =====================================================
# # 8. CONFUSION MATRICES & CLASSIFICATION REPORTS
# # =====================================================
# if rf_trained:
#     st.subheader("Confusion Matrices & Classification Reports")

#     y_pred_train = rf.predict(X_train)
#     y_pred_test = rf.predict(X_test)

#     fig, ax = plt.subplots(1,2,figsize=(14,5))
#     disp_train = ConfusionMatrixDisplay(confusion_matrix(y_train, y_pred_train))
#     disp_train.plot(ax=ax[0], colorbar=False)
#     ax[0].set_title("In-Sample Confusion Matrix")

#     disp_test = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_test))
#     disp_test.plot(ax=ax[1], colorbar=False)
#     ax[1].set_title("Out-of-Sample Confusion Matrix")
#     st.pyplot(fig)

#     report_train = classification_report(y_train, y_pred_train, output_dict=True)
#     report_test = classification_report(y_test, y_pred_test, output_dict=True)

#     st.write("### In-Sample Classification Report")
#     st.dataframe(pd.DataFrame(report_train).transpose().round(3))

#     st.write("### Out-of-Sample Classification Report")
#     st.dataframe(pd.DataFrame(report_test).transpose().round(3))

#     # Latest Prediction
#     latest_row = df_rf.iloc[[-1]]
#     latest_date = latest_row.index[0]
#     latest_close = latest_row['Close'].values[0]
#     X_latest = scaler.transform(latest_row[FEATURES].values)
#     prob_latest = rf.predict_proba(X_latest)[:,1][0]
#     position_latest = 'HOLD'
#     if prob_latest >= 0.6: position_latest='LONG'
#     elif prob_latest <=0.4: position_latest='SHORT'

#     st.write(f"**Latest Date:** {latest_date.date()}, **Close Price:** {latest_close:.2f}")
#     st.write(f"Predicted Probability of LONG: {prob_latest:.4f}")
#     st.write(f"Recommended Position: {position_latest}")
