import os
import tempfile
import json

from datetime import datetime, UTC
from typing import List, Dict, Optional

from plotly.subplots import make_subplots
import plotly.graph_objects as go

import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st

# --- Helpers to split buys and sells for plotting ---
from typing import Tuple

def split_buys_sells_lists(tx: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """Return (buys_list, sells_list) from the raw transactions.
    - Buys: qty > 0
    - Sells: qty < 0
    Dates are parsed to datetime and made tz-naive to match price indices.
    """
    if not tx:
        return [], []
    df = pd.DataFrame(tx).copy()
    if "ticker" not in df:
        return [], []
    df["ticker"] = df["ticker"].astype(str)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    # Drop timezone info so it matches df indices that are tz-naive
    try:
        if hasattr(pd, "DatetimeTZDtype") and isinstance(df["date"].dtype, pd.DatetimeTZDtype):  # type: ignore[attr-defined]
            df["date"] = df["date"].dt.tz_convert(None)
        else:
            df["date"] = df["date"].dt.tz_localize(None)
    except Exception:
        pass
    df["value"] = pd.to_numeric(df.get("value", df.get("price", 0.0)), errors="coerce").fillna(0.0)
    df["qty"] = pd.to_numeric(df.get("qty", 0.0), errors="coerce").fillna(0.0)
    df["currency"] = df.get("currency", "EUR").fillna("EUR").astype(str)

    buys_df = df[df["qty"] > 0].copy()
    sells_df = df[df["qty"] < 0].copy()

    return buys_df.to_dict(orient="records"), sells_df.to_dict(orient="records")


# === FX and quote currency helpers ===


def _prices_to_eur(last_prices: pd.Series) -> pd.Series:
    """Convert a last-prices Series (native quote currencies) to EUR using latest FX."""
    if last_prices is None or len(last_prices) == 0:
        return last_prices
    out = {}
    for ticker, px in last_prices.items():
        try:
            qccy = _get_quote_currency(ticker)
        except Exception:
            qccy = "USD"
        fx = _fx_rate(qccy, "EUR")
        out[ticker] = float(px) * fx
    return pd.Series(out)


_fx_cache = {}


def _get_quote_currency(symbol: str) -> str:
    try:
        return yf.Ticker(symbol).fast_info.get("currency") or "USD"
    except Exception:
        return "USD"


def _fx_rate(src: str, dst: str) -> float:
    """Convert 1 unit of src into dst (e.g., EUR->USD)."""
    src = (src or "USD").upper()
    dst = (dst or "USD").upper()
    if src == dst:
        return 1.0
    PAIRS = {
        ("EUR", "USD"): "EURUSD=X",
        ("USD", "EUR"): "USDEUR=X",
    }
    sym = PAIRS.get((src, dst), f"{src}{dst}=X")
    try:
        px = yf.download(sym, period="10d", interval="1d", progress=False,
                         auto_adjust=True, actions=False, threads=False)
        if px is not None and not px.empty and "Close" in px:
            rate = px["Close"].dropna().iloc[-1].item()
        else:
            raise RuntimeError("empty FX series")
    except Exception:
        # Fallback: reciprocal
        sym2 = PAIRS.get((dst, src), f"{dst}{src}=X")
        try:
            px2 = yf.download(sym2, period="10d", interval="1d", progress=False,
                              auto_adjust=True, actions=False, threads=False)
            rate2 = px2["Close"].dropna().iloc[-1].item()
            rate = 1.0 / rate2 if rate2 not in (0, None) else 1.0
        except Exception:
            rate = 1.0
    if rate and rate < 0.2:
        rate = 1.0 / rate
    return rate


st.set_page_config(page_title="Dashboard", layout="wide")

# Defaults & State
DEFAULT_TICKERS = ["AMD", "ASML", "NVDA", "STM", "TSM", "SNPS", "QCI.HA",
                   "INTC", "AVGO", "ARM", "TXN", "CDNS", "NXP", "ADI", "AFX.DE", "MCHP", "IBM"]

DATA_DIR = os.path.abspath(os.path.dirname(__file__))
TRANSACTIONS_FILE = os.path.join(DATA_DIR, "transactions.json")


def load_transactions() -> List[Dict]:
    if not os.path.exists(TRANSACTIONS_FILE):
        return []
    try:
        with open(TRANSACTIONS_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, list):
            # Normalize records to ensure backward compatibility (add currency if missing)
            normalized = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                itm = item.copy()
                if "currency" not in itm or not itm.get("currency"):
                    itm["currency"] = "EUR"
                normalized.append(itm)
            return normalized
        # If file contains a dict or other structure, ignore it and return empty
        return []
    except Exception:
        # If loading fails (corrupt file, permission), ignore and start fresh
        return []


def save_transactions(transactions: List[Dict]) -> None:
    try:
        # Ensure data dir exists
        os.makedirs(DATA_DIR, exist_ok=True)
        # Create temp file in same directory to allow atomic replace
        with tempfile.NamedTemporaryFile(mode="w", delete=False, dir=DATA_DIR, encoding="utf-8") as tmp:
            json.dump(transactions, tmp, indent=2, ensure_ascii=False)
            tmp_name = tmp.name
        os.replace(tmp_name, TRANSACTIONS_FILE)
    except Exception:
        # If persisting fails, don't crash the app; continue in memory
        return


if "transactions" not in st.session_state:
    # Basic schema: list of dicts with ticker, date (YYYY-MM-DD), qty, price
    st.session_state.transactions = load_transactions()

# Data Helpers


def fetch_history(symbols, period="1y", interval="1d") -> pd.DataFrame:
    data = yf.download(
        tickers=" ".join(symbols),
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        actions=False,
        threads=False,
    )
    if data is None or len(data) == 0:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        close = data.get("Close", pd.DataFrame()).copy()
        ordered = [s for s in symbols if s in close.columns]
        close = close[ordered]
    else:
        if "Close" in data:
            close = data[["Close"]].copy()
            close.columns = [symbols[0]]
        else:
            close = pd.DataFrame()

    # Forward-fill per ticker to avoid gaps due to mixed holidays/missing prints
    close = close.sort_index().ffill()

    # Drop rows where *all* tickers are still NaN after ffill
    close = close.dropna(how="all")
    return close


def fetch_ohlc(symbol, period="6mo", interval="1d") -> pd.DataFrame:
    df = yf.download(
        tickers=symbol,
        period=period,
        interval=interval,
        auto_adjust=False,  # keep raw OHLC for candles
        progress=False,
        actions=False,
        threads=False,
    )
    if df is None or df.empty:
        return pd.DataFrame()
    # Ensure numeric & flatten potential MultiIndex (yfinance can return ('Open','AAPL') etc.)
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = df.columns.get_level_values(0)
        if any(x in set(lvl0) for x in ["Open", "High", "Low", "Close", "Volume"]):
            df.columns = lvl0
    cols = [c for c in ["Open", "High", "Low",
                        "Close", "Volume"] if c in df.columns]
    df = df[cols].copy()
    for c in cols:
        series_or_df = df[c]
        # If duplicate column names resulted in a DataFrame, take first column
        if hasattr(series_or_df, 'ndim') and getattr(series_or_df, 'ndim', 1) > 1:
            series_or_df = series_or_df.iloc[:, 0]
        df[c] = pd.to_numeric(series_or_df, errors="coerce")
    df = df.dropna(subset=cols, how="any")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[~df.index.isna()]
    return df


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    norm = df.copy()
    for col in norm.columns:
        ser = norm[col]
        if ser.notna().any():
            first = ser.dropna().iloc[0]
            norm[col] = (ser / first) * 100.0
        else:
            norm[col] = np.nan
    norm = norm.ffill()
    return norm


def compute_performance(close: pd.DataFrame) -> pd.DataFrame:
    if close.empty:
        return pd.DataFrame(columns=["Return %"])
    last = close.ffill().iloc[-1]
    first = close.apply(lambda s: s.dropna(
    ).iloc[0] if s.notna().any() else np.nan)
    perf = (last / first - 1.0) * 100.0
    return perf.to_frame("Return %").sort_values(by="Return %", ascending=False)


def cost_basis_summary(transactions, last_prices) -> pd.DataFrame:
    cols = ["Qty", "Avg Cost", "Last", "Unrealized P/L", "P/L %"]
    if not transactions:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(transactions).copy()
    df["ticker"] = df["ticker"].astype(str)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["qty"] = pd.to_numeric(df.get("qty", 0), errors="coerce").fillna(0.0)
    df["price"] = pd.to_numeric(
        df.get("value", df.get("price", 0)), errors="coerce").fillna(0.0)
    df["currency"] = df.get("currency", "EUR").fillna("EUR").astype(str)

    rows = {}
    for ticker, tdf in df.groupby("ticker", sort=False):
        tdf = tdf.sort_values("date", kind="stable")
        avg_cost = float('nan')
        for _, r in tdf.iterrows():
            q = float(r["qty"])
            px_src = float(r["price"])
            src_ccy = str(r["currency"]) if pd.notna(r["currency"]) else "EUR"
            fx = _fx_rate(src_ccy, "EUR")
            px = px_src * fx
            if q > 0:  # buy
                if not np.isfinite(avg_cost):
                    avg_cost = px
                    qty = q
                else:
                    new_qty = qty + q
                    avg_cost = (qty * avg_cost + q * px) / \
                        new_qty if new_qty != 0 else float('nan')
                    qty = new_qty
            elif q < 0:  # sell
                sell_q = -q
                qty = qty - sell_q
                if qty <= 1e-12:
                    qty = 0.0
                    avg_cost = float('nan')
        last = float(last_prices.get(ticker, float('nan'))) if isinstance(
            last_prices, (pd.Series, dict)) else float('nan')
        if qty == 0:
            unreal = 0.0
            plpct = float('nan')
            avg_display = None
        else:
            avg_display = avg_cost if np.isfinite(avg_cost) else None
            unreal = (last - (avg_cost if np.isfinite(avg_cost) else last)) * qty
            plpct = ((last / avg_cost) - 1.0) * 100.0 if (np.isfinite(avg_cost)
                                                          and avg_cost != 0) else float('nan')
        rows[ticker] = {"Qty": qty, "Avg Cost": avg_display,
                        "Last": last, "Unrealized P/L": unreal, "P/L %": plpct}
    res = pd.DataFrame.from_dict(rows, orient="index")
    res.index.name = "Ticker"
    return res[cols].sort_index()

# Plotly helpers



def plot_indexed_plotly(norm: pd.DataFrame, buys: List[Dict], sells: List[Dict], height: int = 360):
    norm = norm.ffill()
    fig = go.Figure()
    for col in norm.columns:
        fig.add_trace(go.Scatter(x=norm.index, y=norm[col], mode="lines", name=col))

    # Buys ▲
    if buys:
        txb = pd.DataFrame(buys).copy()
        txb["date"] = pd.to_datetime(txb["date"], errors="coerce")
        for ticker in norm.columns:
            tdf = txb[txb["ticker"] == ticker]
            if tdf.empty:
                continue
            yb = norm[ticker].reindex(tdf["date"]).dropna()
            if not yb.empty:
                mask = tdf["date"].isin(yb.index)
                hb = [
                    f"Date: {d.strftime('%Y-%m-%d')}<br>Qty: {q}<br>Price: {v} {c}"
                    for d, q, v, c in zip(tdf.loc[mask, "date"], tdf.loc[mask, "qty"],
                                          tdf.loc[mask, "value"], tdf.loc[mask, "currency"])
                ]
                fig.add_trace(go.Scatter(
                    x=yb.index, y=yb.values, mode="markers", name=f"{ticker} buys",
                    hovertext=hb, hoverinfo="text",
                    marker=dict(symbol="triangle-up", size=9, line=dict(width=1, color="black"))
                ))

    # Sells ▼
    if sells:
        txs = pd.DataFrame(sells).copy()
        txs["date"] = pd.to_datetime(txs["date"], errors="coerce")
        for ticker in norm.columns:
            tdf = txs[txs["ticker"] == ticker]
            if tdf.empty:
                continue
            ys = norm[ticker].reindex(tdf["date"]).dropna()
            if not ys.empty:
                mask = tdf["date"].isin(ys.index)
                hs = [
                    f"Date: {d.strftime('%Y-%m-%d')}<br>Qty: {abs(q)}<br>Price: {v} {c}"
                    for d, q, v, c in zip(tdf.loc[mask, "date"], tdf.loc[mask, "qty"],
                                          tdf.loc[mask, "value"], tdf.loc[mask, "currency"])
                ]
                fig.add_trace(go.Scatter(
                    x=ys.index, y=ys.values, mode="markers", name=f"{ticker} sells",
                    hovertext=hs, hoverinfo="text",
                    marker=dict(symbol="triangle-down", size=9, line=dict(width=1, color="black"))
                ))

    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis_title="Index",
        xaxis_title="Date",
    )
    fig.update_yaxes(type="linear")
    fig.update_xaxes(rangeslider=dict(visible=False))
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})



def plot_candles_plotly(symbol: str, period="6mo", interval="1d", height: int = 520,
                        buys: Optional[List[Dict]] = None, sells: Optional[List[Dict]] = None):
    df = fetch_ohlc(symbol, period=period, interval=interval)
    if df.empty:
        st.info("No data for this period/interval.")
        return
    # Moving averages
    ma20 = df["Close"].rolling(20).mean()
    ma50 = df["Close"].rolling(50).mean()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                        row_heights=[0.75, 0.25], specs=[[{"secondary_y": False}], [{"secondary_y": False}]])

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name=symbol
    ), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=ma20, mode="lines", name="MA20"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=ma50, mode="lines", name="MA50"), row=1, col=1)

    # Buys overlay (▲)
    if buys:
        tb = pd.DataFrame(buys)
        tb = tb[(tb["ticker"] == symbol)].copy()
        if not tb.empty:
            tb["date"] = pd.to_datetime(tb["date"], errors="coerce")
            try:
                idxer = df.index.get_indexer(tb["date"].to_numpy(), method="ffill")
            except Exception:
                y = df["Close"].reindex(tb["date"]).dropna()
                if not y.empty:
                    fig.add_trace(go.Scatter(
                        x=y.index, y=y.values, mode="markers", name="Buys",
                        marker=dict(symbol="triangle-up", size=10, line=dict(width=1, color="black"))
                    ), row=1, col=1)
            else:
                valid = idxer != -1
                if valid.any():
                    mapped_idx = df.index[idxer[valid]]
                    mapped_close = df["Close"].iloc[idxer[valid]].values
                    hover_texts = [
                        f"Date: {od}<br>Qty: {q}<br>Price: {v} {c}"
                        for od, q, v, c in zip(
                            tb.loc[valid, "date"].dt.strftime("%Y-%m-%d"),
                            tb.loc[valid, "qty"], tb.loc[valid, "value"], tb.loc[valid, "currency"]
                        )
                    ]
                    fig.add_trace(go.Scatter(
                        x=mapped_idx, y=mapped_close, mode="markers", name="Buys",
                        hovertext=hover_texts, hoverinfo="text",
                        marker=dict(symbol="triangle-up", size=10, line=dict(width=1, color="black"))
                    ), row=1, col=1)

    # Sells overlay (▼)
    if sells:
        ts = pd.DataFrame(sells)
        ts = ts[(ts["ticker"] == symbol)].copy()
        if not ts.empty:
            ts["date"] = pd.to_datetime(ts["date"], errors="coerce")
            try:
                idxer = df.index.get_indexer(ts["date"].to_numpy(), method="ffill")
            except Exception:
                y = df["Close"].reindex(ts["date"]).dropna()
                if not y.empty:
                    fig.add_trace(go.Scatter(
                        x=y.index, y=y.values, mode="markers", name="Sells",
                        marker=dict(symbol="triangle-down", size=10, line=dict(width=1, color="black"))
                    ), row=1, col=1)
            else:
                valid = idxer != -1
                if valid.any():
                    mapped_idx = df.index[idxer[valid]]
                    mapped_close = df["Close"].iloc[idxer[valid]].values
                    hover_texts = [
                        f"Date: {od}<br>Qty: {abs(q)}<br>Price: {v} {c}"
                        for od, q, v, c in zip(
                            ts.loc[valid, "date"].dt.strftime("%Y-%m-%d"),
                            ts.loc[valid, "qty"], ts.loc[valid, "value"], ts.loc[valid, "currency"]
                        )
                    ]
                    fig.add_trace(go.Scatter(
                        x=mapped_idx, y=mapped_close, mode="markers", name="Sells",
                        hovertext=hover_texts, hoverinfo="text",
                        marker=dict(symbol="triangle-down", size=10, line=dict(width=1, color="black"))
                    ), row=1, col=1)

    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume"), row=2, col=1)

    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(rangeslider=dict(visible=False)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


# UI: Sidebar
with st.sidebar:
    st.header("Controls")
    period = st.selectbox(
        "Period", ["1mo", "3mo", "6mo", "ytd", "1y", "2y", "5y", "10y", "max"], index=4)
    interval = st.selectbox("Interval", ["1d", "1wk", "1mo"], index=0)
    chart_height = 350
    st.markdown("---")

# Tabs
tab_overview, tab_transactions, tab_single = st.tabs(
    ["Overview", "Transactions", "Single Ticker"])

# Overview Tab
with tab_overview:
    st.title("Stoncks Dashboard")
    st.caption("Buy high, sell low 🚀💰📈")
    # Image on the right upper side (under the title/caption)
    left_col_top, right_col_top = st.columns([3, 1])
    with right_col_top:
        img_path = os.path.join(DATA_DIR, "lib", "Stonks.jpg")
        if os.path.exists(img_path):
            st.image(img_path, caption="Stoncks")
        else:
            st.warning(f"Image not found: {os.path.relpath(img_path)}")

    with st.spinner("Fetching data..."):
        close = fetch_history(
            DEFAULT_TICKERS, period=period, interval=interval)
    if close.empty:
        st.error("No data returned. Try a different period/interval.")
        st.stop()

    norm = normalize(close)

    # KPIs row (window returns)
    perf = compute_performance(close)
    c1, c2, c3, c4 = st.columns(4)
    cols = [c1, c2, c3, c4]
    for i, (ticker, row) in enumerate(perf.itertuples()):
        cols[i % 4].metric(ticker, f"{row:.2f}%")

    # (Image moved to top-right above)

    st.subheader("Variation")
    buys_list, sells_list = split_buys_sells_lists(st.session_state.transactions)
    plot_indexed_plotly(norm, buys_list, sells_list, height=chart_height)

    with st.expander("Window performance table"):
        # calculate a reasonable height based on rows so the table doesn't force a small scrollbox
        perf_height = min(800, 80 + 28 * max(1, len(perf)))
        st.dataframe(perf.style.format(
            {"Return %": "{:.2f}%"}), width="stretch", height=perf_height)

# Transactions tab
with tab_transactions:
    st.subheader("Add acquisition")
    with st.form("add_tx", clear_on_submit=True):
        col1, col2, col3, col4 = st.columns(4)
        ticker = col1.selectbox("Ticker", DEFAULT_TICKERS, index=0)
        date = col2.date_input("Date", value=datetime.now(UTC))
        qty = col3.number_input(
            "Quantity", min_value=0.0, step=1.0, format="%.4f")
        value = col4.number_input(
            "Value per share", min_value=0.0, step=0.01, format="%.4f")
        currency = col4.selectbox("Currency", ["EUR", "USD"], index=0)
        submitted = st.form_submit_button("Add")
        if submitted:
            st.session_state.transactions.append({
                "ticker": ticker,
                "date": date.isoformat(),
                "qty": qty,
                "value": value,
                "currency": currency,
            })
            # Persist ledger to disk
            try:
                save_transactions(st.session_state.transactions)
            except Exception:
                # If saving fails, continue silently - the app still works in memory
                pass
            st.success(f"Added: {ticker} {qty} @ {value} on {date}")

    if st.session_state.transactions:
        st.subheader("Ledger")
        tx_df = pd.DataFrame(st.session_state.transactions)
        ledger_height = min(800, 80 + 28 * max(1, len(tx_df)))
        tx_df = tx_df.copy()
        tx_df.index = tx_df.index + 1
        tx_df.index.name = "#"
        st.dataframe(tx_df, width="stretch", height=ledger_height)

        with st.spinner("Computing P/L..."):
            latest_prices = fetch_history(
                DEFAULT_TICKERS, period=period, interval=interval)
            last_native = latest_prices.iloc[-1] if not latest_prices.empty else pd.Series(
                dtype=float)
            last = _prices_to_eur(last_native)
            summary = cost_basis_summary(st.session_state.transactions, last)
        st.subheader("Summary")
        summary_height = min(800, 80 + 28 * max(1, len(summary)))
        st.dataframe(summary.style.format({"Avg Cost": "{:.2f} €", "Last": "{:.2f} €",
                                           "Unrealized P/L": "{:.2f} €", "P/L %": "{:.2f}%"}),
                     width="stretch", height=summary_height)
    else:
        st.info("No transactions yet. Add buys above to see them on the charts.")

# Single ticker tab
with tab_single:
    t = st.selectbox("Ticker", DEFAULT_TICKERS, index=0, key="single_ticker")
    sub_period = st.selectbox(
        "Period (single)", ["1mo", "3mo", "6mo", "ytd", "1y", "2y"], index=2)
    sub_interval = st.selectbox(
        "Interval (single)", ["1d", "1wk", "1mo"], index=0)
    buys_list, sells_list = split_buys_sells_lists(st.session_state.transactions)
    plot_candles_plotly(t, period=sub_period, interval=sub_interval,
                        height=chart_height+140, buys=buys_list, sells=sells_list)
