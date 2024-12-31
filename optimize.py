import streamlit as st
import ccxt
import ccxt.async_support as ccxt_async
import pandas as pd
from backtesting import Backtest, Strategy
import numpy as np
import time
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import asyncio

warnings.filterwarnings("ignore")

# ----------------------------------------
# HELPER FUNCTIONS -
# ----------------------------------------

def timeframe_to_pandas_freq(timeframe):
    """
    Converts a ccxt timeframe string to a Pandas frequency string.
    """
    mapping = {
        '1m': 'T',
        '5m': '5T',
        '15m': '15T',
        '30m': '30T',
        '1h': 'H',
        '4h': '4H',
        '1d': 'D',
        '1w': 'W',
        '1M': 'M',
    }
    return mapping.get(timeframe, 'D')  # Default to daily if not found

def verify_data_completeness(data, timeframe):
    """
    Verifies the completeness of the fetched data.
    """
    freq = timeframe_to_pandas_freq(timeframe)
    expected_index = pd.date_range(start=data.index.min(), end=data.index.max(), freq=freq)
    missing = expected_index.difference(data.index)
    if not missing.empty:
        st.warning(f"Data has {len(missing)} missing periods.")
        return False
    return True

def pine_crossover(a, b):
    """
    Matches PineScript's ta.crossover(source, ref):
    (source[1] < ref[1]) AND (source[0] > ref[0])
    """
    return a[-2] < b[-2] and a[-1] > b[-1]

def pine_crossunder(a, b):
    """
    Matches PineScript's ta.crossunder(source, ref):
    (source[1] > ref[1]) AND (source[0] < ref[0])
    """
    return a[-2] > b[-2] and a[-1] < b[-1]

def calculate_atr(high, low, close, window=14):
    """
    Standard ATR with rolling(window).mean().
    """
    hl = high - low
    hc = (high - close.shift()).abs()
    lc = (low - close.shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(window).mean()

# ----------------------------------------
# ASYNC DATA FETCHING
# ----------------------------------------

async def fetch_symbol_data(
    exchange_name, symbol, timeframe, length_max,
    override_check=False,  # <-- ADDED parameter
    max_retries=5, backoff_factor=2
):
    """
    Asynchronously fetches all historical OHLCV data for a given symbol.
    """
    exchange_class = getattr(ccxt_async, exchange_name, None)
    if exchange_class is None:
        st.error(f"Exchange '{exchange_name}' is not supported by ccxt.")
        return None

    exchange = exchange_class({
        'enableRateLimit': True,
        'timeout': 30000,
    })

    all_data = []
    limit = 1000

    try:
        st.write(f"Loading markets for {exchange_name}...")
        await exchange.load_markets()
    except Exception as e:
        st.error(f"Error loading markets for {symbol}: {e}")
        await exchange.close()
        return None

    if symbol not in exchange.symbols:
        st.error(f"Symbol '{symbol}' not found on '{exchange_name}'.")
        await exchange.close()
        return None

    start_date = exchange.parse8601('2017-01-01T00:00:00Z')
    since = start_date
    max_timestamp = exchange.milliseconds()

    st.write(f"Fetching data for {symbol} from {exchange_name}...")
    attempt = 0

    while True:
        try:
            ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
            if not ohlcv:
                st.write(f"No more data returned for {symbol}.")
                break

            if all_data and ohlcv[0][0] <= all_data[-1][0]:
                st.warning(f"Duplicate or overlapping data for {symbol}.")
                break

            all_data.extend(ohlcv)
            latest_timestamp = ohlcv[-1][0]
            st.write(
                f"Fetched {len(all_data)} candles for {symbol}. "
                f"Latest date: {pd.to_datetime(latest_timestamp, unit='ms')}"
            )

            since = latest_timestamp + 1
            if since > max_timestamp:
                st.write(f"Reached current timestamp for {symbol}.")
                break

            await asyncio.sleep(exchange.rateLimit / 1000)

        except ccxt.NetworkError as e:
            st.warning(f"Network error for {symbol}: {e}. Retrying...")
            attempt += 1
            if attempt >= max_retries:
                st.error(f"Failed after {max_retries} retries for {symbol}.")
                await exchange.close()
                return None
            await asyncio.sleep(backoff_factor ** attempt)
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {e}")
            await exchange.close()
            return None

    await exchange.close()

    if not all_data:
        st.warning(f"No data fetched for {symbol}.")
        return None

    data = pd.DataFrame(all_data, columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"])
    data["Date"] = pd.to_datetime(data["Timestamp"], unit="ms")
    data.set_index("Date", inplace=True)
    data.drop(columns=["Timestamp"], inplace=True)
    data.sort_index(inplace=True)

    # Check completeness
    if not verify_data_completeness(data, timeframe):
        st.warning(f"There are missing data points for {symbol}.")

    required_length = (length_max * 2) + 10
    if len(data) < required_length:
        # If override is NOT checked, do the original logic:
        if not override_check:
            st.warning(
                f"Only {len(data)} candles for {symbol}, need ≥ {required_length} for full ATR warm-up."
            )
            return None
        else:
            # If override IS checked, continue but warn
            st.warning(
                f"Data for {symbol} is insufficient ({len(data)} vs {required_length}), "
                f"but override is enabled. Proceeding anyway."
            )

    return data

async def fetch_all_symbols_data(exchange_name, symbols, timeframe, length_max, override_check=False):
    # Pass override_check to each fetch
    tasks = [
        fetch_symbol_data(exchange_name, s, timeframe, length_max, override_check=override_check)
        for s in symbols
    ]
    return await asyncio.gather(*tasks)

def run_concurrent_fetch(exchange_name, symbols, timeframe, length_max, override_check=False):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    # Pass override_check here too
    data_list = loop.run_until_complete(fetch_all_symbols_data(exchange_name, symbols, timeframe, length_max, override_check))
    loop.close()
    return data_list

# ----------------------------------------
# DATA CACHING
# ----------------------------------------

@st.cache_data(show_spinner=True, ttl=86400)
def fetch_all_historical_data_cached(exchange_name, symbols, timeframe, length_max, override_check=False):
    # Pass override_check to run_concurrent_fetch
    return run_concurrent_fetch(exchange_name, symbols, timeframe, length_max, override_check)

# ----------------------------------------
# PINE-REPLICATED STRATEGY
# ----------------------------------------

class PineReplicatedStrategy(Strategy):
    """
    This replicates your TradingView script logic:
    - length = 10
    - atr_value = ta.sma(ta.atr(200), 200) * 0.8
    - sma_high = ta.sma(high, length) + atr_value
    - sma_low  = ta.sma(low,  length) - atr_value
    - trend = false initially
    - if ta.crossover(close, sma_high) => trend = true
    - if ta.crossunder(close, sma_low) => trend = false
    - 'Buy' on same bar if trend changes from false->true
    - 'Sell' on same bar if trend changes from true->false
    """
    # Default to length=10 for alignment with your script
    length = 10  
    atr_window = 200
    atr_multiplier = 0.8

    def init(self):
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        close = pd.Series(self.data.Close)

        # 1) Calculate 200-bar ATR
        atr_raw = calculate_atr(high, low, close, window=self.atr_window)
        # 2) Then a 200-bar SMA of that ATR
        atr_sma = atr_raw.rolling(self.atr_window).mean()
        # 3) Multiply by 0.8
        self.atr_value = atr_sma * self.atr_multiplier

        # Sma over 'length'
        window_len = int(self.length)
        sma_of_high = high.rolling(window_len).mean()
        sma_of_low  = low.rolling(window_len).mean()

        self.sma_high = self.I(lambda: sma_of_high + self.atr_value)
        self.sma_low  = self.I(lambda: sma_of_low  - self.atr_value)

        # Equivalent to `var bool trend = false`
        self.current_trend = False

    def next(self):
        # Store old trend
        old_trend = self.current_trend

        # Check cross conditions: if crossover => trend = True
        if pine_crossover(self.data.Close, self.sma_high):
            self.current_trend = True
        elif pine_crossunder(self.data.Close, self.sma_low):
            self.current_trend = False

        # If 'trend' changed on this bar, we signal
        if self.current_trend != old_trend:
            if self.current_trend:
                # Turned True => buy on same bar
                self.buy()
            else:
                # Turned False => close position on same bar
                self.position.close()

# ----------------------------------------
# EXHAUSTIVE SEARCH
# ----------------------------------------

def exhaustive_search(data, length_range):
    results = []

    for length in length_range:
        PineReplicatedStrategy.length = length
        bt = Backtest(
            data,
            PineReplicatedStrategy,
            cash=1_000_000,
            commission=0.002,
            exclusive_orders=True
        )
        stats = bt.run()

        sharpe = stats.get("Sharpe Ratio", None)
        sortino = stats.get("Sortino Ratio", None)
        calmar = stats.get("Calmar Ratio", None)
        win_rate = stats.get("Win Rate [%]", None)
        profit_factor = stats.get("Profit Factor", None)
        avg_trade = stats.get("Avg Trade [%]", None)
        expectancy = stats.get("Expectancy", None)
        number_of_trades = stats.get("Total Trades", None)
        recovery_factor = stats.get("Recovery Factor", None)

        # Skip if no trades or NaN metrics
        if pd.isna(sharpe) or pd.isna(sortino) or pd.isna(calmar):
            continue

        combined_score = sharpe * 0.3 + sortino * 0.3 + calmar * 0.2
        if profit_factor and profit_factor > 1:
            combined_score += (profit_factor - 1) * 0.1
        if expectancy:
            combined_score += expectancy * 0.1

        results.append({
            "length": length,
            "sharpe": sharpe,
            "sortino": sortino,
            "calmar": calmar,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_trade": avg_trade,
            "expectancy": expectancy,
            "number_of_trades": number_of_trades,
            "recovery_factor": recovery_factor,
            "combined_score": combined_score,
            "final_equity": stats.get("Equity Final [$]", None),
            "return": stats.get("Return [%]", None),
            "max_drawdown": stats.get("Max. Drawdown [%]", None),
        })

    df = pd.DataFrame(results)
    if df.empty:
        raise ValueError("No valid results found (possibly no trades).")

    best_result = df.loc[df["combined_score"].idxmax()]
    return best_result, df

# ----------------------------------------
# PLOTTING
# ----------------------------------------

def plot_global_results(global_results_df):
    if 'length' not in global_results_df.columns:
        st.error("Missing 'length' column in results.")
        return

    grouped = global_results_df.groupby("length")["combined_score"].mean().reset_index()
    grouped.rename(columns={"combined_score": "avg_combined_score"}, inplace=True)

    best_idx = grouped["avg_combined_score"].idxmax()
    best_length = grouped.loc[best_idx, "length"]
    best_score = grouped.loc[best_idx, "avg_combined_score"]

    st.write("\n## Global Best Length")
    st.write(f"**Best Length (Avg across all tested symbols)** = {best_length}")
    st.write(f"**Average Combined Score** = {best_score:.2f}")

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=grouped, x="length", y="avg_combined_score", marker="o")
    plt.title("Average Combined Score by Length (All Symbols)")
    plt.xlabel("Length")
    plt.ylabel("Avg. Combined Score")
    plt.axvline(x=best_length, color="red", linestyle="--", label=f"Best Length: {best_length}")
    plt.legend()
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    st.image(buf, caption="Average Combined Score by Length")

# ----------------------------------------
# BACKTEST RUNNER
# ----------------------------------------

def run_backtest(exchange_name, symbols, timeframe, length_range, length_max, override_check=False):
    global_results_list = []
    total_symbols = len(symbols)
    progress_bar = st.progress(0)
    symbol_counter = 0

    st.write("Fetching data for all symbols...")
    # Pass override_check here
    data_list = fetch_all_historical_data_cached(exchange_name, symbols, timeframe, length_max, override_check)

    for idx, symbol in enumerate(symbols):
        symbol_counter += 1
        st.subheader(f"Processing {symbol}...")
        data = data_list[idx]
        if data is None:
            st.warning(f"Skipping {symbol} due to insufficient data or errors.\n")
            progress_bar.progress(symbol_counter / total_symbols)
            continue

        st.write("Analyzing all 'length' parameters...")
        try:
            best_result, all_results = exhaustive_search(data, length_range)
            csv_data = all_results.to_csv(index=False)
            st.download_button(
                label="Download Full Results CSV",
                data=csv_data,
                file_name=f"{symbol.replace('/', '')}_results.csv",
                mime="text/csv"
            )
            st.write(f"Results for {symbol} saved to CSV.")

            # Final run with best length
            PineReplicatedStrategy.length = best_result["length"]
            bt = Backtest(
                data,
                PineReplicatedStrategy,
                cash=1_000_000,
                commission=0.002,
                exclusive_orders=True
            )
            final_stats = bt.run()

            st.markdown("**Performance Metrics:**")
            def safe_fmt(v, d=2):
                return f"{v:.{d}f}" if isinstance(v, (int, float)) else str(v)

            metrics = {
                "Best Length": best_result['length'],
                "Sharpe Ratio": final_stats.get('Sharpe Ratio', 'N/A'),
                "Sortino Ratio": final_stats.get('Sortino Ratio', 'N/A'),
                "Calmar Ratio": final_stats.get('Calmar Ratio', 'N/A'),
                "Win Rate (%)": final_stats.get('Win Rate [%]', 'N/A'),
                "Profit Factor": final_stats.get('Profit Factor', 'N/A'),
                "Average Trade (%)": final_stats.get('Avg Trade [%]', 'N/A'),
                "Expectancy": final_stats.get('Expectancy', 'N/A'),
                "Number of Trades": final_stats.get('Total Trades', 'N/A'),
                "Recovery Factor": final_stats.get('Recovery Factor', 'N/A'),
                "Max Drawdown (%)": final_stats.get('Max. Drawdown [%]', 'N/A'),
                "Final Equity ($)": final_stats.get('Equity Final [$]', 'N/A'),
                "Return (%)": final_stats.get('Return [%]', 'N/A'),
            }
            for k, v in metrics.items():
                st.write(f"- **{k}**: {safe_fmt(v)}")

            # Grab trades
            trades = final_stats._trades if hasattr(final_stats, '_trades') else pd.DataFrame()
            if not isinstance(trades, pd.DataFrame):
                trades = trades.to_dataframe()

            if not trades.empty:
                # Minimal renaming for old vs. new backtesting.py versions
                rename_map = {}
                col_set = set(trades.columns)
                if 'EntryTime' in col_set:
                    rename_map['EntryTime'] = 'Entry Time'
                if 'ExitTime' in col_set:
                    rename_map['ExitTime'] = 'Exit Time'
                if 'EntryPrice' in col_set:
                    rename_map['EntryPrice'] = 'Entry Price'
                if 'ExitPrice' in col_set:
                    rename_map['ExitPrice'] = 'Exit Price'
                if 'Shares' in col_set:
                    rename_map['Shares'] = 'Size'
                if 'PnL%' in col_set:
                    rename_map['PnL%'] = 'Profit (%)'
                if 'PnL' in col_set:
                    rename_map['PnL'] = 'Profit ($)'

                trades_display = trades.copy()
                trades_display.rename(columns=rename_map, inplace=True)

                # Convert times if columns exist
                if 'Entry Time' in trades_display.columns:
                    trades_display['Entry Time'] = pd.to_datetime(
                        trades_display['Entry Time'], unit='ms', errors='coerce'
                    )
                if 'Exit Time' in trades_display.columns:
                    trades_display['Exit Time'] = pd.to_datetime(
                        trades_display['Exit Time'], unit='ms', errors='coerce'
                    )

                # Round profit columns
                if 'Profit (%)' in trades_display.columns:
                    trades_display['Profit (%)'] = trades_display['Profit (%)'].round(2)
                if 'Profit ($)' in trades_display.columns:
                    trades_display['Profit ($)'] = trades_display['Profit ($)'].round(2)

                # Final subset
                possible_cols = [
                    'Entry Time', 'Exit Time', 'Size',
                    'Entry Price', 'Exit Price', 'Profit (%)', 'Profit ($)'
                ]
                final_cols = [c for c in possible_cols if c in trades_display.columns]
                trades_display = trades_display[final_cols]

                # Show aggregated stats if "Profit (%)" is present
                if 'Profit (%)' in trades_display.columns:
                    st.markdown("**Aggregated Trade Stats:**")
                    total_trades = len(trades_display)
                    profitable_trades = len(trades_display[trades_display['Profit (%)'] > 0])
                    losing_trades = len(trades_display[trades_display['Profit (%)'] <= 0])
                    total_profit = trades_display['Profit (%)'].sum()
                    avg_profit = trades_display['Profit (%)'].mean()
                    avg_loss = (
                        trades_display.loc[trades_display['Profit (%)'] < 0, 'Profit (%)'].mean()
                        if losing_trades > 0 else float('nan')
                    )

                    st.write(f"- **Total Trades**: {total_trades}")
                    if total_trades > 0:
                        st.write(f"- **Profitable Trades**: {profitable_trades} "
                                 f"({profitable_trades / total_trades * 100:.2f}%)")
                        st.write(f"- **Losing Trades**: {losing_trades} "
                                 f"({losing_trades / total_trades * 100:.2f}%)")
                        st.write(f"- **Total Profit (%)**: {total_profit:.2f}%")
                        st.write(f"- **Average Profit per Trade (%)**: {avg_profit:.2f}%")
                        if not pd.isna(avg_loss):
                            st.write(f"- **Average Loss per Trade (%)**: {avg_loss:.2f}%")

                def highlight_profits(row):
                    if 'Profit (%)' in row and row['Profit (%)'] > 0:
                        return ['background-color: #d4edda'] * len(row)
                    elif 'Profit (%)' in row and row['Profit (%)'] < 0:
                        return ['background-color: #f8d7da'] * len(row)
                    else:
                        return [''] * len(row)

                styled_trades = trades_display.style.apply(highlight_profits, axis=1)
                trades_html = styled_trades.to_html()
                st.markdown("**Trade Log:**")
                st.markdown(trades_html, unsafe_allow_html=True)

                # Download trades
                trades_csv = trades_display.to_csv(index=False)
                st.download_button(
                    label="Download Trades CSV",
                    data=trades_csv,
                    file_name=f"{symbol.replace('/', '')}_trades.csv",
                    mime="text/csv"
                )
            else:
                st.info("No trades were executed for this symbol.")

            # Append best result
            best_dict = best_result.to_dict()
            best_dict["symbol"] = symbol
            if 'length' not in best_dict:
                best_dict['length'] = PineReplicatedStrategy.length
            global_results_list.append(best_dict)

        except ValueError as ve:
            st.error(f"No valid results found for {symbol}: {ve}")
        except Exception as e:
            st.error(f"Unexpected error for {symbol}: {e}")

        progress_bar.progress(symbol_counter / total_symbols)

    if global_results_list:
        global_df = pd.DataFrame(global_results_list)
        plot_global_results(global_df)
    else:
        st.error("No results to plot. Possibly no trades were found for any symbol.")

# ----------------------------------------
# STREAMLIT UI
# ----------------------------------------
def main():
    st.title("📈 Pine-Replicated Trend Strategy [Multi-Asset]")

    st.sidebar.header("🔧 Configuration")

    exchange_name = st.sidebar.selectbox(
        "Select Exchange",
        ["binance", "kraken", "bitfinex", "coinbasepro"]
    )

    symbols_input = st.sidebar.text_input(
        "Enter Symbols (comma-separated)",
        "BTC/USDT,ETH/USDT"
    )
    symbols = [s.strip().upper() for s in symbols_input.split(",")]

    timeframe = st.sidebar.selectbox(
        "Select Timeframe",
        ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
    )

    process_all = st.sidebar.checkbox("Process All Lengths (10-1000)")
    if not process_all:
        length_min, length_max = st.sidebar.slider(
            "Select Length Range",
            min_value=10,
            max_value=1000,
            value=(10, 200)
        )
        length_range = range(length_min, length_max + 1)
    else:
        st.sidebar.info("Processing all lengths 10–1000. This may take time.")
        length_range = range(10, 1001)
        length_max = 1000

    # --- NEW override checkbox for insufficient data
    override_check = st.sidebar.checkbox("Override Minimum Data Requirement?", value=False)

    if st.sidebar.button("🚀 Run Backtest"):
        run_backtest(
            exchange_name,
            symbols,
            timeframe,
            length_range,
            length_max=length_max,
            override_check=override_check  # Pass it down
        )

if __name__ == "__main__":
    main()
