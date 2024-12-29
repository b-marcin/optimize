import streamlit as st
import ccxt
import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import time
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import requests
import sys

warnings.filterwarnings("ignore")

# Define helper functions
def crossunder(a, b):
    return a[-2] > b[-2] and a[-1] < b[-1]

def calculate_atr(high, low, close, window=14):
    high_low = high - low
    high_close = abs(high - close.shift())
    low_close = abs(low - close.shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(window=window).mean()

def fetch_all_historical_data(exchange_name, symbol, timeframe, max_retries=5, backoff_factor=2):
    exchange = getattr(ccxt, exchange_name)({
        'enableRateLimit': True,  # Enable rate limit handling
    })
    
    # Load markets with retry logic
    attempt = 0
    while attempt < max_retries:
        try:
            st.write(f"Loading markets for {exchange_name}...")
            exchange.load_markets()
            break  # Exit loop if successful
        except ccxt.NetworkError as e:
            st.warning(f"Network error while loading markets: {e}. Retrying in {backoff_factor ** attempt} seconds...")
        except ccxt.ExchangeError as e:
            st.error(f"Exchange error while loading markets: {e}.")
            return None
        except Exception as e:
            st.error(f"Unexpected error while loading markets: {e}.")
            return None
        attempt += 1
        time.sleep(backoff_factor ** attempt)
    else:
        st.error(f"Failed to load markets for {exchange_name} after {max_retries} attempts.")
        return None

    # Set the initial 'since' parameter
    # Attempt to fetch the earliest available data by setting 'since' to 0
    since = 0  # Start from epoch; adjust if necessary based on exchange
    all_data = []
    limit = 1000  # Number of bars per API call

    st.write(f"Fetching data for {symbol} from {exchange_name}...")
    
    attempt = 0
    max_timestamp = exchange.milliseconds()  # Current timestamp in ms to prevent fetching future data

    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
            if not ohlcv:
                st.write(f"No more data returned for {symbol}. Finished fetching.")
                break
            # Append fetched data
            all_data.extend(ohlcv)
            st.write(
                f"Fetched {len(all_data)} candles for {symbol}. "
                f"Latest date: {pd.to_datetime(ohlcv[-1][0], unit='ms')}"
            )
            # Update 'since' to the last timestamp plus one millisecond
            since = ohlcv[-1][0] + 1

            # Safety check to prevent fetching beyond the current time
            if since > max_timestamp:
                st.write(f"Reached current timestamp for {symbol}.")
                break

            # Respect rate limits
            time.sleep(exchange.rateLimit / 1000)
            # Reset retry attempt after a successful fetch
            attempt = 0
        except ccxt.NetworkError as e:
            st.warning(f"Network error fetching data for {symbol}: {e}. Retrying in {backoff_factor ** attempt} seconds...")
            time.sleep(backoff_factor ** attempt)
            attempt += 1
            if attempt >= max_retries:
                st.error(f"Failed to fetch data for {symbol} after {max_retries} attempts.")
                return None
            continue
        except ccxt.ExchangeError as e:
            st.error(f"Exchange error fetching data for {symbol}: {e}. Skipping symbol.")
            return None
        except Exception as e:
            st.error(f"Unexpected error fetching data for {symbol}: {e}. Skipping symbol.")
            return None

    if not all_data:
        st.warning(f"No data fetched for {symbol}.")
        return None

    # Convert to DataFrame
    data = pd.DataFrame(all_data, columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"])
    data["Date"] = pd.to_datetime(data["Timestamp"], unit="ms")
    data.set_index("Date", inplace=True)
    data.drop(columns=["Timestamp"], inplace=True)
    data.sort_index(inplace=True)

    # Validate data length
    if len(data) < 200:
        st.warning(f"Insufficient data retrieved for {symbol}. Only {len(data)} candles fetched.")
        return None

    return data

# Define the trading strategy
class TrendStrategy(Strategy):
    atr_multiplier = 0.8
    length = 10

    def init(self):
        high = pd.Series(self.data.High)
        low  = pd.Series(self.data.Low)
        close = pd.Series(self.data.Close)

        window_atr = int(200)
        atr = calculate_atr(high, low, close, window=window_atr)
        atr_adjusted = atr * self.atr_multiplier

        window_len = int(self.length)
        sma_high = high.rolling(window=window_len).mean() + atr_adjusted
        sma_low  = low.rolling(window=window_len).mean() - atr_adjusted

        self.sma_high = self.I(lambda: sma_high)
        self.sma_low = self.I(lambda: sma_low)

    def next(self):
        if pd.isna(self.sma_high[-1]) or pd.isna(self.sma_low[-1]):
            return

        if crossover(self.data.Close, self.sma_high):
            self.buy(size=0.1)
        elif crossunder(self.data.Close, self.sma_low):
            self.sell(size=0.1)

def exhaustive_search(data, length_range):
    results = []

    for length in length_range:
        TrendStrategy.length = length
        bt = Backtest(
            data,
            TrendStrategy,
            cash=1_000_000,
            commission=0.002,
            exclusive_orders=True
        )
        stats = bt.run()

        sharpe = stats["Sharpe Ratio"]
        sortino = stats["Sortino Ratio"]
        calmar = stats["Calmar Ratio"]
        if pd.isna(sharpe) or pd.isna(sortino) or pd.isna(calmar):
            continue

        combined_score = sharpe * 0.4 + sortino * 0.4 + calmar * 0.2

        results.append({
            "length": length,
            "sharpe": sharpe,
            "sortino": sortino,
            "calmar": calmar,
            "combined_score": combined_score,
            "final_equity": stats["Equity Final [$]"],
            "return": stats["Return [%]"],
            "max_drawdown": stats["Max. Drawdown [%]"],
        })

    results_df = pd.DataFrame(results)
    if results_df.empty:
        raise ValueError("No valid results found during exhaustive search.")

    best_result = results_df.loc[results_df["combined_score"].idxmax()]
    return best_result, results_df

def plot_global_results(global_results_df):
    grouped = global_results_df.groupby("length", as_index=False)["combined_score"].mean()
    grouped.rename(columns={"combined_score": "avg_combined_score"}, inplace=True)
    best_idx = grouped["avg_combined_score"].idxmax()
    best_length = grouped.loc[best_idx, "length"]
    best_score = grouped.loc[best_idx, "avg_combined_score"]

    st.write("\n## Global Best Length")
    st.write(f"**Best Length (Avg across all assets)** = {best_length}")
    st.write(f"**Average Combined Score at best length** = {best_score:.2f}")

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=grouped, x="length", y="avg_combined_score", marker="o")
    plt.title("Average Combined Score by Length (All Symbols)")
    plt.xlabel("Length Parameter")
    plt.ylabel("Average Combined Score")
    plt.axvline(x=best_length, color="red", linestyle="--", label=f"Best Length: {best_length}")
    plt.legend()
    plt.tight_layout()

    # Convert plot to bytes
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    st.image(buf, caption="Average Combined Score by Length")

# Streamlit App
def main():
    st.title("Crypto Backtesting Tool")

    st.sidebar.header("Configuration")

    exchange_name = st.sidebar.selectbox(
        "Select Exchange",
        ["binance", "kraken", "bitfinex", "coinbasepro"]
    )

    symbols_input = st.sidebar.text_input(
        "Enter Symbols (comma-separated, e.g., BTC/USDT,ETH/USDT)",
        "BTC/USDT,ETH/USDT,XRP/USDT,SOL/USDT,DOT/USDT,AVAX/USDT,ARB/USDT,UNI/USDT,SUI/USDT"
    )
    symbols = [symbol.strip().upper() for symbol in symbols_input.split(",")]

    timeframe = st.sidebar.selectbox(
        "Select Timeframe",
        ["1m", "5m", "15m", "1h", "4h", "1d"]
    )

    length_min, length_max = st.sidebar.slider(
        "Select Length Range",
        min_value=10,
        max_value=200,
        value=(10, 200)
    )
    length_range = range(length_min, length_max + 1)

    if st.sidebar.button("Run Backtest"):
        run_backtest(exchange_name, symbols, timeframe, length_range)

def run_backtest(exchange_name, symbols, timeframe, length_range):
    global_results_list = []

    for symbol in symbols:
        st.subheader(f"Processing {symbol}...")
        data = fetch_all_historical_data(exchange_name, symbol, timeframe)
        if data is None:
            st.warning(f"Skipping {symbol} due to insufficient data or errors.\n")
            continue

        st.write("Analyzing all 'length' parameters...")
        try:
            best_result, all_results = exhaustive_search(data, length_range)
            csv = all_results.to_csv(index=False)
            st.download_button(
                label="Download Results CSV",
                data=csv,
                file_name=f"{symbol.replace('/', '')}_length_analysis.csv",
                mime="text/csv",
            )
            st.write(f"Results for {symbol} saved to CSV.")

            TrendStrategy.length = best_result["length"]
            bt = Backtest(
                data,
                TrendStrategy,
                cash=1_000_000,
                commission=0.002,
                exclusive_orders=True
            )
            final_stats = bt.run()

            st.markdown("**Performance Metrics:**")
            st.write(f"- **Best Length**: {best_result['length']}")
            st.write(f"- **Sharpe Ratio**: {final_stats['Sharpe Ratio']:.2f}")
            st.write(f"- **Sortino Ratio**: {final_stats['Sortino Ratio']:.2f}")
            st.write(f"- **Calmar Ratio**: {final_stats['Calmar Ratio']:.2f}")
            st.write(f"- **Max Drawdown (%)**: {final_stats['Max. Drawdown [%]']:.2f}")
            st.write(f"- **Final Equity ($)**: {final_stats['Equity Final [$]']:.2f}")
            st.write(f"- **Return (%)**: {final_stats['Return [%]']:.2f}")

            all_results["symbol"] = symbol
            global_results_list.append(all_results)

        except ValueError as ve:
            st.error(f"No valid results found for {symbol}: {ve}")
        except Exception as e:
            st.error(f"An unexpected error occurred while processing {symbol}: {e}")

    if global_results_list:
        global_results_df = pd.concat(global_results_list, ignore_index=True)
        plot_global_results(global_results_df)
    else:
        st.error("No results to plot. All symbols skipped or had no valid trades.")

if __name__ == "__main__":
    main()
