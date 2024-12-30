import streamlit as st
import ccxt
import ccxt.async_support as ccxt_async  # For asynchronous fetching
import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder
import time
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import asyncio
import re
import logging
from sklearn.preprocessing import StandardScaler

# -------------------------------
# Configuration and Logging
# -------------------------------

# Configure logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# -------------------------------
# Helper Functions
# -------------------------------

def timeframe_to_pandas_freq(timeframe):
    """
    Converts a ccxt timeframe string to a Pandas frequency string.

    Parameters:
        timeframe (str): Timeframe string (e.g., '1m', '5m', '1h', '1d').

    Returns:
        str: Pandas frequency string (e.g., 'T', '5T', 'H', 'D').
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
    if timeframe not in mapping:
        st.error(f"Unsupported timeframe: {timeframe}")
        return None
    return mapping[timeframe]

def verify_data_completeness(data, timeframe):
    """
    Verifies the completeness of the fetched data.

    Parameters:
        data (pd.DataFrame): Historical OHLCV data.
        timeframe (str): Timeframe for OHLCV data.

    Returns:
        bool: True if data is complete, False otherwise.
    """
    freq = timeframe_to_pandas_freq(timeframe)
    if freq is None:
        return False
    expected_index = pd.date_range(
        start=data.index.min(),
        end=data.index.max(),
        freq=freq
    )
    missing = expected_index.difference(data.index)
    if not missing.empty:
        st.warning(f"Data has {len(missing)} missing periods.")
        return False
    return True

def calculate_atr(high, low, close, window=14):
    """
    Calculates the Average True Range (ATR).

    Parameters:
        high (pd.Series): High prices.
        low (pd.Series): Low prices.
        close (pd.Series): Close prices.
        window (int): Rolling window for ATR.

    Returns:
        pd.Series: ATR values.
    """
    high_low = high - low
    high_close = (high - close.shift()).abs()
    low_close = (low - close.shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(window=window).mean()

def safe_format(value, decimals=2):
    """
    Safely formats a value to a string with specified decimals.

    Parameters:
        value: The value to format.
        decimals (int): Number of decimal places.

    Returns:
        str: Formatted string.
    """
    if isinstance(value, (int, float)):
        return f"{value:.{decimals}f}"
    else:
        return value

# -------------------------------
# Asynchronous Data Fetching with Controlled Concurrency
# -------------------------------

# Semaphore to limit concurrent requests
SEM_MAX = 5
semaphore = asyncio.Semaphore(SEM_MAX)

async def fetch_symbol_data(exchange_name, symbol, timeframe, length_max, max_retries=5, backoff_factor=2):
    """
    Asynchronously fetches all historical OHLCV data for a given symbol from the specified exchange.

    Parameters:
        exchange_name (str): Name of the exchange (e.g., 'binance').
        symbol (str): Trading pair symbol (e.g., 'BTC/USDT').
        timeframe (str): Timeframe for OHLCV data (e.g., '1d').
        length_max (int): Maximum length parameter to determine data requirements.
        max_retries (int): Maximum number of retry attempts for network/exchange errors.
        backoff_factor (int): Factor for exponential backoff between retries.

    Returns:
        pd.DataFrame or None: DataFrame containing historical data or None if failed.
    """
    async with semaphore:
        exchange_class = getattr(ccxt_async, exchange_name, None)
        if exchange_class is None:
            st.error(f"Exchange '{exchange_name}' is not supported by ccxt.")
            return None

        exchange = exchange_class({
            'enableRateLimit': True,  # Enable rate limit handling
            'timeout': 30000,         # Set a longer timeout if necessary
        })

        all_data = []
        limit = 1000  # Number of bars per API call

        try:
            await exchange.load_markets()
        except ccxt_async.NetworkError as e:
            st.error(f"Network error while loading markets for {symbol}: {e}")
            await exchange.close()
            return None
        except ccxt_async.ExchangeError as e:
            st.error(f"Exchange error while loading markets for {symbol}: {e}")
            await exchange.close()
            return None
        except Exception as e:
            st.error(f"Unexpected error while loading markets for {symbol}: {e}")
            await exchange.close()
            return None

        # Verify if the symbol exists on the exchange
        if symbol not in exchange.symbols:
            st.error(f"Symbol '{symbol}' not found on exchange '{exchange_name}'.")
            await exchange.close()
            return None

        # Determine the earliest possible 'since' timestamp
        start_date = exchange.parse8601('2017-01-01T00:00:00Z')
        since = start_date

        # Get current timestamp in ms to prevent fetching future data
        max_timestamp = exchange.milliseconds()

        attempt = 0  # Initialize attempt counter

        while True:
            try:
                ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
                if not ohlcv:
                    break

                # Check for duplicates or overlapping data
                if all_data and ohlcv[0][0] <= all_data[-1][0]:
                    st.warning(f"Duplicate or overlapping data detected for {symbol}.")
                    break

                all_data.extend(ohlcv)
                latest_timestamp = ohlcv[-1][0]

                # Update 'since' to the last timestamp plus one millisecond
                since = latest_timestamp + 1

                # Safety check to prevent fetching beyond the current time
                if since > max_timestamp:
                    break

                # Respect rate limits
                await asyncio.sleep(exchange.rateLimit / 1000)

            except ccxt_async.NetworkError as e:
                st.warning(f"Network error fetching data for {symbol}: {e}. Retrying in {backoff_factor ** attempt} seconds...")
                attempt += 1
                if attempt >= max_retries:
                    st.error(f"Failed to fetch data for {symbol} after {max_retries} attempts.")
                    await exchange.close()
                    return None
                await asyncio.sleep(backoff_factor ** attempt)
                continue
            except ccxt_async.ExchangeError as e:
                st.error(f"Exchange error fetching data for {symbol}: {e}. Skipping symbol.")
                await exchange.close()
                return None
            except Exception as e:
                st.error(f"Unexpected error fetching data for {symbol}: {e}. Skipping symbol.")
                await exchange.close()
                return None

        await exchange.close()

        if not all_data:
            st.warning(f"No data fetched for {symbol}.")
            return None

        # Convert to DataFrame
        data = pd.DataFrame(all_data, columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"])
        data["Date"] = pd.to_datetime(data["Timestamp"], unit="ms")
        data.set_index("Date", inplace=True)
        data.drop(columns=["Timestamp"], inplace=True)
        data.sort_index(inplace=True)

        # Verify data completeness
        if not verify_data_completeness(data, timeframe):
            st.warning(f"There are missing data points for {symbol}.")

        # Validate data length
        required_length = (length_max * 2) + 10  # window_atr = 2 × length_max + buffer
        if len(data) < required_length:
            st.warning(
                f"Insufficient data retrieved for {symbol}. "
                f"Only {len(data)} candles fetched, which is less than the required {required_length}."
            )
            return None

        return data

async def fetch_all_symbols_data(exchange_name, symbols, timeframe, length_max):
    """
    Asynchronously fetches data for all symbols concurrently.

    Parameters:
        exchange_name (str): Name of the exchange.
        symbols (list): List of trading pair symbols.
        timeframe (str): Timeframe for OHLCV data.
        length_max (int): Maximum length parameter.

    Returns:
        list: List of DataFrames or None for each symbol.
    """
    tasks = [fetch_symbol_data(exchange_name, symbol, timeframe, length_max) for symbol in symbols]
    return await asyncio.gather(*tasks)

def run_concurrent_fetch(exchange_name, symbols, timeframe, length_max):
    """
    Runs the asynchronous fetching of all symbols data.

    Parameters:
        exchange_name (str): Name of the exchange.
        symbols (list): List of trading pair symbols.
        timeframe (str): Timeframe for OHLCV data.
        length_max (int): Maximum length parameter.

    Returns:
        list: List of DataFrames or None for each symbol.
    """
    try:
        return asyncio.run(fetch_all_symbols_data(exchange_name, symbols, timeframe, length_max))
    except Exception as e:
        logger.error("Error in run_concurrent_fetch", exc_info=True)
        st.error(f"An error occurred while fetching data: {e}")
        return [None] * len(symbols)

# -------------------------------
# Data Fetching Function with Caching
# -------------------------------

@st.cache_data(show_spinner=True, ttl=86400)  # Cache data for 1 day
def fetch_all_historical_data_cached(exchange_name, symbols, timeframe, length_max):
    """
    Cached function to fetch historical data for all symbols.

    Parameters:
        exchange_name (str): Name of the exchange.
        symbols (list): List of trading pair symbols.
        timeframe (str): Timeframe for OHLCV data.
        length_max (int): Maximum length parameter.

    Returns:
        list: List of DataFrames or None for each symbol.
    """
    return run_concurrent_fetch(exchange_name, symbols, timeframe, length_max)

# -------------------------------
# Trading Strategy Class
# -------------------------------

class TrendStrategy(Strategy):
    """
    A trend-following strategy based on SMA and ATR.
    """
    atr_multiplier = 0.8
    length = 10

    def init(self):
        high = self.data.High
        low = self.data.Low
        close = self.data.Close

        # Dynamic ATR window based on the current 'length'
        window_atr = self.length * 2  # Example: ATR window = 2 × length
        atr = calculate_atr(high, low, close, window=window_atr)
        atr_adjusted = atr * self.atr_multiplier

        window_len = self.length
        sma_high = self.I(lambda: high.rolling(window=window_len).mean() + atr_adjusted)
        sma_low = self.I(lambda: low.rolling(window=window_len).mean() - atr_adjusted)

        self.sma_high = sma_high
        self.sma_low = sma_low

    def next(self):
        if pd.isna(self.sma_high[-1]) or pd.isna(self.sma_low[-1]):
            return

        if crossover(self.data.Close, self.sma_high):
            self.buy(size=0.1)
        elif crossunder(self.data.Close, self.sma_low):
            self.sell(size=0.1)

# -------------------------------
# Exhaustive Search Function
# -------------------------------

def exhaustive_search(data, length_range):
    """
    Performs exhaustive search over a range of 'length' parameters to find the best strategy.

    Parameters:
        data (pd.DataFrame): Historical OHLCV data.
        length_range (range): Range of 'length' parameters to test.

    Returns:
        tuple: Best result as a Series and all results as a DataFrame.
    """
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
        try:
            stats = bt.run()
        except Exception as e:
            logger.error(f"Backtest failed for length {length}: {e}")
            continue

        # Extract additional metrics if available
        sharpe = stats.get("Sharpe Ratio", None)
        sortino = stats.get("Sortino Ratio", None)
        calmar = stats.get("Calmar Ratio", None)
        win_rate = stats.get("Win Rate [%]", None)
        profit_factor = stats.get("Profit Factor", None)
        avg_trade = stats.get("Avg Trade [%]", None)
        expectancy = stats.get("Expectancy", None)
        number_of_trades = stats.get("Total Trades", None)
        recovery_factor = stats.get("Recovery Factor", None)

        # Ensure essential metrics are present
        if pd.isna(sharpe) or pd.isna(sortino) or pd.isna(calmar):
            continue

        # Normalize metrics
        scaler = StandardScaler()
        metrics_scaled = scaler.fit_transform([[sharpe, sortino, calmar, profit_factor if profit_factor else 0, expectancy if expectancy else 0]])
        scaled_sharpe, scaled_sortino, scaled_calmar, scaled_profit_factor, scaled_expectancy = metrics_scaled[0]

        # Calculate combined score with normalized metrics and weights
        combined_score = (scaled_sharpe * 0.3 +
                          scaled_sortino * 0.3 +
                          scaled_calmar * 0.2 +
                          scaled_profit_factor * 0.1 +
                          scaled_expectancy * 0.1)

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

    results_df = pd.DataFrame(results)
    if results_df.empty:
        raise ValueError("No valid results found during exhaustive search.")

    # Identify the best result based on combined score
    best_result = results_df.loc[results_df["combined_score"].idxmax()]
    return best_result, results_df

# -------------------------------
# Plotting Function
# -------------------------------

def plot_global_results(global_results_df):
    """
    Plots the average combined score by length across all symbols.

    Parameters:
        global_results_df (pd.DataFrame): DataFrame containing all backtest results.
    """
    grouped = global_results_df.groupby("length", as_index=False)["combined_score"].mean()
    grouped.rename(columns={"combined_score": "avg_combined_score"}, inplace=True)
    best_idx = grouped["avg_combined_score"].idxmax()
    best_length = grouped.loc[best_idx, "length"]
    best_score = grouped.loc[best_idx, "avg_combined_score"]

    st.write("\n## Global Best Length")
    st.write(f"**Best Length (Avg across all assets)** = {best_length}")
    st.write(f"**Average Combined Score at Best Length** = {best_score:.2f}")

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=grouped, x="length", y="avg_combined_score", marker="o")
    plt.title("Average Combined Score by Length (All Symbols)")
    plt.xlabel("Length Parameter")
    plt.ylabel("Average Combined Score")
    plt.axvline(x=best_length, color="red", linestyle="--", label=f"Best Length: {best_length}")
    plt.legend()
    plt.tight_layout()

    # Convert plot to bytes and display
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    st.image(buf, caption="Average Combined Score by Length")

# -------------------------------
# Backtesting Function
# -------------------------------

def run_backtest(exchange_name, symbols, timeframe, length_range, length_max, initial_cash, commission, exclusive_orders):
    """
    Runs backtests for all specified symbols and aggregates the results.

    Parameters:
        exchange_name (str): Name of the exchange.
        symbols (list): List of trading pair symbols.
        timeframe (str): Timeframe for OHLCV data.
        length_range (range): Range of 'length' parameters to test.
        length_max (int): Maximum 'length' parameter value.
        initial_cash (float): Initial cash for backtesting.
        commission (float): Commission percentage.
        exclusive_orders (bool): Whether to use exclusive orders.
    """
    global_results_list = []
    total_symbols = len(symbols)
    progress_bar = st.progress(0)
    symbol_counter = 0

    # Fetch all data concurrently
    st.write("Fetching data for all symbols...")
    data_list = fetch_all_historical_data_cached(exchange_name, symbols, timeframe, length_max)

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
            csv = all_results.to_csv(index=False)
            st.download_button(
                label="Download Results CSV",
                data=csv,
                file_name=f"{symbol.replace('/', '')}_length_analysis.csv",
                mime="text/csv",
            )
            st.write(f"Results for {symbol} saved to CSV.")

            # Set the best length and run the final backtest
            TrendStrategy.length = best_result["length"]
            bt = Backtest(
                data,
                TrendStrategy,
                cash=initial_cash,
                commission=commission,
                exclusive_orders=exclusive_orders
            )
            final_stats = bt.run()

            st.markdown("**Performance Metrics:**")

            # Display each metric with safe formatting
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

            for metric, value in metrics.items():
                formatted_value = safe_format(value)
                st.write(f"- **{metric}**: {formatted_value}")

            # Extract and display trade details
            # Access 'trades' after running the backtest
            if hasattr(bt, 'trades') and not bt.trades.empty:
                trades = bt.trades.copy()
                trades['Entry Time'] = pd.to_datetime(trades['Entry Time'], unit='ms')
                trades['Exit Time'] = pd.to_datetime(trades['Exit Time'], unit='ms')
                trades['Profit (%)'] = trades['Profit [%]'].round(2)
                trades['Return (%)'] = trades['Return [%]'].round(2)
                trades_display = trades[[
                    'Entry Time', 'Exit Time', 'Size', 'Entry Price', 'Exit Price',
                    'Profit (%)', 'Return (%)'
                ]]
                trades_display.rename(columns={
                    'Size': 'Position Size',
                    'Entry Price': 'Buy Price',
                    'Exit Price': 'Sell Price'
                }, inplace=True)

                # Display aggregated statistics
                st.markdown("**Aggregated Trade Statistics:**")
                total_trades = len(trades_display)
                profitable_trades = len(trades_display[trades_display['Profit (%)'] > 0])
                losing_trades = len(trades_display[trades_display['Profit (%)'] <= 0])
                total_profit = trades_display['Profit (%)'].sum()
                average_profit = trades_display['Profit (%)'].mean()
                average_loss = trades_display[trades_display['Profit (%)'] < 0]['Profit (%)'].mean()
                profit_factor = final_stats.get("Profit Factor", "N/A")
                expectancy = final_stats.get("Expectancy", "N/A")

                st.write(f"- **Total Trades**: {total_trades}")
                st.write(f"- **Profitable Trades**: {profitable_trades} ({(profitable_trades/total_trades*100):.2f}%)")
                st.write(f"- **Losing Trades**: {losing_trades} ({(losing_trades/total_trades*100):.2f}%)")
                st.write(f"- **Total Profit (%)**: {total_profit:.2f}%")
                st.write(f"- **Average Profit per Trade (%)**: {average_profit:.2f}%")
                st.write(f"- **Average Loss per Trade (%)**: {average_loss:.2f}%")
                st.write(f"- **Profit Factor**: {profit_factor}")
                st.write(f"- **Expectancy**: {expectancy:.2f}")

                st.markdown("**Trade Log:**")

                # Limit the number of trades displayed to prevent UI overload
                max_display = 100
                if len(trades_display) > max_display:
                    st.write(f"Displaying first {max_display} trades out of {len(trades_display)}")
                    trades_display = trades_display.head(max_display)

                # Apply conditional formatting
                def highlight_profits(row):
                    if row['Profit (%)'] > 0:
                        color = 'background-color: #d4edda'  # Light green
                    elif row['Profit (%)'] < 0:
                        color = 'background-color: #f8d7da'  # Light red
                    else:
                        color = ''
                    return [color] * len(row)

                styled_trades = trades_display.style.apply(highlight_profits, axis=1)

                # Display trades with pagination
                st.dataframe(styled_trades, use_container_width=True)

                # Enable CSV download of trades
                trades_csv = trades_display.to_csv(index=False)
                st.download_button(
                    label="Download Trades CSV",
                    data=trades_csv,
                    file_name=f"{symbol.replace('/', '')}_trades.csv",
                    mime="text/csv",
                )
            else:
                st.info("No trades were executed for this symbol.")

            all_results = best_result.copy()
            all_results["symbol"] = symbol
            global_results_list.append(all_results)

        except ValueError as ve:
            st.error(f"No valid results found for {symbol}: {ve}")
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}", exc_info=True)
            st.error(f"An unexpected error occurred while processing {symbol}: {e}")

        # Update progress bar
        progress_bar.progress(symbol_counter / total_symbols)

    if global_results_list:
        global_results_df = pd.concat(global_results_list, ignore_index=True)
        plot_global_results(global_results_df)
    else:
        st.error("No results to plot. All symbols skipped or had no valid trades.")

# -------------------------------
# Streamlit App
# -------------------------------

def main():
    """
    Main function to run the Streamlit app.
    """
    st.title("📈 Crypto Backtesting Tool")

    st.sidebar.header("🔧 Configuration")

    # Exchange Selection
    exchange_name = st.sidebar.selectbox(
        "Select Exchange",
        ["binance", "kraken", "bitfinex", "coinbasepro"]
    )

    # Symbols Input with Validation
    symbols_input = st.sidebar.text_input(
        "Enter Symbols (comma-separated, e.g., BTC/USDT,ETH/USDT)",
        "BTC/USDT,ETH/USDT,XRP/USDT,SOL/USDT,DOT/USDT,AVAX/USDT,ARB/USDT,UNI/USDT,SUI/USDT"
    )
    # Validate symbol format
    symbol_pattern = re.compile(r"^[A-Z0-9]+/[A-Z0-9]+$")
    symbols = [symbol.strip().upper() for symbol in symbols_input.split(",") if symbol_pattern.match(symbol.strip())]

    if not symbols:
        st.error("No valid symbols entered. Please enter symbols in the format BASE/QUOTE, e.g., BTC/USDT.")
        st.stop()

    # Timeframe Selection
    timeframe = st.sidebar.selectbox(
        "Select Timeframe",
        ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
    )
    pandas_freq = timeframe_to_pandas_freq(timeframe)
    if pandas_freq is None:
        st.stop()

    # Process All Lengths Checkbox
    process_all = st.sidebar.checkbox("Process All Lengths (10-1000)")

    if not process_all:
        # Length Range Slider with Reasonable Limits
        length_min, length_max = st.sidebar.slider(
            "Select Length Range",
            min_value=10,
            max_value=200,  # Adjusted max value for manageability
            value=(10, 100)
        )
        length_range = range(length_min, length_max + 1)
    else:
        # Set to entire possible range with user warning
        st.sidebar.info("Processing all lengths from 10 to 1000. This may take some time.")
        length_range = range(10, 1001)

    # Expose Backtesting Parameters
    initial_cash = st.sidebar.number_input(
        "Initial Cash ($)",
        min_value=1000,
        value=1_000_000,
        step=10_000
    )
    commission = st.sidebar.number_input(
        "Commission (%)",
        min_value=0.0,
        max_value=5.0,
        value=0.2,
        step=0.1
    ) / 100  # Convert to decimal
    exclusive_orders = st.sidebar.checkbox("Exclusive Orders", value=True)

    # Run Backtest Button
    if st.sidebar.button("🚀 Run Backtest"):
        with st.spinner('Running backtests...'):
            run_backtest(
                exchange_name=exchange_name,
                symbols=symbols,
                timeframe=timeframe,
                length_range=length_range,
                length_max=1000 if process_all else length_max,
                initial_cash=initial_cash,
                commission=commission,
                exclusive_orders=exclusive_orders
            )

# -------------------------------
# Entry Point
# -------------------------------

if __name__ == "__main__":
    main()
