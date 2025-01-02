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
# HELPER FUNCTIONS
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
    return mapping.get(timeframe, 'D')

def verify_data_completeness(data, timeframe):
    """
    Enhanced verification of data completeness with detailed reporting.
    """
    freq = timeframe_to_pandas_freq(timeframe)
    expected_index = pd.date_range(start=data.index.min(), end=data.index.max(), freq=freq)
    missing = expected_index.difference(data.index)
    
    if not missing.empty:
        # Group missing periods to find gaps
        missing_series = pd.Series(missing)
        gaps = missing_series.diff() > pd.Timedelta(hours=4 if timeframe == '4h' else 24)
        gap_starts = missing_series[gaps].index
        
        st.warning(f"""Data Quality Report:
        - Total missing periods: {len(missing)}
        - Number of gaps: {len(gap_starts)}
        - First missing: {missing[0]}
        - Last missing: {missing[-1]}
        """)
        
        # Display gaps if there are any
        if len(gap_starts) > 0:
            st.write("Major gaps found:")
            for start in gap_starts:
                end = missing_series[missing_series > start].iloc[0]
                st.write(f"- Gap from {start} to {end}")
                
        return False
    return True

def preprocess_data(data, timeframe):
    """
    Handles data preprocessing including gap filling and cleaning.
    """
    # Remove any duplicate indices
    data = data[~data.index.duplicated(keep='first')]
    
    # Sort by index
    data = data.sort_index()
    
    # Forward fill gaps shorter than 3 periods
    freq = timeframe_to_pandas_freq(timeframe)
    max_fill_periods = 3
    data = data.asfreq(freq)
    data = data.fillna(method='ffill', limit=max_fill_periods)
    
    # Drop any remaining NaN values
    data = data.dropna()
    
    return data

def pine_crossover(a, b):
    """
    Matches PineScript's ta.crossover(source, ref).
    """
    return a[-2] < b[-2] and a[-1] > b[-1]

def pine_crossunder(a, b):
    """
    Matches PineScript's ta.crossunder(source, ref).
    """
    return a[-2] > b[-2] and a[-1] < b[-1]

def calculate_atr(high, low, close, window=14):
    """
    Standard ATR calculation.
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
    override_check=False,
    max_retries=5, backoff_factor=2
):
    """
    Enhanced fetch function with better error handling and data quality checks.
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

    try:
        # Create DataFrame and process data
        data = pd.DataFrame(all_data, columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"])
        data["Date"] = pd.to_datetime(data["Timestamp"], unit="ms")
        data.set_index("Date", inplace=True)
        data.drop(columns=["Timestamp"], inplace=True)
        
        # Add preprocessing step
        data = preprocess_data(data, timeframe)
        
        # Log data quality information
        st.write(f"""
        Data Summary for {symbol}:
        - Time Range: {data.index.min()} to {data.index.max()}
        - Total Periods: {len(data)}
        - Trading Days: {len(data.index.date.unique())}
        """)
        
        # Verify completeness with enhanced reporting
        verify_data_completeness(data, timeframe)
        
        required_length = (length_max * 2) + 10
        if len(data) < required_length and not override_check:
            st.warning(
                f"Insufficient data for {symbol}: {len(data)} < {required_length} periods required"
            )
            return None
            
        return data
        
    except Exception as e:
        st.error(f"Error processing {symbol}: {str(e)}")
        return None

async def fetch_all_symbols_data(exchange_name, symbols, timeframe, length_max, override_check=False):
    tasks = [
        fetch_symbol_data(exchange_name, s, timeframe, length_max, override_check=override_check)
        for s in symbols
    ]
    return await asyncio.gather(*tasks)

def run_concurrent_fetch(exchange_name, symbols, timeframe, length_max, override_check=False):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    data_list = loop.run_until_complete(
        fetch_all_symbols_data(exchange_name, symbols, timeframe, length_max, override_check)
    )
    loop.close()
    return data_list

@st.cache_data(show_spinner=True, ttl=86400)
def fetch_all_historical_data_cached(exchange_name, symbols, timeframe, length_max, override_check=False):
    return run_concurrent_fetch(exchange_name, symbols, timeframe, length_max, override_check)
# ----------------------------------------
# STRATEGY IMPLEMENTATION
# ----------------------------------------

class PineReplicatedStrategy(Strategy):
    length = 10
    atr_window = 200
    atr_multiplier = 0.8

    def init(self):
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        close = pd.Series(self.data.Close)

        atr_raw = calculate_atr(high, low, close, window=self.atr_window)
        atr_sma = atr_raw.rolling(self.atr_window).mean()
        self.atr_value = atr_sma * self.atr_multiplier

        window_len = int(self.length)
        sma_of_high = high.rolling(window_len).mean()
        sma_of_low = low.rolling(window_len).mean()

        self.sma_high = self.I(lambda: sma_of_high + self.atr_value)
        self.sma_low = self.I(lambda: sma_of_low - self.atr_value)

        self.current_trend = False

    def next(self):
        old_trend = self.current_trend

        if pine_crossover(self.data.Close, self.sma_high):
            self.current_trend = True
        elif pine_crossunder(self.data.Close, self.sma_low):
            self.current_trend = False

        if self.current_trend != old_trend:
            if self.current_trend:
                self.buy()
            else:
                self.position.close()

# ----------------------------------------
# BACKTEST AND ANALYSIS
# ----------------------------------------

def run_backtest(exchange_name, symbols, timeframe, length_range, length_max, override_check=False):
    """
    Runs backtests for all symbols and analyzes results with enhanced visualization.
    """
    global_results_list = []
    total_symbols = len(symbols)
    progress_bar = st.progress(0)
    symbol_counter = 0

    st.write("Fetching data for all symbols...")
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
            stats = bt.run()

            st.markdown("**Performance Metrics:**")
            metrics = {
                "Best Length": best_result['length'],
                "Sharpe Ratio": stats.get('Sharpe Ratio', 'N/A'),
                "Sortino Ratio": stats.get('Sortino Ratio', 'N/A'),
                "Calmar Ratio": stats.get('Calmar Ratio', 'N/A'),
                "Win Rate (%)": stats.get('Win Rate [%]', 'N/A'),
                "Profit Factor": stats.get('Profit Factor', 'N/A'),
                "Max Drawdown (%)": stats.get('Max. Drawdown [%]', 'N/A'),
                "Final Equity ($)": stats.get('Equity Final [$]', 'N/A'),
                "Return (%)": stats.get('Return [%]', 'N/A'),
            }
            
            for k, v in metrics.items():
                st.write(f"- **{k}**: {v:.2f}" if isinstance(v, (int, float)) else f"- **{k}**: {v}")

            # Process trades if available
            trades = stats._trades if hasattr(stats, '_trades') else pd.DataFrame()
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

                # Calculate % Return
                if 'Entry Price' in trades_display.columns and 'Exit Price' in trades_display.columns:
                    trades_display['% Return'] = ((trades_display['Exit Price'] - trades_display['Entry Price']) / trades_display['Entry Price']) * 100
                    trades_display['% Return'] = trades_display['% Return'].round(2)

                # Round profit columns
                if 'Profit (%)' in trades_display.columns:
                    trades_display['Profit (%)'] = trades_display['Profit (%)'].round(2)
                if 'Profit ($)' in trades_display.columns:
                    trades_display['Profit ($)'] = trades_display['Profit ($)'].round(2)

                # Calculate Holding Time in minutes
                if 'Entry Time' in trades_display.columns and 'Exit Time' in trades_display.columns:
                    trades_display['Holding Time (min)'] = (trades_display['Exit Time'] - trades_display['Entry Time']).dt.total_seconds() / 60
                    trades_display['Holding Time (min)'] = trades_display['Holding Time (min)'].round(2)

                # Select final columns for display
                possible_cols = [
                    'Entry Time', 'Exit Time', 'Size',
                    'Entry Price', 'Exit Price', '% Return', 'Profit (%)', 'Profit ($)', 'Holding Time (min)'
                ]
                final_cols = [c for c in possible_cols if c in trades_display.columns]
                trades_display = trades_display[final_cols]

                # Show aggregated stats
                if 'Profit (%)' in trades_display.columns and '% Return' in trades_display.columns:
                    st.markdown("**Aggregated Trade Stats:**")
                    total_trades = len(trades_display)
                    profitable_trades = len(trades_display[trades_display['% Return'] > 0])
                    losing_trades = len(trades_display[trades_display['% Return'] <= 0])
                    total_profit = trades_display['% Return'].sum()
                    avg_profit = trades_display['% Return'].mean()
                    avg_loss = (
                        trades_display.loc[trades_display['% Return'] < 0, '% Return'].mean()
                        if losing_trades > 0 else float('nan')
                    )

                    st.write(f"- **Total Trades**: {total_trades}")
                    if total_trades > 0:
                        st.write(f"- **Profitable Trades**: {profitable_trades} "
                                 f"({profitable_trades / total_trades * 100:.2f}%)")
                        st.write(f"- **Losing Trades**: {losing_trades} "
                                 f"({losing_trades / total_trades * 100:.2f}%)")
                        st.write(f"- **Total % Return**: {total_profit:.2f}%")
                        st.write(f"- **Average % Return per Trade**: {avg_profit:.2f}%")
                        if not pd.isna(avg_loss):
                            st.write(f"- **Average Loss per Trade (%)**: {avg_loss:.2f}%")
                
                # Add enhanced visualization
                st.header("📊 Enhanced Trade Analysis")
                display_enhanced_metrics(trades_display)
                
                try:
                    fig = plot_strategy_analysis(trades_display, stats)
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error creating visualizations: {str(e)}")

                # Style and display trades table
                def highlight_profits(row):
                    if '% Return' in row and row['% Return'] > 0:
                        return ['background-color: #d4edda'] * len(row)
                    elif '% Return' in row and row['% Return'] < 0:
                        return ['background-color: #f8d7da'] * len(row)
                    else:
                        return [''] * len(row)

                styled_trades = trades_display.style.apply(highlight_profits, axis=1)\
                    .set_properties(**{
                        'text-align': 'center',
                        'border': '1px solid black'
                    })\
                    .set_table_styles([{
                        'selector': 'th',
                        'props': [('background-color', '#f2f2f2'), ('color', 'black'), ('font-weight', 'bold')]
                    }])

                trades_html = styled_trades.to_html()
                st.markdown("**Trade Log:**")
                st.markdown(trades_html, unsafe_allow_html=True)

                # Provide download options
                csv_trades = trades_display.to_csv(index=False)
                st.download_button(
                    label="Download Trades CSV",
                    data=csv_trades,
                    file_name=f"{symbol.replace('/', '')}_trades.csv",
                    mime="text/csv"
                )
            else:
                st.info("No trades were executed for this symbol.")

            # Append results for global analysis
            best_dict = best_result.to_dict()
            best_dict["symbol"] = symbol
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

        # Extract metrics
        metrics = {
            "length": length,
            "sharpe": stats.get("Sharpe Ratio", None),
            "sortino": stats.get("Sortino Ratio", None),
            "calmar": stats.get("Calmar Ratio", None),
            "win_rate": stats.get("Win Rate [%]", None),
            "profit_factor": stats.get("Profit Factor", None),
            "avg_trade": stats.get("Avg Trade [%]", None),
            "expectancy": stats.get("Expectancy", None),
            "number_of_trades": stats.get("Total Trades", None),
            "recovery_factor": stats.get("Recovery Factor", None),
            "final_equity": stats.get("Equity Final [$]", None),
            "return": stats.get("Return [%]", None),
            "max_drawdown": stats.get("Max. Drawdown [%]", None),
        }

        # Skip if critical metrics are missing
        if pd.isna(metrics["sharpe"]) or pd.isna(metrics["sortino"]) or pd.isna(metrics["calmar"]):
            continue

        # Calculate combined score
        combined_score = (
            metrics["sharpe"] * 0.3 +
            metrics["sortino"] * 0.3 +
            metrics["calmar"] * 0.2
        )
        
        if metrics["profit_factor"] and metrics["profit_factor"] > 1:
            combined_score += (metrics["profit_factor"] - 1) * 0.1
        if metrics["expectancy"]:
            combined_score += metrics["expectancy"] * 0.1

        metrics["combined_score"] = combined_score
        results.append(metrics)

    df = pd.DataFrame(results)
    if df.empty:
        raise ValueError("No valid results found (possibly no trades).")

    best_result = df.loc[df["combined_score"].idxmax()]
    return best_result, df

def compare_timeframes(exchange_name, symbols, length_range, length_max, override_check=False):
    """
    Enhanced timeframe comparison with better error handling and reporting.
    """
    timeframes = ['4h', '1d']
    all_results = {}
    progress_bar = st.progress(0)
    total_steps = len(timeframes) * len(symbols)
    current_step = 0

    for timeframe in timeframes:
        st.subheader(f"Fetching {timeframe} data...")
        data_list = fetch_all_historical_data_cached(
            exchange_name, symbols, timeframe, length_max, override_check
        )
        
        timeframe_results = []
        for idx, symbol in enumerate(symbols):
            current_step += 1
            progress_bar.progress(current_step / total_steps)
            
            data = data_list[idx]
            if data is None:
                st.warning(f"Skipping {symbol} for {timeframe} timeframe due to insufficient data.")
                continue
                
            try:
                best_result, _ = exhaustive_search(data, length_range)
                result_dict = best_result.to_dict()
                result_dict["symbol"] = symbol
                result_dict["timeframe"] = timeframe
                timeframe_results.append(result_dict)
            except Exception as e:
                st.error(f"Error processing {symbol} for {timeframe}: {e}")
                
        if timeframe_results:  # Only create DataFrame if we have results
            all_results[timeframe] = pd.DataFrame(timeframe_results)
        else:
            st.error(f"No valid results for {timeframe} timeframe")
            all_results[timeframe] = pd.DataFrame()  # Empty DataFrame
    
    return all_results

def plot_timeframe_comparison(results_dict):
    """
    Enhanced visualization of timeframe comparisons with better formatting.
    """
    if not all(tf in results_dict for tf in ['4h', '1d']):
        st.error("Missing data for one or both timeframes.")
        return
        
    # Aggregate Comparison
    st.header("📊 Aggregate Timeframe Comparison")
    metrics = ['sharpe', 'sortino', 'calmar', 'win_rate', 'profit_factor', 'return']
    metric_names = {
        'sharpe': 'Sharpe Ratio',
        'sortino': 'Sortino Ratio',
        'calmar': 'Calmar Ratio',
        'win_rate': 'Win Rate (%)',
        'profit_factor': 'Profit Factor',
        'return': 'Return (%)'
    }
    
    # Create summary table
    summary_data = []
    for tf in ['4h', '1d']:
        df = results_dict[tf]
        metrics_dict = {
            'Timeframe': tf,
            **{metric_names[m]: df[m].mean() for m in metrics}
        }
        summary_data.append(metrics_dict)
    
    summary_df = pd.DataFrame(summary_data)
    
    # Create formatter that handles both numeric and non-numeric columns
    def format_value(val):
        try:
            if pd.isna(val):
                return "N/A"
            elif isinstance(val, (int, float)):
                return f"{val:.2f}"
            return str(val)
        except:
            return str(val)
    
    # Format all columns except 'Timeframe'
    formatted_df = summary_df.copy()
    for col in formatted_df.columns:
        if col != 'Timeframe':
            formatted_df[col] = formatted_df[col].apply(format_value)
    
    st.write("Summary Statistics (Averages across all symbols):")
    st.dataframe(formatted_df)
    
    # Plotting
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (metric, metric_name) in enumerate(metric_names.items()):
        ax = axes[idx]
        data = []
        for tf in ['4h', '1d']:
            avg_value = results_dict[tf][metric].mean()
            data.append({
                'timeframe': tf,
                'value': avg_value
            })
        metric_df = pd.DataFrame(data)
        
        sns.barplot(
            data=metric_df,
            x='timeframe',
            y='value',
            ax=ax
        )
        ax.set_title(metric_name)
        ax.set_ylabel('')
        # Format y-axis ticks to 2 decimal places
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.2f}"))
        
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # Individual Symbol Comparison
    st.header("🔍 Individual Symbol Performance")
    common_symbols = set(results_dict['4h']['symbol']) & set(results_dict['1d']['symbol'])
    
    for symbol in common_symbols:
        st.subheader(f"{symbol} Comparison")
        
        symbol_data = []
        for tf in ['4h', '1d']:
            symbol_row = results_dict[tf][results_dict[tf]['symbol'] == symbol].iloc[0]
            metrics_dict = {
                'Timeframe': tf,
                **{metric_names[m]: symbol_row[m] for m in metrics}
            }
            symbol_data.append(metrics_dict)
            
        symbol_df = pd.DataFrame(symbol_data)
        # Format individual symbol dataframes
        formatted_symbol_df = symbol_df.copy()
        for col in formatted_symbol_df.columns:
            if col != 'Timeframe':
                formatted_symbol_df[col] = formatted_symbol_df[col].apply(format_value)
                
        st.dataframe(formatted_symbol_df)
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, (metric, metric_name) in enumerate(metric_names.items()):
            ax = axes[idx]
            data = []
            for tf in ['4h', '1d']:
                value = results_dict[tf][results_dict[tf]['symbol'] == symbol][metric].iloc[0]
                data.append({
                    'timeframe': tf,
                    'value': value
                })
            metric_df = pd.DataFrame(data)
            
            sns.barplot(
                data=metric_df,
                x='timeframe',
                y='value',
                ax=ax
            )
            ax.set_title(f"{metric_name}")
            ax.set_ylabel('')
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.2f}"))
            
        plt.suptitle(f"{symbol} Metrics Comparison", y=1.02)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

def plot_strategy_analysis(trades_df, stats):
    """
    Creates comprehensive strategy visualization with multiple subplots.
    """
    if trades_df.empty:
        st.warning("No trades to visualize.")
        return
        
    # Convert Entry Time if needed
    if 'Entry Time' in trades_df.columns and not isinstance(trades_df['Entry Time'].iloc[0], pd.Timestamp):
        trades_df['Entry Time'] = pd.to_datetime(trades_df['Entry Time'])
        
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1])

    # 1. Returns Distribution
    ax1 = fig.add_subplot(gs[0, 0])
    returns = trades_df['% Return'] if '% Return' in trades_df.columns else trades_df['Profit (%)']
    sns.histplot(returns, kde=True, ax=ax1, color='blue', alpha=0.6)
    ax1.axvline(x=0, color='r', linestyle='--', alpha=0.5)
    ax1.set_title('Returns Distribution')
    ax1.set_xlabel('Return %')
    ax1.set_ylabel('Frequency')

    # 2. Cumulative Returns
    ax2 = fig.add_subplot(gs[0, 1])
    cumulative_returns = (1 + returns/100).cumprod()
    ax2.plot(range(len(cumulative_returns)), cumulative_returns, color='green')
    ax2.set_title('Cumulative Returns Growth')
    ax2.set_xlabel('Trade Number')
    ax2.set_ylabel('Growth Multiple')
    ax2.grid(True)

    # 3. Monthly Performance Heatmap
    ax3 = fig.add_subplot(gs[1, :])
    monthly_returns = trades_df.set_index('Entry Time')['% Return'].resample('M').sum()
    monthly_returns = monthly_returns.to_frame().pivot_table(
        index=monthly_returns.index.year,
        columns=monthly_returns.index.month,
        values='% Return'
    )
    sns.heatmap(monthly_returns, cmap='RdYlGn', center=0, ax=ax3, annot=True, fmt='.1f')
    ax3.set_title('Monthly Returns Heatmap (%)')
    ax3.set_xlabel('Month')
    ax3.set_ylabel('Year')

    # 4. Trade Duration Analysis
    ax4 = fig.add_subplot(gs[2, 0])
    if 'Holding Time (min)' in trades_df.columns:
        holding_times = trades_df['Holding Time (min)']
        sns.boxplot(y=holding_times, ax=ax4, color='lightblue')
        ax4.set_title('Trade Duration Distribution')
        ax4.set_ylabel('Duration (minutes)')

    # 5. Win/Loss Analysis by Month
    ax5 = fig.add_subplot(gs[2, 1])
    monthly_wins = trades_df[trades_df['% Return'] > 0].set_index('Entry Time').resample('M').size()
    monthly_losses = trades_df[trades_df['% Return'] <= 0].set_index('Entry Time').resample('M').size()
    
    width = 0.35
    months = range(len(monthly_wins))
    ax5.bar(months, monthly_wins, width, label='Wins', color='green', alpha=0.6)
    ax5.bar([x + width for x in months], monthly_losses, width, label='Losses', color='red', alpha=0.6)
    ax5.set_title('Monthly Win/Loss Distribution')
    ax5.set_xlabel('Month')
    ax5.set_ylabel('Number of Trades')
    ax5.legend()

    plt.tight_layout()
    return fig

def display_enhanced_metrics(trades_df):
    """
    Displays enhanced trading metrics in a formatted way.
    """
    if trades_df.empty:
        st.warning("No trades to analyze.")
        return

    # Calculate metrics
    returns = trades_df['% Return'] if '% Return' in trades_df.columns else trades_df['Profit (%)']
    profitable_trades = len(returns[returns > 0])
    losing_trades = len(returns[returns <= 0])
    total_trades = len(returns)
    win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
    avg_win = returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0
    avg_loss = returns[returns < 0].mean() if len(returns[returns < 0]) > 0 else 0

    # Display metrics in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Trades", total_trades)
        st.metric("Win Rate", f"{win_rate:.2f}%")
        st.metric("Profit Factor", f"{abs(avg_win/avg_loss):.2f}" if avg_loss != 0 else "∞")
        
    with col2:
        st.metric("Profitable Trades", profitable_trades)
        st.metric("Average Win", f"{avg_win:.2f}%")
        st.metric("Best Trade", f"{returns.max():.2f}%")
        
    with col3:
        st.metric("Losing Trades", losing_trades)
        st.metric("Average Loss", f"{avg_loss:.2f}%")
        st.metric("Worst Trade", f"{returns.min():.2f}%")

# Add this to your run_backtest function:
def enhanced_trade_analysis(trades_df, stats):
    """
    Performs enhanced trade analysis and displays results.
    """
    st.header("📊 Enhanced Trade Analysis")
    
    # Display enhanced metrics
    display_enhanced_metrics(trades_df)
    
    # Create and display enhanced visualizations
    try:
        fig = plot_strategy_analysis(trades_df, stats)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error creating visualizations: {str(e)}")
        
    # Provide detailed trade data download
    if not trades_df.empty:
        csv = trades_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Detailed Trade Analysis",
            data=csv,
            file_name="trade_analysis.csv",
            mime="text/csv"
        )

    # Add comparison summary download
    comparison_summary = pd.concat([
        results_dict['4h'].assign(timeframe='4h'),
        results_dict['1d'].assign(timeframe='1d')
    ])
    
    csv_data = comparison_summary.to_csv(index=False)
    st.download_button(
        label="Download Complete Comparison Results",
        data=csv_data,
        file_name="timeframe_comparison_results.csv",
        mime="text/csv"
    )

def plot_global_results(global_results_df):
    """
    Plot overall results across all symbols.
    """
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
# MAIN UI AND EXECUTION
# ----------------------------------------

def main():
    st.title("📈 Optimal Trend Length Calculator")
    
    st.sidebar.header("🔧 Configuration")
    
    exchange_name = st.sidebar.selectbox(
        "Select Exchange",
        ["binance", "kraken", "bitfinex", "coinbasepro"]
    )
    
    symbols_input = st.sidebar.text_input(
        "Enter Symbols (comma-separated)",
        "BTC/USDT,ETH/USDT,SOL/USDT,AVAX/USDT"
    )
    symbols = [s.strip().upper() for s in symbols_input.split(",")]
    
    # Create tabs for different functionalities
    tab1, tab2 = st.tabs(["Single Timeframe Analysis", "Timeframe Comparison"])
    
    with tab1:
        timeframe = st.sidebar.selectbox(
            "Select Timeframe",
            ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
        )
        
        process_all = st.sidebar.checkbox("Process All Lengths (10-1000)")
        if not process_all:
            length_min, length_max = st.sidebar.slider(
                "Select Length Range",
                min_value=10,
                max_value=300,
                value=(10, 250)
            )
            length_range = range(length_min, length_max + 1)
        else:
            st.sidebar.info("Processing all lengths 10–1000. This may take time.")
            length_range = range(10, 1001)
            length_max = 1000
            
        override_check = st.sidebar.checkbox("Override Minimum Data Requirement?", value=False)
        
        if st.sidebar.button("🚀 Run Single Timeframe Analysis"):
            run_backtest(
                exchange_name,
                symbols,
                timeframe,
                length_range,
                length_max=length_max,
                override_check=override_check
            )
    
    with tab2:
        st.header("4H vs 1D Timeframe Comparison")
        
        comparison_info = """
        This analysis will compare performance metrics between 4H and 1D timeframes for all selected symbols.
        It provides both aggregate statistics and individual symbol comparisons.
        """
        st.info(comparison_info)
        
        if st.button("🔄 Run Timeframe Comparison"):
            length_max = 1000 if process_all else length_max
            results = compare_timeframes(
                exchange_name,
                symbols,
                length_range,
                length_max,
                override_check
            )
            
            if results and all(not df.empty for df in results.values()):
                plot_timeframe_comparison(results)
                
                # Allow downloading comparison results
                for tf in ['4h', '1d']:
                    if tf in results:
                        csv_data = results[tf].to_csv(index=False)
                        st.download_button(
                            label=f"Download {tf} Results CSV",
                            data=csv_data,
                            file_name=f"comparison_{tf}_results.csv",
                            mime="text/csv"
                        )
            else:
                st.error("Unable to generate comparison. Please check the data and try again.")

if __name__ == "__main__":
    main()