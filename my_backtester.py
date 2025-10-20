import backtrader as bt
import operator 
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt

def create_pandasdata_class(custom_columns):
    """
    Dynamically create a custom PandasData subclass
    with user-defined extra lines and params.
    """
    # Default params (OHLCV already handled by Backtrader)
    base_params = [
        ('datetime', None),
        ('open', 'open'),
        ('high', 'high'),
        ('low', 'low'),
        ('close', 'close'),
        ('volume', 'volume'),
    ]

    # Add your custom columns
    for col in custom_columns:
        base_params.append((col, col))

    # Create and return a new subclass dynamically
    return type(
        'DynamicPandasData',             # name of new class
        (bt.feeds.PandasData,),          # parent class
        {
            'lines': tuple(custom_columns),
            'params': tuple(base_params),
        }
    )
    
class DynamicStrategy(bt.Strategy):
    params = (
        ("strategy_rules", None),
    )

    def __init__(self):
        self.rules = self.p.strategy_rules

        # One LineBoolean per side (one rule each)
        self.long_entry_condition = None
        self.long_exit_condition  = None
        self.short_entry_condition = None
        self.short_exit_condition  = None
        self.portfolio_values, self.entries, self.exits = [], [], []
        self.order = None
        self.current_conditions = []
        # Operator mappings
        self.get_op = {
            '>': operator.gt,
            '<': operator.lt,
            '>=': operator.ge,
            '<=': operator.le
        }
        self.bt_logical_op = {
            'AND': bt.And,
            'OR': bt.Or
        }

        # Build conditions for each side
        self.build_conditions()

    def build_conditions(self):
        # Long entry
        if self.rules["LONG"]["ENTRY"]["conditions"]:
            self.long_entry_condition = self.build_rule(
                self.rules["LONG"]["ENTRY"]["conditions"],
                self.rules["LONG"]["ENTRY"]["connectors"]
            )
            print(f'long entry condition: {self.long_entry_condition}')
        # Long exit
        if self.rules["LONG"]["EXIT"]["conditions"]:
            self.long_exit_condition = self.build_rule(
                self.rules["LONG"]["EXIT"]["conditions"],
                self.rules["LONG"]["EXIT"]["connectors"]
            )
            print(f'long exit condition: {self.long_exit_condition}')
        # Short entry
        if self.rules["SHORT"]["ENTRY"]["conditions"]:
            self.short_entry_condition = self.build_rule(
                self.rules["SHORT"]["ENTRY"]["conditions"],
                self.rules["SHORT"]["ENTRY"]["connectors"]
            )
        # Short exit
        if  self.rules["SHORT"]["ENTRY"]["conditions"]:
            
            self.short_exit_condition = self.build_rule(
                self.rules["SHORT"]["EXIT"]["conditions"],
                self.rules["SHORT"]["EXIT"]["connectors"]
            )

    def build_rule(self, conditions, connectors):
        """
        Combine all conditions in a single rule using the connectors
        """
        if not conditions:
            return None

        # First condition
        lhs, op, rhs, rhs_is_num = conditions[0]
        lhs_line = getattr(self.data.lines, lhs)
        rhs_val = rhs if rhs_is_num else getattr(self.data.lines, rhs)
        final_cond = self.get_op[op](lhs_line, rhs_val)

        # Combine remaining conditions using connectors
        for i in range(1, len(conditions)):
            lhs, op, rhs, rhs_is_num = conditions[i]
            lhs_line = getattr(self.data.lines, lhs)
            rhs_val = rhs if rhs_is_num else getattr(self.data.lines, rhs)
            next_cond = self.get_op[op](lhs_line, rhs_val)

            connector = connectors[i-1]  # connectors are one less than conditions
            final_cond = self.bt_logical_op[connector](final_cond, next_cond)
        
        return final_cond

    def log(self, msg):
        print(f'{self.datas[0].datetime.datetime(0)} {msg}')
        
    def notify_order(self, order):
        if order.status == order.Completed:    
            self.entry_price = order.executed.price 
            dt = self.datetime.datetime(0)
            if order.isbuy():
                if self.position.size > 0:
                    self.entries.append((dt, self.data.close[0], "LONG"))
                    self.log(f'LONG PLACED AT {self.entry_price} on {dt}')
                
                else:
                    self.exits.append((dt, self.data.close[0], "SHORT"))
                    self.log(f'EXIT SHORT Placed at {self.entry_price} on {dt}')     
            
            elif order.issell():
                if self.position.size < 0:
                    self.log(f'SHORT PLACED AT {self.entry_price} on {dt}')
                
                else:
                    self.exits.append((dt, self.data.close[0], "LONG"))
                    self.log(f'LONG EXIT AT {self.entry_price} on {dt}')
        
        elif order.status in [order.Rejected, order.Margin, order.Cancelled]:
            self.log(f'Order REJECTED/MARGIN/CANCELLED')
        
        self.order = None
            
    def next(self):
        dt = self.data.datetime.date(0)
        self.portfolio_values.append({
            'datetime': self.datetime.datetime(0),
            'value': self.broker.get_value()
        })
        if self.order:
            return
        # Long entry
        if self.position.size == 0:
            if self.long_entry_condition is not None and self.long_entry_condition[0]:
                self.current_conditions.append((dt, self.data.close[0], self.data.SMA_close_20_1T[0], "ENTRY"))
                self.log(f"{dt} - Long Entry Triggered")
                self.buy(size=1, exectype=bt.Order.Market)
            
            elif self.short_entry_condition is not None and self.short_entry_condition[0]:
                self.log(f"{dt} - Short Entry Triggered")
                self.buy(size=1, exectype=bt.Order.Market)

        if self.position.size > 0 and self.long_exit_condition is not None and  self.long_exit_condition[0]:
            self.current_conditions.append((dt, self.data.close[0], self.data.SMA_close_20_1T[0], "EXIT"))
            self.log(f"{dt} - Long Exit Triggered")
            self.close()

        if self.position.size < 0 and self.short_exit_condition is not None and self.short_exit_condition[0]:
            self.log(f"{dt} - Short Exit Triggered")
            self.close()

                


def backtest(df, strategy_cls, initial_cash, custom_cols, strategy_rules, commission=0.04, slippage=0.1):
    cerebro = bt.Cerebro()
    cerebro.broker.set_slippage_perc(slippage)  # 0.1% slippage
    cerebro.broker.set_coc(True)  # Market orders execute on current bar close

    # cerebro.addstrategy(strategy_cls, adx_mean=df['ADX_14'].mean(), adx_std=df['ADX_14'].std(), diff_mean=(df['DMN_14'] - df['DMP_14']).mean(), diff_std=(df['DMN_14'] - df['DMP_14']).std() )
    # cerebro.addstrategy(strategy_cls, vwap_mean=df_filtered['vwap'].mean(), vwap_std=df_filtered['vwap'].std())
    cerebro.addstrategy(strategy_cls, strategy_rules=strategy_rules)
    # Load data
    DynamicDataClass = create_pandasdata_class(custom_cols)

# Now you can use it like a normal Backtrader data feed
    data = DynamicDataClass(dataname=df)
    cerebro.adddata(data)

    # Broker
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=commission)
    cerebro.broker.set_shortcash(True)  # allow shorts

    # Analyzers
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')

    results = cerebro.run()
    strat = results[0]
    return strat

def get_detailed_metrics(strat):
    trades = strat.analyzers.trades.get_analysis()
    sharpe = strat.analyzers.sharpe.get_analysis()
    drawdown = strat.analyzers.drawdown.get_analysis()
    returns = strat.analyzers.returns.get_analysis()

    total_trades = safe_get(trades, 'total', 'closed')
    won_trades = safe_get(trades, 'won', 'total')
    lost_trades = safe_get(trades, 'lost', 'total')

    win_rate = (won_trades / total_trades * 100) if total_trades else 0
    loss_rate = (lost_trades / total_trades * 100) if total_trades else 0

    avg_win = safe_get(trades, 'won', 'pnl', 'average')
    avg_loss = safe_get(trades, 'lost', 'pnl', 'average')
    max_win = safe_get(trades, 'won', 'pnl', 'max')
    max_loss = safe_get(trades, 'lost', 'pnl', 'max')

    # pnl = []
    # for i in range(min(len(strat.entries), len(strat.exits))):
    #     pnl.append(strat.exits[i][1] - strat.entries[i][1])  # long: exit - entry
    # total_pnl = sum(pnl)

    metrics = {
        'total_trades': total_trades,
        'winning_trades': won_trades,
        'losing_trades': lost_trades,
        'win_rate_%': win_rate,
        'loss_rate_%': loss_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'max_win': max_win,
        'max_loss': max_loss,
        # 'pnl': total_pnl,
        'sharpe_ratio': sharpe.get('sharperatio', 0),
        'max_drawdown_%': safe_get(drawdown, 'max', 'drawdown'),
        'max_drawdown_duration_bars': safe_get(drawdown, 'max', 'len'),
        'cumulative_return_%': returns.get('rtot', 0) * 100,
        'average_return_%': returns.get('ravg', 0) * 100,
        'volatility_%': returns.get('rstd', 0) * 100,
        'calmar_ratio': (returns.get('ravg',0)*100 / safe_get(drawdown,'max','drawdown')) if safe_get(drawdown,'max','drawdown') else 0,
        'profit_factor': (avg_win * won_trades) / abs(avg_loss * lost_trades) if lost_trades > 0 else float('inf'),
        'expectancy_per_trade': (win_rate/100)*avg_win - (loss_rate/100)*abs(avg_loss)
    }

    df_portfolio = pd.DataFrame(strat.portfolio_values)
    return metrics, df_portfolio


def safe_get(d, *keys, default=0):
    """Safely get nested keys from AutoOrderedDict"""
    for key in keys:
        if key in d:
            d = d[key]
        else:
            return default
    return d

import matplotlib.pyplot as plt

def plot_strategy_metrics(metrics):
    figs = []

    # ---- 1. Key performance metrics ----
    key_perf = {
        'Calmar Ratio': metrics.get('calmar_ratio', 0) or 0,
        'Cumulative Return %': metrics.get('cumulative_return_%', 0) or 0,
        'Volatility %': metrics.get('volatility_%', 0) or 0,
        'Average Return %': metrics.get('average_return_%', 0) or 0
    }

    fig1, ax1 = plt.subplots(figsize=(12,5))
    ax1.bar(key_perf.keys(), key_perf.values(), color='skyblue', alpha=0.8)
    ax1.set_title('Key Performance Metrics')
    ax1.set_ylabel('Value')
    ax1.set_xticklabels(key_perf.keys(), rotation=45, ha='right')
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    figs.append(fig1)

    # ---- 2. Winning vs Losing Trades ----
    trades_counts = [metrics.get('win_rate_%', 0), metrics.get('loss_rate_%', 0)]
    trades_labels = ['Winning Trades','Losing Trades']

    fig2, ax2 = plt.subplots(figsize=(6,6))
    ax2.pie(trades_counts, labels=trades_labels, autopct='%1.1f%%', colors=['green','red'], startangle=90)
    ax2.set_title('Winning vs Losing Trades')
    figs.append(fig2)

    # ---- 3. Trade profitability metrics ----
    trade_stats = {
        'Avg Win': metrics.get('avg_win', 0),
        'Avg Loss': metrics.get('avg_loss', 0),
        'Max Win': metrics.get('max_win', 0),
        'Max Loss': metrics.get('max_loss', 0),
        'Profit Factor': metrics.get('profit_factor', 0),
        'Expectancy/Trade': metrics.get('expectancy_per_trade', 0)
    }

    fig3, ax3 = plt.subplots(figsize=(12,5))
    ax3.bar(trade_stats.keys(), trade_stats.values(), color='orange', alpha=0.8)
    ax3.set_title('Trade Profitability Metrics')
    ax3.set_ylabel('PnL / Ratio')
    ax3.set_xticklabels(trade_stats.keys(), rotation=45, ha='right')
    ax3.grid(axis='y', linestyle='--', alpha=0.5)
    figs.append(fig3)

    # ---- 4. Return vs Drawdown ----
    fig4, ax4 = plt.subplots(figsize=(8,5))
    ax4.bar(['Cumulative Return %','Max Drawdown %'],
            [metrics.get('cumulative_return_%', 0), metrics.get('max_drawdown_%', 0)],
            color=['blue','red'], alpha=0.7)
    ax4.set_title('Return vs Drawdown')
    ax4.set_ylabel('%')
    ax4.grid(axis='y', linestyle='--', alpha=0.5)
    figs.append(fig4)

    return figs

def plot_candles_with_entries_and_exits(df, entries=None, exits=None):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import pandas as pd

    df = df.sort_index()
    full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq="1T")
    df = df.reindex(full_idx)
    df[['open','high','low','close']] = df[['open','high','low','close']].ffill()
    # df[['ADX_14','DMP_14','DMN_14']] = df[['ADX_14','DMP_14','DMN_14']].ffill()

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
        row_heights=[0.6, 0.4],
        subplot_titles=('Price with Entries/Exits')
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'],
        increasing_line_color='green', decreasing_line_color='red', name='Price'
    ), row=1, col=1)

    # --- Entries ---
    if entries:
        entry_times = [pd.to_datetime(t) for t, _, _ in entries]
        entry_prices = [p for _, p, _ in entries]
        entry_labels = [lbl for _, _, lbl in entries]
        entry_colors = ['blue' if lbl=='LONG' else 'purple' for lbl in entry_labels]

        # 1) Marker only (always visible)
        fig.add_trace(go.Scatter(
            x=entry_times, y=entry_prices, mode='markers',
            marker=dict(symbol='triangle-up', size=10, color=entry_colors),
            name='Entry Marker'
        ), row=1, col=1)

        # 2) Marker + text (hidden by default)
        fig.add_trace(go.Scatter(
            x=entry_times, y=entry_prices, mode='markers+text',
            marker=dict(symbol='triangle-up', size=10, color=entry_colors),
            text=entry_labels, textposition='top center',
            hoverinfo='text', visible=False,  # hidden initially
            name='Entry Label'
        ), row=1, col=1)

    # --- Exits ---
    if exits:
        exit_times = [pd.to_datetime(t) for t, _, _ in exits]
        exit_prices = [p for _, p, _ in exits]
        exit_labels = [lbl for _, _, lbl in exits]
        exit_colors = ['orange' if lbl=='LONG' else 'red' for lbl in exit_labels]

        # 1) Marker only
        fig.add_trace(go.Scatter(
            x=exit_times, y=exit_prices, mode='markers',
            marker=dict(symbol='triangle-down', size=10, color=exit_colors),
            name='Exit Marker'
        ), row=1, col=1)

        # 2) Marker + text (hidden initially)
        fig.add_trace(go.Scatter(
            x=exit_times, y=exit_prices, mode='markers+text',
            marker=dict(symbol='triangle-down', size=10, color=exit_colors),
            text=exit_labels, textposition='bottom center',
            hoverinfo='text', visible=False,
            name='Exit Label'
        ), row=1, col=1)

    # --- Buttons to toggle labels ---
    
    fig.update_layout(
        title='Candlestick with Entries/Exits',
        xaxis_title='Time', yaxis_title='Price',
        xaxis_rangeslider_visible=False, height=750,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )

    
    # --- Display in Streamlit ---
    return fig
