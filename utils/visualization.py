import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mplfinance.original_flavor import candlestick_ohlc
import logging

class PhysicsFinanceVisualizer:
    def __init__(self, logger=None):
        """初始化可视化工具"""
        self.logger = logger or logging.getLogger('PhysicsFinanceVisualizer')

    def plot_analysis(self, df: pd.DataFrame, backtest_engine):
        """可视化分析结果，包括K线和盈亏曲线"""
        # 创建子图，使用gridspec_kw来设置高度比例
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(15, 20), 
                                                gridspec_kw={'height_ratios': [3, 1, 1, 1, 1.5]})
        fig.suptitle('交易策略分析', fontsize=16)
        
        # 首先计算需要的技术指标
        df = df.copy()  # 创建副本以免修改原始数据
        
        # 计算移动平均线
        df['MA20'] = df['close'].rolling(window=20).mean()
        df['MA60'] = df['close'].rolling(window=60).mean()
        df['VOL_MA120'] = df['volume'].rolling(window=120).mean()
        
        # 计算MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_SIGNAL'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_HIST'] = df['MACD'] - df['MACD_SIGNAL']
        
        # 计算RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 准备K线数据
        ohlc_data = []
        for i in range(len(df)):
            ohlc_data.append([i, df['open'].iloc[i], df['high'].iloc[i], 
                             df['low'].iloc[i], df['close'].iloc[i]])
        
        # 绘制K线图
        candlestick_ohlc(ax1, ohlc_data, width=0.6, colorup='red', colordown='green', alpha=0.8)
        
        # 添加均线
        ax1.plot(range(len(df)), df['MA20'].values, label='MA20', color='blue', alpha=0.7)
        ax1.plot(range(len(df)), df['MA60'].values, label='MA60', color='orange', alpha=0.7)
        
        # 标记买卖点
        for i in range(len(df)):
            if df['signal'].iloc[i] == 1:  # 买入信号
                ax1.scatter(i, df['high'].iloc[i] * 1.02, 
                           marker='^', color='red', s=100)
                ax1.annotate('B', (i, df['high'].iloc[i] * 1.02), 
                            xytext=(0, 5), textcoords='offset points', 
                            ha='center', va='bottom', color='red')
            elif df['signal'].iloc[i] == -1:  # 卖出信号
                ax1.scatter(i, df['low'].iloc[i] * 0.98, 
                           marker='v', color='green', s=100)
                ax1.annotate('S', (i, df['low'].iloc[i] * 0.98), 
                            xytext=(0, -5), textcoords='offset points', 
                            ha='center', va='top', color='green')
        
        # 设置x轴刻度
        date_ticks = range(0, len(df), 20)
        ax1.set_xticks(date_ticks)
        ax1.set_xticklabels([df.index[i].strftime('%Y-%m-%d') for i in date_ticks], 
                            rotation=45)
        
        ax1.set_ylabel('价格', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')
        
        # 绘制成交量
        volume_colors = ['red' if c >= o else 'green' 
                        for o, c in zip(df['open'], df['close'])]
        ax2.bar(range(len(df)), df['volume'], color=volume_colors, alpha=0.7)
        ax2.plot(range(len(df)), df['VOL_MA120'].values, 
                 color='blue', alpha=0.7, label='VOL MA120')
        
        ax2.set_xticks(date_ticks)
        ax2.set_xticklabels([])  # 移除中间子图的日期标签
        ax2.set_ylabel('成交量', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # 绘制MACD
        ax3.plot(range(len(df)), df['MACD'], label='MACD', color='blue', alpha=0.7)
        ax3.plot(range(len(df)), df['MACD_SIGNAL'], label='Signal', color='orange', alpha=0.7)
        ax3.bar(range(len(df)), df['MACD_HIST'], 
                color=['red' if x >= 0 else 'green' for x in df['MACD_HIST']], 
                alpha=0.5)
        
        ax3.set_xticks(date_ticks)
        ax3.set_xticklabels([])
        ax3.set_ylabel('MACD', fontsize=10)
        ax3.legend(loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        # 绘制RSI
        ax4.plot(range(len(df)), df['RSI'], label='RSI', color='purple', alpha=0.7)
        ax4.axhline(y=70, color='red', linestyle='--', alpha=0.3)
        ax4.axhline(y=30, color='green', linestyle='--', alpha=0.3)
        
        ax4.set_xticks(date_ticks)
        ax4.set_xticklabels([df.index[i].strftime('%Y-%m-%d') for i in date_ticks], 
                            rotation=45)
        ax4.set_ylabel('RSI', fontsize=10)
        ax4.set_ylim(0, 100)
        ax4.grid(True, alpha=0.3)
        
        # 绘制盈亏曲线
        self.logger.info("绘制盈亏曲线...")
        equity_curve = self.calculate_equity_curve(df, backtest_engine)
        
        ax5.plot(equity_curve.index, equity_curve['total_value'], 
                 label='账户价值', color='blue', linewidth=1.5)
        ax5.plot(equity_curve.index, equity_curve['drawdown_line'], 
                 label='最高价值', color='green', linestyle='--', alpha=0.5)
        
        # 标记买卖点
        for trade in backtest_engine.trade_log:
            if trade['action'] == 'BUY':
                ax5.scatter(trade['datetime'], trade['total_value'], 
                           marker='^', color='red', s=100)
            elif trade['action'] == 'SELL':
                ax5.scatter(trade['datetime'], trade['total_value'], 
                           marker='v', color='green', s=100)
        
        # 设置x轴刻度
        date_ticks = pd.date_range(start=equity_curve.index[0], 
                                  end=equity_curve.index[-1], 
                                  periods=10)
        ax5.set_xticks(date_ticks)
        ax5.set_xticklabels([d.strftime('%Y-%m-%d') for d in date_ticks], rotation=45)
        
        ax5.set_ylabel('账户价值', fontsize=10)
        ax5.grid(True, alpha=0.3)
        ax5.legend(loc='upper left')
        
        # 调整布局
        plt.tight_layout()
        plt.show()

    def calculate_equity_curve(self, df: pd.DataFrame, backtest_engine) -> pd.DataFrame:
        """计算并返回权益曲线数据"""
        # 创建日期索引的DataFrame
        equity_curve = pd.DataFrame(index=df.index)
        equity_curve['total_value'] = backtest_engine.initial_capital
        
        # 记录每日账户价值
        current_position = 0
        current_cash = backtest_engine.initial_capital
        
        for date in df.index:
            # 更新持仓价值
            if current_position > 0:
                position_value = current_position * df.loc[date, 'close']
                total_value = current_cash + position_value
            else:
                total_value = current_cash
            
            equity_curve.loc[date, 'total_value'] = total_value
            
            # 检查是否有交易发生
            day_trades = [t for t in backtest_engine.trade_log 
                         if t['datetime'] == date]
            
            for trade in day_trades:
                if trade['action'] == 'BUY':
                    current_position = trade['shares']
                    current_cash -= (trade['price'] * trade['shares'] + trade['commission'])
                elif trade['action'] == 'SELL':
                    current_position = 0
                    current_cash += (trade['price'] * trade['shares'] - trade['commission'])
        
        # 计算回撤
        equity_curve['peak'] = equity_curve['total_value'].expanding().max()
        equity_curve['drawdown'] = (equity_curve['total_value'] - equity_curve['peak']) / equity_curve['peak'] * 100
        equity_curve['drawdown_line'] = equity_curve['peak']
        
        # 计算收益率统计
        returns = equity_curve['total_value'].pct_change()
        annual_return = (returns.mean() * 252) * 100
        annual_volatility = (returns.std() * np.sqrt(252)) * 100
        sharpe_ratio = annual_return / annual_volatility if annual_volatility != 0 else 0
        max_drawdown = equity_curve['drawdown'].min()
        
        self.logger.info("\n=== 收益统计 ===")
        self.logger.info(f"年化收益率: {annual_return:.2f}%")
        self.logger.info(f"年化波动率: {annual_volatility:.2f}%")
        self.logger.info(f"夏普比率: {sharpe_ratio:.2f}")
        self.logger.info(f"最大回撤: {max_drawdown:.2f}%")
        
        return equity_curve

    def plot_surge_analysis(self, df: pd.DataFrame, start_idx=None, window_size=100):
        """绘制带有surge检测结果的分析图"""
        if start_idx is None:
            start_idx = len(df) - window_size
        end_idx = min(start_idx + window_size, len(df))
        
        # 创建子图
        fig = plt.figure(figsize=(15, 12))
        gs = GridSpec(4, 1, height_ratios=[2, 1, 1, 1], figure=fig)
        ax1 = fig.add_subplot(gs[0])  # K线图
        ax2 = fig.add_subplot(gs[1])  # Surge得分
        ax3 = fig.add_subplot(gs[2])  # Surge角度
        ax4 = fig.add_subplot(gs[3])  # Surge区间

        plot_data = df.iloc[start_idx:end_idx]
        
        # 绘制K线图
        candlestick_ohlc(ax1, 
                         zip(range(len(plot_data)),
                             plot_data['open'],
                             plot_data['high'],
                             plot_data['low'],
                             plot_data['close']),
                         width=0.6, colorup='red', colordown='green')
        
        # 标记买卖点
        buy_points = plot_data[plot_data['buy_signal'] == 1]
        sell_points = plot_data[plot_data['sell_signal'] == 1]
        
        for idx, row in buy_points.iterrows():
            ax1.scatter(idx - start_idx, row['low'], color='r', marker='^', s=100)
        for idx, row in sell_points.iterrows():
            ax1.scatter(idx - start_idx, row['high'], color='g', marker='v', s=100)
        
        # 绘制Surge得分
        ax2.plot(plot_data['surge_score'], label='Surge得分', color='blue')
        ax2.axhline(y=4, color='r', linestyle='--', label='得分阈值')
        ax2.set_ylabel('得分')
        ax2.legend()
        
        # 绘制Surge角度
        ax3.plot(plot_data['surge_angle'], label='Surge角度', color='purple')
        ax3.axhline(y=45, color='r', linestyle='--', label='角度阈值')
        ax3.set_ylabel('角度')
        ax3.legend()
        
        # 绘制Surge区间标记
        surge_periods = []
        for i in range(len(plot_data)):
            if plot_data['surge_start'].iloc[i] != 0:
                start = plot_data['surge_start'].iloc[i] - start_idx
                end = plot_data['surge_end'].iloc[i] - start_idx
                surge_periods.append((start, end))
        
        for start, end in surge_periods:
            ax4.axv