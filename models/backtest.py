import pandas as pd
import numpy as np
import logging
from datetime import datetime

class BacktestEngine:
    def __init__(self, initial_capital=1e6, logger=None):
        """
        初始化回测引擎
        
        参数:
            initial_capital: 初始资金
            logger: 日志记录器
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position = 0  # 当前持仓数量
        self.trade_log = []  # 交易记录
        self.current_position = 0  # 当前持仓状态：0表示空仓，1表示持仓
        self.logger = logger or logging.getLogger('BacktestEngine')
        self.logger.info(f"初始化回测引擎，初始资金: {initial_capital:,.2f}")
        
    def execute_trade(self, signal, price, signal_time, df=None):
        """
        执行交易
        
        参数:
            signal: 交易信号，'BUY' 或 'SELL'
            price: 交易价格
            signal_time: 信号时间
            df: 数据框架，用于模式验证（可选）
            
        返回:
            bool: 是否执行了交易
        """
        commission_rate = 0.0003  # 手续费率
        min_commission = 5  # 最低手续费
        
        # 增加模式验证（如果提供了数据框架）
        if df is not None and signal_time in df.index:
            current_data = df.loc[signal_time]
            if signal == 'BUY' and hasattr(current_data, 'pattern_buy') and current_data['pattern_buy'] != 1:
                return False
            if signal == 'SELL' and hasattr(current_data, 'pattern_sell') and current_data['pattern_sell'] != 1:
                return False
        
        # 买入信号处理
        if signal == 'BUY':
            # 检查是否已经持仓
            if self.current_position == 1:
                self.logger.info(f"已有持仓，忽略买入信号 - 时间: {signal_time}, 价格: {price:.2f}")
                return False
                
            # 计算可买入数量（使用90%资金）
            available_capital = self.capital * 0.9
            shares = int(available_capital / price)
            
            if shares > 0:
                cost = shares * price
                commission = max(min_commission, cost * commission_rate)
                total_cost = cost + commission
                
                if total_cost <= self.capital:
                    self.capital -= total_cost
                    self.position = shares
                    self.current_position = 1  # 更新持仓状态
                    
                    self.trade_log.append({
                        'datetime': signal_time,
                        'action': 'BUY',
                        'price': price,
                        'shares': shares,
                        'commission': commission,
                        'capital': self.capital,
                        'total_value': self.capital + (shares * price)
                    })
                    self.logger.info(f"执行买入: {signal_time}, 价格: {price:.2f}, 数量: {shares}")
                    return True
                else:
                    self.logger.info(f"资金不足，无法买入 - 时间: {signal_time}, 所需资金: {total_cost:.2f}")
            else:
                self.logger.info(f"计算买入数量为0，无法买入 - 时间: {signal_time}")
                
        # 卖出信号处理
        elif signal == 'SELL':
            # 检查是否有持仓可卖
            if self.current_position == 0:
                self.logger.info(f"无持仓，忽略卖出信号 - 时间: {signal_time}, 价格: {price:.2f}")
                return False
                
            revenue = self.position * price
            commission = max(min_commission, revenue * commission_rate)
            net_revenue = revenue - commission
            
            self.capital += net_revenue
            old_position = self.position
            self.position = 0
            self.current_position = 0  # 更新持仓状态
            
            self.trade_log.append({
                'datetime': signal_time,
                'action': 'SELL',
                'price': price,
                'shares': old_position,
                'commission': commission,
                'capital': self.capital,
                'total_value': self.capital
            })
            self.logger.info(f"执行卖出: {signal_time}, 价格: {price:.2f}, 数量: {old_position}")
            return True
            
        return False

    def get_current_position(self):
        """获取当前持仓状态"""
        return self.current_position

    def get_position_value(self, current_price):
        """计算当前持仓市值"""
        return self.position * current_price if self.position > 0 else 0
    
    def get_statistics(self):
        """计算回测统计数据"""
        if not self.trade_log:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'total_return': 0,
                'max_drawdown': 0
            }
        
        # 计算交易统计
        buy_trades = [t for t in self.trade_log if t['action'] == 'BUY']
        sell_trades = [t for t in self.trade_log if t['action'] == 'SELL']
        
        # 确保交易配对
        paired_trades = min(len(buy_trades), len(sell_trades))
        
        # 计算每笔交易的盈亏
        profits = []
        for i in range(paired_trades):
            buy_price = buy_trades[i]['price']
            sell_price = sell_trades[i]['price']
            shares = buy_trades[i]['shares']
            buy_commission = buy_trades[i]['commission']
            sell_commission = sell_trades[i]['commission']
            
            profit = (sell_price - buy_price) * shares - buy_commission - sell_commission
            profits.append(profit)
        
        # 计算胜率
        winning_trades = sum(1 for p in profits if p > 0)
        win_rate = winning_trades / paired_trades if paired_trades > 0 else 0
        
        # 计算盈亏比
        gains = sum(p for p in profits if p > 0)
        losses = abs(sum(p for p in profits if p < 0))
        profit_factor = gains / losses if losses > 0 else float('inf')
        
        # 计算总回报率
        total_return = (self.capital - self.initial_capital) / self.initial_capital * 100
        
        # 计算最大回撤
        peak = self.initial_capital
        max_drawdown = 0
        
        for trade in self.trade_log:
            total_value = trade['total_value']
            if total_value > peak:
                peak = total_value
            else:
                drawdown = (peak - total_value) / peak * 100
                max_drawdown = max(max_drawdown, drawdown)
        
        return {
            'total_trades': paired_trades,
            'win_rate': win_rate * 100,  # 转换为百分比
            'profit_factor': profit_factor,
            'total_return': total_return,
            'max_drawdown': max_drawdown
        }

class BacktestSystem:
    def __init__(self, df, initial_capital=1e6, logger=None):
        """
        初始化回测系统
        
        参数:
            df: 包含价格和信号的DataFrame
            initial_capital: 初始资金
            logger: 日志记录器
        """
        self.df = df.copy()
        self.initial_capital = initial_capital
        self.engine = BacktestEngine(initial_capital, logger)
        self.logger = logger or logging.getLogger('BacktestSystem')
        
    def run_backtest(self):
        """
        运行回测
        """
        self.logger.info("开始回测...")
        
        # 确保DataFrame有日期索引
        if not isinstance(self.df.index, pd.DatetimeIndex):
            if 'trade_date' in self.df.columns:
                self.df['trade_date'] = pd.to_datetime(self.df['trade_date'])
                self.df.set_index('trade_date', inplace=True)
        
        # 遍历每一行数据
        for i in range(1, len(self.df)):
            current_row = self.df.iloc[i]
            current_date = self.df.index[i]
            
            # 检查是否有交易信号
            if 'signal' in current_row:
                signal = current_row['signal']
                
                if signal == 1 and self.engine.get_current_position() == 0:
                    # 买入信号
                    self.engine.execute_trade('BUY', current_row['close'], current_date)
                    
                elif signal == -1 and self.engine.get_current_position() == 1:
                    # 卖出信号
                    self.engine.execute_trade('SELL', current_row['close'], current_date)
        
        # 如果结束时仍有持仓，执行平仓
        if self.engine.get_current_position() == 1:
            last_price = self.df['close'].iloc[-1]
            last_date = self.df.index[-1]
            self.engine.execute_trade('SELL', last_price, last_date)
        
        self.logger.info("回测完成")
    
    def get_statistics(self):
        """获取回测统计数据"""
        return self.engine.get_statistics()
    
    def plot_results(self):
        """绘制回测结果"""
        # 这个方法将在visualization模块中实现
        pass

def run_backtest_analysis(df):
    """运行回测分析"""
    # 创建回测系统
    backtest = BacktestSystem(df)
    
    # 运行回测
    backtest.run_backtest()
    
    # 打印统计数据
    stats = backtest.get_statistics()
    logger = logging.getLogger('BacktestAnalysis')
    logger.info("\n回测统计:")
    for key, value in stats.items():
        logger.info(f"{key}: {value}")
    
    # 绘制结果
    backtest.plot_results()
    
    return backtest