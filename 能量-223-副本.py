import numpy as np
import pandas as pd
from typing import Tuple, Dict
from dataclasses import dataclass
import talib
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import importlib.util
import sys
import akshare as ak
from datetime import datetime, timedelta
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mplfinance.original_flavor import candlestick_ohlc
import mplfinance as mpf
import akshare as ak
from datetime import datetime
import talib

import py_compile
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
from tqdm import tqdm
import akshare as ak
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import random
import logging
import os
# 生成带时间戳的日志文件名
log_filename = f"logs/trading_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# 创建logs目录
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler()  # 同时输出到控制台
    ]
)
# 为每个类创建独立的logger
physics_logger = logging.getLogger('PhysicsFinanceModel')
backtest_logger = logging.getLogger('BacktestEngine')
pattern_logger = logging.getLogger('MarketPatternAnalyzer')
# 1. 将编译文件路径添加到搜索路径最前面
compiled_dir = r"D:\compiled_files"
if compiled_dir not in sys.path:
    sys.path.insert(0, compiled_dir)
    print(f"已将 {compiled_dir} 添加到搜索路径最前面")

# 2. 打印当前搜索路径顺序
print("\nPython搜索路径顺序:")
for i, path in enumerate(sys.path):
    print(f"{i}: {path}")

# 3. 确保模块未被加载
modules_to_remove = [
    'chan_fs_jiasu',
    'chan_fs_jiasu_compiled',
    'chan_fs_jiasu_compiled_unique'
]

for mod in modules_to_remove:
    if mod in sys.modules:
        del sys.modules[mod]

# 4. 导入编译后的模块
try:
    print("\n正在导入编译后的缠论模块...")
    compiled_file_path = os.path.join(compiled_dir, "chan_fs_jiasu_compiled_unique.pyc")
    
    def load_unique_pyc():
        unique_module_name = "chan_fs_jiasu_compiled_unique"

        # 移除可能存在的同名模
        if unique_module_name in sys.modules:
            del sys.modules[unique_module_name]

        spec = importlib.util.spec_from_file_location(unique_module_name, compiled_file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[unique_module_name] = module
        spec.loader.exec_module(module)

        return module

    module = load_unique_pyc()
    
    # 5. 验证导入的模块路径
    print(f"\n模块文件路径: {module.__file__}")
    
    # 6. 获取所需函数
    cal_fenbi = getattr(module, 'cal_fenbi')
    xd_js = getattr(module, 'xd_js')
    kxian_baohan_js_0 = getattr(module, 'kxian_baohan_js_0')
    fenbi_js = getattr(module, 'fenbi_js')
    repeat_bi_js = getattr(module, 'repeat_bi_js')
    xianDuan_js = getattr(module, 'xianDuan_js')
    Xian_Duan = getattr(module, 'Xian_Duan')
    
    # 7. 验证函数来源
    print("\nXian_Duan函数的来源:", Xian_Duan.__module__)
    print("Xian_Duan函数的文件:", getattr(Xian_Duan, '__code__', None).co_filename 
          if hasattr(Xian_Duan, '__code__') else "Unknown")
    
    # 8. 定义并注入所需的辅助函数
    def LogInfo(*args):
        self.logger.info(*args)
    module.LogInfo = LogInfo
    
    print("\n模块导入成功")
    
except Exception as e:
    print(f"\n导入模块时出错: {str(e)}")
    raise

@dataclass
class MarketEnergy:
    """市场能量状态"""
    ACCUMULATION = "能量积聚"
    RELEASE = "能量释放"
    DISSIPATION = "能量耗散"
    EQUILIBRIUM = "能量平衡"

class PhysicsFinanceModel:
    def __init__(self, 
                 atr_period: int = 14,
                 lookback_period: int = 252,  # 一年交易日
                 energy_threshold: float = 0.1,
                 equilibrium_epsilon: float = 0.005):
        """
        初始化物理-金融模型
        
        参数:
            atr_period: ATR计算周期
            lookback_period: 回溯期间（用于计算ATR分位数）
            energy_threshold: 能量阈值
            equilibrium_epsilon: 平衡状态阈值
        """
        self.atr_period = atr_period
        self.lookback_period = lookback_period
        self.energy_threshold = energy_threshold
        self.equilibrium_epsilon = equilibrium_epsilon
        self.logger = physics_logger
        self.logger.info("初始化 PhysicsFinanceModel")
        
    def calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """计算ATR"""
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=self.atr_period).mean()
        
        return atr
    
    def calculate_elasticity(self, df: pd.DataFrame) -> float:
        """
        计算市场弹性系数k
        k = Σ(ΔP_i * Vol_i) / Σ(Vol_i)
        """
        price_change = df['close'].diff()
        volume = df['volume']
        
        k = (price_change * volume).sum() / volume.sum()
        return abs(k)  # 使用绝对值，因为我们关注的是弹性大小而非方向
    
    def calculate_energy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算能量特征
        """
        # 计算ATR
        df['atr'] = self.calculate_atr(df)
        
        # 计算弹性系数k（20日滚动计算）
        price_change = df['close'].diff().abs()
        df['k'] = price_change.rolling(20).sum() / np.sqrt(df['volume'].rolling(20).mean())
        
        # 计算标准化波动幅度A
        annual_atr = df['atr'].rolling(252).quantile(0.25)  # 1年25%分位数
        df['A'] = df['atr'] / annual_atr 
        
        # 计算市场能量E
        df['energy'] = 0.5 * df['k'] * df['A'] ** 2
        
        # 计算能量导数
        df['energy_derivative'] = df['energy'].diff()
        df['energy_second_derivative'] = df['energy_derivative'].diff()
        
        return df
    
    def calculate_energy_derivative(self, 
                                  energy_series: pd.Series) -> pd.Series:
        """计算能量变化率 dE/dt"""
        return energy_series.diff()
    
    def calculate_energy_second_derivative(self, 
                                         energy_derivative: pd.Series) -> pd.Series:
        """计算能量二阶导数 d²E/dt²"""
        return energy_derivative.diff()
    
    def determine_market_state(self, 
                             energy: float,
                             energy_derivative: float,
                             energy_second_derivative: float,
                             k: float,
                             A: float,
                             volume: float = None,
                             avg_volume: float = None) -> str:
        """判断市场状态"""
        # 极度简化的市场状态判断
        if energy > 0 and energy_derivative > 0:
            return MarketEnergy.ACCUMULATION
        elif energy_derivative < 0:
            return MarketEnergy.DISSIPATION
        else:
            return MarketEnergy.EQUILIBRIUM
    
    def analyze(self, df: pd.DataFrame) -> Dict:
        """
        分析市场状态
        
        参数:
            df: DataFrame，包含OHLCV数据
        返回:
            市场状态分析结果
        """
        # 计算能量特征
        df = self.calculate_energy(df)
        
        # 获取最新状态
        current_state = self.determine_market_state(
            df['energy'].iloc[-1],
            df['energy_derivative'].iloc[-1],
            df['energy_second_derivative'].iloc[-1],
            df['k'].iloc[-1],
            df['A'].iloc[-1],
            df['volume'].iloc[-1],
            df['volume'].rolling(20).mean().iloc[-1]
        )
        
        return {
            'market_state': current_state,
            'current_energy': df['energy'].iloc[-1],
            'energy_derivative': df['energy_derivative'].iloc[-1],
            'elasticity': df['k'].iloc[-1],
            'normalized_atr': df['A'].iloc[-1]
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号，添加过滤条件减少交易频率"""
        df['signal'] = 0  # 0: 无信号, 1: 买入, -1: 卖出
        
        # 计算技术指标
        df['ma20'] = df['close'].rolling(window=20).mean()  # 20日均线
        df['ma60'] = df['close'].rolling(window=60).mean()  # 60日均线
        df['volume_ma20'] = df['volume'].rolling(window=20).mean()  # 20日成交量均线
        
        # 计算波动率
        df['volatility'] = df['close'].rolling(window=20).std()
        
        # 添加趋势过滤
        df['trend'] = np.where(df['ma20'] > df['ma60'], 1, -1)
        
        # 打印调试信息
        self.logger.info("\n=== 能量特征统计 ===")
        self.logger.info(f"能量值范围: {df['energy'].min():.4f} 到 {df['energy'].max():.4f}")
        self.logger.info(f"能量导数范围: {df['energy_derivative'].min():.4f} 到 {df['energy_derivative'].max():.4f}")
        
        # 设置最小交易间隔（至少20个交易日）
        min_trade_interval = 20
        last_trade_idx = -min_trade_interval
        
        for i in range(60, len(df)):  # 从第60天开始，确保有足够的历史数据
            # 检查是否满足最小交易间隔
            if i - last_trade_idx < min_trade_interval:
                continue
            
            # 买入条件：
            # 1. 能量积聚状态
            # 2. 能量值显著为正且在增加
            # 3. 20日均线在60日均线之上（上升趋势）
            # 4. 成交量放大
            # 5. 价格高于20日均线
            if (df['market_state'].iloc[i] == MarketEnergy.ACCUMULATION and 
                df['energy'].iloc[i] > 0.02 and  # 提高能量阈值
                df['energy_derivative'].iloc[i] > 0.001 and  # 要求更强的能量增长
                df['trend'].iloc[i] == 1 and  # 上升趋势
                df['close'].iloc[i] > df['ma20'].iloc[i] and  # 价格在均线上方
                df['volume'].iloc[i] > df['volume_ma20'].iloc[i] * 1.2):  # 成交量显著放大
                
                df.iloc[i, df.columns.get_loc('signal')] = 1
                last_trade_idx = i
            
            # 卖出条件：
            # 1. 能量耗散状态
            # 2. 能量开始下降
            # 3. 价格跌破20日均线
            # 4. 成交量萎缩或暴增（反转信号）
            elif (df['market_state'].iloc[i] == MarketEnergy.DISSIPATION and 
                  df['energy_derivative'].iloc[i] < -0.001 and  # 要求更明显的能量下降
                  df['close'].iloc[i] < df['ma20'].iloc[i] and  # 价格跌破均线
                  (df['volume'].iloc[i] < df['volume_ma20'].iloc[i] * 0.8 or  
                  df['volume'].iloc[i] > df['volume_ma20'].iloc[i] * 1.5)):
                
                df.iloc[i, df.columns.get_loc('signal')] = -1
                last_trade_idx = i
        
        # 打印信号统计
        buy_signals = len(df[df['signal'] == 1])
        sell_signals = len(df[df['signal'] == -1])
        self.logger.info(f"\n=== 交易信号统计 ===")
        self.logger.info(f"买入信号数量: {buy_signals}")
        self.logger.info(f"卖出信号数量: {sell_signals}")
        
        # 如果有信号，打印一些示例
        if buy_signals > 0:
            self.logger.info("\n=== 买入信号示例 ===")
            buy_examples = df[df['signal'] == 1].head()
            self.logger.info(buy_examples[['energy', 'energy_derivative', 'market_state', 'signal']].head())
        
        if sell_signals > 0:
            self.logger.info("\n=== 卖出信号示例 ===")
            sell_examples = df[df['signal'] == -1].head()
            self.logger.info(sell_examples[['energy', 'energy_derivative', 'market_state', 'signal']].head())
        
        return df

def example_usage():
    """使用示例"""
    # 创建模型实例
    model = PhysicsFinanceModel()
    
    # 假设数据格式
    df = pd.DataFrame({
        'open': [...],
        'high': [...],
        'low': [...],
        'close': [...],
        'volume': [...]
    })
    
    # 分析市场状态
    result = model.analyze(df)
    
    self.logger.info(f"市场状态: {result['market_state']}")
    self.logger.info(f"当前能量: {result['current_energy']:.4f}")
    self.logger.info(f"能量变化率: {result['energy_derivative']:.4f}")
    self.logger.info(f"市场弹性: {result['elasticity']:.4f}")
    self.logger.info(f"标准化ATR: {result['normalized_atr']:.4f}")

import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

plt.style.use('default')  # 使用 matplotlib 默认样式
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class PhysicsFinanceVisualizer(PhysicsFinanceModel):
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
        
        # 标记突破点
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

    def plot_surge_analysis(self, start_idx=None, window_size=100):
        """
        绘制带有surge检测结果的分析图
        """
        if start_idx is None:
            start_idx = len(self.df) - window_size
        end_idx = min(start_idx + window_size, len(self.df))
        
        # 创建子图
        fig = plt.figure(figsize=(15, 12))
        gs = GridSpec(4, 1, height_ratios=[2, 1, 1, 1], figure=fig)
        ax1 = fig.add_subplot(gs[0])  # K线图
        ax2 = fig.add_subplot(gs[1])  # Surge得分
        ax3 = fig.add_subplot(gs[2])  # Surge角度
        ax4 = fig.add_subplot(gs[3])  # Surge区间

        plot_data = self.df.iloc[start_idx:end_idx]
        
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
            ax4.axvspan(start, end, alpha=0.3, color='yellow')
        ax4.set_ylabel('Surge区间')
        
        # 设置标题和标签
        ax1.set_title('Surge检测分析')
        ax4.set_xlabel('K线序号')
        
        # 调整布局
        plt.tight_layout()
        return fig

    def calculate_bi_score(self, prev_bi, current_bi):
        """计算单个笔的评分"""
        time_diff = current_bi.name - prev_bi.name
        price_diff = current_bi['high'] - prev_bi['low']
        angle = np.degrees(np.arctan(price_diff/(time_diff*0.1)))
        
        # 获取历史统计量
        history_bi = self.df[(self.df['fenxing_type_last'].isin([1, -1])) &
                           (self.df.index < prev_bi.name)].tail(20)
        
        if len(history_bi) >= 5:
            angles = []
            time_ratios = []
            price_ratios = []
            for i in range(0, len(history_bi)-1, 2):
                p = history_bi.iloc[i]
                c = history_bi.iloc[i+1]
                t_diff = c.name - p.name
                p_diff = c['high'] - p['low']
                atr = self.df['ATR'].iloc[p.name]
                
                angles.append(np.degrees(np.arctan(p_diff/(t_diff*0.1))))
                time_ratios.append(t_diff / self.avg_bi_length(history_bi))
                price_ratios.append(p_diff / atr)
            
            angle_threshold = np.mean(angles) + np.std(angles)
            time_ratio_threshold = np.mean(time_ratios) + np.std(time_ratios)
            price_ratio_threshold = np.mean(price_ratios) + np.std(price_ratios)
        else:
            angle_threshold = 45
            time_ratio_threshold = 1.2
            price_ratio_threshold = 2.0
        
        # 计算评分
        score = 0
        if angle >= angle_threshold:
            score += 2
        elif angle >= angle_threshold * 0.7:
            score += 1
        
        time_ratio = time_diff / self.avg_bi_length(history_bi)
        time_z = (time_ratio - 1) / (np.std(time_ratios) + 1e-5)
        if abs(time_z) <= 0.5:
            score += 1.5
        elif time_z < -1.0:
            score += 0.5
        
        atr = self.df['ATR'].iloc[prev_bi.name]
        price_ratio = price_diff / atr
        price_levels = [
            (price_ratio_threshold * 1.5, 3),
            (price_ratio_threshold, 2),
            (price_ratio_threshold * 0.7, 1)
        ]
        for threshold, points in price_levels:
            if price_ratio >= threshold:
                score += points
                break
        
        return score

class BacktestEngine:
    def __init__(self, initial_capital=1e6):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position = 0  # 当前持仓数量
        self.trades = []
        self.current_position = 0  # 当前持仓状态：0表示空仓，1表示持仓
        self.logger.info(f"初始化回测引擎，初始资金: {initial_capital:,.2f}")
        self.logger = backtest_logger
        self.logger.info(f"初始化回测引擎，初始资金: {initial_capital:,.2f}")
    def execute_trade(self, signal, price, signal_time):
        """
        执行交易
        :param signal: 交易信号，'BUY' 或 'SELL'
        :param price: 交易价格
        :param signal_time: 信号时间
        :return: bool, 是否执行了交易
        """
        commission_rate = 0.0003  # 手续费率
        min_commission = 5  # 最低手续费
        
        # 增加模式验证
        current_data = self.df.loc[self.df.index == signal_time]
        if signal == 'BUY' and current_data['pattern_buy'].iloc[0] != 1:
            return False
        if signal == 'SELL' and current_data['pattern_sell'].iloc[0] != 1:
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
                    
                    self.trades.append({
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
            
            self.trades.append({
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

class MarketPatternAnalyzer:
    CACHE_DIR = "analysis_cache"
    
    def __init__(self, df: pd.DataFrame, code: str):
        self.df = df.copy()
        self.code = code
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.CACHE_DIR = f"analysis_cache"
        self.cache_file = os.path.join(self.CACHE_DIR, f"{self.code}_analysis.csv")
        
        # 创建缓存目录
        os.makedirs(self.CACHE_DIR, exist_ok=True)
        self.pattern_cache = []
        self.logger = pattern_logger
        self.logger.info(f"初始化市场模式分析器，股票代码: {code}")
        self.params = {
            'surge_days': 5,          # 大涨持续时间
            'surge_pct': 0.15,        # 涨幅阈值
            'consolidation_days': 7,  # 最小横盘天数
            'volatility_ratio': 0.3,  # 波动率比率
            'breakout_pct': 0.03,     # 突破幅度
            'bi_min_length': 3,       # 笔的最小长度
            'bi_angle_threshold': 45  # 笔的角度阈值(度)
        }
        # 尝试加载缓存
        if not self._load_cache():
            self.setup_indicators()
            self.setup_analysis()
            self._save_cache()
    def _load_cache(self):
        """加载缓存数据，支持增量更新"""
        if os.path.exists(self.cache_file):
            try:
                # 加载缓存数据
                cache_df = pd.read_csv(self.cache_file)
                #cache_df = pd.read_parquet(self.cache_file)
                #将
                # 比较最新日期
                cache_last_date = pd.to_datetime(cache_df['trade_date'].iloc[-1])
                current_last_date = pd.to_datetime(self.df['trade_date'].iloc[-1])
                
                if cache_last_date == current_last_date:
                    # 日期一致，直接使用缓存
                    self.df = cache_df
                    self.logger.info(f"已加载缓存数据: {self.code}")
                    return True
                elif cache_last_date < current_last_date:
                    # 需要增量更新
                    self.logger.info(f"检测到新数据，进行增量更新: {self.code}")
                    
                # 获取倒数第四个 fenxing_plot 不为0的位置
                fenxing_points = cache_df[cache_df['fenxing_type_last'] != 0]
                if len(fenxing_points) >=4:
                    start_idx = fenxing_points.iloc[-4].name
                    print('start_idx',start_idx)
                    # 获取新数据
                    new_data = self.df[self.df['trade_date'] > cache_df.loc[start_idx, 'trade_date']]
                    
                    # 包含前3天数据以确保计算的连续性
                    calc_data = self.df.loc[new_data.index[0]-4:, :]
                    
                    # 对新数据进行分型计算
                    calc_data = self.setup_indicators_for_data(calc_data)
                    calc_data = calc_data.fillna(0)
                    #calc_data = calc_data[4:]  # 去掉前3天的数据
                    
                    # 获取原有数据
                    old_data =  self.df[ self.df['trade_date'] <=  self.df.loc[start_idx, 'trade_date']]
                    calc_data  = cache_df[cache_df['trade_date'] > cache_df.loc[start_idx, 'trade_date']]
                    
                    # 合并数据
                    self.df = pd.concat([old_data, calc_data], axis=0, ignore_index=True)

                    # 计算ATR
                    
                    tr1 = self.df['high'] - self.df['low']
                    tr2 = abs(self.df['high'] - self.df['close'].shift(1))
                    tr3 = abs(self.df['low'] - self.df['close'].shift(1))
                    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                    self.df['ATR'] = tr.rolling(window=14).mean()
                    
                    # 计算RSI
                    self.df['RSI'] = talib.RSI(df['close'], timeperiod=14)
                    
                    # 计算ADX
                    self.df['ADX'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
                    
                    # 计算成交量均线
                    self.df['volume_ma20'] = df['volume'].rolling(window=20).mean()
                                
                    # 保存更新后的缓存
                    self._save_cache()
                    return True
                else:
                    self.setup_indicators()
                    self.setup_analysis()
                    self._save_cache()
            except Exception as e:

                self.logger.info(f"缓存加载失败: {str(e)}")
                #删除self.cache_file
                os.remove(self.cache_file)
                #删除
        return False

    def setup_indicators_for_data(self, data):
        """对指定数据段进行技术指标计算"""
        df = data.copy()
        df_copy = df[['trade_date', 'open', 'close', 'high', 'low', 'volume']].copy()
        # 计算ATR
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift(1))
        tr3 = abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(window=14).mean()
        
        # 计算RSI
        df['RSI'] = talib.RSI(df['close'], timeperiod=14)
        
        # 计算ADX
        df['ADX'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        
        # 计算成交量均线
        df['volume_ma20'] = df['volume'].rolling(window=20).mean()

        # 计算分型
        after_fenxing = kxian_baohan_js_0(df)
        baohan = after_fenxing.copy()
        bi_data0 = fenbi_js(after_fenxing)
        fenxing_chongfu = bi_data0.copy()
        
        bi_data2 = repeat_bi_js(bi_data0)
        bi_data_1 = xianDuan_js(bi_data2)
        
        # 删除不需要的列
        bi_data_1 = bi_data_1.drop(
            labels=['open', 'high', 'low', 'close', 'volume', 'fenxing_type', 
                    'range_high', 'fenxing_high', 'fenxing_high_less', 
                    'fenxing_low', 'fenxing_low_less'], axis=1)
        
        # 合并数据
        df = pd.merge(df, bi_data_1, how='left', on=['trade_date'])
        df = pd.merge(df, baohan[['trade_date', 'baohan']], 
                            how='left', on=['trade_date'])
        df = pd.merge(df, fenxing_chongfu[['trade_date', 'fenxing_type']], 
                            how='left', on=['trade_date'])
        
        # 添加顶底分型的高低点
        df['Bottom_high'] = df.apply(
            lambda row: row['high'] if row['fenxing_type'] != 0 else 0, axis=1)
        df['Top_low'] = df.apply(
            lambda row: row['low'] if row['fenxing_type'] != 0 else 0, axis=1)
        
        # 填充空值
        df = df.fillna(0)

        # 计算线段
        #df = Xian_Duan(df)
        #
        # 计算ATR
        #self.logger.info('xxx1',df_copy.columns)

        df = df[['trade_date', 'fenxing_type_last', 'fenxing_plot', 
                        'zoushi', 'baohan', 'fenxing_type',
                        'Bottom_high', 'Top_low']]
        df = pd.merge(df_copy, df, how='left', on=['trade_date'])
        
        return df
    '''
    def _load_cache(self):
        """加载缓存数据"""
        if os.path.exists(self.cache_file):
            
            try:
                cache_df = pd.read_parquet(self.cache_file)
                # 验证缓存有效性
                if len(cache_df) == len(self.df) and \
                   cache_df.index.equals(self.df.index):
                    self.df = cache_df
                    self.logger.info(f"已加载缓存数据: {self.code}")
                    return True
            except Exception as e:
                self.logger.info(f"缓存加载失败: {str(e)}")
            return False
    ‘’‘'''
    def _save_cache(self):
        """保存分析结果"""
        try:
            self.df.to_csv(self.cache_file, index=False)
            #self.df.to_parquet(self.cache_file)
            self.logger.info(f"已保存分析缓存: {self.code}")
        except Exception as e:
            self.logger.info(f"缓存保存失败: {str(e)}")

        """设置技术指标"""


    def setup_indicators(self):
        """设置技术指标"""
        # 计算ATR
        tr1 = self.df['high'] - self.df['low']
        tr2 = abs(self.df['high'] - self.df['close'].shift(1))
        tr3 = abs(self.df['low'] - self.df['close'].shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        self.df['ATR'] = tr.rolling(window=14).mean()
        
        # 计算RSI
        self.df['RSI'] = talib.RSI(self.df['close'], timeperiod=14)
        
        # 计算ADX
        self.df['ADX'] = talib.ADX(self.df['high'], self.df['low'], 
                                  self.df['close'], timeperiod=14)
        
        # 计算成交量均线
        self.df['volume_ma20'] = self.df['volume'].rolling(window=20).mean()     

    def setup_analysis(self):
        """设置技术分析，计算笔和线段"""
        try:
            self.logger.info("开始计算笔和线段分析...")
            
            # 确保数据索引正确
            self.df = self.df.reset_index()
            df_copy=self.df.copy()
            self.logger.info("数据预处理完成")
            
            # 使用导入的函数计算笔和线段
            self.logger.info("开始K线包含处理...")
            after_fenxing = kxian_baohan_js_0(self.df)
            baohan = after_fenxing.copy()
            self.logger.info("K线包含处理完成")
            
            bi_data0 = fenbi_js(after_fenxing)
            fenxing_chongfu = bi_data0.copy()
            
            bi_data2 = repeat_bi_js(bi_data0)
            bi_data_1 = xianDuan_js(bi_data2)
            
            # 删除不需要的列
            bi_data_1 = bi_data_1.drop(
                labels=['open', 'high', 'low', 'close', 'volume', 'fenxing_type', 
                       'range_high', 'fenxing_high', 'fenxing_high_less', 
                       'fenxing_low', 'fenxing_low_less'], axis=1)
            
            # 合并数据
            self.df = pd.merge(self.df, bi_data_1, how='left', on=['trade_date'])
            self.df = pd.merge(self.df, baohan[['trade_date', 'baohan']], 
                             how='left', on=['trade_date'])
            self.df = pd.merge(self.df, fenxing_chongfu[['trade_date', 'fenxing_type']], 
                             how='left', on=['trade_date'])
            
            # 添加顶底分型的高低点
            self.df['Bottom_high'] = self.df.apply(
                lambda row: row['high'] if row['fenxing_type'] != 0 else 0, axis=1)
            self.df['Top_low'] = self.df.apply(
                lambda row: row['low'] if row['fenxing_type'] != 0 else 0, axis=1)
            
            # 填充空值
            self.df = self.df.fillna(0)

            # 计算线段
            #self.df = Xian_Duan(self.df)
            #
            # 计算ATR
            #self.logger.info('xxx1',df_copy.columns)
            df_copy = df_copy[['trade_date', 'open', 'close', 'high', 'low', 'volume']].copy()
            self.df = self.df[['trade_date', 'fenxing_type_last', 'fenxing_plot', 
                          'zoushi', 'baohan', 'fenxing_type',
                          'Bottom_high', 'Top_low']]
            self.df = pd.merge(df_copy, self.df, how='left', on=['trade_date'])
            #self.logger.info('xxx2',self.df.columns)
            tr1 = self.df['high'] - self.df['low']
            tr2 = abs(self.df['high'] - self.df['close'].shift(1))
            tr3 = abs(self.df['low'] - self.df['close'].shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            self.df['ATR'] = tr.rolling(window=14).mean()
            
            # 计算RSI
            self.df['RSI'] = talib.RSI(self.df['close'], timeperiod=14)
            
            # 计算ADX
            self.df['ADX'] = talib.ADX(self.df['high'], self.df['low'], 
                                     self.df['close'], timeperiod=14)
            
            # 计算成交量均线
            self.df['volume_ma20'] = self.df['volume'].rolling(window=20).mean()
        # 准备买卖点数据
            self.logger.info(self.df)
   
            self.logger.info("笔和线段分析计算完成")
            try:
                self._save_cache()
                self.logger.info("缓存保存成功")
            except Exception as e:
                self.logger.error(f"缓存保存失败: {str(e)}")
        except Exception as e:
            self.logger.error(f"设置分析时出错: {str(e)}")
            self.logger.error(f"错误类型: {type(e).__name__}")
            self.logger.error(f"错误详情: {e.__dict__}")
            import traceback
            self.logger.error(f"完整堆栈跟踪:\n{traceback.format_exc()}")
            raise

    def avg_bi_length(self, bi_data):
        """计算平均笔长"""
        if len(bi_data) < 2:
            return 10  # 默认10个K线长度
        return (bi_data.index[-1] - bi_data.index[0]) / (len(bi_data)//2)

    def detect_surge(self, start_idx: int) -> dict:
        """基于最近6笔的三维度趋势验证"""
        # 获取最近6个有效笔端点（3组高低点）
        
        self.logger.info('~~~~~~~surge~~~~~~~~%s',self.df.at[start_idx,'trade_date'])
        history_bi = self.df[(self.df['fenxing_type_last'].isin([1, -1])) &
                          (self.df.index <= start_idx)].tail(20)  # 获取最近3组笔
        bi_points=history_bi 
        surge_candidates = []
        surge_bi= []
        if len(history_bi) >= 5:
            angles = []
            time_ratios = []
            price_ratios = []
            for i in range(0, len(history_bi)-1):
                #打印trade_date
                self.logger.info(f'~~11 index:{i} date:{history_bi.iloc[i]["trade_date"]}')
       
                prev = history_bi.iloc[i]
                current = history_bi.iloc[i+1]
                time_diff = current.name - prev.name
                if current['high']>prev['low']:

                    price_diff = current['high'] - prev['low']
                else:
                    price_diff =-current['high']+ prev['low']
                atr = self.df['ATR'].iloc[prev.name]
                angle= np.degrees(np.arctan(price_diff/(time_diff*0.1)))
                angles.append(angle)
                time_ratio=time_diff / self.avg_bi_length(history_bi)
                time_ratios.append(time_ratio)
                price_ratio=price_diff / atr
                price_ratios.append(price_diff / atr)
                self.df.loc[current.name, 'price_diff'] = price_diff
                surge_candidates.append({
                'start': prev.name,
                'end': current.name,
                'bihigh': current['high'],
                'bilow': prev['low'],
                'angle': angle,
                'time_ratio': time_ratio,
                'price_ratio': price_ratio,
    
            })
            # 动态阈值计算（均值+1标准差）
            angle_threshold = np.mean(angles) + 0.5*np.std(angles)
            time_ratio_threshold = np.mean(time_ratios)+ 0.5*np.std(time_ratios)
            price_ratio_threshold = np.mean(price_ratios)+ 0.5* np.std(price_ratios)
        else:  # 数据不足时使用默认值
            angle_threshold = 45
            time_ratio_threshold = 1.2
            price_ratio_threshold = 2.0

        if len(history_bi) < 6:
            return None
        
                # ... existing code ...
        hist_scores = []
        for candidate in surge_candidates:
            angle = candidate['angle']
            time_ratio = candidate['time_ratio']
            price_ratio = candidate['price_ratio']
            
            # 角度得分（最高2分）
            angle_score = (angle / angle_threshold) * 2
            
            # 时间比率得分（最高1分）
            time_score =2* time_ratio 
            
            # 价格比率得分（最高2分）
            price_score = (price_ratio / price_ratio_threshold) * 2
            
            # 计算总分
            total_score = angle_score + time_score + price_score

                        # 将评分存入候选字典
            candidate.update({
                'angle_score': angle_score,
                'time_score': time_score,
                'price_score': price_score,
                'total_score': total_score
            })

            hist_scores.append(total_score)
            self.logger.info(f'~~113~ 角度得分: {angle_score:.2f} 时间得分: {time_score:.2f} 价格得分: {price_score:.2f} | 总分: {total_score:.2f}')
        # ... existing code ...
                    

        score_threshold = np.mean(hist_scores)# + 1.1 * np.std(hist_scores)
        #self.logger.info('~~12-',i,history_bi.iloc[i]['trade_date'],total_score, score_threshold)
        for candidate in surge_candidates:
            total_score= candidate['total_score']
            
            if total_score>= 1.3*score_threshold and self.df.at[candidate['start'],'close']<self.df.at[candidate['end'],'close']:
                print('total_score',total_score,score_threshold,total_score>= score_threshold)
                surge_bi.append({
                    'start':candidate['start'],
                    'end': candidate['end'],
                    'start_date':self.df.at[candidate['start'],'trade_date'],
                    'end_date':self.df.at[candidate['end'],'trade_date'],
                    'now_index':len(self.df),
                    'bihigh':candidate['bihigh'],
                    'bilow': candidate['bilow'],
                    'angle':candidate['angle'],
                    'time_ratio': candidate['time_ratio'],
                    'price_ratio': candidate['price_ratio'],
                    'score': candidate['total_score']
                })
                #start对应的trade_date

                print('ss',self.df.loc[candidate['start'], 'trade_date'],self.df.loc[current.name, 'trade_date'])
                print('ss',self.df.loc[candidate['end'], 'trade_date'])
                self.df.loc[candidate['end'], 'surge_start'] = candidate['start']
                self.df.loc[candidate['end'], 'surge_end'] = candidate['end']
                self.df.loc[candidate['end'], 'surge_score'] = candidate['total_score']

        self.logger.info('~~suege-%s',surge_bi)
        
        # 返回最后一个符合条件的大涨
        return surge_bi[-1] if surge_bi else []
        #return len(surge_candidates[-1]) if surge_candidates else None

    def detect_consolidation(self, surge_end: int,now_index: int) -> dict:
        """基于笔端点检测横盘"""
        consolidation_bi = []
        resistance = self.df['high'].iloc[surge_end]
        support = self.df['low'].iloc[surge_end]
        threshold = 0.05   # 降低阈值到1.5%
        breakout = False
        overlap_count = 0  # 新增：连续重叠K线计数器
        overlap_window = []  # 新增：存储最近4根K线
        # 初始化标记列
        #d打印now_index对应的trade_date
        self.logger.info(f"开始检测横盘区间，起始位置：{self.df.loc[surge_end, 'trade_date']} 当前索引：{self.df.loc[now_index, 'trade_date']}")
        #d打印now_index对应的trade_date
        self.logger.info(f"开始检测横盘区间，起始位置：{surge_end} 当前索引：{now_index}")

        if pd.Timestamp(self.df.at[now_index, 'trade_date']) >= pd.Timestamp('2017-09-27 00:00:00'):
            #zz
            self.logger.info(f"'---consolidation_bi{ consolidation_bi}")
        #找到df的now_index之前surge_end不为0的最后一个值的trade_date
        
        self.df.to_csv('cc.csv')
        if len(self.df.loc[(self.df.index < now_index) & (self.df['surge_end'] != 0), 'trade_date'])>0:
                        # 找到df的now_index之前surge_end不为0的最后一个值的trade_date
            previous_surge_end_trade_date = self.df.loc[(self.df.index < now_index) & (self.df['surge_end']>= 0), 'trade_date'].iloc[-1]
            print('previous_surge_end_trade_date',previous_surge_end_trade_date,surge_end)
            previous_surge_end_trade_date = pd.Timestamp(previous_surge_end_trade_date)
            self.logger.info(f"Previous surge_end trade_date before now_index: {previous_surge_end_trade_date}")


                                       
        # 标记横盘开始
        consolidations = []
        
        is_hp=0
        # 收集后续笔端点
        condition = (
            self.df['fenxing_type'].isin([1, -1]) &  # 分型类型过滤
            (self.df.index >= surge_end-1) &            # 大于起始索引
            (self.df.index < now_index)              # 小于当前索引
        )
        bi_points = self.df[condition].copy()
        continuous_df = self.df[(self.df.index >= surge_end) & 
                        (self.df.index <= now_index)].copy()
        # 滑动窗口检测6根K线中4根重叠
            # 改进1：增加允许的间断参数
        MAX_ALLOWED_GAPS = 1  # 允许最多1根不重叠的K线
        MIN_CONSECUTIVE = 4   # 至少4根有效重叠
        window_size = 6
        min_overlap = 4
        overlap_ranges = []
                # 改进2：动态跟踪最后有效区间
        current_range = {'high': self.df['high'].iloc[surge_end], 
                        'low': self.df['low'].iloc[surge_end]}
        consecutive_count = 0
        gap_count = 0

        for idx in range(surge_end, now_index+1):
            print('idx',idx,self.df.at[idx, 'trade_date'])      
            current_high = self.df.at[idx, 'high']
            current_low = self.df.at[idx, 'low']
            
            # 判断是否与当前区间重叠
            if (current_low < current_range['high']) and (current_high > current_range['low']):
                consecutive_count += 1
                gap_count = 0
                # 更新区间范围
                current_range['high'] = min(current_range['high'], current_high)
                current_range['low'] = max(current_range['low'], current_low)
            else:#不重叠
                if gap_count < MAX_ALLOWED_GAPS:
                    gap_count += 1
                    consecutive_count += 1  # 即使有间隔也计数
                else:
                    # 重置检测
                    current_range = {'high': current_high, 'low': current_low}
                    consecutive_count = 0
                    gap_count = 0
            print('overlap_count',consecutive_count)
            # 满足最小连续条件时标记
            if consecutive_count >= MIN_CONSECUTIVE:
                start_idx = idx - MIN_CONSECUTIVE + 1
                #self.df.loc[start_idx:idx, 'consolidation'] = 1
                #self.df.loc[start_idx:idx, 'consolidation_support'] = current_range['low']
                #self.df.loc[start_idx:idx, 'consolidation_resistance'] = current_range['high']
            '''
        for i in range(len(continuous_df) - window_size + 1):
            print('i',i,window_size)
            window = continuous_df.iloc[i:i+window_size]
            highs = window['high'].values
            lows = window['low'].values
            
            # 计算所有4根组合
            from itertools import combinations
            has_overlap = False
            
            for combo in combinations(range(window_size), 4):
                combo_high = min(highs[list(combo)])
                combo_low = max(lows[list(combo)])
                
                if combo_high > combo_low:
                    # 检查剩余2根是否在区间内
                    remaining = [x for x in range(window_size) if x not in combo]
                    if all((lows[x] <= combo_high) and (highs[x] >= combo_low) for x in remaining):
                        has_overlap = True
                        break
                        
            if has_overlap:
                start_idx = window.index[0]
                end_idx = window.index[-1]
                support = max(lows)
                resistance = min(highs)
                
                # 标记到原始df
                self.df.loc[start_idx:end_idx, 'consolidation'] = 1
                self.df.loc[start_idx:end_idx, 'consolidation_support'] = support
                self.df.loc[start_idx:end_idx, 'consolidation_resistance'] = resistance
                
                overlap_ranges.append({
                    'start': start_idx,
                    'end': end_idx,
                    'support': support,
                    'resistance': resistance
                })
        self.df.to_csv('c1c.csv')


        zzz
        for idx, row in bi_points.iterrows():
            # 更新阻力支撑
            print('---row',self.df.loc[idx, 'trade_date'],idx)
            current_high = row['high']
            current_low = row['low']
                    # 维护最近4根K线的窗口
            overlap_window.append({'high': current_high, 'low': current_low})
        '''
        
        continuous_df.to_csv('c1.csv')
        # 获取笔的DataFrame

        bi_points['bi_vol']=0
        # 找到连续的下降笔（fenxing_type为-1）
        for i in range(len(bi_points) - 1):  # 遍历索引，避免越界
            print('i', i)
            
            current = bi_points.iloc[i + 1]  # 获取当前行
            prev = bi_points.iloc[i]           # 获取前一行
            
            if current['high'] > prev['low']:
                price_diff = current['high'] - prev['low']
            else:
                price_diff = -current['high'] + prev['low']
            
            bi_points.loc[bi_points.index[i], 'bi_vol'] = price_diff  # 使用index更新bi_vol
            print('price_diff', price_diff)  
            #print('bi_points', bi_points)
            print('！！！！！！！！！！！！！！')

    # 选取有效的下降笔
            valid_negative_bi_points = bi_points[(bi_points['fenxing_type'] == -1) & (bi_points['bi_vol'] != 0)]
            num_valid_negative_bi = len(valid_negative_bi_points)
            print('num_valid_negative_bi', num_valid_negative_bi)
  # 检查是否为连续的下降笔
            if num_valid_negative_bi>=2:
                # 计算这两个下降笔的波动率与ATR的比值
                last_bi_vol = valid_negative_bi_points['bi_vol'].iloc[-1]
                last_second_bi_vol = valid_negative_bi_points['bi_vol'].iloc[-2]
                current_atr = bi_points['ATR'].iloc[i]

                print('last_bi_date', valid_negative_bi_points['trade_date'].iloc[-1])
                print('last_second_bi_dae', valid_negative_bi_points['trade_date'].iloc[-2])     

                print('last_bi_vol',last_bi_vol)
                print('last_second_bi_vol',last_second_bi_vol)
                vol_atr_ratio =last_bi_vol/ current_atr
                print('vol_atr_ratio',vol_atr_ratio,last_bi_vol,current_atr)
                
                self.logger.info(f'连续下降笔波动率/ATR比值: {vol_atr_ratio:.2f}')
                print(bi_points['fenxing_type'].iloc[-1])
                print(bi_points['fenxing_plot'].iloc[i],bi_points['fenxing_plot'].iloc[i-3])
                if vol_atr_ratio < 6 and bi_points['fenxing_type'].iloc[-1]==-1 and\
                    bi_points['fenxing_plot'].iloc[i]<bi_points['fenxing_plot'].iloc[i-3]:
                    
                    
                    
                    # 标记横盘
                    print('----------------',i,i-3)
                    print('last_bi_date', valid_negative_bi_points['trade_date'].iloc[-1])
                    print('last_second_bi_dae', valid_negative_bi_points['trade_date'].iloc[-2]) 
                    if i-3<=0:
                        start_idx = bi_points.index[0] 
                    else:
                        start_idx = bi_points.index[i-3] 
                         # 使用index作为start_idx
                    end_idx = bi_points.index[i]      # 使用index作为end_idx
                    print('bi_points',bi_points)
                    #self.logger.info(f'检测到盘整 | 开始日期:{current_bi["trade_date"]} 结束日期:{next_bi["trade_date"]}')
                    self.logger.info(f'波动率/ATR比值: {vol_atr_ratio:.2f} < 阈值: {threshold}')
                    
                    # 标记区间
                    print('start_idx',start_idx,end_idx)
                    self.df.loc[start_idx:end_idx, 'consolidation'] = 1

                    
            self.df.to_csv('c2.csv')
        # 存储开始和结束索引的列表


                                                                                                                                                                                                                                                                                                                                                            
        #如果当前index为1402，则打印相关数据做调试

        #self.df.to_csv('test.csv')
        
        
        return self.df

                        

    def confirm_breakout(self, consolidation: dict) -> bool:
        """验证突破"""
        # 获取突破笔
        breakout_bi = self.df[(self.df['fenxing_type_last'] == 1) & 
                            (self.df.index > consolidation['end'])].head(3)
        
        for idx, row in breakout_bi.iterrows():
            # 突破条件
            if (row['high'] > consolidation['resistance'] * 1.03 and
                row['volume'] > self.df['volume_ma20'].iloc[idx] * 1.2):
                # 验证后续笔不跌破
                next_bi = self.df[(self.df['fenxing_type_last'].isin([1, -1])) & 
                                (self.df.index > idx)].head(1)
                if not next_bi.empty and next_bi['low'].iloc[0] > consolidation['resistance']:
                    return {
                        'breakout_index': idx,
                        'breakout_price': row['high'],
                        'confirm_index': next_bi.index[0]
                    }
        return None

    def generate_signals(self):
        """生成交易信号并记录每个K线的surge检测结果"""
        df = self.df.copy()
        patterns = []  
        self.df['pattern_buy'] = 0
        self.df['pattern_sell'] = 0
        self.df['surge_start'] = 0
        self.df['surge_end'] = 0
        self.df['surge_score'] = 0
        self.df['surge_angle'] = 0
        self.df['buy_signal'] = 0
        self.df['sell_signal'] = 0
        self.df['surge_score'] = 0
        self.df['surge_start'] = 0
        self.df['surge_end'] = 0
        self.df['surge_score'] = 0
        self.df['surge_angle'] = 0
        self.df['consolidation_start'] = 0
        self.df['consolidation_end'] = 0
        self.df['consolidation_support'] = np.nan
        self.df['consolidation_resistance'] = np.nan
        # 遍历每个K线位置
        for i in range(len(df)):
            surge = self.detect_surge(i)
            self.logger.info('-iii-',self.df.loc[i, 'trade_date'],surge)
            if pd.Timestamp(self.df.at[i, 'trade_date']) >= pd.Timestamp('2017-07-18 00:00:00'):
                
                self.logger.info('---consolidation_bi')
            if surge:
                # 检测横盘阶段
                self.logger.info('-',i,surge,surge['end'])
                self.logger.info('----surgex',self.df.loc[i, 'trade_date'])
                self.logger.info('----surgey',self.df.loc[surge['end'], 'trade_date'])
                self.df.loc[i, 'surge_score'] = surge['score']
                consolidations   = self.detect_consolidation(surge['end'],i) or []
                if not isinstance(consolidations, list):
                    self.logger.info(f"警告：在位置{i}检测到无效的横盘类型：{type(consolidations)}")
                    consolidations = []
                for consolidation in consolidations:
                    breakout = self.confirm_breakout(consolidation)  # 确保传入字典
                    # 确认突破
                    #breakout = self.confirm_breakout(consolidation)
                    if breakout:
                        # 生成买入信号
                        self.df.loc[self.df.index[breakout['breakout_index']], 'pattern_buy'] = 1

                        # 生成卖出信号（示例：跌破支撑止损）
                        stop_loss = consolidation['support']
                        for j in range(breakout['breakout_index'], len(self.df)):
                            if self.df['low'].iloc[j] < stop_loss:
                                self.df.loc[self.df.index[j], 'pattern_sell'] = 1
                                
                                
                        patterns.append({
                            'surge': surge,
                            'consolidation': consolidation,
                            'breakout': breakout
                        })
                        #i = j  # 跳转到止损点之后
                        #continue
                    #i = consolidation['end']
               # else:
                    #i = surge['end']
            i += 1

        # 合并信号
        #
        if self.df['pattern_buy'] is not None:
            self.df['buy_signal'] = self.df['pattern_buy']
        if self.df['pattern_sell'] is not None:
            self.df['sell_signal'] = self.df['pattern_sell']
        
        return self.df

    def plot_bi_pattern(self, pattern):
        """绘制笔结构模式"""
        plt.figure(figsize=(12,6))
        
        # 绘制K线
        plot_candles(self.df.iloc[pattern['surge']['start']-5:pattern['breakout']['confirm_index']+5])        
        # 标注笔端点
        bi_points = self.df[self.df['fenxing_type_last'].isin([1,-1])]
        plt.scatter(bi_points.index, bi_points['high'], c='red', marker='^', s=100)
        plt.scatter(bi_points.index, bi_points['low'], c='green', marker='v', s=100)
        
        # 绘制趋势线
        plt.plot([pattern['surge']['start'], pattern['surge']['end']], 
                 [pattern['surge']['low'], pattern['surge']['high']], 'b--')
        
        # 绘制横盘区间
        plt.axhline(y=pattern['consolidation']['resistance'], color='purple', linestyle='--')
        plt.axhline(y=pattern['consolidation']['support'], color='purple', linestyle='--')
    def plot_surge_analysis(self, start_idx=None, window_size=100):
        """
        绘制带有surge检测结果的分析图
        """
        if start_idx is None:
            start_idx = len(self.df) - window_size
        end_idx = min(start_idx + window_size, len(self.df))
        
        # 创建子图
        fig = plt.figure(figsize=(15, 12))
        gs = GridSpec(4, 1, height_ratios=[2, 1, 1, 1], figure=fig)
        ax1 = fig.add_subplot(gs[0])  # K线图
        ax2 = fig.add_subplot(gs[1])  # Surge得分
        ax3 = fig.add_subplot(gs[2])  # Surge角度
        ax4 = fig.add_subplot(gs[3])  # Surge区间

        plot_data = self.df.iloc[start_idx:end_idx]
        
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
        consolidation_ranges = self.df[(self.df['consolidation'] == 1) & 
                                      (self.df.index >= start_idx) & 
                                      (self.df.index <= end_idx)]
        
        # 绘制支撑阻力线
        for idx, row in consolidation_ranges.iterrows():
            ax1.axhline(y=row['consolidation_support'], color='gray', alpha=0.3, linestyle='--')
            ax1.axhline(y=row['consolidation_resistance'], color='gray', alpha=0.3, linestyle='--')
        
        # 高亮横盘区间
        for _, group in consolidation_ranges.groupby((consolidation_ranges.index.to_series().diff() != 1).cumsum()):
            start = group.index[0]
            end = group.index[-1]
            ax1.axvspan(start, end, alpha=0.1, color='blue')
        
        # 设置标题和标签
        ax1.set_title('Surge检测分析')
        ax4.set_xlabel('K线序号')
        
        # 调整布局
        plt.tight_layout()
        return fig
def get_stock_data(stock_code='000300', start_date='20210101'):
    """获取股票数据"""
    try:
        end_date = datetime.now().strftime('%Y%m%d')
        if stock_code == '000300':
            df = ak.stock_zh_index_daily_em(symbol='sz399300')
            # 重命名列
            self.logger.info('df',df)
            df = df.rename(columns={
                'date': 'trade_date',

            })
        else:
                    # 转换股票代码格式
            if '.SZ' in stock_code:
                symbol = stock_code.replace('.SZ', '')
            elif '.SH' in stock_code:
                symbol = stock_code.replace('.SH', '')
            else:
                symbol = stock_code
                
            self.logger.info(f"正在获取 {symbol} 的数据...")
            # 普通股票的处理逻辑保持不变
            df = ak.stock_zh_a_hist(symbol=stock_code, 
                                  start_date=start_date,
                                  end_date=datetime.now().strftime('%Y%m%d'),
                                  adjust="qfq")
            df = df.rename(columns={
                '日期': 'trade_date',
                '开盘': 'open',
                '最高': 'high',
                '最低': 'low',
                '收盘': 'close',
                '成交量': 'volume'
            })

        

        self.logger.info('df',df)
 
        # 确保数据类型正确
        numeric_columns = ['trade_date','open', 'high', 'low', 'close', 'volume']
        #df[numeric_columns] = df[numeric_columns].astype(float)
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        self.logger.info(f"成功获取数据，共{len(df)}条记录")
        return df
        
    except Exception as e:
        self.logger.info(f"获取数据时出错: {str(e)}")
        raise

class BacktestSystem:
    def __init__(self, df, initial_capital=1000000):
        """初始化回测系统"""
        self.df = df.copy()
        self.initial_capital = initial_capital
        self.positions = pd.DataFrame(index=df.index)
        self.trades = []
        self.current_position = 0
        
    def run_backtest(self):
        """执行回测"""
        self.logger.info("开始回测...")
        
        # 初始化资金和持仓
        capital = self.initial_capital
        position = 0
        self.positions['capital'] = 0
        self.positions['position'] = 0
        self.positions['equity'] = capital
        
        for i in range(len(self.df)):
            date = self.df.index[i]
            row = self.df.iloc[i]
            
            # 获取当前价格
            current_price = row['close']
            
            # 检查买入信号
            if row.get('buy_signal', 0) == 1 and position == 0:
                # 计算可买入数量（假设使用90%资金）
                shares = int((capital * 0.9) / current_price)
                cost = shares * current_price
                
                if cost <= capital:
                    position = shares
                    capital -= cost
                    self.trades.append({
                        'date': date,  # 直接使用索引日期
                        'type': 'buy',
                        'price': current_price,
                        'shares': shares,
                        'cost': cost
                    })
                    self.logger.info(f"买入: 日期={date}, 价格={current_price:.2f}, 数量={shares}")
            
            # 检查卖出信号
            elif row.get('sell_signal', 0) == 1 and position > 0:
                # 全部卖出
                revenue = position * current_price
                capital += revenue
                self.trades.append({
                    'date': date,  # 直接使用索引日期
                    'type': 'sell',
                    'price': current_price,
                    'shares': position,
                    'revenue': revenue
                })
                self.logger.info(f"卖出: 日期={date}, 价格={current_price:.2f}, 数量: {position}")
                position = 0
            
            # 更新持仓信息
            self.positions.loc[date, 'capital'] = capital
            self.positions.loc[date, 'position'] = position
            self.positions.loc[date, 'equity'] = capital + (position * current_price)
    
    def plot_results(self):
        """绘制回测结果"""
        import mplfinance as mpf
        from matplotlib.font_manager import FontProperties
            # 准备附加图表
        apds = []
        
        # 计算实际需要的面板数量
        num_panels = 2  # 默认有主图和成交量
        
            # 设置中文字体
        try:
            font = FontProperties(fname=r"C:\Windows\Fonts\SimHei.ttf")
            plt.rcParams['font.family'] = font.get_name()
        except:
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建绘图数据
        df_plot = self.df.copy()
        #导出csv


        self.df.to_csv('dfplot.csv')                                 
        # 打印调试信息
        self.logger.info("DataFrame columns:", df_plot.columns)
        self.logger.info("DataFrame index type:", type(df_plot.index))
        
        # 确保日期索引是 DatetimeIndex 类型
        if 'trade_date' in df_plot.columns:
            # 确保 trade_date 列是 datetime 类型
            df_plot['trade_date'] = pd.to_datetime(df_plot['trade_date'])
            df_plot.set_index('trade_date', inplace=True)
        else:
            # 如果已经是索引，确保是 datetime 类型
            df_plot.index = pd.to_datetime(df_plot.index)
        
        # 再次打印调试信息
        self.logger.info("After conversion - index type:", type(df_plot.index))
        self.logger.info("Index sample:", df_plot.index[:5])
        
        # 确保数据列名符合 mplfinance 要求
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df_plot.columns for col in required_columns):
            self.logger.info("Missing required columns. Current columns:", df_plot.columns)
            return None
                    # 添加Surge分析指标作为一个面板
        if 'surge_score' in df_plot.columns:
            # 将score归一化到0-100区间以便于显示
            normalized_score = df_plot['surge_score'] * 20  # 假设score最大值约为5
            apds.append(mpf.make_addplot(normalized_score, panel=3, 
                                        color='blue', title='Surge Score'))
            # 添加阈值线
            threshold_line = pd.Series(80, index=df_plot.index)  # 4*20=80
            apds.append(mpf.make_addplot(threshold_line, panel=3,
                                        color='r', linestyle='--'))
        
            num_panels += 1
        # 确保有交易记录·
        if not self.trades:
            self.logger.info("没有交易记录，仅显示K线图")
            # 继续绘制K线图，但不添加交易相关的标记
            # 修正后的样式设置
            mc = mpf.make_marketcolors(
                up='red', 
                down='green',
                edge='inherit',
                wick='inherit',
                volume={'up': 'red', 'down': 'green'},  # 正确设置成交量颜色
                ohlc='i'
            )
            
            s = mpf.make_mpf_style(
                base_mpl_style='seaborn',
                marketcolors=mc,
                gridstyle=':',
                y_on_right=False
            )
            


        
        # 以下是原有的交易记录处理代码
        trades_df = pd.DataFrame(self.trades)
        
        # 添加买卖点
        df_plot['buy_signal'] = 0
        df_plot['sell_signal'] = 0
        
        # 填充买卖信号
        for trade in self.trades:
            try:
                # 获取交易日期
                if isinstance(trade['date'], (int, float)):
                    date = df_plot.index[int(trade['date'])]
                elif isinstance(trade['date'], str):
                    date = pd.to_datetime(trade['date'])
                else:
                    date = trade['date']
                
                # 设置买卖点
                if trade['type'] == 'buy':
                    df_plot.loc[date, 'buy_signals'] = df_plot.loc[date, 'low'] * 0.995
                else:
                    df_plot.loc[date, 'sell_signals'] = df_plot.loc[date, 'high'] * 1.005
                
            except Exception as e:
                self.logger.info(f"处理交易记录时出错: {e}, 交易记录: {trade}")
                continue
        
        # 准备附加图表

        
        # 添加均线
        df_plot['MA20'] = df_plot['close'].rolling(window=20).mean()
        df_plot['MA60'] = df_plot['close'].rolling(window=60).mean()
        apds.append(mpf.make_addplot(df_plot['MA20'], color='blue', width=0.7))
        apds.append(mpf.make_addplot(df_plot['MA60'], color='orange', width=0.7))
                # 填充买卖信号
        for trade in self.trades:
            try:
                date = trade['date']

                #如果有buy_signals列，则添加买入标记

                if 'buy_signals' in df_plot.columns:
                    apds.append(mpf.make_addplot(df_plot['buy_signals'], type='scatter',
                            markersize=100, marker='^', color='red'))
                if 'sell_signals' in df_plot.columns:
                    apds.append(mpf.make_addplot(df_plot['sell_signals'], type='scatter',
                                   markersize=100, marker='v', color='lime'))
            except KeyError as e:
                self.logger.info(f"无效日期: {date}, 错误: {str(e)}")
                continue
        
        # 添加买卖点标记


        
        df_plot.to_csv('dfplot1.csv')
        # 标记surge笔
        # 标记surge笔
        # 标记surge笔
        surge_lines = set()  # 使用集合来存储唯一的笔
        
        # 收集所有需要连接的线段
        aline_segments = []  # 存储线段数据
        
        for i in range(len(df_plot)):
            if df_plot['surge_start'].iloc[i] != 0:
                start_idx = int(df_plot['surge_start'].iloc[i])
                end_idx = int(df_plot['surge_end'].iloc[i])
                
                # 将笔的起止点转换为元组，用于去重
                surge_line = (start_idx, end_idx)
                if surge_line not in surge_lines:
                    surge_lines.add(surge_line)
                    # 收集线段坐标
                    start_point = (df_plot.index[start_idx], df_plot['close'].iloc[start_idx])
                    end_point = (df_plot.index[end_idx], df_plot['close'].iloc[end_idx])
                    aline_segments.append([start_point, end_point])


        # 调试信息
        self.logger.info("\n交易记录:")
        for trade in self.trades:
            self.logger.info(trade)
        
        # 修改资金曲线计算部分
        equity_data = []
        current_equity = self.initial_capital
        current_position = 0
        
        # 按日期排序交易记录
        sorted_trades = sorted(self.trades,      key=lambda x: x['date'])
        
        for date in df_plot.index:
            # 获取当日交易
            day_trades = [t for t in sorted_trades if (
                isinstance(t['date'], (int, float)) and df_plot.index[int(t['date'])] == date
            ) or (
                isinstance(t['date'], (str, pd.Timestamp)) and pd.to_datetime(t['date']) == date
            )]
            
            # 处理当日交易
            for trade in day_trades:
                if trade['type'] == 'buy':
                    current_position = trade['shares']
                    current_equity -= trade['cost']
                elif trade['type'] == 'sell':
                    current_position = 0
                    current_equity += trade['revenue']
            
            # 计算当前持仓的市值
            if current_position > 0:
                position_value = current_position * df_plot.loc[date, 'close']
            else:
                position_value = 0
            
            # 计算当日总资产
            total_equity = current_equity + position_value
            equity_data.append(total_equity)
        
        # 创建资金曲线Series
        equity_series = pd.Series(equity_data, index=df_plot.index)
        
        # 打印资金曲线统计
        self.logger.info("\n资金曲线统计:")
        self.logger.info(f"起始值: {equity_series.iloc[0]:,.2f}")
        self.logger.info(f"结束值: {equity_series.iloc[-1]:,.2f}")
        self.logger.info(f"最大值: {equity_series.max():,.2f}")
        self.logger.info(f"最小值: {equity_series.min():,.2f}")
        self.logger.info(f"变化范围: {equity_series.max() - equity_series.min():,.2f}")
        
        # 添加资金曲线到图表
    # 添加资金曲线
        if len(equity_series) > 0:  # 修改这里
            apds.append(mpf.make_addplot(equity_series, panel=2, 
                                    color='b', title='Portfolio Value'))
            num_panels += 1
        # 设置样式
        mc = mpf.make_marketcolors(up='red', down='green',
                                edge='inherit',
                                wick='inherit',
                                volume='inherit')
        s = mpf.make_mpf_style(marketcolors=mc, gridstyle=':')
        # 绘图参数
        panel_ratios = [2, 0.5] + [1] * (num_panels - 2) if num_panels > 2 else [2, 0.5]
        kwargs = dict(
                type='candle',
                volume=True,
                figsize=(15, 12),
                style=s,
                addplot=apds if apds else None,
                            alines=dict(alines=aline_segments, colors=['red'], linewidths=2, alpha=0.7),  # 添加到这里
                        volume_panel=1,  # 指定成交量面板
                panel_ratios=panel_ratios,
                title='Market Data'
            )
            
        self.logger.info(f"Number of addplots: {len(apds)}")  # 调试信息
        fig, axes = mpf.plot(df_plot, **kwargs, returnfig=True)
        self.logger.info(f"Number of axes: {len(axes)}")  # 调试信息
                
        # 如果有Surge面板，添加标签
    # 如果有Surge面板，添加标签
        if 'surge_score' in df_plot.columns:
            axes[-1].set_ylabel('Surge Score')
            axes[-1].legend(['Score', 'Threshold'])
        #return fig, axes

                # 绘制横盘区间

        # 获取所有横盘区间
        # 添加调试输出
        self.logger.info(f"总横盘K线数量: {len(self.df[self.df['consolidation'] == 1])}")
        
        consolidation_ranges = self.df[self.df['consolidation'] == 1]
        if not consolidation_ranges.empty:
            self.logger.info("前5个横盘K线数据:")
            self.logger.info(consolidation_ranges[['consolidation_support', 'consolidation_resistance']].head())
        else:
            self.logger.info("警告：没有检测到横盘数据")
            return
        
        
        # 合并连续区间
        consolidation_groups = []
        if not consolidation_ranges.empty:
            # 生成连续区间分组
            group_key = (consolidation_ranges.index.to_series().diff() != 1).cumsum()
            
            # 遍历每个独立区间
            for _, group in consolidation_ranges.groupby(group_key):
                start = group.index[0]
                end = group.index[-1]
                y_high = group['consolidation_resistance'].max()
                y_low = group['consolidation_support'].min()
                consolidation_groups.append( (start, end, y_low, y_high) )
        
        # 在主K线图上绘制所有横盘区间
        if consolidation_groups:
            # 获取主K线图的ax对象
            ax = axes[0] if isinstance(axes, (list, np.ndarray)) else axes
            
            # 检查索引类型（处理日期索引和整数索引）
            is_datetime = isinstance(self.df.index, pd.DatetimeIndex)
            
            for start, end, y_low, y_high in consolidation_groups:
                if start > self.df.index[-1] or end < self.df.index[0]:
                    self.logger.info(f"区间 {start}-{end} 超出显示范围")
                    continue
                # 转换日期索引为数值坐标（适用于mplfinance）
                if is_datetime:
                    x_start = mdates.date2num(start)
                    x_end = mdates.date2num(end)
                else:
                    x_start = start
                    x_end = end
                    
                # 绘制半透明矩形
                rect = plt.Rectangle(
                    (x_start, y_low),
                    x_end - x_start,
                    y_high - y_low,
                    facecolor='skyblue',
                    alpha=0.3,
                    edgecolor='navy',
                    linewidth=0.8,
                    zorder=3  # 确保显示在K线上方
                )
                ax.add_patch(rect)
                self.logger.info(f"已绘制区间：{start}~{end} 支撑{y_low:.2f}-阻力{y_high:.2f}")
                self.logger.info(f"主坐标轴类型：{type(ax)}")
                ax.plot([0], [0], marker='x', color='red')  # 测试能否绘制元素
                self.logger.info(f"日期转换示例：{start} → {x_start}")
                self.logger.info(f"价格范围：{y_low}-{y_high} vs K线范围：{ax.get_ylim()}")
        

        plt.show()
        return fig, axes

    def get_statistics(self):
        """计算回测统计数据"""
        initial_equity = self.initial_capital
        final_equity = self.positions['equity'].iloc[-1]
        returns = (final_equity - initial_equity) / initial_equity
        
        # 计算最大回撤
        equity_curve = self.positions['equity']
        running_max = equity_curve.expanding().max()
        drawdowns = (equity_curve - running_max) / running_max
        max_drawdown = drawdowns.min()
        
        # 计算胜率
        if self.trades:
            profits = [t.get('revenue', 0) - t.get('cost', 0) for t in self.trades if 'revenue' in t]
            win_trades = len([p for p in profits if p > 0])
            win_rate = win_trades / len(profits) if profits else 0
        else:
            win_rate = 0
        
        stats = {
            '初始资金': f"{initial_equity:,.2f}",
            '最终权益': f"{final_equity:,.2f}",
            '总收益率': f"{returns:.2%}",
            '最大回撤': f"{max_drawdown:.2%}",
            '交易次数': len(self.trades),
            '胜率': f"{win_rate:.2%}"
        }
        
        return stats

def run_backtest_analysis(df):
    """运行回测分析"""
    # 创建回测系统
    backtest = BacktestSystem(df)
    
    # 运行回测
    backtest.run_backtest()
    
    # 打印统计数据
    stats = backtest.get_statistics()
    self.logger.info("\n回测统计:")
    for key, value in stats.items():
        self.logger.info(f"{key}: {value}")
    
    # 绘制结果
    backtest.plot_results()
    
    return backtest

def main():
    try:
        # 获取沪深300数据
       # df = get_stock_data('000300')
        df = get_stock_data('300718')
        
        # 初始化分析器
        analyzer = MarketPatternAnalyzer(df)
        
        # 生成交易信号
        df_with_signals = analyzer.generate_signals()
        
        # 运行回测
        backtest = run_backtest_analysis(df_with_signals)
        
        # 绘制最近100根K线的分析图
        #analyzer.plot_surge_analysis(window_size=100)
        plt.show()

        # 绘制指定位置的分析图
        #analyzer.plot_surge_analysis(start_idx=1000, window_size=100)
        #plt.show()
        
    except Exception as e:
        self.logger.info(f"运行过程中出错: {str(e)}")
        raise
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
from tqdm import tqdm
import akshare as ak
class PatternScanner:
    def __init__(self):
        self.data_dir = "./data"
        self.chart_dir = "./charts"
        self._prepare_directories()
        
    
    def _prepare_directories(self):
        """创建必要的存储目录"""
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.chart_dir, exist_ok=True)
    
    def get_all_stocks(self):
        """获取沪深股票列表"""
        try:
            # 获取深圳证券交易所的股票列表
            sz_stock = ak.stock_info_sz_name_code()
            # 获取上海证券交易所的股票列表
            sh_stock = ak.stock_info_sh_name_code()
            
            # 打印列名以确认它们是什么
            print("深圳证券交易所股票列表的列名:", sz_stock.columns.tolist())
            print("上海证券交易所股票列表的列名:", sh_stock.columns.tolist())

            # 根据实际列名进行重命名（假设列名为 'code' 和 'name'）
            # 根据实际列名进行重命名
            sz_stock_renamed = sz_stock.rename(columns={'A股代码': 'code', 'A股简称': 'name'})
            sh_stock_renamed = sh_stock.rename(columns={'证券代码': 'code', '证券简称': 'name'})

            print(len(sz_stock_renamed), len(sh_stock_renamed))

            # 合并两个数据框并去重
            df = pd.concat([sz_stock_renamed, sh_stock_renamed])[['code', 'name']].drop_duplicates()

            # 过滤掉科创板（68开头）、创业板（30开头）和北交所（8开头）
            #df = df[~df['code'].str.startswith(('68', '6'))]
            print(df,len(df))
            
            return df
        except Exception as e:
            print(f"获取股票列表时出错: {str(e)}")
            raise
    def get_stock_data(stock_code='000300', start_date='20210101'):
        """获取股票数据"""
        try:
            
            end_date = datetime.now().strftime('%Y%m%d')
            if stock_code == '000300':
                df = ak.stock_zh_index_daily_em(symbol='sz399300')
                # 重命名列
                #self.logger.info('df',df)
                df = df.rename(columns={
                    'date': 'trade_date',

                })
            else:
                        # 转换股票代码格式
                if '.SZ' in stock_code:
                    symbol = stock_code.replace('.SZ', '')
                elif '.SH' in stock_code:
                    symbol = stock_code.replace('.SH', '')
                else:
                    symbol = stock_code
                    
                print(f"正在获取 {symbol} 的数据...")
                # 普通股票的处理逻辑保持不变
                df = ak.stock_zh_a_hist(symbol=stock_code, 
                                    start_date=start_date,
                                    end_date=datetime.now().strftime('%Y%m%d'),
                                    adjust="qfq")
                df = df.rename(columns={
                    '日期': 'trade_date',
                    '开盘': 'open',
                    '最高': 'high',
                    '最低': 'low',
                    '收盘': 'close',
                    '成交量': 'volume'
                })

            

            print('df',df)
    
            # 确保数据类型正确
            numeric_columns = ['trade_date','open', 'high', 'low', 'close', 'volume']
            #df[numeric_columns] = df[numeric_columns].astype(float)
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            self.logger.info(f"成功获取数据，共{len(df)}条记录")
            return df
            
        except Exception as e:
            self.logger.info(f"获取数据时出错: {str(e)}")
            raise
    def download_single_stock(self, stock_code, max_retries=3):
        """带重试机制的数据下载"""
        start_date='20210101'
        file_path = os.path.join(self.data_dir, f"{stock_code}.csv")
        if os.path.exists(file_path):
            return
        base_wait =5  # 基础等待时间15秒
        for attempt in range(max_retries):
            try:
                wait_time = base_wait + (attempt * 10)  # 递增等待时间
                time.sleep(wait_time)
                if stock_code == '000300':
                    df = ak.stock_zh_index_daily_em(symbol='sz399300')
                    # 重命名列
                    #self.logger.info('df',df)
                    df = df.rename(columns={
                        'date': 'trade_date',

                    })
                else:
                            # 转换股票代码格式
                    if '.SZ' in stock_code:
                        symbol = stock_code.replace('.SZ', '')
                    elif '.SH' in stock_code:
                        symbol = stock_code.replace('.SH', '')
                    else:
                        symbol = stock_code
                try:
                    stock_info = ak.stock_individual_info_em(symbol=stock_code)
                    print(stock_info)
                    if stock_info.empty:
                        print(f"股票 {stock_code} 代码无效，跳过下载")
                        return
                except:
                    print(f"股票 {stock_code} 代码1无效，跳过下载")
                    pass  # 即使获取信息失败也继续尝试下载数据
                df = ak.stock_zh_a_hist(symbol=stock_code, 
                                    start_date=start_date,
                                    end_date=datetime.now().strftime('%Y%m%d'),
                                    adjust="qfq")
                print('df',df)
                if len(df) < 100:
                    return

                # 成功获取数据后保存
                df['trade_date'] = pd.to_datetime(df['日期'])
                df.set_index('trade_date', inplace=True)
                column_mapping = {
                    '日期': 'date',
                    '开盘': 'open',
                    '最高': 'high',
                    '最低': 'low',
                    '收盘': 'close',
                    '成交量': 'volume',
                    '成交额': 'amount',
                    '振幅': 'amplitude',
                    '涨跌幅': 'pct_change',
                    '换手率': 'turnover_rate'
                }
                
                # 执行列名转换（仅转换存在的列）
                existing_cols = [col for col in column_mapping.keys() if col in df.columns]
                df = df.rename(columns={cn: en for cn, en in column_mapping.items() if cn in existing_cols})
                df[['date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'amplitude', 'pct_change', 'turnover_rate']].to_csv(file_path)
                #df[['日期','开盘', '最高', '最低', '收盘', '成交量','成交额','振幅','涨跌幅','换手率']].to_csv(file_path)
                
                # 动态调整延迟：失败次数越多等待越长
                sleep_time = random.uniform(0.3 + attempt*0.2, 0.5 + attempt*0.3)
                #time.sleep(sleep_time)
                return
                
            except Exception as e:
                print(f"第{attempt+1}次尝试下载{stock_code}失败: {str(e)}")
                error_type = type(e).__name__
                error_msg = str(e)
            
                # 更详细的错误信息打印
                print(f"第{attempt+1}次尝试下载{stock_code}失败:")
                print(f"错误类型: {error_type}")
                print(f"错误信息: {error_msg}")
                print(f"错误详情: {e.__dict__ if hasattr(e, '__dict__') else '无详细信息'}")
                if attempt < max_retries - 1:
                    # 指数退避等待
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    time.sleep(wait_time)
                else:
                    # 记录失败股票
                    with open("failed_downloads.txt", "a") as f:
                        f.write(f"{stock_code}\n")
                    return
        time.sleep(10)

    def download_all_data(self, workers=8):
        """带失败重试的下载"""
        # 先尝试正常下载
        print("开始首次下载...")
        self._download_batch(workers=workers)
        
        # 处理失败案例
        if os.path.exists("failed_downloads.txt"):
            print("开始重试失败股票...")
            with open("failed_downloads.txt") as f:
                failed_codes = [line.strip() for line in f]
            
            # 清空失败记录
            os.remove("failed_downloads.txt")
            
            # 使用更保守的参数重试
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {executor.submit(self.download_single_stock, code, max_retries=5): code 
                          for code in failed_codes}
                for future in tqdm(as_completed(futures), total=len(futures), desc="重试进度"):
                    pass

    def _download_batch(self, workers=8):
        """批量下载核心逻辑"""
        print('_download_batch',os.listdir(self.data_dir))
        self.stock_list = self.get_all_stocks()
        codes = self.stock_list['code'].tolist()
        existing = [f.split('.')[0] for f in os.listdir(self.data_dir)]

        #todo_codes = [c for c in codes if c not in existing]

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(self.download_single_stock, code): code 
                      for code in codes}
            for future in tqdm(as_completed(futures), total=len(codes), desc="下载进度"):
                pass

    def process_single_stock(self, file_name):
        """整合MarketPatternAnalyzer的检测逻辑"""
        try:
            code = file_name.split('.')[0]
            df = pd.read_csv(os.path.join(self.data_dir, file_name))
            print('df',df.columns,code)
            df.reset_index(inplace=True)
            #df.rename(columns={'index': 'trade_date'}, inplace=True)

            
            
            # 添加数据完整性检查
            if len(df) < 100:
                print(f"{file_name} 数据不足100条")
                return None
            if 'close' not in df.columns:
                print(f"{file_name} 缺少必要列")
                return None
                
            # 强制类型转换
            df = df.astype({
                'open': 'float64',
                'high': 'float64', 
                'low': 'float64',
                'close': 'float64',
                'volume': 'float64'
            })
            
            # 统一索引处理
            df = df.reset_index(drop=True)
            df['position_index'] = range(len(df))
            
            # 初始化分析器
            analyzer = MarketPatternAnalyzer(df, code)
            
            #i为df的最后一行的索引

            # 调用分析器方法
            surges = analyzer.detect_surge(df.tail(1).index[0])
            con_df= analyzer.detect_consolidation(surges['end'],df.tail(1).index[0])
            print('con_df',df.columns )

            print(len(con_df[con_df['consolidation'] == 1]))
            ranges = []
            if len(con_df[con_df['consolidation'] == 1])>0:
                    # 假设self.df已经包含了'consolidation'列
                bi_start_idx=surges['start'] 
                bi_end_idx=surges['end'] 
                recent_prices = df.loc[bi_end_idx:, 'close']
                    # 计算最大跌幅
                peak_price = df.loc[bi_end_idx:, 'high'].max()  # end_idx时的价格
                min_price = df.loc[bi_end_idx:, 'low'].min()     # 期间最低价
                min_price_idx = df.loc[bi_end_idx:, 'low'].idxmin()  # 最低价对应的索引
                
                # 计算跌幅百分比
                max_drawdown =abs (min_price - peak_price)# / peak_price * 100
                print('max_drawdown',max_drawdown,min_price,peak_price)

                """计算从start_idx到end_idx的涨幅"""
                # 获取价格
                start_price = df.loc[bi_start_idx:bi_end_idx, 'low'].min()
                end_price = df.loc[bi_start_idx:bi_end_idx, 'high'].max()
                
                # 计算涨幅百分比
                price_change = abs(end_price - start_price) #/ start_price * 100
                print('price_change',price_change,end_price,start_price)
                
                consolidation_groups =con_df[con_df['consolidation'] == 1].index.to_series().groupby((con_df['consolidation'] != 1).cumsum()).agg(['first', 'last'])
                pt=PatternScanner()
                # 打印开始和结束索引
                for start_idx, end_idx in zip(consolidation_groups['first'], consolidation_groups['last']):
                    print(f'开始索引: {start_idx}, 结束索引: {end_idx}')
                    ranges.append((code,df.loc[start_idx,'trade_date'], df.loc[end_idx,'trade_date'], surges,max_drawdown,price_change,peak_price ) )
                    print('range--------',ranges) # 将元组添加到列表中
                    print(ranges,type(ranges))
                    self._plot_pattern(code, df,bi_start_idx,bi_end_idx,start_idx, end_idx)


            return ranges
            
        except Exception as e:
            print(f"处理{file_name}出错: {str(e)}")
            return None

    def _plot_pattern(self, code, df,bi_start_idx,bi_end_idx,start_idx, end_idx):
        """生成交互式K线图HTML文件，支持缩放和平移"""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import os

        # 准备数据
        plot_df = df.copy()

        # 创建图形
        fig = go.Figure(data=[
            # K线图
            go.Candlestick(
                x=plot_df['trade_date'],
                open=plot_df['open'],
                high=plot_df['high'],
                low=plot_df['low'],
                close=plot_df['close'],
                increasing_line_color='red',
                decreasing_line_color='green',
                name='K线图'
            )
        ])

        # 添加笔的线段
        fig.add_trace(
            go.Scatter(
                x=[df.loc[bi_start_idx, 'trade_date'], df.loc[bi_end_idx, 'trade_date']],
                y=[df.loc[bi_start_idx, 'close'], df.loc[bi_end_idx, 'close']],
                mode='lines',
                name='笔',
                line=dict(color='blue', width=2)
            )
        )

        # 添加矩形框
        start_price = df.loc[start_idx, 'close']
        end_price = df.loc[end_idx, 'close']
        min_price = min(start_price, end_price)
        max_price = max(start_price, end_price)
        rect_height = max_price - min_price
        padding = rect_height * 0.1

        # 使用shape添加矩形
        fig.add_shape(
            type="rect",
            x0=df.loc[start_idx, 'trade_date'],
            y0=min_price - padding,
            x1=df.loc[end_idx, 'trade_date'],
            y1=max_price + padding,
            line=dict(color="red", width=2),
            fillcolor="rgba(255,0,0,0)",
        )

        # 更新布局
        fig.update_layout(
            # 主标题
            title=dict(
                text=f'{code} Pattern Analysis',
                font=dict(size=20, family='Microsoft YaHei')
            ),
            # Y轴设置
            yaxis=dict(
                title=dict(
                    text='价格',
                    font=dict(family='Microsoft YaHei', size=14)
                ),
                tickfont=dict(family='Microsoft YaHei'),
                                fixedrange=False  # 允许y轴缩放
            ),
            # X轴设置
            xaxis=dict(
                title=dict(
                    text='日期',
                    font=dict(family='Microsoft YaHei', size=14)
                ),
                rangeslider=dict(visible=True),
                type='category',
                tickangle=45,
                tickfont=dict(family='Microsoft YaHei'),
                  fixedrange=False  # 允许x轴缩放
            
            ),
            # 图例设置
                       dragmode='zoom',  # 设置默认的拖动模式为框选缩放
            showlegend=True,
            legend=dict(
                font=dict(family='Microsoft YaHei', size=12)
            ),
            # 按钮设置
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    buttons=[
                        dict(
                            label="重置视图",
                            method="relayout",
                            args=[{"xaxis.autorange": True,
                                  "yaxis.autorange": True}]
                        ),
                    ],
                    font=dict(family='Microsoft YaHei')
                )
            ],
            # 边距设置
            margin=dict(
                l=50,
                r=50,
                b=100,
                t=100
            )
        )

        # 确保输出目录存在
        os.makedirs('charts', exist_ok=True)

        # 保存为HTML文件
        output_path = f'charts/{code}_pattern.html'
        
        config = {
            'scrollZoom': True,
            'displayModeBar': True,
            'toImageButtonOptions': {
                'format': 'png',
                'filename': f'{code}_pattern',
                'height': 800,
                'width': 1200,
                'scale': 2
            },
             'modeBarButtonsToAdd': [
                'drawline',
                'drawopenpath', 
                'eraseshape',
                'zoom2d',        # 添加二维缩放按钮
                'pan2d',         # 添加平移按钮
                'select2d',      # 添加选择按钮
                'lasso2d',       # 添加套索选择按钮
                'zoomIn2d',      # 添加放大按钮
                'zoomOut2d',     # 添加缩小按钮
                'autoScale2d',   # 添加自动缩放按钮
            ],
           
            'locale': 'zh-CN' , # 设置为中文
                        'doubleClick': 'reset+autosize'  # 双击重置视图
        }
        # 'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'eraseshape'],
        # 添加自定义CSS样式
        with open(output_path, 'w', encoding='utf-8') as f:
            html_content = fig.to_html(
                config=config,
                include_plotlyjs=True,
                full_html=True,
                include_mathjax=False,
            )
             # 添加键盘控制代码
            keyboard_control = """

            <script>
            // 等待页面完全加载
            document.addEventListener('DOMContentLoaded', function() {
                // 等待Plotly图表加载完成
                setTimeout(function() {
                    var gd = document.querySelector('.js-plotly-plot');
                    console.log('Plot element found:', gd);  // 调试信息
                    
                    document.addEventListener('keydown', function(e) {
                        console.log('Key pressed:', e.key);  // 调试信息
                        
                        if (!gd || !gd._fullLayout) {
                            console.log('Plot not ready');  // 调试信息
                            return;
                        }
                        
                        var xRange = gd._fullLayout.xaxis.range;
                        var yRange = gd._fullLayout.yaxis.range;
                        var xDiff = xRange[1] - xRange[0];
                        var yDiff = yRange[1] - yRange[0];
                        
                        var update = {};
                        
                        switch(e.key) {
                            case 'ArrowUp':
                                update = {
                                    'yaxis.range': [
                                        yRange[0] + yDiff * 0.1,
                                        yRange[1] - yDiff * 0.1
                                    ]
                                };
                                break;
                            case 'ArrowDown':
                                update = {
                                    'yaxis.range': [
                                        yRange[0] - yDiff * 0.1,
                                        yRange[1] + yDiff * 0.1
                                    ]
                                };
                                break;
                            case 'ArrowLeft':
                                update = {
                                    'xaxis.range': [
                                        xRange[0] + xDiff * 0.1,
                                        xRange[1] - xDiff * 0.1
                                    ]
                                };
                                break;
                            case 'ArrowRight':
                                update = {
                                    'xaxis.range': [
                                        xRange[0] - xDiff * 0.1,
                                        xRange[1] + xDiff * 0.1
                                    ]
                                };
                                break;
                            case 'Home':
                                update = {
                                    'xaxis.autorange': true,
                                    'yaxis.autorange': true
                                };
                                break;
                            default:
                                return;
                        }
                        
                        console.log('Updating plot with:', update);  // 调试信息
                        Plotly.relayout(gd, update).catch(function(err) {
                            console.error('Update failed:', err);  // 调试信息
                        });
                    });
                    
                    console.log('Keyboard controls initialized');  // 调试信息
                }, 1000);  // 等待1秒确保图表加载完成
            });
            </script>
            """
            # 添加字体设置
            html_content = html_content.replace(
                '</head>',
                '<style>body { font-family: "Microsoft YaHei", Arial, sans-serif; }</style></head>'
            )
            html_content = html_content.replace('</body>', keyboard_control + '</body>')
            f.write(html_content)

        print(f"已生成交互式图表：{output_path}")
    def _get_stock_name(self, code):
        """根据代码获取股票名称"""
        return self.stock_list[self.stock_list['code'] == code]['name'].values[0]
    
    def run_analysis(self, workers=8):
        """并行处理分析"""
        print("开始分析数据...")
        files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        results = []
        from multiprocessing import Pool
        import multiprocessing
        optimal_workers = min(multiprocessing.cpu_count(), workers)
        #self.process_single_stock('300577.csv')
      
        with Pool(processes=optimal_workers) as pool:
            # 使用 imap 替代 submit 以减少内存占用

            for result in tqdm(pool.imap(self.process_single_stock, files), 
                            total=len(files), 
                            desc="分析进度"):
                if result:
                    results.append(result)


        flattened_results = []
        for result_list in results:
            if isinstance(result_list, list):
                flattened_results.extend(result_list)  # 如果是列表，展开它
            else:
                flattened_results.append(result_list)  # 如果不是列表，直接添加

        # 创建DataFrame
        if flattened_results:
            codes = [item[0] for item in flattened_results]
            hp_start=[item[1] for item in flattened_results]
            hp_end=[item[2] for item in flattened_results]
            data = [item[3] for item in flattened_results]
            max_drawdown=[item[4] for item in flattened_results]
            price_change=[item[5] for item in flattened_results]
            peak_price = [item[6] for item in flattened_results]
            # 创建DataFrame
            result_df = pd.DataFrame(data)  # 从字典列表创建DataFrame
            result_df['hp_start'] = hp_start     # 添加code列
            result_df['hp_end'] =hp_end 
            result_df['code'] =codes
            result_df['max_drawdown'] =max_drawdown
            result_df['price_change'] =price_change
            result_df['peak_price'] =peak_price
            # 重新排列列的顺序，将'code'放在最前面
            cols = ['code'] + [col for col in result_df.columns if col != 'code']
            result_df = result_df[cols]
            print(result_df.columns)
            
            # 创建一个新列表示 now_index 与 end 的差值
            result_df['now_to_end'] = result_df['now_index']- result_df['end']

            # 创建一个新列表示 start 与 end 的差值
            result_df['start_to_end'] = -result_df['start'] +result_df['end']

            # 筛选满足条件的行 vbvbvbvbvbvbvbvb     
            result_df = result_df[result_df['now_to_end'] <1.2*result_df['start_to_end']]
            result_df = result_df[result_df['peak_price'] >result_df['bihigh']]
            # 如果不需要显示这两个辅助列，可以删除它们
            #选择'max_drawdown'/'price_change'小于30%的行

            result_df['drawdown_price_ratio'] = abs(result_df['max_drawdown'] / result_df['price_change'])

            # 筛选满足条件的行
            #result_df = result_df[result_df['drawdown_price_ratio'] < 0.3]

            #filtered_df = filtered_df.drop(['now_to_end', 'start_to_end'], axis=1)

            print("\nFiltered DataFrame:")

            # 如果需要，可以指定列的顺序
            #desired_cols = ['code', 'start', 'end', 'high', 'low', 'angle', 'time_ratio', 'price_ratio', 'score']
            #result_df = result_df[desired_cols]
        else:
            print("No valid results found!")
            result_df = pd.DataFrame()  # 创建空的DataFrame

        # 打印结果
        print("\nResult DataFrame:")
        print(result_df)

        

  
        if not result_df.empty:
            result_df.to_csv("pattern_results.csv", index=False)
            print(f"发现{len(result_df)}只符合形态的股票，结果已保存")
            self.copy_matched_charts()
        else:
            print("未发现符合形态的股票")
    def copy_matched_charts(self):
        """复制符合条件的股票图表到新文件夹"""
        import shutil
        import os
        
        # 创建目标文件夹
        target_dir = "matched_charts"
        os.makedirs(target_dir, exist_ok=True)
        
        # 读取结果文件
        result_df = pd.read_csv("pattern_results.csv")
        
        # 获取所有符合条件的股票代码
        matched_codes = result_df['code'].tolist()
        
        # 源文件夹
        source_dir = "charts"
        
        # 复制文件
        copied_count = 0
        for code in matched_codes:
            source_file = os.path.join(source_dir, f"{code}_pattern.html")
            target_file = os.path.join(target_dir, f"{code}_pattern.html")
            
            if os.path.exists(source_file):
                shutil.copy2(source_file, target_file)
                copied_count += 1
                print(f"已复制: {code}_pattern.html")
            else:
                print(f"未找到文件: {code}_pattern.html")
        
        print(f"\n共复制了 {copied_count} 个文件到 {target_dir} 文件夹")


def convert_parquet_to_csv(base_dirs=['data', 'analysis_cache']):
    """将指定目录下的所有parquet文件转换为csv格式"""
    for base_dir in base_dirs:
        if not os.path.exists(base_dir):
            print(f"目录不存在: {base_dir}")
            continue
            
        # 遍历目录下的所有文件
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                if file.endswith('.parquet'):
                    parquet_path = os.path.join(root, file)
                    csv_path = parquet_path.replace('.parquet', '.csv')
                    
                    try:
                        # 读取parquet文件
                        df = pd.read_parquet(parquet_path)
                        
                        # 保存为csv
                        df.to_csv(csv_path, index=False)
                        
                        print(f"已转换: {parquet_path} -> {csv_path}")
                        
                        # 删除原parquet文件（可选）
                        #os.remove(parquet_path)
                        print(f"已删除原文件: {parquet_path}")
                        
                    except Exception as e:
                        
                        print(f"转换失败 {parquet_path}: {str(e)}")
import pandas as pd
import os
from datetime import datetime

def validate_date(date_str):
    """多格式日期验证"""
    try:
        # 尝试常见日期格式
        return pd.to_datetime(date_str, format='%Y-%m-%d', errors='coerce').date()
    except:
        return None

def fix_cache_dates(stock_code):
    """修复缓存日期数据"""
    # 路径配置
    cache_path = os.path.join('analysis_cache', f"{stock_code}.csv")
    #code为stock_code中“_”前面的字符
    code = stock_code.split('_')[0]
    #print(f"正在处1理 {code} 的数据...")
    data_path = os.path.join('data', f"{code}.csv")
    
    if not os.path.exists(cache_path) or not os.path.exists(data_path):
        print(f"文件不存在: {cache_path} 或 {data_path}")
        return

    # 读取数据
    cache_df = pd.read_csv(cache_path)
    data_df = pd.read_csv(data_path)
    
    # 检查最后一个日期是否有效
    last_date = cache_df['trade_date'].iloc[-1]
    print(f"{stock_code} 最后一个日期: {last_date}")
    print( validate_date(last_date))
    
    if validate_date(last_date) is pd.NaT:
        #print(f"{stock_code} 日期格式正常")
        
        print(f"{stock_code} 最后一个日期无效，正在修复...")
        # 数据预处理
        def preprocess(df):
            df = df.copy()
            # 四舍五入到4位小数并创建唯一键
            print(df.columns)
            df['price_key'] = df[['open', 'high', 'low', 'close']].apply(tuple, axis=1)
            return df

        cache_processed = preprocess(cache_df)
        data_processed = preprocess(data_df)

        # 合并数据获取正确日期
        merged = pd.merge(
            cache_processed,
            data_processed[['price_key', 'trade_date']],
            on='price_key',
            how='left',
            suffixes=('_cache', '_correct')
        )

        # 替换日期列
        cache_df['trade_date'] = merged['trade_date_correct'].combine_first(cache_df['trade_date'])
        
        # 保存修复后的数据
        cache_df.to_csv(cache_path, index=False)
        print(f"已修复 {stock_code} 的日期数据，更新 {len(merged)} 条记录")

# 使用示例：修复所有缓存文件
def convert_column_names(data_dir='data'):
    """统一转换数据目录下的列名为英文"""
    # 列名映射表
    column_mapping = {
        '开盘': 'open',
        '最高': 'high',
        '最低': 'low',
        '收盘': 'close',
        '成交量': 'volume'
    }

    # 遍历数据目录
    files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    for file in tqdm(files, desc='转换列名'):
        file_path = os.path.join(data_dir, file)
        try:
            # 读取数据
            df = pd.read_csv(file_path)
            
            # 执行列名转换（保留未映射的列）
            df = df.rename(columns=column_mapping)
            
            # 保存时保留原始索引
            df.to_csv(file_path, index=False)
            
        except Exception as e:
            print(f"处理 {file} 失败: {str(e)}")

if __name__ == "__main__":
 
    print("欢迎使用股票分析工具！")

    scanner = PatternScanner()
    
    # 第一步：下载数据（只需运行一次）
    #scanner.download_all_data(workers=30)
    
    # 第二步：分析数据
    scanner.run_analysis(workers=30)



    ''''
    


        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(self.process_single_stock, f) for f in files]
            for future in tqdm(as_completed(futures), total=len(files), desc="分析进度"):
                result = future.result()
                if result:
                    results.append(result)
        convert_parquet_to_csv(base_dirs=['data'])
    zzz
                  for cache_file in os.listdir('analysis_cache'):
        if cache_file.endswith('.csv'):
            stock_code = cache_file.split('.')[0]
            #print(f"正在修复 {stock_code} 的日期数据...")
            
            fix_cache_dates(stock_code) 
    zz     
    zzz
    df = ak.stock_zh_a_hist(symbol= "601727", 
    period= "daily",
                        start_date='20210101',
                        end_date=datetime.now().strftime('%Y%m%d'),
                        adjust="qfq")
    '''