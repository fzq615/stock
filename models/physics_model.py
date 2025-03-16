import numpy as np
import pandas as pd
from typing import Dict
from dataclasses import dataclass
import logging

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
                 equilibrium_epsilon: float = 0.005,
                 logger=None):
        """
        初始化物理-金融模型
        
        参数:
            atr_period: ATR计算周期
            lookback_period: 回溯期间（用于计算ATR分位数）
            energy_threshold: 能量阈值
            equilibrium_epsilon: 平衡状态阈值
            logger: 日志记录器
        """
        self.atr_period = atr_period
        self.lookback_period = lookback_period
        self.energy_threshold = energy_threshold
        self.equilibrium_epsilon = equilibrium_epsilon
        self.logger = logger or logging.getLogger('PhysicsFinanceModel')
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
        df = df.copy()
        df['signal'] = 0  # 0: 无信号, 1: 买入, -1: 卖出
        
        # 计算技术指标
        df['ma20'] = df['close'].rolling(window=20).mean()  # 20日均线
        df['ma60'] = df['close'].rolling(window=60).mean()  # 60日均线
        df['volume_ma20'] = df['volume'].rolling(window=20).mean()  # 20日成交量均线
        
        # 计算波动率
        df['volatility'] = df['close'].rolling(window=20).std()
        
        # 添加趋势过滤
        df['trend'] = np.where(df['ma20'] > df['ma60'], 1, -1)
        
        # 计算能量特征
        df = self.calculate_energy(df)
        
        # 添加市场状态
        df['market_state'] = ''
        for i in range(len(df)):
            if i < 60:  # 跳过前60行，确保有足够的历史数据
                continue
                
            df.loc[df.index[i], 'market_state'] = self.determine_market_state(
                df['energy'].iloc[i],
                df['energy_derivative'].iloc[i],
                df['energy_second_derivative'].iloc[i],
                df['k'].iloc[i],
                df['A'].iloc[i],
                df['volume'].iloc[i],
                df['volume_ma20'].iloc[i]
            )
        
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