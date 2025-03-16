import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime
import talib

# 导入缠论模块包装器
from utils.chan_wrapper import load_chan_module, get_chan_functions

class MarketPatternAnalyzer:
    CACHE_DIR = "analysis_cache"
    
    def __init__(self, df: pd.DataFrame, code: str, logger=None):
        """
        初始化市场模式分析器
        
        参数:
            df: 股票数据DataFrame
            code: 股票代码
            logger: 日志记录器
        """
        self.df = df.copy()
        self.code = code
        self.CACHE_DIR = f"analysis_cache"
        self.cache_file = os.path.join(self.CACHE_DIR, f"{self.code}_analysis.csv")
        
        # 创建缓存目录
        os.makedirs(self.CACHE_DIR, exist_ok=True)
        self.pattern_cache = []
        self.logger = logger or logging.getLogger('MarketPatternAnalyzer')
        self.logger.info(f"初始化市场模式分析器，股票代码: {code}")
        
        # 加载缠论模块
        try:
            self.chan_module = load_chan_module()
            self.chan_functions = get_chan_functions(self.chan_module)
            self.kxian_baohan_js_0 = self.chan_functions['kxian_baohan_js_0']
            self.fenbi_js = self.chan_functions['fenbi_js']
            self.repeat_bi_js = self.chan_functions['repeat_bi_js']
            self.xianDuan_js = self.chan_functions['xianDuan_js']
            self.Xian_Duan = self.chan_functions['Xian_Duan']
        except Exception as e:
            self.logger.error(f"加载缠论模块失败: {str(e)}")
            raise
        
        # 设置参数
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
                    if len(fenxing_points) >= 4:
                        start_idx = fenxing_points.iloc[-4].name
                        
                        # 获取新数据
                        new_data = self.df[self.df['trade_date'] > cache_df.loc[start_idx, 'trade_date']]
                        
                        # 包含前3天数据以确保计算的连续性
                        calc_data = self.df.loc[new_data.index[0]-4:, :]
                        
                        # 对新数据进行分型计算
                        calc_data = self.setup_indicators_for_data(calc_data)
                        calc_data = calc_data.fillna(0)
                        
                        # 获取原有数据
                        old_data = self.df[self.df['trade_date'] <= self.df.loc[start_idx, 'trade_date']]
                        calc_data = cache_df[cache_df['trade_date'] > cache_df.loc[start_idx, 'trade_date']]
                        
                        # 合并数据
                        self.df = pd.concat([old_data, calc_data], axis=0, ignore_index=True)
                        
                        # 更新技术指标
                        self._update_technical_indicators()
                        
                        # 保存更新后的缓存
                        self._save_cache()
                        return True
                    else:
                        self.setup_indicators()
                        self.setup_analysis()
                        self._save_cache()
            except Exception as e:
                self.logger.error(f"缓存加载失败: {str(e)}")
                if os.path.exists(self.cache_file):
                    os.remove(self.cache_file)
        return False
    
    def _update_technical_indicators(self):
        """更新技术指标"""
        # 计算ATR
        tr1 = self.df['high'] - self.df['low']
        tr2 = abs(self.df['high'] - self.df['close'].shift(1))
        tr3 = abs(self.df['low'] - self.df['close'].shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        self.df['ATR'] = tr.rolling(window=14).mean()
        
        # 计算RSI
        self.df['RSI'] = talib.RSI(self.df['close'], timeperiod=14)
        
        # 计算ADX
        self.df['ADX'] = talib.ADX(self.df['high'], self.df['low'], self.df['close'], timeperiod=14)
        
        # 计算成交量均线
        self.df['volume_ma20'] = self.df['volume'].rolling(window=20).mean()
    
    def _save_cache(self):
        """保存分析结果"""
        try:
            self.df.to_csv(self.cache_file, index=False)
            self.logger.info(f"已保存分析缓存: {self.code}")
        except Exception as e:
            self.logger.error(f"缓存保存失败: {str(e)}")
    
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
        after_fenxing = self.kxian_baohan_js_0(df)
        baohan = after_fenxing.copy()
        bi_data0 = self.fenbi_js(after_fenxing)
        fenxing_chongfu = bi_data0.copy()
        
        bi_data2 = self.repeat_bi_js(bi_data0)
        bi_data_1 = self.xianDuan_js(bi_data2)
        
        # 删除不需要的列
        bi_data_1 = bi_data_1.drop(
            labels=['open', 'high', 'low', 'close', 'volume', 'fenxing_type', 
                    'range_high', 'fenxing_high', 'fenxing_high_less', 
                    'fenxing_low', 'fenxing_low_less'], axis=1)
        
        # 合并数据
        df = pd.merge(df, bi_data_1, how='left', on=['trade_date'])
        df = pd.merge(df, baohan[['trade_date', 'baohan']], how='left', on=['trade_date'])
        df = pd.merge(df, fenxing_chongfu[['trade_date', 'fenxing_type']], how='left', on=['trade_date'])
        
        # 添加顶底分型的高低点
        df['Bottom_high'] = df.apply(
            lambda row: row['high'] if row['fenxing_type'] != 0 else 0, axis=1)
        df['Top_low'] = df.apply(
            lambda row: row['low'] if row['fenxing_type'] != 0 else 0, axis=1)
        
        # 填充空值
        df = df.fillna(0)
        
        # 提取需要的列
        df = df[['trade_date', 'fenxing_type_last', 'fenxing_plot', 
                 'zoushi', 'baohan', 'fenxing_type',
                 'Bottom_high', 'Top_low']]
        df = pd.merge(df_copy, df, how='left', on=['trade_date'])
        
        return df
    
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
        self.df['ADX'] = talib.ADX(self.df['high'], self.df['low'], self.df['close'], timeperiod=14)
        
        # 计算成交量均线
        self.df['volume_ma20'] = self.df['volume'].rolling(window=20).mean()
    
    def setup_analysis(self):
        """设置技术分析，计算笔和线段"""
        try:
            self.logger.info("开始计算笔和线段分析...")
            
            # 确保数据索引正确
            self.df = self.df.reset_index()
            df_copy = self.df.copy()
            
            # 使用导入的函数计算笔和线段
            self.logger.info("开始K线包含处理...")
            after_fenxing = self.kxian_baohan_js_0(self.df)
            baohan = after_fenxing.copy()
            self.logger.info("K线包含处理完成")
            
            bi_data0 = self.fenbi_js(after_fenxing)
            fenxing_chongfu = bi_data0.copy()
            
            bi_data2 = self.repeat_bi_js(bi_data0)
            bi_data_1 = self.xianDuan_js(bi_data2)
            
            # 删除不需要的列
            bi_data_1 = bi_data_1.drop(
                labels=['open', 'high', 'low', 'close', 'volume', 'fenxing_type', 
                       'range_high', 'fenxing_high', 'fenxing_high_less', 
                       'fenxing_low', 'fenxing_low_less'], axis=1)
            
            # 合并数据
            self.df = pd.merge(self.df, bi_data_1, how='left', on=['trade_date'])
            self.df = pd.merge(self.df, baohan[['trade_date', 'baohan']], how='left', on=['trade_date'])
            self.df = pd.merge(self.df, fenxing_chongfu[['trade_date', 'fenxing_type']], how='left', on=['trade_date'])
            
            # 添加顶底分型的高低点
            self.df['Bottom_high'] = self.df.apply(
                lambda row: row['high'] if row['fenxing_type'] != 0 else 0, axis=1)
            self.df['Top_low'] = self.df.apply(
                lambda row: row['low'] if row['fenxing_type'] != 0 else 0, axis=1)
            
            # 填充空值
            self.df = self.df.fillna(0)
            
            # 提取需要的列并合并
            df_copy = df_copy[['trade_date', 'open', 'close', 'high', 'low', 'volume']].copy()
            self.df = self.df[['trade_date', 'fenxing_type_last', 'fenxing_plot', 
                          'zoushi', 'baohan', 'fenxing_type',
                          'Bottom_high', 'Top_low']]
            self.df = pd.merge(df_copy, self.df, how='left', on=['trade_date'])
            
            # 更新技术指标
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
        history_bi = self.df[(self.df['fenxing_type_last'].isin([1, -1])) &
                          (self.df.index <= start_idx)].tail(20)  # 获取最近3组笔
        surge_candidates = []
        surge_bi = []
        
        if len(history_bi) >= 5:
            angles = []
            time_ratios = []
            price_ratios = []
            
            for i in range(0, len(history_bi)-1):
                prev = history_bi.iloc[i]
                current = history_bi.iloc[i+1]
                time_diff = current.name - prev.name
                
                if current['high'] > prev['low']:
                    price_diff = current['high'] - prev['low']
                else:
                    price_diff = -current['high'] + prev['low']
                    
                atr = self.df['ATR'].iloc[prev.name]
                angle = np.degrees(np.arctan(price_diff/(time_diff*0.1)))
                angles.append(angle)
                
                time_ratio = time_diff / self.avg_bi_length(history_bi)
                time_ratios.append(time_ratio)
                
                price_ratio = price_diff / atr
                price_ratios.append(price_diff / atr)
                
                surge_candidates.append({
                    'start': prev.name,
                    'end': current.name,
                    'bihigh': current['high'],
                    'bilow': prev['low'],
                    'angle': angle,
                    'time_ratio': time_ratio,
                    'price_ratio': price_ratio,
                })
            
            # 动态阈值计算（均值+0.5标准差）
            angle_threshold = np.mean(angles) + 0.5*np.std(angles)
            time_ratio_threshold = np.mean(time_ratios) + 0.5*np.std(time_ratios)
            price_ratio_threshold = np.mean(price_ratios) + 0.5*np.std(price_ratios)
        else:  # 数据不足时使用默认值
            angle_threshold = 45
            time_ratio_threshold = 1.2
            price_ratio_threshold = 2.0
        
        if len(history_bi) < 6:
            return None
        
        # 计算评分
        hist_scores = []
        for candidate in surge_candidates:
            angle = candidate['angle']
            time_ratio = candidate['time_ratio']
            price_ratio = candidate['price_ratio']
            
            # 角度得分（最高2分）
            angle_score = (angle / angle_threshold) * 2
            
            # 时间比率得分（最高1分）
            time_score = 2 * time_ratio 
            
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
        
        # 设置评分阈值
        score_threshold = np.mean(hist_scores)
        
        # 筛选符合条件的大涨
        for candidate in surge_candidates:
            total_score = candidate['total_score']
            
            if total_score >= 1.3*score_threshold and self.df.at[candidate['start'],'close'] < self.df.at[candidate['end'],'close']:
                surge_bi.append({
                    'start': candidate['start'],
                    'end': candidate['end'],
                    'start_date': self.df.at[candidate['start'],'trade_date'],
                    'end_date': self.df.at[candidate['end'],'trade_date'],
                    'now_index': len(self.df),
                    'bihigh': candidate['bihigh'],
                    'bilow': candidate['bilow'],
                    'angle': candidate['angle'],
                    'time_ratio': candidate['time_ratio'],
                    'price_ratio': candidate['price_ratio'],
                    'score': candidate['total_score']
                })
                
                # 记录大涨信息
                self.df.loc[candidate['end'], 'surge_start'] = candidate['start']
                self.df.loc[candidate['end'], 'surge_end'] = candidate['end']
                self.df.loc[candidate['end'], 'surge_score'] = candidate['total_score']
        
        # 返回最后一个符合条件的大涨
        return surge_bi[-1] if surge_bi else None
    
    def detect_consolidation(self, surge_end: int, now_index: int) -> dict:
        """基于笔端点检测横盘"""
        consolidation_bi = []
        resistance = self.df['high'].iloc[surge_end]
        support = self.df['low'].iloc[surge_end]
        threshold = 0.05   # 降低阈值到1.5%
        breakout = False
        overlap_count = 0  # 新增：连续重叠K线计数器
        overlap_window = []  # 新增：存储最近4根K线
        
        # 初始化标记列
        self.logger.info(f"开始检测横盘区间，起始位置：{self.df.loc[surge_end, 'trade_date']} 当前索引：{self.df.loc[now_index, 'trade_date']}")
        self.logger.info(f"开始检测横盘区间，起始位置：{surge_end} 当前索引：{now_index}")
        
        # 标记横盘开始
        consolidations = []
        
        is_hp = 0
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
        MAX_ALLOWED_GAPS = 1  # 允许最多1根不重叠的K线
        MIN_CONSECUTIVE = 4   # 至少4根有效重叠
        window_size = 6
        min_overlap = 4
        overlap_ranges = []
        
        # 动态跟踪最后有效区间
        current_range = {'high': self.df['high'].iloc[surge_end], 
                        'low': self.df['low'].iloc[surge_end]}
        consecutive_count = 0
        gap_count = 0

        for idx in range(surge_end, now_index+1):
            current_high = self.df.at[idx, 'high']
            current_low = self.df.at[idx, 'low']
            
            # 判断是否与当前区间重叠
            if (current_low < current_range['high']) and (current_high > current_range['low']):
                consecutive_count += 1
                gap_count = 0
                # 更新区间范围
                current_range['high'] = min(current_range['high'], current_high)
                current_range['low'] = max(current_range['low'], current_low)
            else:
                if gap_count < MAX_ALLOWED_GAPS:
                    gap_count += 1
                    consecutive_count += 1  # 即使有间隔也计数
                else:
                    # 重置检测
                    current_range = {'high': current_high, 'low': current_low}
                    consecutive_count = 0
                    gap_count = 0
            
            # 满足最小连续条件时标记
            if consecutive_count >= MIN_CONSECUTIVE:
                start_idx = idx - MIN_CONSECUTIVE + 1
                self.df.loc[start_idx:idx, 'consolidation'] = 1
                self.df.loc[start_idx:idx, 'consolidation_support'] = current_range['low']
                self.df.loc[start_idx:idx, 'consolidation_resistance'] = current_range['high']
        
        return self.df

    def generate_signals(self, df=None):
        """生成交易信号"""
        # 如果传入了df参数，则使用传入的df，否则使用self.df
        if df is not None:
            self.df = df.copy()
            
        # 初始化信号列
        self.df['signal'] = 0
        self.df['buy_signal'] = 0
        self.df['sell_signal'] = 0
        
        # 计算均线
        self.df['ma20'] = self.df['close'].rolling(window=20).mean()
        self.df['ma60'] = self.df['close'].rolling(window=60).mean()
        
        # 设置最小交易间隔
        min_trade_interval = 20
        last_trade_idx = -min_trade_interval
        
        # 遍历数据生成信号
        for i in range(60, len(self.df)):
            # 检查是否满足最小交易间隔
            if i - last_trade_idx < min_trade_interval:
                continue
            
            # 买入条件
            if (self.df['ma20'].iloc[i] > self.df['ma60'].iloc[i] and  # 均线多头排列
                self.df['close'].iloc[i] > self.df['ma20'].iloc[i] and  # 价格在均线上方
                self.df['volume'].iloc[i] > self.df['volume_ma20'].iloc[i] * 1.2):  # 成交量放大
                
                self.df.iloc[i, self.df.columns.get_loc('signal')] = 1
                self.df.iloc[i, self.df.columns.get_loc('buy_signal')] = 1
                last_trade_idx = i
            
            # 卖出条件
            elif (self.df['ma20'].iloc[i] < self.df['ma60'].iloc[i] and  # 均线空头排列
                  self.df['close'].iloc[i] < self.df['ma20'].iloc[i]):  # 价格在均线下方
                
                self.df.iloc[i, self.df.columns.get_loc('signal')] = -1
                self.df.iloc[i, self.df.columns.get_loc('sell_signal')] = 1
                last_trade_idx = i
        
        return self.df

