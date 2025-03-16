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
from datetime import datetime
import plotly.graph_objects as go

# 导入自定义模块
from models.market_pattern import MarketPatternAnalyzer

# 在文件开头添加导入
import tushare as ts

class PatternScanner:
    def __init__(self, logger=None):
        """初始化模式扫描器"""
        self.data_dir = "./data"
        self.chart_dir = "./charts"
        self.logger = logger or logging.getLogger('PatternScanner')
        # 初始化 tushare
        ts.set_token('49023af1c21e051f1695472c7a4e2113dfbc476c4028369fac63da88')
        self.pro = ts.pro_api()
        self._prepare_directories()
    
    def _prepare_directories(self):
        """创建必要的存储目录"""
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.chart_dir, exist_ok=True)

    def download_single_stock(self, stock_code, max_retries=5, timeout=30):
        """带重试机制和超时控制的数据下载，如果已有数据不是最新的则更新"""
        default_start_date='20210101'
        file_path = os.path.join(self.data_dir, f"{stock_code}.csv")
        current_date = datetime.now().date()
        

        try:
            df = ak.stock_zh_index_daily_em(symbol='sh000001', start_date= "20250101", end_date=datetime.now().strftime('%Y%m%d'))
            latest_trade_date = pd.to_datetime(df['date'].iloc[-1])
        except Exception as e:
            self.logger.error(f"获取指数数据失败: {str(e)}")
            raise
        # 检查文件是否存在，如果存在则检查是否需要更新
        if os.path.exists(file_path):
            try:
                existing_df = pd.read_csv(file_path)
                if 'trade_date' in existing_df.columns:
                    existing_df['trade_date'] = pd.to_datetime(existing_df['trade_date'])
                    last_date = existing_df['trade_date'].max()
                    
                    if last_date.date() >= latest_trade_date.date():
                        self.logger.info(f"股票 {stock_code} 数据已是最新，无需更新")
                        return existing_df
                    else:
                        self.logger.info(f"股票 {stock_code} 数据需要更新")
                        start_date = (last_date + pd.Timedelta(days=1)).strftime('%Y%m%d')
                else:
                    self.logger.info(f"股票 {stock_code} 数据格式异常，将重新下载")
                    start_date = default_start_date
            except Exception as e:
                self.logger.error(f"检查股票 {stock_code} 数据时出错: {str(e)}，将重新下载")
                start_date = default_start_date
            
            if 'start_date' not in locals():
                if os.path.exists(file_path):
                    os.remove(file_path)
                start_date = default_start_date
        else:
            start_date = default_start_date
    
        base_wait = 5
        for attempt in range(max_retries):
            try:
                wait_time = base_wait + (attempt * 10)
                time.sleep(wait_time)
                
                if stock_code == '000300':
                    df = self.pro.index_daily(ts_code='399300.SZ',
                                                start_date=start_date,
                                                end_date=datetime.now().strftime('%Y%m%d'))
                else:
                    # 转换股票代码格式为tushare格式
                    if '.SZ' in stock_code:
                        ts_code = stock_code.replace('.SZ', '.SZ')
                    elif '.SH' in stock_code:
                        ts_code = stock_code.replace('.SH', '.SH')
                    else:
                        # 根据股票代码判断市场
                        if stock_code.startswith(('6', '688')):
                            ts_code = f"{stock_code}.SH"
                        else:
                            ts_code = f"{stock_code}.SZ"
                    
                    # 获取日线数据
                    df = self.pro.daily(ts_code=ts_code,
                                          start_date=start_date,
                                          end_date=datetime.now().strftime('%Y%m%d'))
                    
                    # 获取复权因子
                    adj_factor = self.pro.adj_factor(ts_code=ts_code,
                                                       start_date=start_date,
                                                       end_date=datetime.now().strftime('%Y%m%d'))
                    
                    # 合并数据并计算前复权价格
                    if not adj_factor.empty:
                        df = pd.merge(df, adj_factor[['trade_date', 'adj_factor']], on='trade_date', how='left')
                        df['adj_factor'] = df['adj_factor'].fillna(method='ffill')
                        for price_col in ['open', 'high', 'low', 'close']:
                            df[price_col] = df[price_col] * df['adj_factor']
    
                # 处理日期列
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df = df.sort_values('trade_date')
                
                # 如果文件已存在，合并数据
                if os.path.exists(file_path):
                    existing_df = pd.read_csv(file_path)
                    existing_df['trade_date'] = pd.to_datetime(existing_df['trade_date'])
                    df = pd.concat([existing_df, df]).drop_duplicates(subset=['trade_date']).sort_values('trade_date')
                
                # 保存数据
                df.to_csv(file_path, index=False)
                
                sleep_time = random.uniform(0.3 + attempt*0.2, 0.5 + attempt*0.3)
                time.sleep(sleep_time)
                return df
                
            except Exception as e:
                error_type = type(e).__name__
                error_msg = str(e)
                
                self.logger.error(f"第{attempt+1}次尝试下载{stock_code}失败:")
                self.logger.error(f"错误类型: {error_type}")
                self.logger.error(f"错误信息: {error_msg}")
                
                if attempt >= 2:
                    is_suspended, reason = self.is_stock_suspended(stock_code)
                    if is_suspended:
                        self.logger.info(f"股票 {stock_code} 处于停牌状态: {reason}")
                        with open("suspended_stocks.txt", "a") as f:
                            f.write(f"{stock_code},{datetime.now().strftime('%Y-%m-%d')},{reason}\n")
                        return None
                
                if attempt < max_retries - 1:
                    wait_time = min((2 ** attempt) + random.uniform(1, 3), 60)
                    self.logger.info(f"等待 {wait_time:.2f} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    with open("failed_downloads.txt", "a") as f:
                        f.write(f"{stock_code}\n")
                    self.logger.error(f"股票 {stock_code} 下载失败，已达到最大重试次数")
                    return None
        
        self.logger.error(f"股票 {stock_code} 下载失败，已记录到失败列表")
        return None
    
    def download_all_data(self, workers=8):
        """带失败重试的下载"""
        # 先尝试正常下载
        self.logger.info("开始首次下载...")
        self._download_batch(workers=workers)
        
        # 处理失败案例
        if os.path.exists("failed_downloads.txt"):
            self.logger.info("开始重试失败股票...")
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
    def get_all_stocks(self):
        """获取沪深股票列表"""
        try:
            # 获取深圳证券交易所的股票列表
            sz_stock = ak.stock_info_sz_name_code()
            # 获取上海证券交易所的股票列表
            sh_stock = ak.stock_info_sh_name_code()
            
            # 打印列名以确认它们是什么
            self.logger.info("深圳证券交易所股票列表的列名: %s", sz_stock.columns.tolist())
            self.logger.info("上海证券交易所股票列表的列名: %s", sh_stock.columns.tolist())

            # 根据实际列名进行重命名
            sz_stock_renamed = sz_stock.rename(columns={'A股代码': 'code', 'A股简称': 'name'})
            sh_stock_renamed = sh_stock.rename(columns={'证券代码': 'code', '证券简称': 'name'})

            self.logger.info("股票数量: 深圳 %d, 上海 %d", len(sz_stock_renamed), len(sh_stock_renamed))

            # 合并两个数据框并去重
            df = pd.concat([sz_stock_renamed, sh_stock_renamed])[['code', 'name']].drop_duplicates()
            
            return df
        except Exception as e:
            self.logger.error(f"获取股票列表时出错: {str(e)}")
            raise
    
    def _download_batch(self, workers=8):
        """批量下载核心逻辑"""
        self.logger.info(f"当前数据目录内容: {os.listdir(self.data_dir)}")
        self.stock_list = self.get_all_stocks()
        codes = self.stock_list['code'].tolist()
        existing = [f.split('.')[0] for f in os.listdir(self.data_dir)]
        print('ex',codes)
        # 可以选择只下载不存在的股票数据
        # todo_codes = [c for c in codes if c not in existing]
        
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
            self.logger.info(f"处理股票 {code}, 数据列: {df.columns}")
            df.reset_index(inplace=True)
            
            # 添加数据完整性检查
            if len(df) < 100:
                self.logger.info(f"{file_name} 数据不足100条")
                return None
            if 'close' not in df.columns:
                self.logger.info(f"{file_name} 缺少必要列")
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
            analyzer = MarketPatternAnalyzer(df, code, self.logger)
            
            # 调用分析器方法
            surges = analyzer.detect_surge(df.tail(1).index[0])
            if surges is None:
                return None
                
            # 检测横盘整理
            con_df = analyzer.detect_consolidation(surges['end'], df.tail(1).index[0])
            if con_df is None or len(con_df[con_df['consolidation'] == 1]) == 0:
                return None
                
            ranges = []
            if len(con_df[con_df['consolidation'] == 1]) > 0:
                bi_start_idx = surges['start'] 
                bi_end_idx = surges['end'] 
                recent_prices = df.loc[bi_end_idx:, 'close']
                
                # 计算最大跌幅
                peak_price = df.loc[bi_end_idx:, 'high'].max()  # end_idx时的价格
                min_price = df.loc[bi_end_idx:, 'low'].min()     # 期间最低价
                min_price_idx = df.loc[bi_end_idx:, 'low'].idxmin()  # 最低价对应的索引
                
                # 计算跌幅百分比
                max_drawdown = abs(min_price - peak_price)
                
                # 计算从start_idx到end_idx的涨幅
                start_price = df.loc[bi_start_idx:bi_end_idx, 'low'].min()
                end_price = df.loc[bi_start_idx:bi_end_idx, 'high'].max()
                price_change = abs(end_price - start_price)
                
                # 分析横盘区间
                consolidation_groups = con_df[con_df['consolidation'] == 1].index.to_series().groupby(
                    (con_df['consolidation'] != 1).cumsum()).agg(['first', 'last'])
                
                # 记录每个横盘区间
                for start_idx, end_idx in zip(consolidation_groups['first'], consolidation_groups['last']):
                    self.logger.info(f'横盘区间: 开始索引: {start_idx}, 结束索引: {end_idx}')
                    ranges.append((code, 
                                  df.loc[start_idx,'trade_date'], 
                                  df.loc[end_idx,'trade_date'], 
                                  surges,
                                  max_drawdown,
                                  price_change,
                                  peak_price))
                    
                    # 绘制图表
                    self._plot_pattern(code, df, bi_start_idx, bi_end_idx, start_idx, end_idx)
            
            return ranges
            
        except Exception as e:
            self.logger.error(f"处理{file_name}出错: {str(e)}")
            return None
    
    def _plot_pattern(self, code, df, bi_start_idx, bi_end_idx, start_idx, end_idx):
        """生成交互式K线图HTML文件，支持缩放和平移"""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # 准备数据
        plot_df = df.copy()
        
        # 尝试获取股票名称
        stock_name = ""
        try:
            # 如果self.stock_list存在，尝试从中获取股票名称
            if hasattr(self, 'stock_list') and self.stock_list is not None:
                stock_info = self.stock_list[self.stock_list['code'] == code]
                if not stock_info.empty:
                    stock_name = stock_info['name'].values[0]
        except Exception as e:
            self.logger.warning(f"获取股票名称失败: {str(e)}")
        
        # 创建子图布局，2行1列，第一行是K线图，第二行是成交量图
        fig = make_subplots(
            rows=2, 
            cols=1, 
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3],
            subplot_titles=(f'{code} {stock_name} - 价格走势', '成交量')
        )
        
        # 添加K线图
        fig.add_trace(
            go.Candlestick(
                x=plot_df['trade_date'],
                open=plot_df['open'],
                high=plot_df['high'],
                low=plot_df['low'],
                close=plot_df['close'],
                increasing_line_color='red',
                decreasing_line_color='green',
                name='K线图'
            ),
            row=1, col=1
        )
        
        # 添加笔的线段
        fig.add_trace(
            go.Scatter(
                x=[df.loc[bi_start_idx, 'trade_date'], df.loc[bi_end_idx, 'trade_date']],
                y=[df.loc[bi_start_idx, 'close'], df.loc[bi_end_idx, 'close']],
                mode='lines',
                name='笔',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # 添加成交量图
        colors = ['red' if row['close'] >= row['open'] else 'green' for _, row in plot_df.iterrows()]
        fig.add_trace(
            go.Bar(
                x=plot_df['trade_date'],
                y=plot_df['volume'],
                name='成交量',
                marker_color=colors
            ),
            row=2, col=1
        )
        
        # 添加成交量均线（如果需要）
        if 'volume_ma20' in plot_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=plot_df['trade_date'],
                    y=plot_df['volume_ma20'],
                    name='成交量MA20',
                    line=dict(color='blue', width=1)
                ),
                row=2, col=1
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
            fillcolor="rgba(255,0,0,0.1)",  # 添加半透明填充
            row=1, col=1
        )
        
        # 更新布局
        fig.update_layout(
            title=dict(
                text=f'{code} {stock_name} - 模式分析',
                font=dict(size=20, family='Microsoft YaHei'),
                x=0.5,  # 居中标题
                xanchor='center'
            ),
            dragmode='zoom',  # 设置默认的拖动模式为框选缩放
            showlegend=True,
            legend=dict(
                font=dict(family='Microsoft YaHei', size=12),
                bgcolor='rgba(255,255,255,0.8)',  # 添加半透明背景
                bordercolor='rgba(0,0,0,0.3)',
                borderwidth=1
            ),
            height=800,  # 设置图表高度
            margin=dict(t=100, b=50, l=50, r=50),
            plot_bgcolor='white',  # 设置绘图区背景色为白色
            paper_bgcolor='white'  # 设置整体背景色为白色
        )
        
        # 更新Y轴
        fig.update_yaxes(
            title=dict(
                text='价格',
                font=dict(family='Microsoft YaHei', size=14)
            ),
            tickfont=dict(family='Microsoft YaHei'),
            fixedrange=False,  # 允许y轴缩放
            gridcolor='rgba(128,128,128,0.2)',  # 添加网格线
            row=1, col=1
        )
        
        # 更新X轴
        fig.update_xaxes(
            title=dict(
                text='日期',
                font=dict(family='Microsoft YaHei', size=14)
            ),
            rangeslider=dict(visible=True),
            type='category',
            tickangle=45,
            tickfont=dict(family='Microsoft YaHei'),
            fixedrange=False,  # 允许x轴缩放
            gridcolor='rgba(128,128,128,0.2)',  # 添加网格线
            row=2, col=1
        )
        
        # 更新成交量Y轴
        fig.update_yaxes(
            title=dict(
                text='成交量',
                font=dict(family='Microsoft YaHei', size=12)
            ),
            tickfont=dict(family='Microsoft YaHei', size=10)
        )
    
    def run_analysis(self, workers=30):
        """并行处理分析"""
        self.logger.info("开始分析数据...")
        files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        results = []
        
        from multiprocessing import Pool
        import multiprocessing
        optimal_workers = min(multiprocessing.cpu_count(), workers)
        
        with Pool(processes=optimal_workers) as pool:
            # 使用 imap 替代 submit 以减少内存占用
            for result in tqdm(pool.imap(self.process_single_stock, files), 
                            total=len(files), 
                            desc="分析进度"):
                if result:
                    results.extend(result if isinstance(result, list) else [result])
        
        # 创建DataFrame
        if results:
            result_df = pd.DataFrame([
                {
                    'code': item[0],
                    'hp_start': item[1],
                    'hp_end': item[2],
                    'start': item[3]['start'],
                    'end': item[3]['end'],
                    'start_date': item[3]['start_date'],
                    'end_date': item[3]['end_date'],
                    'now_index': item[3]['now_index'],
                    'bihigh': item[3]['bihigh'],
                    'bilow': item[3]['bilow'],
                    'angle': item[3]['angle'],
                    'time_ratio': item[3]['time_ratio'],
                    'price_ratio': item[3]['price_ratio'],
                    'score': item[3]['score'],
                    'max_drawdown': item[4],
                    'price_change': item[5],
                    'peak_price': item[6]
                } for item in results
            ])
            
            # 创建一个新列表示 now_index 与 end 的差值
            result_df['now_to_end'] = result_df['now_index'] - result_df['end']
            
            # 创建一个新列表示 start 与 end 的差值
            result_df['start_to_end'] = result_df['end'] - result_df['start']
            
            # 筛选满足条件的行
            result_df = result_df[result_df['now_to_end'] < 1.2 * result_df['start_to_end']]
            result_df = result_df[result_df['peak_price'] > result_df['bihigh']]
            
            # 计算回撤与涨幅比率
            result_df['drawdown_price_ratio'] = abs(result_df['max_drawdown'] / result_df['price_change'])
            
            # 保存结果
            result_df.to_csv("pattern_results.csv", index=False)
            self.logger.info(f"发现{len(result_df)}只符合形态的股票，结果已保存")
            return result_df
        else:
            self.logger.info("未发现符合形态的股票")
            return pd.DataFrame()  # 返回空DataFrame


