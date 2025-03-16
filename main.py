import os
import logging
import pandas as pd
from datetime import datetime

# 导入配置
from config.logging_config import setup_logging

# 导入模型
from models.physics_model import PhysicsFinanceModel
from models.market_pattern import MarketPatternAnalyzer
from models.backtest import BacktestSystem, run_backtest_analysis

# 导入扫描器
from scanner.pattern_scanner import PatternScanner

# 设置日志
loggers = setup_logging()
logger = logging.getLogger('Main')



def analyze_single_stock(stock_code='000300'):
    """分析单只股票"""
    try:
        # 获取数据
        logger.info(f"开始分析股票: {stock_code}")
        df = get_stock_data(stock_code)
        
        # 初始化分析器
        analyzer = MarketPatternAnalyzer(df, stock_code, loggers['pattern_logger'])
        
        # 生成交易信号
        df_with_signals = analyzer.generate_signals()
        
        # 运行回测
        backtest = run_backtest_analysis(df_with_signals)
        
        # 创建物理金融模型
        physics_model = PhysicsFinanceModel(logger=loggers['physics_logger'])
        
        # 分析市场状态
        market_state = physics_model.analyze(df)
        
        logger.info(f"市场状态: {market_state['market_state']}")
        logger.info(f"当前能量: {market_state['current_energy']:.4f}")
        logger.info(f"能量变化率: {market_state['energy_derivative']:.4f}")
        logger.info(f"市场弹性: {market_state['elasticity']:.4f}")
        logger.info(f"标准化ATR: {market_state['normalized_atr']:.4f}")
        
        return {
            'df': df_with_signals,
            'backtest': backtest,
            'market_state': market_state
        }
    except Exception as e:
        logger.error(f"分析过程中出错: {str(e)}")
        raise

def scan_market_patterns(workers=30):
    """扫描市场模式"""
    try:
        # 初始化扫描器
        scanner = PatternScanner(loggers['scanner_logger'])
        
        # 下载数据
        logger.info("开始下载股票数据...")
        scanner.download_all_data(workers=workers)
        
                # 分析模式
        logger.info("开始分析市场模式...") 
        results = scanner.run_analysis(workers=workers)
        
        logger.info(f"分析完成，发现{len(results)}个符合条件的模式")
        return results
    except Exception as e:
        logger.error(f"扫描过程中出错: {str(e)}")
        raise

def main():
    """主函数"""
    try:
        # 创建必要的目录
        os.makedirs("data", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        os.makedirs("analysis_cache", exist_ok=True)
        os.makedirs("charts", exist_ok=True)
        
        # 分析单只股票示例
        logger.info("=== 开始单只股票分析示例 ===")
        #result = analyze_single_stock('000300')  # 沪深300指数
        
        # 扫描市场模式示例
        # 注意：这个过程可能需要较长时间
        logger.info("\n=== 开始市场模式扫描示例 ===")
        logger.info("注意：完整的市场扫描可能需要较长时间")
        #response = input("是否执行完整的市场扫描？(y/n): ")
        response= 'y'
        if response == 'y':
            patterns = scan_market_patterns(workers=30)
            logger.info(f"扫描完成，结果已保存到 pattern_results.csv")
        else:
            logger.info("跳过市场扫描")
        
        logger.info("程序执行完成")
    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}")
        raise
def copy_matched_charts():
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

if __name__ == "__main__":
    #copy_matched_charts()
    import akshare as ak
    #下载000002行情
    #df = ak.stock_zh_a_hist(symbol="688693", period="daily", start_date="20200101", end_date="20231231", adjust="qfq")
    #print(df)
    #zzz
    main()