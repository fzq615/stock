import logging
import os
from datetime import datetime

# 创建logs目录
def setup_logging():
    """设置日志配置"""
    # 生成带时间戳的日志文件名
    log_filename = f"logs/trading_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # 创建logs目录
    os.makedirs('logs', exist_ok=True)
    
    # 配置基本日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )
    
    # 创建各模块的logger
    physics_logger = logging.getLogger('PhysicsFinanceModel')
    backtest_logger = logging.getLogger('BacktestEngine')
    pattern_logger = logging.getLogger('MarketPatternAnalyzer')
    scanner_logger = logging.getLogger('PatternScanner')
    
    return {
        'physics_logger': physics_logger,
        'backtest_logger': backtest_logger,
        'pattern_logger': pattern_logger,
        'scanner_logger': scanner_logger
    }