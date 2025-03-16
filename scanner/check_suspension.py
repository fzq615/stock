import akshare as ak
import pandas as pd
import logging
from datetime import datetime
import time
import random

def is_stock_suspended(stock_code, logger=None):
    """
    实时查询股票是否处于停牌状态
    
    参数:
    stock_code (str): 股票代码
    logger (logging.Logger, optional): 日志记录器
    
    返回:
    tuple: (是否停牌(bool), 停牌原因(str))
    """
    if logger is None:
        logger = logging.getLogger('suspension_checker')
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    # 转换股票代码格式
    if '.SZ' in stock_code:
        symbol = stock_code.replace('.SZ', '')
    elif '.SH' in stock_code:
        symbol = stock_code.replace('.SH', '')
    else:
        symbol = stock_code
    
    # 添加重试机制
    max_retries = 3
    for attempt in range(max_retries):
        try:
            logger.info(f"正在查询股票 {symbol} 的停牌状态...")
            
            # 使用akshare的stock_individual_info_em函数获取股票信息
            stock_info = ak.stock_individual_info_em(symbol=symbol)
            
            # 检查股票信息是否为空
            if stock_info.empty:
                logger.warning(f"股票 {symbol} 代码无效或不存在")
                return True, "股票代码无效或不存在"
            
            # 检查是否有交易状态列
            if '交易状态' in stock_info.columns:
                status = str(stock_info['交易状态'].values[0])
                logger.info(f"股票 {symbol} 当前交易状态: {status}")
                
                # 判断是否包含停牌关键词
                if '停牌' in status:
                    reason = status
                    logger.warning(f"股票 {symbol} 处于停牌状态: {reason}")
                    return True, reason
                else:
                    logger.info(f"股票 {symbol} 正常交易中")
                    return False, ""
            else:
                # 尝试通过其他方式判断
                # 如果能获取到最新交易数据，则认为未停牌
                try:
                    # 获取最近的交易日期，而不是使用当前日期
                    try:
                        # 使用上证指数获取最新交易日期
                        index_df = ak.stock_zh_index_daily_em(symbol='sh000001', start_date="20250101", end_date=datetime.now().strftime('%Y%m%d'))
                        latest_trade_date = pd.to_datetime(index_df['date'].iloc[-1]).strftime('%Y%m%d')
                        logger.info(f"获取到最新交易日期: {latest_trade_date}")
                    except Exception as e:
                        logger.warning(f"获取最新交易日期失败，使用当前日期: {str(e)}")
                        latest_trade_date = datetime.now().strftime('%Y%m%d')
                    
                    # 使用最新交易日期查询股票数据
                    df = ak.stock_zh_a_hist(symbol=symbol, 
                                          start_date=latest_trade_date,
                                          end_date=latest_trade_date,
                                          adjust="qfq")
                    if not df.empty:
                        logger.info(f"股票 {symbol} 最近交易日有交易数据，正常交易中")
                        return False, ""
                    else:
                        # 数据为空可能是停牌
                        logger.warning(f"股票 {symbol} 最近交易日无交易数据，可能处于停牌状态")
                        return True, "最近交易日无交易数据"
                except Exception as e:
                    logger.error(f"获取股票 {symbol} 交易数据时出错: {str(e)}")
                    if '停牌' in str(e):
                        return True, str(e)
            
            # 如果以上方法都无法确定，默认为未停牌
            return False, ""
            
        except Exception as e:
            logger.error(f"第{attempt+1}次查询股票 {symbol} 停牌状态失败: {str(e)}")
            if attempt < max_retries - 1:
                # 指数退避等待
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                logger.info(f"等待 {wait_time:.2f} 秒后重试...")
                time.sleep(wait_time)
            else:
                logger.error(f"查询股票 {symbol} 停牌状态失败，已达到最大重试次数")
                # 无法确定是否停牌，返回错误信息
                return True, f"查询失败: {str(e)}"
    
    return False, ""


def get_suspended_stocks(stock_codes, logger=None):
    """
    批量查询多个股票的停牌状态
    
    参数:
    stock_codes (list): 股票代码列表
    logger (logging.Logger, optional): 日志记录器
    
    返回:
    dict: 停牌股票字典 {股票代码: 停牌原因}
    """
    if logger is None:
        logger = logging.getLogger('suspension_checker')
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    suspended_stocks = {}
    
    for code in stock_codes:
        is_suspended, reason = is_stock_suspended(code, logger)
        if is_suspended:
            suspended_stocks[code] = reason
        # 添加延时避免频繁请求
        time.sleep(random.uniform(0.5, 1.5))
    
    return suspended_stocks


if __name__ == "__main__":
    # 示例用法
    logging.basicConfig(level=logging.INFO, 
                      format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('suspension_test')
    
    # 测试单个股票
    test_code = "000001"
    is_suspended, reason = is_stock_suspended(test_code, logger)
    print(f"股票 {test_code} 是否停牌: {is_suspended}, 原因: {reason}")
    
    # 测试多个股票
    test_codes = ["000001", "000002", "000004"]
    suspended = get_suspended_stocks(test_codes, logger)
    print(f"停牌股票列表: {suspended}")