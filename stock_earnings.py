import akshare as ak
import pandas as pd
import os
from datetime import datetime
import logging
import time

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_stock_earnings_report(stock_code):
    """
    获取股票业绩报表数据
    
    参数:
    stock_code (str): 股票代码
    
    返回:
    pandas.DataFrame: 业绩报表数据
    """
    try:
        # 确保股票代码格式正确
        stock_code = stock_code.strip().zfill(6)
        logger.info(f"正在获取股票 {stock_code} 的业绩报表数据...")
        
        # 获取业绩报表数据
        df = ak.stock_yjbb_em(symbol=stock_code)
        
        if df.empty:
            logger.warning(f"股票 {stock_code} 没有业绩报表数据")
            return None
            
        logger.info(f"成功获取股票 {stock_code} 的业绩报表数据")
        return df
    except Exception as e:
        logger.error(f"获取股票 {stock_code} 的业绩报表数据失败: {str(e)}")
        return None

def get_stock_earnings_express(stock_code):
    """
    获取股票业绩快报数据
    
    参数:
    stock_code (str): 股票代码
    
    返回:
    pandas.DataFrame: 业绩快报数据
    """
    try:
        # 确保股票代码格式正确
        stock_code = stock_code.strip().zfill(6)
        logger.info(f"正在获取股票 {stock_code} 的业绩快报数据...")
        
        # 获取业绩快报数据
        df = ak.stock_yjkb_em(symbol=stock_code)
        
        if df.empty:
            logger.warning(f"股票 {stock_code} 没有业绩快报数据")
            return None
            
        logger.info(f"成功获取股票 {stock_code} 的业绩快报数据")
        return df
    except Exception as e:
        logger.error(f"获取股票 {stock_code} 的业绩快报数据失败: {str(e)}")
        return None

def get_stock_earnings_forecast(stock_code):
    """
    获取股票业绩预告数据
    
    参数:
    stock_code (str): 股票代码
    
    返回:
    pandas.DataFrame: 业绩预告数据
    """
    try:
        # 确保股票代码格式正确
        stock_code = stock_code.strip().zfill(6)
        logger.info(f"正在获取股票 {stock_code} 的业绩预告数据...")
        
        # 获取业绩预告数据
        df = ak.stock_yjyg_em(symbol=stock_code)
        
        if df.empty:
            logger.warning(f"股票 {stock_code} 没有业绩预告数据")
            return None
            
        logger.info(f"成功获取股票 {stock_code} 的业绩预告数据")
        return df
    except Exception as e:
        logger.error(f"获取股票 {stock_code} 的业绩预告数据失败: {str(e)}")
        return None

def save_earnings_data(stock_code, data_type, data, output_dir):
    """
    保存财务数据到CSV文件
    
    参数:
    stock_code (str): 股票代码
    data_type (str): 数据类型（'report', 'express', 'forecast'）
    data (pandas.DataFrame): 要保存的数据
    output_dir (str): 输出目录
    """
    if data is None or data.empty:
        return False
        
    try:
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{stock_code}_{data_type}_{timestamp}.csv"
        filepath = os.path.join(output_dir, filename)
        
        # 保存数据到CSV文件
        data.to_csv(filepath, encoding='utf-8-sig', index=False)
        logger.info(f"成功保存{data_type}数据到文件: {filepath}")
        return True
    except Exception as e:
        logger.error(f"保存{data_type}数据失败: {str(e)}")
        return False

def get_and_save_all_earnings_data(stock_code, output_dir):
    """
    获取并保存股票的所有财务数据
    
    参数:
    stock_code (str): 股票代码
    output_dir (str): 输出目录
    
    返回:
    bool: 是否成功保存所有数据
    """
    success = True
    
    # 获取并保存业绩报表数据
    report_data = get_stock_earnings_report(stock_code)
    if not save_earnings_data(stock_code, 'report', report_data, output_dir):
        success = False
    
    # 获取并保存业绩快报数据
    express_data = get_stock_earnings_express(stock_code)
    if not save_earnings_data(stock_code, 'express', express_data, output_dir):
        success = False
    
    # 获取并保存业绩预告数据
    forecast_data = get_stock_earnings_forecast(stock_code)
    if not save_earnings_data(stock_code, 'forecast', forecast_data, output_dir):
        success = False
    
    return success

def main():
    # 示例：获取并保存单个股票的财务数据
    stock_code = '000001'
    output_dir = 'stock_earnings_data'
    
    if get_and_save_all_earnings_data(stock_code, output_dir):
        logger.info(f"成功获取并保存股票 {stock_code} 的所有财务数据")
    else:
        logger.warning(f"部分数据获取或保存失败")

if __name__ == '__main__':
    main()