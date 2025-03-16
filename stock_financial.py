import akshare as ak
import pandas as pd
from datetime import datetime
import time
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_stock_financial_data(stock_code):
    """
    获取股票的财务指标数据
    
    参数:
    stock_code (str): 股票代码
    
    返回:
    dict: 包含财务指标数据的字典
    """
    logger.info(f"获取股票 {stock_code} 的财务指标数据")
    
    try:
        # 获取财务分析指标
        financial_data = ak.stock_financial_analysis_indicator(symbol=stock_code)
        
        # 如果数据为空，尝试其他方法获取
        if financial_data.empty:
            logger.warning(f"使用stock_financial_analysis_indicator获取 {stock_code} 的财务数据为空，尝试其他方法")
            
            # 尝试获取财务报表数据
            try:
                # 获取最近的财务报表
                financial_report = ak.stock_financial_report_sina(symbol=stock_code)
                if not financial_report.empty:
                    logger.info(f"成功获取 {stock_code} 的财务报表数据")
                    financial_data = financial_report
            except Exception as e:
                logger.error(f"获取财务报表数据失败: {str(e)}")
            
            # 尝试获取财务摘要数据
            if financial_data.empty:
                try:
                    financial_abstract = ak.stock_financial_abstract(symbol=stock_code)
                    if not financial_abstract.empty:
                        logger.info(f"成功获取 {stock_code} 的财务摘要数据")
                        financial_data = financial_abstract
                except Exception as e:
                    logger.error(f"获取财务摘要数据失败: {str(e)}")
        
        # 如果仍然没有数据，尝试从东方财富获取市盈率等基本指标
        if financial_data.empty:
            logger.warning(f"无法获取 {stock_code} 的详细财务数据，尝试获取基本财务指标")
            try:
                # 获取实时行情数据，包含市盈率等
                stock_quote = ak.stock_zh_a_spot_em()
                stock_info = stock_quote[stock_quote['代码'] == stock_code]
                
                if not stock_info.empty:
                    logger.info(f"成功获取 {stock_code} 的基本财务指标")
                    # 提取关键财务指标
                    financial_indicators = {
                        "市盈率": float(stock_info['市盈率'].values[0]) if '市盈率' in stock_info.columns else None,
                        "市净率": float(stock_info['市净率'].values[0]) if '市净率' in stock_info.columns else None,
                        "总市值": float(stock_info['总市值'].values[0]) if '总市值' in stock_info.columns else None,
                        "流通市值": float(stock_info['流通市值'].values[0]) if '流通市值' in stock_info.columns else None,
                        "换手率": float(stock_info['换手率'].values[0]) if '换手率' in stock_info.columns else None,
                        "量比": float(stock_info['量比'].values[0]) if '量比' in stock_info.columns else None,
                    }
                    return {
                        "获取时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "基本财务指标": financial_indicators
                    }
            except Exception as e:
                logger.error(f"获取基本财务指标失败: {str(e)}")
        
        # 如果成功获取了详细财务数据
        if not financial_data.empty:
            logger.info(f"成功获取 {stock_code} 的财务数据，包含 {len(financial_data)} 行记录")
            
            # 转换为字典格式
            # 获取最近的财务数据（前5条记录）
            recent_data = financial_data.head(5).to_dict('records')
            
            return {
                "获取时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "财务指标数据": recent_data
            }
        
        # 如果所有尝试都失败
        logger.warning(f"无法获取 {stock_code} 的任何财务数据")
        return {
            "获取时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "财务指标数据": [],
            "基本财务指标": {
                "市盈率": None,
                "市净率": None,
                "总市值": None,
                "流通市值": None,
                "换手率": None,
                "量比": None
            },
            "错误信息": "无法获取财务数据"
        }
        
    except Exception as e:
        logger.error(f"获取财务指标数据时出错: {str(e)}")
        return {
            "获取时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "财务指标数据": [],
            "基本财务指标": {
                "市盈率": None,
                "市净率": None,
                "总市值": None,
                "流通市值": None,
                "换手率": None,
                "量比": None
            },
            "错误信息": str(e)
        }

# 测试函数
def test_financial_data():
    """
    测试获取财务数据的函数
    """
    test_stocks = ['000001', '600519', '300750']
    
    for stock in test_stocks:
        print(f"\n测试获取 {stock} 的财务数据:")
        result = get_stock_financial_data(stock)
        print(f"结果类型: {type(result)}")
        print(f"结果内容: {result}")
        time.sleep(1)  # 避免频繁请求

if __name__ == "__main__":
    test_financial_data()