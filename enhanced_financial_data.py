import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime
import time
import logging
import requests
from bs4 import BeautifulSoup
import re
import json

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_enhanced_financial_data(stock_code):
    """
    获取股票的增强财务数据，包括市盈率和近三个季度利润同比增速
    
    参数:
    stock_code (str): 股票代码
    
    返回:
    dict: 包含增强财务指标数据的字典
    """
    logger.info(f"获取股票 {stock_code} 的增强财务数据")
    
    # 初始化结果字典
    result = {
        "获取时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "基本财务指标": {
            "市盈率": None,
            "市净率": None,
            "总市值": None,
            "流通市值": None,
            "换手率": None,
            "量比": None
        },
        "季度利润增速": {
            "最近一季度利润同比增速": None,
            "最近二季度利润同比增速": None,
            "最近三季度利润同比增速": None
        },
        "错误信息": ""
    }
    
    try:
        # 1. 获取市盈率等基本财务指标
        try:
            # 确保股票代码格式正确（去除前缀）
            clean_code = stock_code.lstrip("SH").lstrip("SZ").lstrip("sh").lstrip("sz")
            # 确保股票代码是6位数字
            clean_code = clean_code.zfill(6)
            
            logger.info(f"尝试获取股票 {clean_code} 的基本财务指标")
            
            # 方法1: 使用stock_zh_a_spot_em获取实时行情数据
            try:
                stock_quote = ak.stock_zh_a_spot_em()
                # 尝试多种方式匹配股票代码
                stock_info = stock_quote[stock_quote['代码'] == clean_code]
                
                if not stock_info.empty:
                    logger.info(f"成功获取 {clean_code} 的基本财务指标")
                    # 提取关键财务指标
                    result["基本财务指标"] = {
                        "市盈率": float(stock_info['市盈率'].values[0]) if '市盈率' in stock_info.columns and pd.notna(stock_info['市盈率'].values[0]) else None,
                        "市净率": float(stock_info['市净率'].values[0]) if '市净率' in stock_info.columns and pd.notna(stock_info['市净率'].values[0]) else None,
                        "总市值": float(stock_info['总市值'].values[0]) if '总市值' in stock_info.columns and pd.notna(stock_info['总市值'].values[0]) else None,
                        "流通市值": float(stock_info['流通市值'].values[0]) if '流通市值' in stock_info.columns and pd.notna(stock_info['流通市值'].values[0]) else None,
                        "换手率": float(stock_info['换手率'].values[0]) if '换手率' in stock_info.columns and pd.notna(stock_info['换手率'].values[0]) else None,
                        "量比": float(stock_info['量比'].values[0]) if '量比' in stock_info.columns and pd.notna(stock_info['量比'].values[0]) else None,
                    }
                    return result
            except Exception as e:
                logger.error(f"使用stock_zh_a_spot_em获取基本财务指标失败: {str(e)}")
            
            # 方法2: 尝试使用stock_individual_info_em获取个股信息
            try:
                logger.info(f"尝试使用stock_individual_info_em获取 {clean_code} 的基本财务指标")
                stock_info = ak.stock_individual_info_em(symbol=clean_code)
                
                if not stock_info.empty:
                    # 将数据转换为字典格式
                    info_dict = dict(zip(stock_info['item'], stock_info['value']))
                    
                    # 提取关键财务指标
                    result["基本财务指标"] = {
                        "市盈率": float(info_dict.get('市盈率(动态)', '').replace('--', '0')) if '市盈率(动态)' in info_dict and info_dict.get('市盈率(动态)') != '--' else None,
                        "市净率": float(info_dict.get('市净率', '').replace('--', '0')) if '市净率' in info_dict and info_dict.get('市净率') != '--' else None,
                        "总市值": float(info_dict.get('总市值', '').replace('--', '0').replace('亿', '')) * 100000000 if '总市值' in info_dict and info_dict.get('总市值') != '--' else None,
                        "流通市值": float(info_dict.get('流通市值', '').replace('--', '0').replace('亿', '')) * 100000000 if '流通市值' in info_dict and info_dict.get('流通市值') != '--' else None,
                        "换手率": float(info_dict.get('换手率', '').replace('--', '0').replace('%', '')) if '换手率' in info_dict and info_dict.get('换手率') != '--' else None,
                        "量比": float(info_dict.get('量比', '').replace('--', '0')) if '量比' in info_dict and info_dict.get('量比') != '--' else None,
                    }
                    logger.info(f"成功使用stock_individual_info_em获取 {clean_code} 的基本财务指标")
            except Exception as e:
                logger.error(f"使用stock_individual_info_em获取基本财务指标失败: {str(e)}")
            
            # 方法3: 尝试使用stock_zh_a_hist获取历史数据中的部分指标
            try:
                logger.info(f"尝试使用stock_zh_a_hist获取 {clean_code} 的基本财务指标")
                end_date = datetime.now().strftime("%Y%m%d")
                start_date = (datetime.now() - pd.Timedelta(days=7)).strftime("%Y%m%d")
                hist_data = ak.stock_zh_a_hist(symbol=clean_code, period="daily", start_date=start_date, end_date=end_date, adjust="")
                
                if not hist_data.empty:
                    latest_data = hist_data.iloc[-1]
                    result["基本财务指标"]["换手率"] = float(latest_data['换手率']) if '换手率' in latest_data.index and pd.notna(latest_data['换手率']) else None
                    logger.info(f"成功使用stock_zh_a_hist获取 {clean_code} 的部分基本财务指标")
            except Exception as e:
                logger.error(f"使用stock_zh_a_hist获取基本财务指标失败: {str(e)}")
                # 继续执行，不中断流程
        except Exception as e:
            logger.error(f"获取基本财务指标失败: {str(e)}")
            result["错误信息"] += f"获取基本财务指标失败: {str(e)}\n"
        
        # 2. 获取季度利润同比增速数据
        try:
            # 确保股票代码格式正确
            clean_code = stock_code.lstrip("SH").lstrip("SZ").lstrip("sh").lstrip("sz").zfill(6)
            
            # 方法1: 尝试使用stock_financial_abstract获取财务摘要数据
            try:
                logger.info(f"尝试使用stock_financial_abstract获取 {clean_code} 的财务摘要数据")
                financial_abstract = ak.stock_financial_abstract(symbol=clean_code)
                
                if not financial_abstract.empty:
                    logger.info(f"成功获取 {clean_code} 的财务摘要数据")
                    # 提取净利润增长率数据
                    if '净利润同比增长率' in financial_abstract.columns:
                        recent_data = financial_abstract.sort_values(by='截止日期', ascending=False).head(3)
                        if len(recent_data) >= 1:
                            result["季度利润增速"]["最近一季度利润同比增速"] = recent_data.iloc[0]['净利润同比增长率']
                        if len(recent_data) >= 2:
                            result["季度利润增速"]["最近二季度利润同比增速"] = recent_data.iloc[1]['净利润同比增长率']
                        if len(recent_data) >= 3:
                            result["季度利润增速"]["最近三季度利润同比增速"] = recent_data.iloc[2]['净利润同比增长率']
            except Exception as e:
                logger.error(f"使用stock_financial_abstract获取财务摘要数据失败: {str(e)}")
            
            # 方法2: 使用stock_financial_report_sina获取财务报表数据
            try:
                logger.info(f"尝试使用stock_financial_report_sina获取 {clean_code} 的财务报表数据")
                financial_data = ak.stock_financial_report_sina(symbol=clean_code)
                
                if not financial_data.empty:
                    logger.info(f"成功获取 {clean_code} 的财务报表数据")
                    # 处理财务数据，计算季度利润同比增速
                    if '净利润同比增长率' in financial_data.columns:
                        recent_data = financial_data.sort_values(by='报表日期', ascending=False).head(3)
                        if len(recent_data) >= 1:
                            result["季度利润增速"]["最近一季度利润同比增速"] = recent_data.iloc[0]['净利润同比增长率']
                        if len(recent_data) >= 2:
                            result["季度利润增速"]["最近二季度利润同比增速"] = recent_data.iloc[1]['净利润同比增长率']
                        if len(recent_data) >= 3:
                            result["季度利润增速"]["最近三季度利润同比增速"] = recent_data.iloc[2]['净利润同比增长率']
            except Exception as e:
                logger.error(f"使用stock_financial_report_sina获取财务报表数据失败: {str(e)}")
            
            # 方法3: 尝试从东方财富网获取季度利润增速数据
            try:
                logger.info(f"尝试从东方财富网获取 {clean_code} 的季度利润增速数据")
                growth_rates = get_profit_growth_from_eastmoney(clean_code)
                if growth_rates and (growth_rates["最近一季度利润同比增速"] is not None or 
                                    growth_rates["最近二季度利润同比增速"] is not None or 
                                    growth_rates["最近三季度利润同比增速"] is not None):
                    # 只有当至少有一个季度数据不为None时才更新
                    result["季度利润增速"] = growth_rates
                    logger.info(f"成功从东方财富网获取 {clean_code} 的季度利润增速数据")
            except Exception as e:
                logger.error(f"从东方财富网获取季度利润增速数据失败: {str(e)}")
                
            # 检查是否所有季度利润增速数据都为None
            all_none = all(v is None for v in result["季度利润增速"].values())
            if all_none:
                # 尝试使用模拟数据（仅用于演示）
                logger.warning(f"无法获取 {clean_code} 的真实季度利润增速数据，使用模拟数据")
                result["季度利润增速"] = {
                    "最近一季度利润同比增速": "获取失败 - API可能暂时不可用",
                    "最近二季度利润同比增速": "获取失败 - API可能暂时不可用",
                    "最近三季度利润同比增速": "获取失败 - API可能暂时不可用"
                }
                result["数据说明"] = "由于API限制或网络问题，无法获取真实财务数据。建议稍后再试或使用其他数据源。"
        except Exception as e:
            logger.error(f"获取季度利润增速数据失败: {str(e)}")
            result["错误信息"] += f"获取季度利润增速数据失败: {str(e)}\n"
        
        # 如果没有错误信息，则清空错误字段
        if not result["错误信息"]:
            result.pop("错误信息")
        
        return result
        
    except Exception as e:
        logger.error(f"获取增强财务数据时出错: {str(e)}")
        result["错误信息"] = str(e)
        return result

def get_profit_growth_from_eastmoney(stock_code):
    """
    从东方财富网获取股票的季度利润同比增速数据
    
    参数:
    stock_code (str): 股票代码
    
    返回:
    dict: 包含近三个季度利润同比增速的字典
    """
    result = {
        "最近一季度利润同比增速": None,
        "最近二季度利润同比增速": None,
        "最近三季度利润同比增速": None
    }
    
    try:
        # 确保股票代码格式正确（去除前缀）
        clean_code = stock_code.lstrip("SH").lstrip("SZ").lstrip("sh").lstrip("sz")
        
        # 构建东方财富网财务数据URL
        # 根据股票代码判断市场类型
        market = "SH" if clean_code.startswith(('6', '9')) else "SZ"
        url = f"https://emweb.securities.eastmoney.com/PC_HSF10/NewFinanceAnalysis/Index?type=web&code={market}{clean_code.zfill(6)}"
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        # 发送请求获取页面内容
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            # 使用BeautifulSoup解析HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 查找包含财务数据的脚本标签
            script_tags = soup.find_all('script')
            financial_data = None
            
            # 在脚本中查找财务数据
            for script in script_tags:
                if script.string and "var defjson =" in script.string:
                    # 提取JSON数据
                    json_str = re.search(r'var defjson = (.*?);', script.string)
                    if json_str:
                        try:
                            financial_data = json.loads(json_str.group(1))
                            break
                        except:
                            continue
            
            # 如果找到财务数据，提取季度利润增速
            if financial_data and 'REPORT_DATAYEAR_GROWTH' in financial_data:
                growth_data = financial_data['REPORT_DATAYEAR_GROWTH']
                if isinstance(growth_data, list) and len(growth_data) > 0:
                    # 提取净利润同比增长率
                    for item in growth_data:
                        if '净利润同比增长率' in item or '净利润增长率' in item:
                            key = '净利润同比增长率' if '净利润同比增长率' in item else '净利润增长率'
                            values = item[key]
                            
                            # 提取最近三个季度的数据
                            if len(values) >= 1:
                                result["最近一季度利润同比增速"] = values[0]
                            if len(values) >= 2:
                                result["最近二季度利润同比增速"] = values[1]
                            if len(values) >= 3:
                                result["最近三季度利润同比增速"] = values[2]
                            break
        
        return result
    except Exception as e:
        logger.error(f"从东方财富网获取季度利润增速数据失败: {str(e)}")
        return result

def save_enhanced_financial_data(stock_code, financial_data, output_dir):
    """
    将增强财务数据保存到文件
    
    参数:
    stock_code (str): 股票代码
    financial_data (dict): 财务数据字典
    output_dir (str): 输出目录
    """
    import os
    from datetime import datetime
    
    if financial_data is None:
        logger.warning(f"股票 {stock_code} 的增强财务数据为空，跳过保存")
        return False
    
    try:
        # 确保股票代码格式正确
        stock_code = stock_code.strip().zfill(6)
        
        # 创建文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"enhanced_financial_info_{stock_code}_{timestamp}.txt"
        filepath = os.path.join(output_dir, filename)
        
        # 确保输出目录存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 将数据写入文件
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"股票增强财务信息 - 代码: {stock_code}\n")
            f.write(f"获取时间: {financial_data.get('获取时间', '未知')}\n")
            f.write("="*50 + "\n\n")
            
            # 写入基本财务指标
            if '基本财务指标' in financial_data:
                f.write("基本财务指标:\n")
                for key, value in financial_data['基本财务指标'].items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")
            
            # 写入季度利润增速数据
            if '季度利润增速' in financial_data:
                f.write("季度利润增速:\n")
                for key, value in financial_data['季度利润增速'].items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")
            
            # 写入错误信息（如果有）
            if '错误信息' in financial_data and financial_data['错误信息']:
                f.write("错误信息:\n")
                f.write(f"{financial_data['错误信息']}\n")
        
        logger.info(f"股票 {stock_code} 的增强财务数据已保存到 {filepath}")
        return True
    
    except Exception as e:
        logger.error(f"保存股票 {stock_code} 的增强财务数据失败: {str(e)}")
        return False

# 测试函数
def test_enhanced_financial_data():
    """
    测试获取增强财务数据的函数
    """
    test_stocks = ['000001', '600519', '300750']
    output_dir = "E:\\20250305-bi\\stock_info"
    
    for stock in test_stocks:
        print(f"\n测试获取 {stock} 的增强财务数据:")
        result = get_enhanced_financial_data(stock)
        print(f"结果类型: {type(result)}")
        print(f"结果内容: {result}")
        
        # 保存数据
        save_enhanced_financial_data(stock, result, output_dir)
        
        time.sleep(1)  # 避免频繁请求

if __name__ == "__main__":
    test_enhanced_financial_data()