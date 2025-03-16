import os
import sys
import logging
from scanner.check_suspension import is_stock_suspended, get_suspended_stocks

# 配置日志
logging.basicConfig(level=logging.INFO, 
                  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('suspension_checker')

def check_single_stock():
    """检查单个股票的停牌状态"""
    stock_code = input("请输入要查询的股票代码: ")
    print(f"正在查询股票 {stock_code} 的停牌状态...")
    
    is_suspended, reason = is_stock_suspended(stock_code, logger)
    
    if is_suspended:
        print(f"股票 {stock_code} 当前处于停牌状态")
        print(f"停牌原因: {reason}")
    else:
        print(f"股票 {stock_code} 当前正常交易中")

def check_multiple_stocks():
    """批量检查多个股票的停牌状态"""
    stock_codes_input = input("请输入要查询的股票代码列表(用逗号分隔): ")
    stock_codes = [code.strip() for code in stock_codes_input.split(',')]
    
    print(f"正在查询 {len(stock_codes)} 只股票的停牌状态...")
    suspended_stocks = get_suspended_stocks(stock_codes, logger)
    
    if suspended_stocks:
        print(f"发现 {len(suspended_stocks)} 只停牌股票:")
        for code, reason in suspended_stocks.items():
            print(f"  {code}: {reason}")
    else:
        print("所有查询的股票均正常交易中")

def main():
    print("===== 股票停牌状态查询工具 =====")
    print("1. 查询单个股票")
    print("2. 批量查询多个股票")
    print("0. 退出")
    
    choice = input("请选择功能(0-2): ")
    
    if choice == '1':
        check_single_stock()
    elif choice == '2':
        check_multiple_stocks()
    elif choice == '0':
        print("谢谢使用，再见!")
        sys.exit(0)
    else:
        print("无效的选择，请重新输入")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序运行出错: {str(e)}")