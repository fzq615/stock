import os
import sys
import time
from enhanced_financial_data import get_enhanced_financial_data, save_enhanced_financial_data

def test_enhanced_financial_data():
    """
    测试获取增强财务数据的函数
    """
    # 测试多个不同类型的股票
    test_stocks = ['000001', '600519', '300750', '002415', '601318']
    output_dir = "E:\\20250305-bi\\stock_info"
    
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("开始测试增强财务数据获取功能...\n")
    
    for stock in test_stocks:
        print(f"\n{'='*50}")
        print(f"测试获取 {stock} 的增强财务数据:")
        
        # 获取增强财务数据
        result = get_enhanced_financial_data(stock)
        
        # 打印结果
        print(f"\n基本财务指标:")
        if '基本财务指标' in result:
            for key, value in result['基本财务指标'].items():
                print(f"{key}: {value}")
        
        print(f"\n季度利润增速:")
        if '季度利润增速' in result:
            for key, value in result['季度利润增速'].items():
                print(f"{key}: {value}")
        
        # 如果有错误信息，打印出来
        if '错误信息' in result:
            print(f"\n错误信息: {result['错误信息']}")
        
        # 保存数据
        save_result = save_enhanced_financial_data(stock, result, output_dir)
        print(f"\n保存结果: {'成功' if save_result else '失败'}")
        
        # 避免频繁请求
        time.sleep(2)
    
    print("\n测试完成!")

def main():
    test_enhanced_financial_data()

if __name__ == "__main__":
    main()