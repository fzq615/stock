from  akshare import option_shfe_daily

import akshare as ak
import re
import pandas as pd
# 策略参数字典


# 定义交易所映射
CZCE=["白糖期权", "棉花期权", "甲醇期权", "PTA期权",  "菜籽粕期权", "动力煤期权", "短纤期权",
    "菜籽油期权", "花生期权", "纯碱期权", "锰硅期权", "硅铁期权", "尿素期权", "对二甲苯期权", "苹果期权", "红枣期权",
    "烧碱期权", "玻璃期权"]
SHFE=['沪铝期权', '天然橡胶期权', '沪铜期权', '黄金期权', '白银期权', '螺纹钢期权','沪锌期权','铅期权','镍期权','锡期权','氧化铝期权','丁二烯橡胶期权' ]

DEC=["玉米期权", "豆粕期权", "铁矿石期权", "液化石油气期权", "聚乙烯期权", "聚氯乙烯期权",
    "聚丙烯期权", "棕榈油期权", "黄大豆1号期权", "黄大豆2号期权", "豆油期权", "乙二醇期权", "苯乙烯期权",
    "鸡蛋期权", "玉米淀粉期权", "生猪期权"]
GFEX=['工业硅期权', '碳酸锂期权']
symbol_list = []
#symbol_list.extend(CZCE)
#symbol_list.extend(SHFE)
symbol_list.extend(DEC)
symbol_list.extend(GFEX)
exchange_mapping = {}
# 遍历四个列表，创建或更新exchange_mapping字典
for symbol in CZCE:
    if symbol not in exchange_mapping:
        exchange_mapping[symbol] = "CZCE"

for symbol in SHFE:
    if symbol not in exchange_mapping:
        exchange_mapping[symbol] = "SHFE"

for symbol in DEC:
    if symbol not in exchange_mapping:
        exchange_mapping[symbol] = "DCE"

for symbol in GFEX:
    if symbol not in exchange_mapping:
        exchange_mapping[symbol] = "GFEX"

# 输出最终的exchange_mapping字典
print("exchange_mapping:", exchange_mapping)

product_code_mapping = {
    "沪铝期权": "AL",
    "豆粕期权": "M",
    "玉米期权": "C",
    "铁矿石期权": "I",
    "棉花期权": "CF",
    "白糖期权": "SR",
    "PTA期权": "TA",
    "菜籽油期权": "OI",
    "花生期权": "PK",
    "甲醇期权": "MA",
    "天然橡胶期权": "RU",
    "沪铜期权": "CU",
    "黄金期权": "AU",
    "菜籽粕期权": "RM",
    "液化石油气期权": "PG",
    "动力煤期权": "ZC",
    "黄大豆1号期权": "A",
    "黄大豆2号期权": "B",
    "豆油期权": "Y",
    "白银期权": "AG",
    "螺纹钢期权": "RB",
    "工业硅期权": "SI",
    "乙二醇期权": "EG",
    "苯乙烯期权": "EB",
    "碳酸锂期权": "LC",
    "丁二烯橡胶期权": "BR",

    "二甲苯期权": "PX",
    "烧碱期权": "PVC"
}

def get_option_commodity_contract(symbol):

        # 获取该期权的所有合约
    option_commodity_contract_sina_df = ak.option_commodity_contract_sina(symbol=symbol)

    # 检查是否有合约数据
    if not option_commodity_contract_sina_df.empty:
        # 遍历每个合约
        for index, row in option_commodity_contract_sina_df.iterrows():
            contract = row['合约']  # 获取合约名称

            # 获取该合约的具体期权信息
            try:
                option_commodity_contract_table_sina_df = ak.option_commodity_contract_table_sina(symbol=symbol,
                                                                                                  contract=contract)

                def convert_contract_code(code, exchange):
                    match = re.search(r'(\d{4,6})(C|P)(\d+)', code)
                    if match:
                        year_month = match.group(1)
                        option_type = match.group(2)
                        strike_price = match.group(3)
                        if exchange == "ZCE":
                            LogInfo(code, 'year_month', code, year_month, 'zce', year_month[1:])
                            return f"{year_month[1:]}{option_type}{strike_price}"
                        else:
                            LogInfo(code, 'year_month', year_month, '-qt-', year_month)
                            return f"{year_month}{option_type}{strike_price}"
                    return code

                # 转换看涨合约代码
                if '看涨合约-看涨期权合约' in option_commodity_contract_table_sina_df.columns:
                    LogInfo('xxxx')
                    option_commodity_contract_table_sina_df['看涨合约-看涨期权合约'] = \
                    option_commodity_contract_table_sina_df['看涨合约-看涨期权合约'].apply(
                        lambda
                            x: f"{exchange_mapping[symbol]}|O|{product_code_mapping[symbol]}|{convert_contract_code(x, exchange_mapping[symbol])}"
                    )
                    LogInfo(option_commodity_contract_table_sina_df['看涨合约-看涨期权合约'])
                if '看跌合约-看跌期权合约' in option_commodity_contract_table_sina_df.columns:
                    LogInfo('xxx1x')
                    option_commodity_contract_table_sina_df['看跌合约-看跌期权合约'] = \
                    option_commodity_contract_table_sina_df['看跌合约-看跌期权合约'].apply(
                        lambda
                            x: f"{exchange_mapping[symbol]}|O|{product_code_mapping[symbol]}|{convert_contract_code(x, exchange_mapping[symbol])}"
                    )
                    # 打印看涨或看跌期权合约信息
                    LogInfo(option_commodity_contract_table_sina_df['看跌合约-看跌期权合约'])

            except Exception as e:
                LogInfo(f"Failed to fetch data for {symbol} with contract {contract}: {e}")
    else:
        LogInfo(f"No contracts found for symbol: {symbol}")

def get_amount(exchange_mapping, start_date):
    # 初始化空字典来存储每个交易所的期权名称
    #对于exchange_mapping中的每个键值对，分别执行相关函数，SHFE的执行option_shfe_daily，CZCE执行option_czce_daily,DCE执行option_dce_daily,GFEX执行option_gfex_daily
    option_names = {}
    option_list =[]

    def replace_contract_prefix_with_commodity(row):
        contract_name = row['合约代码']
        commodity_name = row['交易所']

        # 找到第一个数字的位置
        for i, char in enumerate(contract_name):
            if char.isdigit():
                #print('xxq')
                break
        else:  # 如果没有找到数字，则整个字符串都保持原样
            #print('x1xq')
            return contract_name

        # 替换前缀并返回新的合约名称
        return commodity_name+"|"+"O" +"|"+contract_name[0:i]+"|"+ contract_name[i:]
    # 应用替换函数到每一行的'合约名称'列
    #建立无数据列表
    no_obj=[]

    for symbol, exchange in exchange_mapping.items():
         print(symbol, exchange)

         if exchange == "SHFE":
            try:
                def update_symbol(symbol):
                    # 创建一个字典，用于映射需要移除"沪"前缀的商品名称
                    commodity_mapping = {
                        "沪铝期权": "铝期权",
                        "沪锌期权": "锌期权",
                        "沪铜期权": "铜期权",
                        "沪铅期权": "铅期权",
                        "沪镍期权": "镍期权",
                        "沪锡期权": "锡期权",
                        "天然橡胶期权": "天胶期权",
                    }
                    # 检查symbol是否在字典中，如果在则返回映射后的值，否则返回原symbol
                    return commodity_mapping.get(symbol, symbol)

                symbol = update_symbol(symbol)
                option_shfe_daily_one, option_shfe_daily_two= ak.option_shfe_daily(symbol=symbol, trade_date=start_date)
                print('xx',option_shfe_daily_one)
                option_shfe_daily_one.columns = option_shfe_daily_one.columns.str.strip()
                #option_list.append(option_shfe_daily_one[['合约代码', '成交额']])

                #将option_shfe_daily_one['合约名称']的值第一个数字前的字母替换成row['商品名称']


                print('xx2',option_shfe_daily_one.columns)
                option_shfe_daily_one['交易所']= exchange
                option_shfe_daily_one['商品名称']= symbol
                print('xx3',option_shfe_daily_one.columns)
                option_shfe_daily_one['合约代码'] = option_shfe_daily_one.apply(replace_contract_prefix_with_commodity, axis=1)

                option_list.append(option_shfe_daily_one[['商品名称','合约代码', '成交额','交易所']])

            except Exception as e:
                print(f"Failed to fetch data for {symbol} with contract ")
                no_obj.append(symbol)
         elif exchange == "CZCE":

             try:
                option_shfe_daily_one= ak.option_czce_daily(symbol=symbol, trade_date=start_date)
                # 去掉列名中的多余空格
                option_shfe_daily_one.columns = option_shfe_daily_one.columns.str.strip()

                print('xx1',option_shfe_daily_one.columns )
                print(option_shfe_daily_one[['合约代码', '成交额(万元)']])
                option_shfe_daily_one.rename(columns={'成交额(万元)': '成交额'}, inplace=True)
                option_shfe_daily_one['交易所']= exchange

                option_shfe_daily_one['商品名称']= symbol
                option_shfe_daily_one['合约代码'] = option_shfe_daily_one.apply(replace_contract_prefix_with_commodity,axis=1)
                print(option_shfe_daily_one.columns )

                option_list.append(option_shfe_daily_one[['商品名称','合约代码', '成交额','交易所']])
                #将'成交额(万元)'改成“成交额”

                #判断option_shfe_daily_one的列名是否包括“合约代码”，因为有可能是“合约代码   ”
             except Exception as e:
                 print(f"Failed to fetch data for {symbol} with contract ")
                 no_obj.append(symbol)
         elif exchange == "DCE":
             try:
                option_shfe_daily_one, option_shfe_daily_two= ak.option_dce_daily(symbol=symbol, trade_date=start_date)
                option_shfe_daily_one.columns = option_shfe_daily_one.columns.str.strip()
                option_shfe_daily_one['合约名称（修改后）'] = option_shfe_daily_one.apply(
                    lambda row: f"{row['合约名称'].replace('-', '')}", axis=1)
                print('zz',option_shfe_daily_one)
                #option_shfe_daily_one['合约名称（修改后）'] = option_shfe_daily_one.apply(
                    #lambda row: f"{row['合约名称（修改后）'].replace('c', row['商品名称'])}", axis=1)
                #把option_shfe_daily_one['合约名称（修改后）']rename成'合约代码'
                option_shfe_daily_one.rename(columns={'合约名称（修改后）': '合约代码'}, inplace=True)
                option_shfe_daily_one['交易所']= exchange
                option_shfe_daily_one['合约代码'] = option_shfe_daily_one.apply(replace_contract_prefix_with_commodity,
                                                                                axis=1)
                print('qq',option_shfe_daily_one)

                option_list.append(option_shfe_daily_one[['商品名称','合约代码', '成交额','交易所']])
             except Exception as e:
                print(f"Failed to fetch data for {symbol} with contract ")
                no_obj.append(symbol)
         elif exchange == "GFEX":
             try:
                 #if symbol为碳酸锂期权，则改为碳酸锂
                symbol = "碳酸锂" if symbol == "碳酸锂期权" else symbol
                symbol = "工业硅" if symbol == "工业硅期权" else symbol
                option_shfe_daily_one= ak.option_gfex_daily(symbol=symbol, trade_date=start_date)
                option_shfe_daily_one.columns = option_shfe_daily_one.columns.str.strip()
                print(option_shfe_daily_one)
                option_shfe_daily_one['合约名称（修改后）'] = option_shfe_daily_one.apply(
                 lambda row: f"{row['合约名称'].replace('-', '')}", axis=1)
                print(option_shfe_daily_one)

                # 把option_shfe_daily_one['合约名称（修改后）']rename成'合约代码'
                option_shfe_daily_one.rename(columns={'合约名称（修改后）': '合约代码'}, inplace=True)
                option_shfe_daily_one['交易所']= exchange
                option_shfe_daily_one['合约代码'] = option_shfe_daily_one.apply(replace_contract_prefix_with_commodity,
                                                                                 axis=1)
                option_list.append(option_shfe_daily_one[['商品名称','合约代码', '成交额','交易所']])
             except Exception as e:
                print(f"Failed to fetch data for {symbol} with contract")
                no_obj.append(symbol)
    print(no_obj)
    pd.DataFrame(no_obj).to_csv('no_obj.csv')
    #将no_obj储存csv



    # 将 option_list 转换为 DataFrame

    full_data = pd.concat(option_list, ignore_index=True)
    #成交额列全变为数字
    full_data['成交额'] = pd.to_numeric(full_data['成交额'], errors='coerce')
    #取合约代码不为空的行
    full_data = full_data[full_data['合约代码'].notna()]
    #按照成交额降序
    full_data = full_data.sort_values(by='成交额', ascending=False)
    # 将数据保存为 CSV 文件

    full_data.to_csv('option_list.csv', index=False)
    print(option_list)
    return full_data

if __name__ == "__main__":
    #选出exchange_mapping中[symbol] = "DCE"的行

   # exchange_mapping = {k: v for k, v in exchange_mapping.items() if v == "SHFE"}
    start_date="20241118"
    end_date="202411119"
    full_data=get_amount(exchange_mapping,start_date)

    #将full_data按商品名称分组，对成交额求和，然后只要商品名称，成交额两列
    grouped_data = full_data.groupby('商品名称')['成交额'].sum().reset_index()
    grouped_data = grouped_data[['商品名称', '成交额']]
#降序
    grouped_data = grouped_data.sort_values(by='成交额', ascending=False)
    grouped_data.to_csv('option_list_start_sum.csv')
    #############3
    end_data=get_amount(exchange_mapping,end_date)
    grouped_end_data = end_data.groupby('商品名称')['成交额'].sum().reset_index()
    grouped_end_data = grouped_end_data[['商品名称', '成交额']]
    grouped_end_data = grouped_end_data.sort_values(by='成交额', ascending=False)
    grouped_end_data.to_csv('option_list_end_sum_end.csv')

    #将grouped_data和grouped_end_data合并，计算grouped_end_data-grouped_data成交额变化，降序排列
    grouped_data = pd.merge(grouped_data, grouped_end_data, on='商品名称')
    grouped_data['成交额变化'] = grouped_data['成交额_y'] - grouped_data['成交额_x']
    grouped_data = grouped_data.sort_values(by='成交额变化', ascending=False)
    grouped_data.to_csv('option_list_sum_change.csv')
    zzz


option_shfe_daily_one, option_shfe_daily_two = option_shfe_daily(
    symbol="铝期权", trade_date="20241111"
)

option_commodity_contract_table_sina_df = ak.option_commodity_contract_table_sina(symbol="沪铝期权", contract="al2501")

option_shfe_daily_one = ak.option_shfe_daily(symbol="铜期权", trade_date="20241106")


print(option_shfe_daily_one.columns)
print(option_shfe_daily_two)
#选出option_shfe_daily_one中所有列名为‘合约代码’'成交额'的列，然后按成交额降序排列

print(option_shfe_daily_one[['合约代码','成交额']].sort_values(by='成交额',ascending=False))

