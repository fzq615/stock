##value:"02216242aeb24aa98f7c2a1b37cdbf91"
from openai import OpenAI
import json
import time
import requests
from bs4 import BeautifulSoup
import json
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
import time
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget
from PySide6.QtCore import Qt
from datetime import datetime
import os
import akshare as ak
from datetime import datetime, timedelta
import pandas as pd
import requests
import pandas as pd
from datetime import datetime
import json
import os
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.firefox import GeckoDriverManager
import undetected_chromedriver as uc
from selenium.webdriver.common.keys import Keys
def save_to_file(data, filename, folder="data"):
    """
    保存数据到本地文件
    """
    try:
        # 获取当前脚本所在目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 构建数据文件夹路径
        data_folder = os.path.join(current_dir, folder)
        
        # 确保文件夹存在
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)
        
        # 生成带时间戳的文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        full_filename = os.path.join(data_folder, f"{filename}_{timestamp}.txt")
        
        # 保存数据
        with open(full_filename, 'w', encoding='utf-8') as f:
            if isinstance(data, dict):
                # 格式化字典数据
                f.write(f"关键词: {data.get('keyword', '')}\n")
                f.write(f"时间: {data.get('timestamp', '')}\n")
                f.write(f"内容:\n{'-'*50}\n")
                
                # 处理内容列表
                if isinstance(data.get('content'), list):
                    for item in data['content']:
                        f.write(f"\n问答 {item['index']}:\n{item['content']}\n")
                        f.write(f"{'-'*50}\n")
                else:
                    f.write(str(data.get('content', '')))
                
                # 如果有分析结果
                if 'analysis' in data:
                    f.write(f"\n分析结果:\n{'-'*50}\n")
                    f.write(f"{data['analysis']}\n")
            else:
                f.write(str(data))
        
        print(f"数据已保存到: {full_filename}")
        return True
    except Exception as e:
        print(f"保存数据失败: {str(e)}")
        return False

def analyze_with_api(content):
    """
    使用 DeepSeek API 分析内容
    """
    try:
        print("正在初始化 DeepSeek API 客户端...")
        
        # 初始化 OpenAI 客户端
        client = OpenAI(
            api_key="sk-cefd9c1f970f4715aa23e4740273a84c",  # 替换为你的 DeepSeek API Key
            base_url="https://api.deepseek.com"
        )
        
        # 构造分析提示词
        prompt = f"""
请分析以下东方财富网的内容，并按以下格式输出：
1. 提取主要主题和热点
2. 对每个主题进行详细分析
3. 总结关键信息和市场影响

内容如下：
{content}

请按以下格式输出：
【主题数量】：发现的主题总数

【主题列表】：
1. 主题1名称
   - 核心内容：
   - 相关信息：
   - 市场影响：

2. 主题2名称
   ...（按此格式列出所有主题）

【总体分析】：
- 市场关注重点：
- 潜在影响：
- 投资建议：
"""
        
        print("发送请求...")
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a professional financial analyst"},
                {"role": "user", "content": prompt},
            ],
            stream=False
        )
        
        # 获取回复内容
        reply = response.choices[0].message.content
        print("成功获取回复")
        
        return reply
        
    except Exception as e:
        error_msg = str(e)
        if "Insufficient Balance" in error_msg:
            print("API 余额不足，请充值后重试")
        elif "invalid_api_key" in error_msg:
            print("API Key 无效，请检查")
        elif "rate_limit_exceeded" in error_msg:
            print("请求频率过高，请稍后重试")
        else:
            print(f"API 请求失败: {error_msg}")
        return None

def get_eastmoney_content(keyword):
    """
    使用Selenium爬取东方财富网搜索结果
    """
    try:
        print("正在初始化浏览器...")
        
        chrome_options = Options()
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--ignore-certificate-errors')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--disable-web-security')
        chrome_options.add_argument('--disable-features=IsolateOrigins,site-per-process')
        chrome_options.binary_location = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
        
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        driver.set_page_load_timeout(30)
        
        print("正在访问网页...")
        driver.get(f"https://so.eastmoney.com/qa/s?keyword={keyword}")
        
        # 等待页面加载
        print("等待页面加载...")
        time.sleep(10)
        
        # 滚动页面多次以加载更多内容
        print("滚动页面以加载更多内容...")
        for _ in range(3):  # 滚动3次
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
        
        # 获取问答内容
        print("获取问答内容...")
        qa_items = driver.find_elements(By.CSS_SELECTOR, "div[class*='qa']")
        print(f"\n找到 {len(qa_items)} 个问答")
        
        results = []
        for i, item in enumerate(qa_items, 1):
            try:
                # 获取整个问答元素的文本内容
                content = item.text.strip()
                if content:
                    results.append({
                        'content': content,
                        'index': i
                    })
                    print(f"成功提取第 {i} 个问答")
                else:
                    print(f"第 {i} 个问答内容为空")
                
            except Exception as e:
                print(f"处理第 {i} 个问答时出错: {str(e)}")
                continue
        
        print(f"\n总共成功提取 {len(results)} 个问答")
        
        # 关闭浏览器
        driver.quit()
        
        return results
        
    except Exception as e:
        print(f"爬取出错: {str(e)}")
        if 'driver' in locals():
            driver.quit()
        return None

def _fetch_org_id(symbol: str = "000001") -> str:
    """
    获取组织ID
    """
    try:
        url = "https://irm.cninfo.com.cn/newircs/index/queryKeyboardInfo"
        params = {"_t": str(int(datetime.now().timestamp()))}
        data = {"keyWord": symbol}
        
        r = requests.post(url, params=params, data=data)
        data_json = r.json()
        org_id = data_json["data"][0]["secid"]
        return org_id
    except Exception as e:
        print(f"获取组织ID失败: {str(e)}")
        return None

def get_all_irm_qa(max_items,start_page):
    """
    使用 undetected_chromedriver 获取互动易问答内容
    max_items: 最大提取条数
    """
    driver = None
    try:
        print("开始获取数据...")
        
        # 配置 Chrome
        options = uc.ChromeOptions()
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        
        # 初始化浏览器
        print("正在启动浏览器...")
        driver = uc.Chrome(options=options)
        driver.set_page_load_timeout(60)
        
        # 访问页面
        url = "https://irm.cninfo.com.cn/views/interactiveAnswer"
        print(f"正在访问: {url}")
        driver.get(url)
        
        print("等待页面加载...")
        time.sleep(20)
        
        # 如果指定了起始页，先跳转到该页
        if start_page > 1:
            print(f"正在跳转到第 {start_page} 页...")
            try:
                # 等待页码输入框出现
                input_box = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, ".el-pagination__editor input.el-input__inner"))
                )
                
                # 使用JavaScript设置值并触发事件
                driver.execute_script("""
                    arguments[0].value = arguments[1];
                    arguments[0].dispatchEvent(new Event('input'));
                    arguments[0].dispatchEvent(new Event('change'));
                """, input_box, str(start_page))
                
                # 发送回车键
                input_box.send_keys(Keys.ENTER)
                print(f"已输入页码 {start_page}")
                
                # 等待页面加载
                time.sleep(10)
                
                # 验证是否成功跳转
                current_page = driver.execute_script("""
                    return document.querySelector('.el-pagination__editor input.el-input__inner').value;
                """)
                
                if str(current_page) != str(start_page):
                    print(f"页面跳转验证失败，当前页码: {current_page}")
                    return None
                    
                print(f"已确认跳转到第 {start_page} 页")
                
            except Exception as e:
                print(f"页码跳转失败: {str(e)}")
                return None
        
        # 初始化数据列表
        qa_items = []
        page_num = start_page
        
        # 创建基础文件名（不包含页码）
        base_filename = f"irm_qa_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        while len(qa_items) < max_items:
            try:
                print(f"\n正在获取第 {page_num} 页...")
                time.sleep(5)
                
                # 获取所有问题
                questions = driver.find_elements(By.CSS_SELECTOR, "[data-v-32770d1c] .question-content span")
                # 获取所有回复
                replies = driver.find_elements(By.CSS_SELECTOR, "[data-v-32770d1c] .reply-content .main span:last-child")
                # 获取公司名称
                companies = driver.find_elements(By.CSS_SELECTOR, "[data-v-32770d1c] .reply-content .main")
                
                print(f"找到 {len(questions)} 个问题和 {len(replies)} 个回复")
                
                # 确保问题和回复数量匹配
                if len(questions) == len(replies) == len(companies):
                    for q, r, c in zip(questions, replies, companies):
                        if len(qa_items) >= max_items:
                            break
                            
                        try:
                            # 提取公司名称（去除回复内容）
                            company_name = c.text.split('：')[0] if '：' in c.text else ''
                            
                            qa_info = {
                                "公司": company_name,
                                "问题": q.text.strip(),
                                "回答": r.text.strip(),
                                "时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "页码": page_num
                            }
                            qa_items.append(qa_info)
                            print(f"成功提取第 {len(qa_items)} 个问答")
                            
                        except Exception as e:
                            print(f"提取问答数据失败: {str(e)}")
                            continue
                            
                    # 每页数据处理完后保存一次
                    df = pd.DataFrame(qa_items)
                    if not df.empty:
                        # 删除空公司
                        df = df.dropna(subset=['公司'])
                        df = df[df['公司'].str.strip() != '']
                        # 计算行数
                        company_counts = df['公司'].value_counts()
                        df['行数'] = df['公司'].map(company_counts)
                        # 保存当前进度
                        temp_filename = f"{base_filename}_to_page_{page_num}.csv"
                        df.to_csv(temp_filename, index=False, encoding='utf-8-sig')
                        print(f"已保存到第 {page_num} 页的数据: {temp_filename}")
                
                else:
                    print(f"问题和回复数量不匹配: 问题={len(questions)}, 回复={len(replies)}, 公司={len(companies)}")
                
                # 检查是否有下一页
                try:
                    # 先检查是否存在禁用的下一页按钮
                    disabled_next = driver.find_elements(By.CSS_SELECTOR, "button.btn-next[disabled]")
                    if disabled_next:
                        print("已到最后一页")
                        break
                        
                    # 如果没有禁用的按钮，再查找可点击的下一页按钮
                    next_button = driver.find_element(By.CSS_SELECTOR, "button.btn-next:not([disabled])")
                    if next_button and next_button.is_displayed():
                        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                        time.sleep(3)
                        next_button.click()
                        print("点击下一页")
                        time.sleep(5)
                        page_num += 1
                    else:
                        print("已到最后一页")
                        break
                except Exception as e:
                    print(f"翻页失败: {str(e)}")
                    print("已到最后一页")
                    break
                
            except Exception as e:
                print(f"处理第 {page_num} 页时出错: {str(e)}")
                time.sleep(5)
                continue
        
        if not qa_items:
            print("未获取到任何数据")
            time.sleep(10)
            return None
        
        # 最终保存完整数据
        df = pd.DataFrame(qa_items)
        if not df.empty:
            df = df.dropna(subset=['公司'])
            df = df[df['公司'].str.strip() != '']
            company_counts = df['公司'].value_counts()
            df['行数'] = df['公司'].map(company_counts)
            #按行数降序
            df = df.sort_values(by='行数', ascending=False)
            final_filename = f"{base_filename}_final.csv"
            df.to_csv(final_filename, index=False, encoding='utf-8-sig')
            print(f"\n最终数据已保存到: {final_filename}")
        
        return df
        
    except Exception as e:
        print(f"获取数据失败: {str(e)}")
        time.sleep(10)
        return None
        
    finally:
        if driver:
            try:
                print("正在关闭浏览器...")
                time.sleep(5)
                driver.quit()
            except:
                pass

def save_to_file(df, folder="data"):
    """
    保存数据到文件
    """
    if df is None or df.empty:
        print("没有数据需要保存")
        return False
        
    try:
        # 创建保存目录
        os.makedirs(folder, exist_ok=True)
        
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(folder, f"irm_qa_{timestamp}.csv")
        
        # 保存为CSV文件
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"数据已保存到: {filename}")
        
        return True
        
    except Exception as e:
        print(f"保存数据失败: {str(e)}")
        return False
def analyze_company_qa(csv_file, api_key=None):
    """
    读取CSV文件，按公司分组并使用大模型分析
    csv_file: CSV文件路径
    api_key: DeepSeek API key
    """
    try:
        # 读取CSV文件
        print(f"正在读取文件: {csv_file}")
        df = pd.read_csv(csv_file)
        
        # 检查数据
        if df.empty:
            print("文件中没有数据")
            return None
            
        print(f"共读取 {len(df)} 条数据")
        
        # 按公司分组
        grouped = df.groupby('公司')
        print(f"共有 {len(grouped)} 个不同的公司")
        
        # 初始化DeepSeek客户端
        from openai import OpenAI
        client = OpenAI(
            api_key="sk-cefd9c1f970f4715aa23e4740273a84c",
            base_url="https://api.deepseek.com"
        )

        # 处理每个公司的数据
        results = []
        for company, group in grouped:
            print(f"\n正在分析 {company} 的数据...")
            
            # 构造问答对文本
            qa_pairs = []
            for _, row in group.iterrows():
                qa_pairs.append(f"问：{row['问题']}\n答：{row['回答']}")
            
            qa_text = "\n\n".join(qa_pairs)
            
            # 构造提示词
            prompt = f"""
请分析以下公司的互动易问答内容，并按以下格式输出：

公司名称：{company}

问答数量：{len(group)}条

【主要问题类别】：
1. 类别1（涉及问题数量）
   - 核心问题概述：
   - 公司回应要点：
   - 分析解读：

2. 类别2（涉及问题数量）
   ...（按此格式列出所有主要类别）

【投资者关注重点】：
1. 重点1
2. 重点2
...


【总体分析】：
- 投资者关注度和关注点是什么：
- 公司经营状况：
- 潜在风险点：
- 发展机遇：

问答原文：
{qa_text}
"""
            
            try:
                print("正在请求API...")
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": "You are a professional financial analyst"},
                        {"role": "user", "content": prompt},
                    ],
                    stream=False
                )
                
                analysis = response.choices[0].message.content
                results.append({
                    "company": company,
                    "qa_count": len(group),
                    "analysis": analysis
                })
                print(f"完成 {company} 的分析")
                
            except Exception as e:
                print(f"分析 {company} 时出错: {str(e)}")
                continue
        
        # 保存分析结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"qa_analysis_{timestamp}.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(f"\n{'='*50}\n")
                f.write(f"公司：{result['company']}\n")
                f.write(f"问答数量：{result['qa_count']}\n")
                f.write(f"\n{result['analysis']}\n")
        
        print(f"\n分析结果已保存到: {output_file}")
        return results
        
    except Exception as e:
        print(f"处理数据时出错: {str(e)}")
        return None
def main():
    # 从第199页开始获取数据
    df=get_eastmoney_content("000001")
    print(df)
    zzz
    df = get_all_irm_qa(max_items=2000, start_page=1)
    if df is not None:
        print("数据获取完成")
        


        
        # 保存数据到CSV文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = f"irm_qa_{timestamp}.csv"
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        print(f"数据已保存到: {csv_file}")
        
        # 分析数据
       # api_key = "your_deepseek_api_key"  # 替换为你的API key
        results = analyze_company_qa(csv_file)
        if results:
            print("数据分析完成")
        else:
            print("数据分析失败")
    else:
        print("数据获取失败")

if __name__ == "__main__":
    main()
