import akshare as ak
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
import time
import psutil
import pandas as pd
from datetime import datetime
import os
import requests
import json
from urllib.parse import quote
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import jieba
import concurrent.futures
import threading

def get_stock_info_em(symbol):
    """
    使用akshare的stock_individual_info_em函数查询股票信息
    
    参数:
    symbol (str): 股票代码
    
    返回:
    dict: 包含股票信息的字典
    """
    print(f"获取股票 {symbol} 的信息")
    try:
        # 获取股票信息
        stock_info = ak.stock_individual_info_em(symbol=symbol)
        return stock_info.to_dict()
    except Exception as e:
        print(f"获取股票信息失败: {str(e)}")
        return None

def get_stock_special_topics(stock_code):
    """
    获取股票的特殊题材信息
    """
    driver = None
    try:
        # 配置 Chrome
        options = uc.ChromeOptions()
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--disable-extensions')
        options.add_argument('--disable-software-rasterizer')
        options.add_argument('--headless')  # 添加无头模式，避免浏览器界面闪现
        options.add_argument('--window-size=1920,1080')  # 设置窗口大小
        options.add_argument('--log-level=3')  # 减少日志输出
        print("正在启动浏览器...")
        try:
            # 直接指定当前Chrome版本
            driver = uc.Chrome(options=options, version_main=122)  # 使用你当前的Chrome版本
            print("浏览器启动成功")
        except Exception as e:
            print(f"浏览器启动失败: {str(e)}")
            try:
                # 尝试其他常见版本
                for version in [122, 121, 120, 119, 118]:
                    try:
                        print(f"尝试使用Chrome版本 {version}")
                        driver = uc.Chrome(options=options, version_main=version)
                        print(f"使用Chrome版本 {version} 启动成功")
                        break
                    except Exception:
                        continue
            except Exception as e2:
                print(f"所有启动方法均失败: {str(e2)}")
                raise
        
            
        driver.set_page_load_timeout(180)  # 增加页面加载超时时间
        
        # 构建URL并访问
        url = f"https://emweb.securities.eastmoney.com/pc_hsf10/pages/index.html?type=web&code={stock_code}&color=b#/hxtc"
        print(f"正在访问: {url}")
        driver.get(url)
        
        print("等待页面加载...")
        time.sleep(20)  # 增加等待时间
        
        # 等待页面主要元素加载完成
        try:
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC
            wait = WebDriverWait(driver, 30)
            wait.until(EC.presence_of_element_located((By.CLASS_NAME, "p_div")))
            print("页面主要元素加载完成")
        except Exception as e:
            print(f"等待页面元素加载失败: {str(e)}")
        
        # 创建结果字典
        result = {
            "所属板块": [],
            "经营范围": "",
            "主营业务": "",
            "行业背景": "",
            "核心竞争力": []
        }
        
        # 获取所属板块信息
        try:
            board_items = driver.find_elements(By.CLASS_NAME, "boardName")
            for item in board_items:
                board_text = item.text.strip()
                if board_text:
                    result["所属板块"].append(board_text)
            print(f"找到 {len(result['所属板块'])} 个板块")
        except Exception as e:
            print(f"获取板块信息失败: {str(e)}")

        # 获取所有p_div元素
        p_divs = driver.find_elements(By.CLASS_NAME, "p_div")
        
        for div in p_divs:
            try:
                # 获取所有文本内容
                all_text = div.text.strip()
                
                # 尝试查找分类标题
                try:
                    title_element = div.find_element(By.CLASS_NAME, "classfiy")
                    title = title_element.text.strip()
                except:
                    # 如果没有classfiy类，尝试获取第一个font元素的文本
                    try:
                        font_element = div.find_element(By.TAG_NAME, "font")
                        title = font_element.text.strip()
                    except:
                        continue
                
                print(f"处理标题: {title}")
                
                # 获取内容（排除标题部分）
                content = all_text.replace(title, '').strip()
                
                # 根据标题存储内容
                if "经营范围" in title:
                    result["经营范围"] = content
                    print("已保存经营范围")
                elif "主营业务" in title:
                    result["主营业务"] = content
                    print("已保存主营业务")
                elif "行业背景" in title:
                    result["行业背景"] = content
                    print("已保存行业背景")
                elif "核心竞争力" in title:
                    result["核心竞争力"].append({
                        "标题": title,
                        "内容": content
                    })
                    print("已保存核心竞争力信息")
                elif not title.startswith("历史题材"):  # 忽略历史题材部分
                    # 检查是否是其他核心竞争力内容
                    if any(keyword in title for keyword in ["优势", "规模", "技术"]):
                        result["核心竞争力"].append({
                            "标题": title,
                            "内容": content
                        })
                        print(f"已保存额外核心竞争力信息: {title}")
                
            except Exception as e:
                print(f"处理div内容时出错: {str(e)}")
                continue
        
        return result
        
    except Exception as e:
        print(f"获取股票题材信息失败: {str(e)}")
        return None
        
    finally:
        if driver:
            try:
                # 强制结束所有Chrome进程
                for proc in psutil.process_iter(['pid', 'name']):
                    try:
                        if 'chrome' in proc.info['name'].lower():
                            psutil.Process(proc.info['pid']).terminate()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                print("Chrome进程已清理")
            except Exception as e:
                print(f"清理Chrome进程时出现异常: {str(e)}")

def get_eastmoney_search_news(keyword):
    """
    获取东方财富搜索页面的新闻
    keyword: 搜索关键词，如 "恒立钻具"
    """
    print(f"开始获取 '{keyword}' 的相关新闻...")
    
    # 尝试使用股票代码直接获取新闻
    if keyword.isdigit() and len(keyword) == 6:
        print(f"检测到输入为股票代码，尝试获取股票 {keyword} 的相关新闻")
        try:
            # 尝试获取股票名称
            import akshare as ak
            try:
                stock_info = ak.stock_zh_a_spot_em()
                stock_name = stock_info[stock_info['代码'] == keyword]['名称'].values[0]
                print(f"获取到股票名称: {stock_name}，将使用此名称搜索新闻")
                keyword = stock_name  # 使用股票名称搜索
            except Exception as e:
                print(f"无法获取股票名称，将继续使用股票代码搜索: {str(e)}")
        except ImportError:
            print("未安装akshare，将继续使用股票代码搜索")
    
    # 构建API URL
    url = "http://search-api-web.eastmoney.com/search/jsonp"
    
    # URL编码处理
    encoded_keyword = quote(keyword)
    json_param = {
        "uid": "",
        "keyword": keyword,
        "type": ["cmsArticleWebOld"],
        "client": "web",
        "clientType": "web",
        "clientVersion": "curr",
        "param": {
            "cmsArticleWebOld": {
                "searchScope": "all",  # 修改为搜索全部内容，不仅限于标题
                "sort": "time",        # 按时间排序
                "pageIndex": 1,
                "pageSize": 100,
                "preTag": "<em>",
                "postTag": "</em>"
            }
        }
    }
    
    params = {
        "cb": "jQuery",
        "param": json.dumps(json_param, ensure_ascii=False),
        "_": str(int(datetime.now().timestamp() * 1000))
    }
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Referer': f'https://so.eastmoney.com/news/s?keyword={encoded_keyword}&type=title',
        'Accept': '*/*'
    }
    
    try:
        print(f"发送请求到东方财富搜索API: {url}")
        response = requests.get(url, params=params, headers=headers, timeout=30)
        response.encoding = 'utf-8'  # 设置响应编码
        
        # 检查响应状态
        if response.status_code != 200:
            print(f"API请求失败，状态码: {response.status_code}")
            return pd.DataFrame()
            
        # 处理返回的jsonp数据
        data = response.text
        print(f"获取到响应数据，长度: {len(data)} 字符")
        
        # 检查响应格式
        if not data.startswith('jQuery(') or not data.endswith(')'):
            print(f"响应格式不正确: {data[:100]}...")
            # 尝试直接解析JSON
            try:
                json_data = json.loads(data)
            except:
                print("无法解析响应数据为JSON")
                return pd.DataFrame()
        else:
            # 正常解析JSONP
            json_str = data.strip('jQuery(')[:-1]
            try:
                json_data = json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"JSON解析失败: {str(e)}")
                return pd.DataFrame()
        
        # 提取新闻列表
        news_list = []
        if 'result' in json_data and 'cmsArticleWebOld' in json_data['result']:
            news_list = json_data['result']['cmsArticleWebOld']
            print(f"从API响应中提取到 {len(news_list)} 条新闻")
        else:
            print("API响应中没有找到新闻列表")
            if 'result' in json_data:
                print(f"可用的结果类型: {list(json_data['result'].keys()) if isinstance(json_data['result'], dict) else '无'}")
        
        # 转换为DataFrame
        news_data = []
        for news in news_list:
            # 完全放宽筛选条件，接受所有返回的新闻
            news_title = news['title'].replace('<em>', '').replace('</em>', '')
            news_content = news['content'].replace('<em>', '').replace('</em>', '') if 'content' in news else ''
            
            # 直接添加所有新闻，不做关键词筛选
            news_data.append({
                '新闻标题': news_title,
                '新闻内容': news_content,
                '发布时间': news.get('date', ''),
                '文章来源': news.get('mediaName', ''),
                '新闻链接': news.get('url', '')
            })
            
            # 打印调试信息
            print(f"添加新闻: {news_title[:30]}...")
        
        df = pd.DataFrame(news_data)
        print(f"\n共获取到 {len(df)} 条相关新闻")
        
        # 确保DataFrame至少包含必要的列
        if df.empty:
            print("未找到相关新闻，尝试使用更宽松的搜索条件")
            
            # 如果没有找到新闻，尝试使用更宽松的搜索条件再次搜索
            # 例如，如果关键词是公司全名，尝试使用公司简称
            if len(keyword) > 4 and not keyword.isdigit():
                short_keyword = keyword[:2]  # 使用前两个字作为简称
                print(f"尝试使用简称 '{short_keyword}' 重新搜索")
                
                # 递归调用自身，但使用简称
                return get_eastmoney_search_news(short_keyword)
            else:
                # 创建空DataFrame
                df = pd.DataFrame(columns=['新闻标题', '新闻内容', '发布时间', '文章来源', '新闻链接'])
                print("未找到相关新闻，返回空DataFrame")
            
        return df
        
    except requests.exceptions.Timeout:
        print("请求超时，东方财富搜索API无响应")
        return pd.DataFrame()
    except requests.exceptions.RequestException as e:
        print(f"请求异常: {str(e)}")
        return pd.DataFrame()
    except Exception as e:
        print(f"获取新闻列表失败: {str(e)}")
        print(f"请求URL: {url}")
        print(f"请求参数: {params}")
        return pd.DataFrame()

def get_full_content(url):
    """
    获取新闻页面的完整内容
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.encoding = 'utf-8'
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 尝试不同的内容定位方式
        content = ''
        
        # 方式1：查找指定ID的div
        content_div = soup.find('div', {'id': 'ContentBody'}) or \
                     soup.find('div', {'id': 'content'}) or \
                     soup.find('div', {'class': 'newsContent'})
        
        if content_div:
            # 获取所有段落
            paragraphs = content_div.find_all('p')
            content = '\n'.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
        
        # 如果没有找到内容，尝试其他方式
        if not content:
            # 方式2：查找文章主体
            article = soup.find('article') or \
                     soup.find('div', {'class': 'article'})
            if article:
                content = article.get_text().strip()

        return content or '无法获取完整内容'
    except Exception as e:
        print(f"获取新闻内容失败: {str(e)}")
        return '获取内容时出错'

def get_stock_news_info(stock_code):
    """
    获取股票的新闻信息
    
    参数:
    stock_code (str): 股票代码
    
    返回:
    dict: 包含新闻信息的字典
    """
    print(f"开始获取股票 {stock_code} 的新闻信息...")
    try:
        # 获取股票名称
        stock_name = stock_code
        try:
            print(f"尝试获取股票 {stock_code} 的名称...")
            stock_info = ak.stock_zh_a_spot_em()
            name_matches = stock_info[stock_info['代码'] == stock_code]['名称']
            if not name_matches.empty:
                stock_name = name_matches.values[0]
                print(f"成功获取股票名称: {stock_name}")
            else:
                print(f"未找到股票 {stock_code} 的名称信息，将使用股票代码作为搜索关键词")
        except Exception as e:
            print(f"获取股票名称时出错: {str(e)}，将使用股票代码作为搜索关键词")
        
        # 尝试直接使用股票代码获取新闻
        print(f"首先尝试使用股票代码 {stock_code} 获取新闻...")
        news_df = get_eastmoney_search_news(stock_code)
        
        # 如果没有找到新闻，并且股票名称与代码不同，则尝试使用股票名称
        if news_df.empty and stock_name != stock_code:
            print(f"使用股票代码未找到新闻，尝试使用股票名称 {stock_name} 获取新闻...")
            news_df = get_eastmoney_search_news(stock_name)
        
        # 如果仍然没有找到新闻，尝试使用更宽松的搜索条件
        if news_df.empty and len(stock_name) > 2 and not stock_name.isdigit():
            short_name = stock_name[:2]  # 使用前两个字作为简称
            print(f"仍未找到新闻，尝试使用简称 {short_name} 获取新闻...")
            news_df = get_eastmoney_search_news(short_name)
        
        # 只保留最近30条新闻
        if not news_df.empty:
            news_df = news_df.head(30)
            print(f"保留最近 {len(news_df)} 条新闻进行处理")
        
        # 如果仍然没有找到新闻，直接返回空结果
        if news_df.empty:
            print("所有尝试均未找到相关新闻，返回空结果")
            return {
                "股票名称": stock_name,
                "获取时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "新闻": []
            }
        
        # 添加新闻内容全列
        news_df['新闻内容全'] = ''
        
        # 使用多线程获取每条新闻的完整内容
        print("开始多线程获取每条新闻的完整内容...")
        success_count = 0
        content_lock = threading.Lock()  # 创建线程锁保护计数器和DataFrame更新
        
        def fetch_news_content(index, row):
            nonlocal success_count
            try:
                url = row['新闻链接']
                content = get_full_content(url)
                with content_lock:
                    news_df.at[index, '新闻内容全'] = content
                    if content and content != '无法获取完整内容' and content != '获取内容时出错':
                        success_count += 1
                        print(f"成功获取第 {index+1} 条新闻内容，当前成功数: {success_count}")
            except Exception as e:
                print(f"获取第 {index+1} 条新闻内容时出错: {str(e)}")
                with content_lock:
                    news_df.at[index, '新闻内容全'] = '获取内容时出错'
        
        # 使用线程池并行获取新闻内容
        max_workers = min(10, len(news_df))  # 最多10个线程，或者新闻数量
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            futures = [executor.submit(fetch_news_content, index, row) for index, row in news_df.iterrows()]
            # 等待所有任务完成
            concurrent.futures.wait(futures)
        
        print(f"成功获取 {success_count} 条新闻的完整内容")
        
        # 如果没有成功获取任何新闻内容，直接返回空结果
        if success_count == 0:
            print("未能成功获取任何新闻的完整内容，返回空结果")
            return {
                "股票名称": stock_name,
                "获取时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "新闻": []
            }
        
        # 定义文本预处理函数
        def preprocess_text(text):
            # 使用jieba分词
            try:
                words = jieba.cut(text)
                return ' '.join(words)
            except Exception as e:
                print(f"文本预处理失败: {str(e)}")
                return text  # 如果分词失败，返回原文本
        
        # 添加标记列用于追踪要保留的新闻
        news_df['keep'] = True
        
        # 获取所有新闻内容并预处理
        contents = []
        for _, row in news_df.iterrows():
            content = row['新闻内容全']
            if content and content.strip() and content != '无法获取完整内容' and content != '获取内容时出错':
                contents.append(content)
            else:
                contents.append('')  # 对于无法获取内容的新闻，添加空字符串
        
        # 检查是否有新闻内容
        valid_contents = [c for c in contents if c.strip()]
        if not valid_contents:
            print("未找到有效的新闻内容")
            return {
                "股票名称": stock_name,
                "获取时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "新闻": []
            }
        
        print(f"开始处理 {len(valid_contents)} 条有效新闻内容")
        processed_contents = [preprocess_text(text) for text in contents]
        
        # 创建TF-IDF向量化器，添加异常处理
        try:
            # 只对有效内容进行向量化
            valid_indices = [i for i, c in enumerate(contents) if c.strip()]
            valid_processed = [processed_contents[i] for i in valid_indices]
            
            if len(valid_processed) >= 2:  # 至少需要两条新闻才能比较相似度
                print("开始计算新闻相似度...")
                vectorizer = TfidfVectorizer(min_df=1, max_df=1.0)
                tfidf_matrix = vectorizer.fit_transform(valid_processed)
                
                # 遍历每条有效新闻，比较相似度
                for i_idx, i in enumerate(valid_indices):
                    # 如果当前新闻已被标记删除，跳过
                    if not news_df.iloc[i]['keep']:
                        continue
                    
                    # 与其他新闻比较
                    for j_idx, j in enumerate(valid_indices):
                        # 跳过自身和已标记删除的新闻
                        if i == j or not news_df.iloc[j]['keep']:
                            continue
                        
                        # 计算余弦相似度
                        try:
                            similarity = cosine_similarity(
                                tfidf_matrix[i_idx:i_idx+1], 
                                tfidf_matrix[j_idx:j_idx+1]
                            )[0][0]
                            
                            # 相似度超过0.5则标记删除较新的一条
                            if similarity > 0.5:
                                try:
                                    time_i = pd.to_datetime(news_df.iloc[i]['发布时间'])
                                    time_j = pd.to_datetime(news_df.iloc[j]['发布时间'])
                                    
                                    if time_i <= time_j:
                                        # 保留较早的新闻i，删除较新的新闻j
                                        news_df.iloc[j, news_df.columns.get_loc('keep')] = False
                                    else:
                                        # 保留较早的新闻j，删除较新的新闻i
                                        news_df.iloc[i, news_df.columns.get_loc('keep')] = False
                                        # 当前新闻被删除，不再继续比较
                                        break
                                except Exception as e:
                                    print(f"比较新闻发布时间时出错: {str(e)}")
                                    continue
                        except Exception as e:
                            print(f"计算新闻相似度时出错: {str(e)}")
                            continue
        except Exception as e:
            print(f"处理新闻相似度时出错: {str(e)}")
            # 如果相似度比较失败，仍然继续处理
        
        # 只保留标记为True的新闻
        news_df = news_df[news_df['keep']].drop('keep', axis=1)
        print(f"过滤后保留 {len(news_df)} 条不相似新闻")
        
        # 整合新闻信息
        news_info = {
            "股票名称": stock_name,
            "获取时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "新闻": []
        }
        
        # 处理过滤后的新闻
        for _, row in news_df.iterrows():
            news_item = {
                "标题": row['新闻标题'],
                "内容": row['新闻内容全'],
                "链接": row['新闻链接'],
                "发布时间": row['发布时间']
            }
            news_info["新闻"].append(news_item)
        
        return news_info
    
    except Exception as e:
        print(f"获取股票新闻信息失败: {str(e)}")
        return {
            "股票名称": stock_code,
            "获取时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "新闻": []
        }

def get_stock_fundamental_info(stock_code):
    """
    获取股票的基本面信息，包括东方财富数据、特殊题材信息和新闻信息
    
    参数:
    stock_code (str): 股票代码
    
    返回:
    dict: 包含所有基本面信息的字典
    """
    # 获取东方财富数据
    em_info = get_stock_info_em(stock_code)
    
    # 获取特殊题材信息
    special_info = get_stock_special_topics(stock_code)
    
    # 获取新闻信息
    news_info = get_stock_news_info(stock_code)
    
    # 合并信息
    fundamental_info = {
        "股票代码": stock_code,
        "股票名称": news_info.get("股票名称", stock_code),
        "获取时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "东方财富数据": em_info if em_info else {},
        "特殊题材信息": special_info if special_info else {},
        "新闻信息": news_info.get("新闻", [])
    }
    
    return fundamental_info

def process_stock(code, output_dir):
    """
    处理单个股票的信息获取和保存
    
    参数:
    code (str): 股票代码
    output_dir (str): 输出目录
    """
    print(f"\n处理股票代码: {code}")
    try:
        # 获取基本面信息
        info = get_stock_fundamental_info(code)
        
        # 生成输出文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(output_dir, f"stock_info_{code}_{timestamp}.txt")
        
        # 保存为TXT文件
        with open(filename, 'w', encoding='utf-8') as f:
            # 写入基本信息
            f.write(f"股票代码：{info['股票代码']}\n")
            f.write(f"股票名称：{info.get('股票名称', info['股票代码'])}\n")
            f.write(f"获取时间：{info['获取时间']}\n\n")
            
            # 写入东方财富数据
            f.write("=== 东方财富数据 ===\n")
            if info['东方财富数据']:
                for key, value in info['东方财富数据'].items():
                    # 将数值转换为字符串
                    value_str = str(value)
                    f.write(f"{key}：{value_str}\n")
            else:
                f.write("无数据\n")
            f.write("\n")
            
            # 写入特殊题材信息
            f.write("=== 特殊题材信息 ===\n")
            if info['特殊题材信息']:
                special_info = info['特殊题材信息']
                
                # 写入所属板块
                f.write("所属板块：\n")
                if special_info['所属板块']:
                    for board in special_info['所属板块']:
                        f.write(f"- {board}\n")
                else:
                    f.write("无数据\n")
                f.write("\n")
                
                # 写入其他基本信息
                for key in ['经营范围', '主营业务', '行业背景']:
                    f.write(f"{key}：\n")
                    content = special_info[key].strip() if special_info[key] else "无数据"
                    f.write(f"{content}\n\n")
                
                # 写入核心竞争力
                f.write("核心竞争力：\n")
                if special_info['核心竞争力']:
                    for item in special_info['核心竞争力']:
                        f.write(f"- {item['标题']}\n")
                        f.write(f"  {item['内容']}\n")
                else:
                    f.write("无数据\n")
            else:
                f.write("无数据\n")
            f.write("\n")
            
            # 写入新闻信息
            f.write("=== 新闻列表 ===\n")
            if info['新闻信息']:
                for i, news in enumerate(info['新闻信息'], 1):
                    f.write(f"\n新闻 {i}\n")
                    f.write(f"标题：{news['标题']}\n")
                    f.write(f"发布时间：{news['发布时间']}\n")
                    f.write(f"链接：{news['链接']}\n")
                    f.write(f"内容：{news['内容']}\n")
                    f.write("\n" + "-"*50 + "\n")  # 分隔线
            else:
                f.write("无数据\n")
            
        print(f"已保存到文件: {filename}")
        return True
        
    except Exception as e:
        print(f"处理股票 {code} 时出错: {str(e)}")
        return False

def batch_get_stock_info(pattern_results_file, max_workers=5):
    """
    从pattern_results.csv文件中读取股票代码并获取基本面信息
    使用多线程并行处理以提高效率
    
    参数:
    pattern_results_file (str): pattern_results.csv文件路径
    max_workers (int): 最大线程数，默认为5
    """
    import pandas as pd
    import os
    from datetime import datetime
    
    # 读取CSV文件并获取唯一的股票代码
    df = pd.read_csv(pattern_results_file)
    df['code'] = df['code'].astype(str).str.zfill(6)
       
    unique_codes = df['code'].unique()
    print(f"共找到 {len(unique_codes)} 个唯一股票代码")
    
    # 创建保存结果的目录
    output_dir = 'stock_info'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 使用线程池并行处理股票信息获取
    print(f"使用 {max_workers} 个线程并行处理股票信息获取...")
    success_count = 0
    fail_count = 0
    
    # 创建线程锁，用于保护计数器
    lock = threading.Lock()
    
    def process_with_counter(code):
        nonlocal success_count, fail_count
        result = process_stock(code, output_dir)
        with lock:
            if result:
                success_count += 1
            else:
                fail_count += 1
    
    # 使用ThreadPoolExecutor并行处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        futures = [executor.submit(process_with_counter, code) for code in unique_codes]
        
        # 等待所有任务完成
        concurrent.futures.wait(futures)
    
    print(f"\n处理完成! 成功: {success_count}, 失败: {fail_count}, 总计: {len(unique_codes)}")
    return success_count, fail_count
if __name__ == "__main__":
    import argparse
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='获取股票基本面信息和新闻')
    parser.add_argument('--file', type=str, default='pattern_results.csv', help='包含股票代码的CSV文件路径')
    parser.add_argument('--threads', type=int, default=20, help='并行处理的最大线程数，默认为5')
    parser.add_argument('--code', type=str, help='单个股票代码，如果提供则只处理这一只股票')
    
    args = parser.parse_args()
    
    if args.code:
        # 如果提供了单个股票代码，只处理这一只股票
        print(f"处理单个股票: {args.code}")
        output_dir = 'stock_info'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        process_stock(args.code, output_dir)
    else:
        # 批量处理CSV文件中的股票
        print(f"使用 {args.threads} 个线程处理文件 {args.file} 中的股票")
        batch_get_stock_info(args.file, max_workers=args.threads)