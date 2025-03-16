from plotly.subplots import make_subplots
import sys
import pandas as pd
import akshare as ak
import plotly.graph_objs as go
from PySide6.QtWidgets import QApplication, QMainWindow, QTableWidget, QTableWidgetItem, QFileDialog, QHBoxLayout, \
    QWidget, QPushButton, QMessageBox, QVBoxLayout, QSplitter, QComboBox, QLabel, QDialog, QTextEdit
from PySide6.QtCore import Qt, QEvent
from PySide6.QtWebEngineWidgets import QWebEngineView
from plotly.offline import plot

from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWebEngineCore import QWebEngineSettings, QWebEnginePage
from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTableWidget, QMessageBox
from PySide6.QtCore import Signal, Slot

from openai import OpenAI
from PySide6.QtCore import Signal, Slot
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
import akshare as ak
import requests
import json
import os
from datetime import datetime
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import jieba
import numpy as np

class CustomWebEnginePage(QWebEnginePage):
    def __init__(self, parent=None):
        super().__init__(parent)

    def javaScriptConsoleMessage(self, level, msg, line_number, source_id):
        levels = {
            QWebEnginePage.InfoMessageLevel: "Info",
            QWebEnginePage.WarningMessageLevel: "Warning",
            QWebEnginePage.ErrorMessageLevel: "Error"
        }
        level_name = levels.get(level, "Unknown")
        print(f"JS Console: {level_name} - {msg} (Line {line_number}, Source {source_id})")

class StockKlineViewer(QMainWindow):
    """
    主窗口类，用于显示股票K线图
    """

    def __init__(self):
        """
        初始化主窗口
        """
        super().__init__()
        self.setWindowTitle("Stock K-line Viewer")
        self.setGeometry(100, 100, 1200, 800)
        
        # 添加缩放状态变量
        self.zoom_level = 100  # 初始缩放级别
        self.current_start_date = None
        self.current_end_date = None
        
        # 添加周期选择下拉框
        self.period_combo = QComboBox()
        self.period_combo.addItems(['日线', '1分钟', '5分钟', '15分钟', '30分钟', '60分钟'])
        self.period_combo.setCurrentText('5分钟')  # 默认选择5分钟
        self.period_combo.currentTextChanged.connect(self.update_chart)
        
        # 添加DeepSeek API配置
        self.deepseek_api_key = "sk-cefd9c1f970f4715aa23e4740273a84c"  # 替换为你的DeepSeek API

        # 添加当前分析的股票代码记录
        self.current_analysis_ticker = None
        self.analysis_dialog = None  # 保存对话框的引用
        
        self.initUI()
        self.api_key="sk-cefd9c1f970f4715aa23e4740273a84c"
        # 安装事件过滤器以捕获键盘事件
        self.web_view.installEventFilter(self)

    def eventFilter(self, obj, event):
        """
        处理键盘事件
        """
        if obj == self.web_view and event.type() == QEvent.KeyPress:
            key = event.key()
            if key in [Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down]:
                self.handle_keyboard_navigation(key)
                return True
        return super().eventFilter(obj, event)

    def handle_keyboard_navigation(self, key):
        """
        处理键盘导航
        """
        if key == Qt.Key_Left:  # 向左移动
            self.update_plot_range('left')
        elif key == Qt.Key_Right:  # 向右移动
            self.update_plot_range('right')
        elif key == Qt.Key_Up:  # 放大
            self.zoom_level = max(50, self.zoom_level - 10)
            self.update_plot_range('zoom')
        elif key == Qt.Key_Down:  # 缩小
            self.zoom_level = min(150, self.zoom_level + 10)
            self.update_plot_range('zoom')

    def update_plot_range(self, action):
        """
        更新图表范围
        """
        # 使用JavaScript更新图表范围
        if action == 'left':
            js_code = """
            var update = {
                'xaxis.range[0]': Plotly.d3.time.day.offset(new Date(Plotly.d3.select('.cartesianlayer').selectAll('.plot')[0][0].__data__[0].x[0]), -1),
                'xaxis.range[1]': Plotly.d3.time.day.offset(new Date(Plotly.d3.select('.cartesianlayer').selectAll('.plot')[0][0].__data__[0].x[1]), -1)
            };
            Plotly.relayout(document.getElementsByClassName('plotly-graph-div')[0], update);
            """
        elif action == 'right':
            js_code = """
            var update = {
                'xaxis.range[0]': Plotly.d3.time.day.offset(new Date(Plotly.d3.select('.cartesianlayer').selectAll('.plot')[0][0].__data__[0].x[0]), 1),
                'xaxis.range[1]': Plotly.d3.time.day.offset(new Date(Plotly.d3.select('.cartesianlayer').selectAll('.plot')[0][0].__data__[0].x[1]), 1)
            };
            Plotly.relayout(document.getElementsByClassName('plotly-graph-div')[0], update);
            """
        elif action == 'zoom':
            js_code = f"""
            var update = {{
                'xaxis.range': [{self.zoom_level}]
            }};
            Plotly.relayout(document.getElementsByClassName('plotly-graph-div')[0], update);
            """
        
        self.web_view.page().runJavaScript(js_code)

    def initUI(self):
        """
        初始化用户界面
        """
        self.setWindowTitle("Stock K-line Viewer")
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        main_layout = QVBoxLayout(self.central_widget)

        # 创建水平分割器（左右分区）
        horizontal_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(horizontal_splitter)

        # 左侧部件
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # 添加周期选择工具栏到左侧
        toolbar_widget = QWidget()
        toolbar_layout = QHBoxLayout(toolbar_widget)
        toolbar_layout.setContentsMargins(5, 5, 5, 5)
        toolbar_layout.addWidget(QLabel("周期:"))
        self.period_combo.setFixedHeight(25)  # 只设置高度，宽度自适应
        toolbar_layout.addWidget(self.period_combo)
        left_layout.addWidget(toolbar_widget)

        # 创建表格控件
        self.table = QTableWidget(self)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.doubleClicked.connect(self.on_row_click)
        left_layout.addWidget(self.table)

        # 创建按钮
        self.open_button = QPushButton("Open Excel", self)
        self.open_button.clicked.connect(self.open_file)
        left_layout.addWidget(self.open_button)

        # 右侧部件（K线图）
        self.web_view = QWebEngineView(self)
        self.web_view.setVisible(True)
        self.web_view.setMinimumSize(600, 400)

        # 将左右部件添加到水平分割器
        horizontal_splitter.addWidget(left_widget)
        horizontal_splitter.addWidget(self.web_view)

        # 设置分割器的初始大小
        horizontal_splitter.setSizes([200, 800])  # 水平分割比例

        # 记录"股票代码"列的索引
        self.ticker_column_index = None

    @Slot(str, str, int, str)
    def on_js_console_message(self, level, msg, line_number, source_id):
        print(f"JS Console: {level} - {msg} (Line {line_number}, Source {source_id})")

    @Slot(str, str, int, str)
    def on_js_console_message(self, level, msg, line_number, source_id):
        print(f"JS Console: {level} - {msg} (Line {line_number}, Source {source_id})")
    def on_load_finished(self, ok):
        if ok:
            print("Chart loaded successfully")
        else:
            print("Failed to load chart")

    @Slot(str, str, int, str)
    def on_js_console_message(self, level, msg, line_number, source_id):
        print(f"JS Console: {level} - {msg} (Line {line_number}, Source {source_id})")
    def load_excel(self, file_path):
        """
        加载Excel文件
        :param file_path: 文件路径
        :return: DataFrame of the Excel file
        """
        try:
            df = pd.read_excel(file_path)
            # 将股票代码列转为字符串，如果小数点前只有四位则加上00
            df['股票代码'] = df['股票代码'].apply(lambda x: f"{x:0>4}" if len(str(x)) == 4 else str(x))
            return df
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load Excel file: {e}")
            return None

    def search_impl(self, arguments):
        """
        处理搜索工具的调用
        """
        return arguments

    def chat_with_deepseek(self, messages):
        """
        与 DeepSeek 进行对话
        """
        try:
            client = OpenAI(
                api_key=self.deepseek_api_key,
                base_url="https://api.deepseek.com"
            )

            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                temperature=0.7,
                max_tokens=2000,
                stream=False
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"DeepSeek API 调用失败: {str(e)}")
            return None

    def get_stock_analysis_from_deepseek(self, ticker, stock_name):
        """
        调用DeepSeek API获取股票分析
        """
        # 如果是相同的股票代码且对话框存在，直接返回None
        if ticker == self.current_analysis_ticker and self.analysis_dialog is not None:
            return None
        
        # 更新当前分析的股票代码
        self.current_analysis_ticker = ticker
        
        try:
            messages = [
                {"role": "system", "content": "你是一个专业的股票分析师。"},
                {"role": "user", "content": f"请分析股票{stock_name}（{ticker}）的基本面情况，包括：\n"
                                          f"1. 公司主营业务\n"
                                          f"2. 最近的财务状况、市盈率、市净率\n"
                                          f"3. 行业地位\n"
                                          f"4. 主要风险因素\n"
                                          f"5. 概念板块、最近有利空和利好新闻消息"}
            ]

            analysis = self.chat_with_deepseek(messages)
            return analysis if analysis else "获取分析失败"

        except Exception as e:
            return f"分析请求出错：{str(e)}"

    def plot_kline(self, ticker):
        try:
            # 获取股票名称
            stock_info = ak.stock_zh_a_spot_em()
            stock_name = stock_info[stock_info['代码'] == ticker]['名称'].values[0]
            
            # 1. 先绘制K线图
            # 根据选择的周期获取数据
            period = self.period_combo.currentText()
            if period == '日线':
                stock_data = ak.stock_zh_a_hist(symbol=ticker, adjust='qfq')
                stock_data = stock_data.rename(columns={
                    '日期': '时间',
                    '开盘': '开盘',
                    '最高': '最高',
                    '最低': '最低',
                    '收盘': '收盘',
                    '成交量': '成交量'
                })
            else:
                # 提取分钟数
                minutes = period.replace('分钟', '')
                stock_data = ak.stock_zh_a_hist_min_em(symbol=ticker, period=minutes, adjust='qfq')
            
            if stock_data.empty:
                QMessageBox.warning(self, "Warning", "No data available for this ticker.")
                return

            # 将数据转换为Plotly需要的格式，并按时间排序
            stock_data['时间'] = pd.to_datetime(stock_data['时间'])
            stock_data = stock_data.sort_values('时间')

            # 计算涨跌幅
            stock_data['涨跌幅'] = (stock_data['收盘'] - stock_data['开盘']) / stock_data['开盘'] * 100

            # 获取每天的第一个和最后一个时间点
            stock_data['日期'] = stock_data['时间'].dt.date
            daily_first = stock_data.groupby('日期')['时间'].first()
            daily_last = stock_data.groupby('日期')['时间'].last()
            tick_values = sorted(list(set(daily_first)))

            # 创建子图布局
            fig = make_subplots(rows=2, cols=1, 
                               shared_xaxes=True,
                               vertical_spacing=0.03,
                               row_heights=[0.7, 0.3])

            # 添加K线图到上面的子图
            fig.add_trace(go.Candlestick(
                x=stock_data['时间'],
                open=stock_data['开盘'],
                high=stock_data['最高'],
                low=stock_data['最低'],
                close=stock_data['收盘'],
                name='K线',
                increasing_line_color='red',     
                decreasing_line_color='green',   
                increasing_fillcolor='white',    
                decreasing_fillcolor='white',    
                line=dict(width=1),
                hovertext=[f'日期: {time.strftime("%Y-%m-%d %H:%M")}<br>' +
                          f'开盘: {open_:.2f}<br>' +
                          f'最高: {high:.2f}<br>' +
                          f'最低: {low:.2f}<br>' +
                          f'收盘: {close:.2f}<br>' +
                          f'涨跌幅: {chg:.2f}%<br>' +
                          f'成交量: {vol:,.0f}'
                          for time, open_, high, low, close, chg, vol in 
                          zip(stock_data['时间'], stock_data['开盘'], stock_data['最高'],
                              stock_data['最低'], stock_data['收盘'], stock_data['涨跌幅'],
                              stock_data['成交量'])],
                hoverinfo='text'
            ), row=1, col=1)

            # 添加成交量图
            colors = ['red' if close >= open_ else 'green' 
                     for close, open_ in zip(stock_data['收盘'], stock_data['开盘'])]
            
            fig.add_trace(go.Bar(
                x=stock_data['时间'],
                y=stock_data['成交量'],
                name='成交量',
                marker_color=colors,
                marker=dict(opacity=1.0),
                hovertext=[f'时间: {time.strftime("%Y-%m-%d %H:%M")}<br>' +
                          f'成交量: {vol:,.0f}'
                          for time, vol in zip(stock_data['时间'], stock_data['成交量'])],
                hoverinfo='text'
            ), row=2, col=1)

            # 更新布局
            title_text = f'{ticker} {stock_name} {period}'
            fig.update_layout(
                title=title_text,
                height=800,
                hovermode='x unified',
                showlegend=False,
                xaxis_rangeslider_visible=False,
                font=dict(family='Microsoft YaHei'),
                plot_bgcolor='white',
                paper_bgcolor='white',
            )

            # 分别设置两个子图的y轴范围和标题，并添加网格线
            fig.update_yaxes(
                title_text="价格", 
                row=1, 
                col=1,
                gridcolor='lightgrey',
                gridwidth=1,
                showgrid=True
            )
            fig.update_yaxes(
                title_text="成交量", 
                row=2, 
                col=1,
                gridcolor='lightgrey',  # 设置网格线颜色
                gridwidth=1,            # 设置网格线宽度
                showgrid=True          # 显示网格
            )

            # 设置x轴网格线
            fig.update_xaxes(
                gridcolor='lightgrey',  # 设置网格线颜色
                gridwidth=1,            # 设置网格线宽度
                showgrid=True          # 显示网格
            )

            # 设置两个子图的独立显示范围
            fig.update_layout(
                yaxis=dict(domain=[0.3, 1.0]),  # K线图占据上方70%
                yaxis2=dict(domain=[0, 0.25])    # 成交量图占据下方25%
            )

            # 更新X轴设置
            fig.update_xaxes(
                type='category',  # 使用分类类型，这样会显示实际有数据的点
                tickformat='%Y-%m-%d %H:%M',
                ticktext=[t.strftime('%Y-%m-%d %H:%M') for t in tick_values],
                tickvals=tick_values,
                tickangle=45,
                row=2, col=1,  # 只在底部子图显示完整的时间轴标签
                rangebreaks=[
                    dict(bounds=["sat", "mon"]),  # 去除周末
                    dict(bounds=[15.5, 9.5], pattern="hour"),  # 去除非交易间
                ]
            )

            # 在上面的子图中隐藏X轴标签，但保持相同的时间轴设置
            fig.update_xaxes(
                showticklabels=False, 
                type='category',
                rangebreaks=[
                    dict(bounds=["sat", "mon"]),
                    dict(bounds=[15.5, 9.5], pattern="hour"),
                ],
                row=1, col=1
            )

            # 配置交互选项
            fig.update_xaxes(showspikes=True, spikemode='across', spikesnap='cursor')
            fig.update_yaxes(showspikes=True, spikemode='across', spikesnap='cursor')

            # 生成HTML并显示
            html_content = f'''
            <html>
                <head>
                    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                    <script>
                        document.addEventListener('keydown', function(e) {{
                            var chart = document.getElementsByClassName('plotly-graph-div')[0];
                            var update = {{}};
                            
                            function getDataInRange(start, end) {{
                                var data = chart.data[0];
                                var prices = [];
                                var volumes = [];
                                for (var i = Math.max(0, Math.floor(start)); i <= Math.min(data.x.length - 1, Math.ceil(end)); i++) {{
                                    prices.push(data.high[i], data.low[i]);
                                    volumes.push(chart.data[1].y[i]);
                                }}
                                return {{
                                    minPrice: Math.min(...prices),
                                    maxPrice: Math.max(...prices),
                                    minVolume: Math.min(...volumes),
                                    maxVolume: Math.max(...volumes)
                                }};
                            }}

                            switch(e.key) {{
                                case 'ArrowLeft':
                                    var currentRange = chart._fullLayout.xaxis.range;
                                    var shift = Math.round((currentRange[1] - currentRange[0]) * 0.1);
                                    var newStart = Math.max(0, currentRange[0] - shift);
                                    var newEnd = Math.max(shift, currentRange[1] - shift);
                                    var rangeData = getDataInRange(newStart, newEnd);
                                    update = {{
                                        'xaxis.range': [newStart, newEnd],
                                        'yaxis.range': [rangeData.minPrice * 0.99, rangeData.maxPrice * 1.01],
                                        'yaxis2.range': [0, rangeData.maxVolume * 1.05]
                                    }};
                                    break;
                                case 'ArrowRight':
                                    var currentRange = chart._fullLayout.xaxis.range;
                                    var shift = Math.round((currentRange[1] - currentRange[0]) * 0.1);
                                    var maxIndex = chart.data[0].x.length - 1;
                                    var newStart = Math.min(maxIndex - shift, currentRange[0] + shift);
                                    var newEnd = Math.min(maxIndex, currentRange[1] + shift);
                                    var rangeData = getDataInRange(newStart, newEnd);
                                    update = {{
                                        'xaxis.range': [newStart, newEnd],
                                        'yaxis.range': [rangeData.minPrice * 0.99, rangeData.maxPrice * 1.01],
                                        'yaxis2.range': [0, rangeData.maxVolume * 1.05]
                                    }};
                                    break;
                                case 'ArrowUp':
                                    var currentRange = chart._fullLayout.xaxis.range;
                                    var center = Math.floor((currentRange[0] + currentRange[1]) / 2);
                                    var span = Math.floor((currentRange[1] - currentRange[0]) * 0.4);
                                    var newStart = Math.max(0, center - span);
                                    var newEnd = Math.min(chart.data[0].x.length - 1, center + span);
                                    var rangeData = getDataInRange(newStart, newEnd);
                                    update = {{
                                        'xaxis.range': [newStart, newEnd],
                                        'yaxis.range': [rangeData.minPrice * 0.99, rangeData.maxPrice * 1.01],
                                        'yaxis2.range': [0, rangeData.maxVolume * 1.05]
                                    }};
                                    break;
                                case 'ArrowDown':
                                    var currentRange = chart._fullLayout.xaxis.range;
                                    var center = Math.floor((currentRange[0] + currentRange[1]) / 2);
                                    var span = Math.floor((currentRange[1] - currentRange[0]) * 0.6);
                                    var newStart = Math.max(0, center - span);
                                    var newEnd = Math.min(chart.data[0].x.length - 1, center + span);
                                    var rangeData = getDataInRange(newStart, newEnd);
                                    update = {{
                                        'xaxis.range': [newStart, newEnd],
                                        'yaxis.range': [rangeData.minPrice * 0.99, rangeData.maxPrice * 1.01],
                                        'yaxis2.range': [0, rangeData.maxVolume * 1.05]
                                    }};
                                    break;
                            }}
                            
                            if (Object.keys(update).length > 0) {{
                                Plotly.relayout(chart, update);
                            }}
                        }});
                    </script>
                </head>
                <body>
                    {fig.to_html(full_html=False, include_plotlyjs=False)}
                </body>
            </html>
            '''
            
            self.web_view.setHtml(html_content)
            
            # 2. 然后异步调用Kimi分析
            QApplication.processEvents()  # 确保K线图先显示出来
            # ... exist


            # 根据股票代码判断交易所
            if ticker.startswith('6'):
                market_code = 'SH'
            elif ticker.startswith(('0', '3')):
                market_code = 'SZ'
            else:
                market_code = 'BJ'  # 北交所
                
            # 构造完整的股票代码
            full_ticker = f"{market_code}{ticker}"
            print(f"构造的完整股票代码: {full_ticker}")
                
                # 获取股票题材信息
            stock_info = get_stock_special_topics(full_ticker)
           
            if stock_info is not None:
                # 如果对话框已存在，先关闭它
                if self.analysis_dialog is not None:
                    self.analysis_dialog.close()
                
                # 创建新的对话框
                self.analysis_dialog = QDialog(self)
                self.analysis_dialog.setWindowTitle(f"{stock_name}题材信息")
                self.analysis_dialog.setMinimumSize(600, 500)
                
                layout = QVBoxLayout()
                
                # 使用QTextEdit显示分析结果
                text_edit = QTextEdit()
                text_edit.setReadOnly(True)
                
                # 格式化显示内容
                formatted_text = f"""【所属板块】
{', '.join(stock_info['所属板块']) if stock_info['所属板块'] else '无数据'}

【经营范围】
{stock_info['经营范围'] if stock_info['经营范围'] else '无数据'}

【主营业务】
{stock_info['主营业务'] if stock_info['主营业务'] else '无数据'}

【行业背景】
{stock_info['行业背景'] if stock_info['行业背景'] else '无数据'}

【核心竞争力】"""

                # 添加核心竞争力的详细信息
                if stock_info['核心竞争力']:
                    for item in stock_info['核心竞争力']:
                        formatted_text += f"\n\n{item['标题']}:\n{item['内容']}"
                else:
                    formatted_text += "\n无数据"
                
                text_edit.setPlainText(formatted_text)
                layout.addWidget(text_edit)
                
                # 添加关闭按钮
                close_button = QPushButton("关闭")
                close_button.clicked.connect(self.analysis_dialog.close)
                layout.addWidget(close_button)
                
                self.analysis_dialog.setLayout(layout)
                self.analysis_dialog.show()
            else:
                # 如果获取信息失败，显示错误消息
                QMessageBox.warning(self, "错误", "获取股票题材信息失败")




        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to fetch or plot data: {str(e)}")
            print(f"Error details: {str(e)}")

    def on_row_click(self, index):
        """
        处理表格行双击事件
        :param index: QModelIndex of the clicked row
        """
        row = index.row()
        column = index.column()

        # 检查是否双击了"股票代码"列
        if column == self.ticker_column_index:
            ticker = self.table.item(row, column).text()


            self.plot_kline(ticker)

    def open_file(self):
        """
        打开Excel文件对话框
        """
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Excel File", "", "Excel files (*.xlsx *.xls)")
        if file_path:
            df = self.load_excel(file_path)
            df['股票代码'] = df['股票代码'].apply(lambda x: f"{x:0>4}" if len(str(x)) == 4 else str(x))

            if df is not None:
                # 设置表格列数和列名
                self.table.setColumnCount(len(df.columns))
                self.table.setHorizontalHeaderLabels(df.columns.tolist())

                # 记录"股票代码"列的索引
                if "股票代码" in df.columns:
                    self.ticker_column_index = df.columns.get_loc("股票代码")
                else:
                    QMessageBox.critical(self, "Error", "Excel file does not contain a '股票代码' column.")
                    return

                # 清空现有数据
                self.table.setRowCount(0)

                # 插入新数据
                for index, row in df.iterrows():
                    self.table.insertRow(self.table.rowCount())
                    for col_index, value in enumerate(row):
                        self.table.setItem(self.table.rowCount() - 1, col_index, QTableWidgetItem(str(value)))
                #

    def update_chart(self):
        """
        当周期选择改变时更新图表
        """
        # 获取当前选中的行
        current_row = self.table.currentRow()
        if current_row >= 0:  # 确保有选中的行
            # 获取股票代码列的值
            ticker = self.table.item(current_row, self.ticker_column_index).text()
            self.plot_kline(ticker)

    def on_analysis_dialog_closed(self):
        """
        当分析对话框关闭时调用
        """
        self.analysis_dialog = None
        self.current_analysis_ticker = None
def is_similar(new_content, existing_contents, api_key, threshold=0.8):
    """
    使用 DeepSeek API 通过提示词检查新内容是否与现有内容相似。
    """
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    for existing_content in existing_contents:
        prompt = f"请判断以下两段文本是否相似，并给出相似度评分（0到1之间）：\n\n文本1：{new_content}\n\n文本2：{existing_content}\n\n相似度评分："
        print('prompt',prompt)
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个文本相似性分析专家。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=10
        )
        
        similarity_score = float(response.choices[0].message.content.strip())
        print('similarity_score',similarity_score)
        sys.exit(1)  # 终止程序，返回状态码 1
        if similarity_score > threshold:
            return True



    return False
def get_eastmoney_search_news(keyword):
    """
    获取东方财富搜索页面的新闻
    keyword: 搜索关键词，如 "恒立钻具"
    """
    import requests
    import pandas as pd
    import json
    from datetime import datetime
    from urllib.parse import quote
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
                "searchScope": "title",
                "sort": "default",
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
        response = requests.get(url, params=params, headers=headers)
        response.encoding = 'utf-8'  # 设置响应编码
        
        # 处理返回的jsonp数据
        data = response.text
        json_str = data.strip('jQuery(')[:-1]
        json_data = json.loads(json_str)
        
        # 提取新闻列表
        news_list = json_data['result']['cmsArticleWebOld']
        
        # 转换为DataFrame
        news_data = []
        for news in news_list:
            # 只保留标题中包含关键词的新闻
            if keyword in news['title'].replace('<em>', '').replace('</em>', ''):
                news_data.append({
                    '新闻标题': news['title'].replace('<em>', '').replace('</em>', ''),
                    '新闻内容': news['content'].replace('<em>', '').replace('</em>', ''),
                    '发布时间': news['date'],
                    '文章来源': news['mediaName'],
                    '新闻链接': news['url']
                })
        
        df = pd.DataFrame(news_data)
        print(f"\n共获取到 {len(df)} 条标题相关新闻")
        return df
        
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

def get_stock_info(symbol, api_key):
    """
    获取股票的新闻和财务指标
    symbol: 股票代码，如 "000001"
    """
    # 导入必要的库
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import jieba
    import numpy as np
    
    # 定义文本预处理函数
    def preprocess_text(text):
        # 使用jieba分词
        words = jieba.cut(text)
        return ' '.join(words)
    
    # 获取股票名称
    try:
        stock_info = ak.stock_zh_a_spot_em()
        stock_name = stock_info[stock_info['代码'] == symbol]['名称'].values[0]
    except:
        stock_name = symbol
    

    news_df = get_eastmoney_search_news(stock_name)
    


    news_df = news_df.head(30)  # 只保留最近30条新闻
    news_df.to_csv('news_df.csv',index=False)
    # 首先在DataFrame中添加新列
    news_df['新闻内容全'] = ''  # 创建空列

    # 然后遍历处理每条新闻
    for index, row in news_df.iterrows():
        url = row['新闻链接']
        print('url',get_full_content(url))
        #将新闻内容全列中添加新闻内容
        news_df.at[index, '新闻内容全'] = get_full_content(url)

    #如果
    for _, row in news_df.iterrows():
        print('-------',row['新闻内容全'])

    print(f"获取到 {len(news_df)} 条原始新闻")
    for index, row in news_df.iterrows():
        # 添加标记列用于追踪要保留的新闻
        news_df['keep'] = True
        
        # 获取所有新闻整内容（标题+内容）并预处理
        contents = [(row['新闻内容全']) for _, row in news_df.iterrows()]
        processed_contents = [preprocess_text(text) for text in contents]
        
        # 创建TF-IDF向量化器
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(processed_contents)
        
        total = len(contents)
        
        # 遍历每条新闻
        for i in range(total):
            # 如果当前新闻已被标记删除，跳过
            if not news_df.iloc[i]['keep']:
                continue
                
            #print(f"\n检查新闻 {i+1}/{total}: {news_df.iloc[i]['新闻标题']}")
            
            # 从第一条新闻开始比较
            for j in range(0, total):
                # 跳过自身和已标记删除的新闻
                if i == j or not news_df.iloc[j]['keep']:
                    continue
                
                # 计算余弦相似度

                
                similarity = cosine_similarity(
                    tfidf_matrix[i:i+1], 
                    tfidf_matrix[j:j+1]
                )[0][0]
                
                print(f"与新闻 {j+1} 相似度: {similarity:.2f}")
                
                # 相似度超过0.8则标记删除较新的一条
                if similarity > 0.5:
                    time_i = pd.to_datetime(news_df.iloc[i]['发布时间'])
                    time_j = pd.to_datetime(news_df.iloc[j]['发布时间'])
                    #print('time_i',news_df.iloc[i]['新闻内容全'])
                    #print('time_j',news_df.iloc[j]['新闻内容全'])
                    if time_i <= time_j:
                        # 保留较早的新闻i，删除较新的新闻j
                        news_df.iloc[j, news_df.columns.get_loc('keep')] = False
                        print(f"标记较新的相似新闻删除: {news_df.iloc[j]['新闻标题']}")
                    else:
                        # 保留较早的新闻j，删除较新的新闻i
                        news_df.iloc[i, news_df.columns.get_loc('keep')] = False
                        print(f"标记较新的相似新闻删除: {news_df.iloc[i]['新闻标题']}")
                        # 当前新闻被删除，不再继续比较


            
            # 如果当前新闻已被标记删除，继续下一条新闻
            if not news_df.iloc[i]['keep']:
                continue
    
    # 只保留标记为True的新闻
    news_df = news_df[news_df['keep']].drop('keep', axis=1)
    print(f"\n过滤后保留 {len(news_df)} 条不相似新闻")
    
    # 整合信息
    info = {
        "股票代码": symbol,
        "股票名称": stock_name,
        "获取时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "新闻": [],
        "财务指标": {}
    }
    
    # 处理过滤后的新闻
    for _, row in news_df.iterrows():
        print(row['新闻内容全'])
        news_info = {
            "标题": row['新闻标题'],
            "内容": row['新闻内容全'],
            "链接": row['新闻链接'],
            "发布时间": row['发布时间']
        }
        info["新闻"].append(news_info)
    
    # 导出为txt文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"stock_info_{symbol}_{timestamp}.txt"
    
    with open(filename, 'w', encoding='utf-8') as f:
        # 写入基本信息
        f.write(f"股票代码：{info['股票代码']}\n")
        f.write(f"股票名称：{info['股票名称']}\n")
        f.write(f"获取时间：{info['获取时间']}\n")
        f.write("\n=== 新闻列表 ===\n\n")
        
        # 写入新闻信息
        for i, news in enumerate(info["新闻"], 1):
            f.write(f"新闻 {i}\n")
            f.write(f"标题：{news['标题']}\n")
            f.write(f"发布时间：{news['发布时间']}\n")
            f.write(f"链接：{news['链接']}\n")
            f.write(f"内容：{news['内容']}\n")
            f.write("\n" + "-"*50 + "\n\n")  # 分隔线
        
        # 写入财务指标（如果有）
        if info["财务指标"]:
            f.write("\n=== 财务指标 ===\n")
            for key, value in info["财务指标"].items():
                f.write(f"{key}：{value}\n")
    
    print(f"\n信息已保存到文件：{filename}")
    return news_df,info["财务指标"]

def get_dongmi_qa(stock_name):
    driver = None
    try:
        # 配置 Chrome
        options = uc.ChromeOptions()
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        
        # 初始化浏览器
        print("正在启动浏览器...")
        driver = uc.Chrome(options=options)
        
        # 设置页面加载超时
        driver.set_page_load_timeout(60)
        
        # 构  URL并访问
        url = f"https://so.eastmoney.com/qa/s?keyword={stock_name}"
        print(f"正在访问: {url}")
        driver.get(url)
        
        print("等待页面加载...")
        time.sleep(10)
        
        qa_items = []
        page_num = 1
        max_pages = 10  # 添加页数限制
        
        while page_num <= max_pages:  # 修改循环条件
            print(f"\n正在获取第 {page_num} 页...")
            
            # 等待问答内容加载
            try:
                # 获取所有问答list

                qa_list_item= WebDriverWait(driver, 10).until(
                    EC.presence_of_all_elements_located((By.CLASS_NAME, "qa_list_item"))
                )
                #qa_pair=
                #qa_pairs = WebDriverWait(driver, 10).until(
                #    EC.presence_of_all_elements_located((By.CLASS_NAME, "qa_content"))
                #)
                print('qa_pairs',qa_list_item)
                for qa_pair in qa_list_item:
                    try:
                        print('qa_pair',qa_pair)
                        name=qa_pair.find_element(By.CLASS_NAME, "avatar_name").text
                        print('name',name)
                        # 提取问题
                        question_element = qa_pair.find_element(By.CLASS_NAME, "qa_question_text")
                        question_text = question_element.find_element(By.TAG_NAME, "a").text
                        
                        # 提取回答
                        answer_element = qa_pair.find_element(By.CLASS_NAME, "qa_answer_text")
                        answer_text = answer_element.find_element(By.TAG_NAME, "a").text
                        date_text = answer_element.find_element(By.CLASS_NAME, "qa_answer_date").text
                        
                        # 清理文本
                        answer_text = answer_text.replace(f'{stock_name}：', '').strip()
                        question_text = question_text.strip()
                        #print('qx',answer_text)
                        
                        qa_info = {
                            "问题": question_text,
                            "回答": answer_text,
                            "日期": date_text,
                            "页码": page_num
                        }
                        if name==stock_name:
                            print('qa_info',name)
                            qa_items.append(qa_info)
                        print(f"成功提取第 {len(qa_items)} 个问答")
                        #print(qa_items)
                    except Exception as e:
                        print(f"提取问答数据失败: {str(e)}")
                        continue
                
                # 检查是否有下一页
                try:
                    next_button = driver.find_element(By.CSS_SELECTOR,"a[title='下一页']")
                    if "disabled" in next_button.get_attribute("class"):
                        print("已到最后一页")
                        break
                    else:
                        next_button.click()
                        print("点击下一页")
                        time.sleep(3)
                        page_num += 1
                except Exception as e:
                    print(f"翻页失败: {str(e)}")
                    break
                    
            except Exception as e:
                print(f"获取问答内容失败: {str(e)}")
                break
            
        # 保存数据


            # 对回答进行相似度对比，保留互相不通的回答
        if qa_items:
            df = pd.DataFrame(qa_items)
            
            # 添加标记列用于追踪要保留的问答
            df['keep'] = True
            
            # 对所有回答内容进行预处理
            def preprocess_text(text):
                # 使用jieba分词
                words = jieba.cut(text)
                return ' '.join(words)
            
            # 预处理所有回答
            processed_answers = [preprocess_text(text) for text in df['回答']]
            
            # 创建TF-IDF向量化器
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(processed_answers)
            
            # 遍历每条问答
            for i in range(len(df)):
                # 如果当前问答已被标记删除，跳过
                if not df.iloc[i]['keep']:
                    continue
                
                # 与其他问答比较相似度
                for j in range(i + 1, len(df)):
                    # 跳过标记删除的问答
                    if not df.iloc[j]['keep']:
                        continue
                    
                    # 计算余弦相似度
                    similarity = cosine_similarity(
                        tfidf_matrix[i:i+1], 
                        tfidf_matrix[j:j+1]
                    )[0][0]
                    
                    # 如果相似度超过阈值，保留较新的一条
                    if similarity > 0.8:
                        date_i = pd.to_datetime(df.iloc[i]['日期'])
                        date_j = pd.to_datetime(df.iloc[j]['日期'])
                        
                        if date_i <= date_j:
                            # 保留较新的问答j，删除较旧的问答i
                            df.iloc[i, df.columns.get_loc('keep')] = False
                            break
                        else:
                            # 保留较新的问答i，删除较旧的问答j
                            df.iloc[j, df.columns.get_loc('keep')] = False
            
            # 只保留标记为True的问答
            df = df[df['keep']].drop('keep', axis=1)
            print(f"\n过滤后保留 {len(df)} 条不相似问答")

# ... existing code ...



            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"dongmi_qa_{stock_name}_{timestamp}.csv"
            df.to_csv(filename, index=False, encoding='utf-8-sig')
            print(f"\n数据已保存到: {filename}")
            return df
        else:
            print("未获取到任何数据")
            return None
            
    except Exception as e:
        print(f"获取数据失败: {str(e)}")
        return None
        
    finally:
        if driver:
            try:
                # 强制结束所有Chrome进程
                import psutil
                for proc in psutil.process_iter(['pid', 'name']):
                    try:
                        if 'chrome' in proc.info['name'].lower():
                            psutil.Process(proc.info['pid']).terminate()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                print("Chrome进程已清理")
            except Exception as e:
                print(f"清理Chrome进程时出现异常: {str(e)}")

def analyze_qa_content(qa_text,money_info, stock_name, api_key):
    """
    分析文本内容，按照指定大小分批处理，保持新闻完整性
    """

    #将new_list转变为dataframe
    news_list =qa_text
    max_chunk_size = 4000
    chunks = []
    current_chunk = ""
    company=stock_name
    # 使用正则表达式提取新闻内容
    import re
    #pattern = r"新闻开始(.*?)新闻结束"
    #news_items = re.findall(pattern, news_list, re.DOTALL)
    #print(f"找到 {len(news_items)} 条新闻",len(str(news_items)))
    qa_results = []
    chunks.append(money_info)
    chunks.append(get_dongmi_qa(stock_name))
    # 遍历DataFrame的每一行
    for idx, row in news_list.iterrows():
        item = str(row['新闻内容全'])
        chunks.append(item)
    
    # 处理每条新闻
    # 如果单条新闻超过最大长度
        #逐块遍历chunks
    # 检查并切分过大的chunks
    final_chunks = []
    for chunk in chunks:
        # 如果当前chunk超过最大长度，需要切分
        if len(str(chunk)) > max_chunk_size:
            # 切分当前chunk
            start_idx = 0
            while start_idx < len(str(chunk)):
                end_idx = min(start_idx + max_chunk_size, len(str(chunk)))
                
                # 查找最后一个完整句子的位置
                if end_idx < len(str(chunk)):
                    last_sentence = max(
                        str(chunk)[start_idx:end_idx].rfind('。'),
                        str(chunk)[start_idx:end_idx].rfind('！'),
                        str(chunk)[start_idx:end_idx].rfind('？')
                    )
                    if last_sentence != -1:
                        end_idx = start_idx + last_sentence + 1
                
                final_chunks.append(str(chunk)[start_idx:end_idx])
                start_idx = end_idx
        else:
            # 如果不超过最大长度，直接添加
            final_chunks.append(chunk)

    print(f"最终分成 {len(final_chunks)} 块进行处理")
    chunks = final_chunks  # 更新chunks列表

    all_analysis = []


    # 处理每个chunk
    for i, chunk_text in enumerate(chunks):
        print(f"\n处理第 {i+1}/{len(chunks)} 批...",len(chunk_text),len(str(chunk_text)))
        
        # 修改提示词,明确说明这是分批输入
        current_prompt = f"""
        我将分{len(chunks)}批输入关于{company}公司的信息,这是第{i+1}批。
        请分析以下内容并提取要点,稍后我会要求你对所有批次进行总结。
        
        本批内容:
        {chunk_text}
        
        请从以下角度提取关键信息:
        1. 业务相关:主要产品、市场地位、发展战略等
        2. 财务相关:营收、利润、现金流等关键指标
        3. 重要事项:重大合同、技术突破、风险提示等
        4. 舆论情况:市场关注点、投资者关心问题、潜在市场影响因素
        5.其他不重合信息或你觉得有价值的信息
        
        注:这只是分批分析,最终我会要求你对全部内容进行综合总结。
        """

        try:
            client = OpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com/"
            )
            
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "你是一个金融分析师。"},
                    {"role": "user", "content": current_prompt}
                ],
                temperature=0.7,
                max_tokens=4000
            )
            
            analysis = response.choices[0].message.content
            all_analysis.append(analysis)
            print(f"完成第 {i+1} 批分析")
            
        except Exception as e:
            print(f"处理第 {i+1} 批时出错: {str(e)}")
            continue
            
        time.sleep(2)  # 请求间隔
        
    if not all_analysis:
        return "所有批次处理均失败"
        
    # 最后进行总结分析
    summary_text = '\n'.join(all_analysis)
    if len(summary_text) > max_chunk_size:
        summary_text = summary_text[:max_chunk_size]
        
    final_prompt = f"""
    请你作为一个信息整理员，严格基于提供的{len(chunks)}批次关于{company}公司的原始材料进行客观汇总。

    要求：
    1. 严格限制在提供的材料范围内进行汇总，不要添加任何推测或编造的内容
    2. 如果某个方面在材料中没有提及，直接标注"材料中未提及"
    3. 保留原始材料中的具体数据和时间点
    4. 对重复信息进行合并但保留最完整的表述
    5. 使用"根据材料显示"、"材料提到"等字样明确信息来源

    {summary_text}

    请按以下维度进行客观汇总：

            【基本面】
            1. 主营业务
            2. 行业地位

            【经营情况】
            1. 业务进展
            2. 财务状况
            3. 风险因素
            【舆论情况】
            1. 市场、网友关注重点内容
            2. 潜在影响

            【投资建议】
            1. 关注重点
            2. 风险提示
            

    特别说明：
    1. 本汇总严格限制在原始材料范围内
    2. 所有数据和结论均需可在原始材料中找到对应内容
    3. 如遇材料中数据矛盾，请标注"材料存在数据差异"
    4. 不进行任何主观评价和预测
    """
    
    try:
        final_response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个金融分析师。"},
                {"role": "user", "content": final_prompt}
            ],
            temperature=0.7,
            max_tokens=4000
        )
        
        final_analysis = final_response.choices[0].message.content
        
    except Exception as e:
        print(f"生成最终分析时出错: {str(e)}")
        final_analysis = "生成最终分析失败"
    
    # 保存分析结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"qa_analysis_{company}_{timestamp}.txt"
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"公司：{company}\n")
        f.write(f"分析时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("=== 最终分析 ===\n")
        f.write(final_analysis)
        f.write("\n\n=== 分批分析详情 ===\n")
        for i, analysis in enumerate(all_analysis):
            f.write(f"\n--- 第 {i+1} 批分析 ---\n")
            f.write(analysis)
    
    print(f"\n分析结果已保存到: {filename}")
    return final_analysis
        
def get_concept_stocks_data():
    """
    获取前10个概念板块的股票数据，并获取其历史成交额
    """
    try:
        # 获取所有概念板块名称
        print("正在获取概念板块列表...")
        concept_df = ak.stock_board_concept_name_em()
        concepts = concept_df['板块名称'].tolist() # 先测试2个概念板块
        print(f"将处理以下{len(concepts)}个概念板块:")
        for i, concept in enumerate(concepts, 1):
            print(f"{i}. {concept}")

        # 创建空的DataFrame用于存储结果
        result_df = pd.DataFrame()
        
        # 遍历每个概念板块
        for concept in concepts:
            try:
                print(f"\n正在处理概念板块: {concept}")
                # 获取该概念板块的成分股
                stocks = ak.stock_board_concept_cons_em(symbol=concept)
                stocks =stocks.drop([col for col in stocks.columns if '序号' in col], axis=1)

                if '序号_x' in stocks.columns:
                    stocks =stocks.drop('序号_x', axis=1)
                if '序号_y' in stocks.columns:
                    stocks= stocks.drop('序号_y', axis=1)
                if '序号' in stocks.columns:
                    stocks= stocks.drop('序号', axis=1)

                # 打印调试信息
                print("stocks的列名:", stocks.columns.tolist())
                print("stocks的前两行数据:")
                print(stocks.head(2))
                if '序号' in stocks.columns:
                    stocks = stocks.drop('序号', axis=1)
                if not stocks.empty:
                    # 添加概念板块列
                    stocks['概念板块_' + concept] = 1
                    
                    # 如果是第一个板块，保留所有信息
                    if result_df.empty:
                        # 保留所有原始列
                        result_df = stocks.copy()
                        # 添加概念板块标记
                        result_df[f'概念板块_{concept}'] = 1
                    else:
                        # 合并数据，保留所有列
                        # 先给新数据添加概念板块标记

                        stocks_with_concept = stocks.copy()
                        stocks_with_concept[f'概念板块_{concept}'] = 1
                        result_df = result_df.drop([col for col in result_df.columns if '序号_x' in col], axis=1)
                        stocks_with_concept = stocks_with_concept.drop([col for col in stocks_with_concept.columns if '序号_y' in col], axis=1)
                        # 打印调试信息
                        print("\nresult_df的列名:", result_df.columns.tolist())
                        print("stocks_with_concept的列名:", stocks_with_concept.columns.tolist())
                        
                        # 使用 outer join 合并，保留所有列
                        result_df = pd.merge(
                            result_df,
                            stocks_with_concept,
                            on=['代码', '名称', '最新价', '涨跌幅', '涨跌额', '成交量', '成交额',
                                 '振幅', '最高', '最低', '今开', '昨收', '换手率', '市盈率-动态', '市净率' ],
                            how='outer'
                        )
                        if '序号_x' in result_df.columns:
                            result_df =result_df.drop('序号_x', axis=1)
                        if '序号_y' in result_df.columns:
                            result_df= result_df.drop('序号_y', axis=1)
                time.sleep(1)  # 添加延时避免请求过快
                
            except Exception as e:
                print(f"处理概念板块 {concept} 时出错: {str(e)}")
                continue

        # 打印调试信息
        print("\n合并后的result_df列名:", result_df.columns.tolist())
        
        # 填充NaN值为0
        concept_columns = [col for col in result_df.columns if col.startswith('概念板块_')]
        result_df[concept_columns] = result_df[concept_columns].fillna(0)
        
        # 获取每只股票的昨日成交额
        print("\n获取股票昨日成交额数据...")
        result_df['昨日成交额'] = None
        
        for index, row in result_df.iterrows():
            try:
                stock_code = row['代码']
                print(f"处理股票: {stock_code}")
                
                # 获取昨日历史数据
                hist_data = ak.stock_zh_a_hist(symbol=stock_code, period="daily", adjust="qfq")
                if not hist_data.empty:
                    # 获取最近第二天的数据（昨天）
                    result_df.loc[index, '昨日成交额'] = hist_data.iloc[-2]['成交额']
                
                time.sleep(0.5)  # 添加延时避免请求过快
                
            except Exception as e:
                print(f"获取股票 {stock_code} 昨日成交额时出错: {str(e)}")
                continue
        print('z1',result_df.columns)
        # 计算成交额变化
        result_df['成交额变化'] = result_df['成交额'] - result_df['昨日成交额']
        
        # 计算每只股票所属的概念板块数量
        result_df['所属概念数'] = result_df[concept_columns].sum(axis=1)
        
        # 整理最终输出的列
        final_columns = ['代码', '名称', '所属概念数', '成交额', '昨日成交额', '成交额变化','涨跌幅',\
         '涨跌额', '成交量','振幅',  '换手率', '市盈率-动态', '市净率'] + concept_columns
        result_df = result_df[final_columns]
        
        # 按成交额变化排序
        result_df = result_df.sort_values('成交额变化', ascending=False)
        
        # 打印概要信息
        print("\n数据统计:")
        print(f"总股票数: {len(result_df)}")
        print(f"处理的概念板块数: {len(concepts)}")
        print("\n成交额变化前5名股票:")
        print(result_df[['代码', '名称', '成交额', '昨日成交额', '成交额变化', '所属概念数']].head())
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'concept_stocks_{timestamp}.csv'
        result_df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"\n数据已保存到: {filename}")
        
        return result_df
        
    except Exception as e:
        print(f"获取数据失败: {str(e)}")
        return None
def analyze_concept_volume_changes(result_df):
    """
    分析各概念板块股票成交量变化情况
    """
    try:
        # 获取所有概念板块列
        concept_columns = [col for col in result_df.columns if col.startswith('概念板块_')]
        
        # 确保数值列为数值类型
        numeric_columns = ['成交额', '昨日成交额', '成交额变化']
        for col in numeric_columns:
            if col in result_df.columns:
                result_df[col] = pd.to_numeric(result_df[col].replace({'-': '0'}), errors='coerce')
        
        analysis_results = []
        
        for concept_col in concept_columns:
            concept_name = concept_col.replace('概念板块_', '')
            
            # 获取该概念板块的股票
            concept_stocks = result_df[result_df[concept_col] == 1].copy()
            
            # 跳过空数据的概念板块
            if concept_stocks.empty:
                print(f"跳过空数据的概念板块: {concept_name}")
                continue
                
            try:
                # 处理可能的NA值
                valid_changes = concept_stocks['成交额变化'].dropna()
                if valid_changes.empty:
                    print(f"概念板块 {concept_name} 没有有效的成交额变化数据")
                    continue
                
                analysis = {
                    '概念板块': concept_name,
                    '股票数量': len(concept_stocks),
                    '总成交额': concept_stocks['成交额'].sum(),
                    '昨日总成交额': concept_stocks['昨日成交额'].sum(),
                    '成交额变化总额': valid_changes.sum(),
                    '成交额变化均值': valid_changes.mean(),
                    '成交额变化中位数': valid_changes.median(),
                    '上涨家数': len(valid_changes[valid_changes > 0]),
                    '下跌家数': len(valid_changes[valid_changes <= 0])
                }
                print('z2',analysis)
                # 安全地获取最大和最小变化的股票信息
                if not valid_changes.empty:
                    try:
                        max_change_idx = valid_changes.idxmax()
                        min_change_idx = valid_changes.idxmin()
                        
                        analysis.update({
                            '最大成交额变化': concept_stocks.loc[max_change_idx, '成交额变化'],
                            '最大变化股票': concept_stocks.loc[max_change_idx, '名称'],
                            '最小成交额变化': concept_stocks.loc[min_change_idx, '成交额变化'],
                            '最小变化股票': concept_stocks.loc[min_change_idx, '名称']
                        })
                    except Exception as e:
                        print(f"获取最大/最小变化股票信息时出错: {concept_name}, {str(e)}")
                        analysis.update({
                            '最大成交额变化': valid_changes.max(),
                            '最大变化股票': 'NA',
                            '最小成交额变化': valid_changes.min(),
                            '最小变化股票': 'NA'
                        })
                
                # 安全计算成交额变化率
                if analysis['昨日总成交额'] != 0:
                    analysis['成交额变化率'] = ((analysis['总成交额'] - analysis['昨日总成交额']) / 
                                          analysis['昨日总成交额'] * 100)
                else:
                    analysis['成交额变化率'] = 0
                    print(f"警告: {concept_name} 昨日总成交额为0")
                
                analysis_results.append(analysis)
                
            except Exception as e:
                print(f"处理概念板块 {concept_name} 时出错: {str(e)}")
                continue
        
        # 转换为DataFrame
        if not analysis_results:
            print("没有有效的分析结果")
            return None
            
        analysis_df = pd.DataFrame(analysis_results)
        
        # 按成交额变化率排序
        analysis_df = analysis_df.sort_values('成交额变化率', ascending=False)
        
        # 格式化数值
        # 定义列名和单位的映射
        column_units = {
            '总成交额': '（亿）',
            '昨日总成交额': '（亿）',
            '成交额变化总额': '（亿）',
            '成交额变化均值': '（亿）',
            '成交额变化中位数': '（亿）',
            '最大成交额变化': '（亿）',
            '最小成交额变化': '（亿）'
        }
        
        # 重命名列，添加单位
        analysis_df = analysis_df.rename(columns={col: col + unit for col, unit in column_units.items()})
        
        # 格式化数值（不添加单位）
        format_columns = [col + unit for col, unit in column_units.items()]
        for col in format_columns:
            if col in analysis_df.columns:
                analysis_df[col] = analysis_df[col].apply(
                    lambda x: f"{x/100000000:.2f}" if pd.notnull(x) else "NA"
                )
        
        # 成交额变化率保持百分比格式
        if '成交额变化率' in analysis_df.columns:
            analysis_df['成交额变化率（%）'] = analysis_df['成交额变化率'].apply(
                lambda x: f"{x:.2f}" if pd.notnull(x) else "NA"
            )
            analysis_df = analysis_df.drop('成交额变化率', axis=1)
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'concept_analysis_{timestamp}.csv'
        analysis_df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"\n分析结果已保存到: {filename}")
        
        # 打印概要信息
        print("\n=== 概念板块成交额变化分析 ===")
        print(f"\n总计分析 {len(analysis_df)} 个概念板块")
        print("\n成交额变化率前5名概念板块:")
        print(analysis_df[['概念板块', '股票数量', '成交额变化率', '上涨家数', '下跌家数']].head())
        print("\n成交额变化率后5名概念板块:")
        print(analysis_df[['概念板块', '股票数量', '成交额变化率', '上涨家数', '下跌家数']].tail())
        
        return analysis_df
        
    except Exception as e:
        print(f"分析概念板块数据时出错: {str(e)}")
        return None
# 使用示例：
# df = get_concept_stocks_data()
# analysis = analyze_concept_volume_changes(df)

def update_today_volume(df):
    """
    更新CSV文件中的今日成交额数据并重新计算变化
    
    Parameters:
    csv_path: str, 昨天生成的CSV文件路径
    
    Returns:
    DataFrame: 更新后的数据
    """
    try:
        # 读取昨天的CSV文件

        print(f"读取到 {len(df)} 条股票数据")
        
        # 获取今天的实时数据
        today_data = ak.stock_zh_a_spot_em()
        print("获取到今日实时数据")
        
        # 确保股票代码格式一致（保持前导零）
        df['代码'] = df['代码'].astype(str).str.zfill(6)
        today_data['代码'] = today_data['代码'].astype(str).str.zfill(6)
        
        # 更新成交额
        for index, row in df.iterrows():
            try:
                stock_code = row['代码']
                print('zz',stock_code)
                # 在今日数据中查找对应股票
                today_stock = today_data[today_data['代码'] == stock_code]

                if not today_stock.empty:
                    # 更新成交额
                    #把df的成交额改为昨日成交额
                    print('today',df.loc[index, '成交额'])
                    df.loc[index, '昨日成交额'] = df.loc[index, '成交额']
                    print('zr',df.loc[index, '昨日成交额'])
                    df.loc[index, '成交额'] = today_stock.iloc[0]['成交额']
                    print('xx',stock_code,today_stock.iloc[0]['成交额'])
                    # 计算成交额变化
                    df.loc[index, '成交额变化'] = df.loc[index, '成交额'] - df.loc[index, '昨日成交额']
                else:
                    print(f"未找到股票 {stock_code} 的今日数据")
                
            except Exception as e:
                print(f"处理股票 {stock_code} 时出错: {str(e)}")
                continue
        
        # 保存更新后的数据
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_filename = f'concept_stocks_updated_{timestamp}.csv'
        df.to_csv(new_filename, index=False, encoding='utf-8-sig')
        print(f"\n更新后的数据已保存到: {new_filename}")
        
        # 打印更新统计
        print("\n数据更新统计:")
        print(f"总股票数: {len(df)}")
        print("\n成交额变化前5名股票:")
        print(df[['代码', '名称', '成交额', '昨日成交额', '成交额变化']].nlargest(5, '成交额变化'))
        
        return df
        
    except Exception as e:
        print(f"更新数据失败: {str(e)}")
        return None
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
# 使用示例：
# updated_df = update_today_volume('concept_stocks_20240101_120000.csv')
# analysis = analyze_concept_volume_changes(updated_df)
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
        
        # 初始化浏览器
        print("正在启动浏览器...")
        driver = uc.Chrome(options=options)
        driver.set_page_load_timeout(60)
        
        # 构建URL并访问
        url = f"https://emweb.securities.eastmoney.com/pc_hsf10/pages/index.html?type=web&code={stock_code}&color=b#/hxtc"
        print(f"正在访问: {url}")
        driver.get(url)
        
        print("等待页面加载...")
        time.sleep(10)
        
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
                import psutil
                for proc in psutil.process_iter(['pid', 'name']):
                    try:
                        if 'chrome' in proc.info['name'].lower():
                            psutil.Process(proc.info['pid']).terminate()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                print("Chrome进程已清理")
            except Exception as e:
                print(f"清理Chrome进程时出现异常: {str(e)}")
# 使用示例
def main():
    app = QApplication(sys.argv)
    window = StockKlineViewer()
    window.show()
    sys.exit(app.exec())
    zzz
    stock_info = get_stock_special_topics('SZ301037')
    if stock_info:
        print(stock_info)

        zzz
        print("主营业务:", stock_info["主营业务"])
        print("行业背景:", stock_info["行业背景"])
        print("\n核心竞争力:")
        for item in stock_info["核心竞争力"]:
            print(f"\n{item['标题']}:")
            print(item['内容'])



    print(f"当前工作目录: {os.getcwd()}")
    stock_individual_info_em_df = get_stock_info_em("000001")
    print(stock_individual_info_em_df)
    zzz
    # 设置工作目录为当前文件所在目录
    print(f"当前文件路径: {__file__}")
    print(f"设置工作目录为: {os.path.dirname(__file__)}")
    os.chdir(os.path.dirname(__file__))
    
    print(f"尝试读取文件: concept_stocks_20241224_013116.csv")
    df=pd.read_csv('concept_stocks_20241224_013116.csv')
    print(f"成功读取文件，共 {len(df)} 行数据")
    df=update_today_volume(df)
    
    analysis = analyze_concept_volume_changes(df)

    xx
    df = get_concept_stocks_data()
    print(df)
    zxzzz


    '''
    app = QApplication(sys.argv)
    window = StockKlineViewer()
    window.show()
    sys.exit(app.exec())
    stock_code = "603893"
    stock_name = "瑞芯微"
    api_key="sk-cefd9c1f970f4715aa23e4740273a84c"
    #dongmi_qa=get_dongmi_qa(stock_name)

    df,tx=get_stock_info(stock_code,api_key)
    print('tx',type(tx))
    analyze_qa_content(df,tx,stock_name,api_key)
    zz
    df = get_dongmi_qa(stock_name)
    if df is not None:
        print("\n获取到的问答数据：")
        print(df)

    '''
#调用main函数
if __name__ == "__main__":
    main()
