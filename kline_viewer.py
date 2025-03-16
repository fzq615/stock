import tkinter as tk
import pandas as pd
import os
from tkinter import ttk, messagebox
from tkhtmlview import HTMLLabel
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import akshare as ak

class KlineViewer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("股票K线图查看器")
        self.geometry("1200x800")
        self.configure(bg="#f0f0f0")
        
        # 创建主框架
        self.main_frame = ttk.Frame(self)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 创建左右分隔窗口
        self.paned_window = ttk.PanedWindow(self.main_frame, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True)
        
        # 左侧面板 - 股票列表
        self.left_frame = ttk.Frame(self.paned_window, width=300)
        self.paned_window.add(self.left_frame, weight=1)
        
        # 右侧面板 - 图表显示
        self.right_frame = ttk.Frame(self.paned_window)
        self.paned_window.add(self.right_frame, weight=3)
        
        # 设置左侧面板组件
        self.setup_left_panel()
        
        # 设置右侧面板组件
        self.setup_right_panel()
        
    def setup_left_panel(self):
        """设置左侧面板"""
        # 创建标题标签
        title_label = ttk.Label(self.left_frame, text="股票列表", font=("Arial", 12, "bold"))
        title_label.pack(pady=10)
        
        # 创建搜索框
        search_frame = ttk.Frame(self.left_frame)
        search_frame.pack(fill=tk.X, padx=5, pady=5)
        
        search_label = ttk.Label(search_frame, text="搜索:")
        search_label.pack(side=tk.LEFT, padx=5)
        
        self.search_var = tk.StringVar()
        self.search_var.trace("w", self.filter_stocks)
        search_entry = ttk.Entry(search_frame, textvariable=self.search_var)
        search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # 创建股票列表
        self.tree_frame = ttk.Frame(self.left_frame)
        self.tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.tree = ttk.Treeview(
            self.tree_frame,
            columns=("code", "name"),
            show="headings",
            height=30
        )
        
        self.tree.heading("code", text="代码")
        self.tree.heading("name", text="名称")
        self.tree.column("code", width=100)
        self.tree.column("name", width=150)
        
        # 添加滚动条
        vsb = ttk.Scrollbar(self.tree_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        
        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        
        self.tree_frame.grid_columnconfigure(0, weight=1)
        self.tree_frame.grid_rowconfigure(0, weight=1)
        
        # 绑定点击事件
        self.tree.bind("<<TreeviewSelect>>", self.on_stock_selected)
        
        # 加载股票列表
        self.load_stock_list()
        
    def setup_right_panel(self):
        """设置右侧面板"""
        # 创建标题标签
        self.chart_title = ttk.Label(self.right_frame, text="请选择股票查看K线图", font=("Arial", 12, "bold"))
        self.chart_title.pack(pady=10)
        
        # 创建周期选择框
        period_frame = ttk.Frame(self.right_frame)
        period_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(period_frame, text="周期:").pack(side=tk.LEFT, padx=5)
        self.period_var = tk.StringVar(value="日线")
        period_combo = ttk.Combobox(period_frame, textvariable=self.period_var, values=["日线", "周线", "月线"])
        period_combo.pack(side=tk.LEFT, padx=5)
        period_combo.bind("<<ComboboxSelected>>", self.on_period_changed)
        
        # 创建HTML显示区域
        self.html_label = HTMLLabel(self.right_frame)
        self.html_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.html_label.set_html("<h2 style='text-align:center; margin-top:100px;'>请从左侧列表选择股票</h2>")
        
    def load_stock_list(self):
        """加载股票列表"""
        try:
            # 获取A股股票列表
            df = ak.stock_info_a_code_name()
            
            # 清空现有项
            for item in self.tree.get_children():
                self.tree.delete(item)
            
            # 添加股票到树形视图
            for _, row in df.iterrows():
                self.tree.insert("", tk.END, values=(row["code"], row["name"]))
                
        except Exception as e:
            messagebox.showerror("错误", f"加载股票列表失败: {str(e)}")
    
    def filter_stocks(self, *args):
        """过滤股票列表"""
        search_text = self.search_var.get().lower()
        
        for item in self.tree.get_children():
            values = self.tree.item(item)["values"]
            if search_text in str(values[0]).lower() or search_text in str(values[1]).lower():
                self.tree.reattach(item, "", tk.END)
            else:
                self.tree.detach(item)
    
    def on_stock_selected(self, event):
        """处理股票选择事件"""
        selected_items = self.tree.selection()
        if not selected_items:
            return
            
        item = selected_items[0]
        code, name = self.tree.item(item)["values"]
        
        self.chart_title.config(text=f"{code} {name}")
        self.show_kline(code)
    
    def on_period_changed(self, event):
        """处理周期变更事件"""
        selected_items = self.tree.selection()
        if not selected_items:
            return
            
        item = selected_items[0]
        code = self.tree.item(item)["values"][0]
        self.show_kline(code)
    
    def show_kline(self, code):
        """显示K线图"""
        try:
            # 获取K线数据
            period = self.period_var.get()
            if period == "日线":
                df = ak.stock_zh_a_hist(symbol=code, adjust="qfq")
            elif period == "周线":
                df = ak.stock_zh_a_hist(symbol=code, period="weekly", adjust="qfq")
            else:
                df = ak.stock_zh_a_hist(symbol=code, period="monthly", adjust="qfq")
            
            # 重命名列
            df = df.rename(columns={
                "日期": "date",
                "开盘": "open",
                "最高": "high",
                "最低": "low",
                "收盘": "close",
                "成交量": "volume"
            })
            
            # 创建子图
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                subplot_titles=("K线图", "成交量"),
                row_heights=[0.7, 0.3]
            )
            
            # 添加K线图
            fig.add_trace(
                go.Candlestick(
                    x=df["date"],
                    open=df["open"],
                    high=df["high"],
                    low=df["low"],
                    close=df["close"],
                    increasing_line_color="red",
                    decreasing_line_color="green"
                ),
                row=1, col=1
            )
            
            # 添加成交量图
            colors = ["red" if close >= open_ else "green"
                     for close, open_ in zip(df["close"], df["open"])]
            
            fig.add_trace(
                go.Bar(
                    x=df["date"],
                    y=df["volume"],
                    marker_color=colors
                ),
                row=2, col=1
            )
            
            # 更新布局
            fig.update_layout(
                height=600,
                showlegend=False,
                xaxis_rangeslider_visible=False
            )
            
            # 生成HTML并显示
            html = fig.to_html(include_plotlyjs=True, full_html=True)
            self.html_label.set_html(html)
            
        except Exception as e:
            messagebox.showerror("错误", f"加载K线数据失败: {str(e)}")
            self.html_label.set_html(f"<h2 style='text-align:center; color:red;'>加载K线数据失败: {str(e)}</h2>")

if __name__ == "__main__":
    # 检查必要的库
    try:
        import tkhtmlview
    except ImportError:
        print("正在安装必要的库...")
        import subprocess
        subprocess.check_call(["pip", "install", "tkhtmlview"])
        print("安装完成，请重新运行程序")
        exit()
    
    app = KlineViewer()
    app.mainloop()