import tkinter as tk
import pandas as pd
import os
from tkinter import ttk
from tkinterweb import HtmlFrame

class ChartViewer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("股票模式图表查看器")
        self.geometry("1200x800")
        self.configure(bg="#f0f0f0")
        
        # 设置应用程序图标和样式
        self.style = ttk.Style()
        self.style.theme_use("clam")
        
        # 创建主框架
        self.main_frame = ttk.Frame(self)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 创建左侧和右侧面板
        self.paned_window = ttk.PanedWindow(self.main_frame, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True)
        
        # 左侧面板 - 股票列表
        self.left_frame = ttk.Frame(self.paned_window, width=300)
        self.paned_window.add(self.left_frame, weight=1)
        
        # 右侧面板 - 图表显示
        self.right_frame = ttk.Frame(self.paned_window)
        self.paned_window.add(self.right_frame, weight=3)
        
        # 设置左侧面板的组件
        self.setup_left_panel()
        
        # 设置右侧面板的组件
        self.setup_right_panel()
        
        # 加载数据
        self.load_data()
    def setup_left_panel(self):
        """设置左侧面板"""
        # 创建标题标签
        title_label = ttk.Label(self.left_frame, text="股票代码列表", font=("Arial", 12, "bold"))
        title_label.pack(pady=10)
        
        # 创建搜索框
        self.search_var = tk.StringVar()
        self.search_var.trace("w", self.filter_stocks)
        search_frame = ttk.Frame(self.left_frame)
        search_frame.pack(fill=tk.X, padx=5, pady=5)
        
        search_label = ttk.Label(search_frame, text="搜索:")
        search_label.pack(side=tk.LEFT, padx=5)
        
        search_entry = ttk.Entry(search_frame, textvariable=self.search_var)
        search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # 创建Treeview用于显示股票列表
        self.tree_frame = ttk.Frame(self.left_frame)
        self.tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建带滚动条的树形视图
        self.tree = ttk.Treeview(
            self.tree_frame, 
            columns=("code", "date", "angle"), 
            show="headings",
            height=30  # 增加显示行数
        )
        
        # 设置列标题和宽度
        self.tree.heading("code", text="股票代码")
        self.tree.heading("date", text="日期")
        self.tree.heading("angle", text="角度")
        self.tree.column("code", width=100)
        self.tree.column("date", width=200)
        self.tree.column("angle", width=100)
        
        # 添加垂直滚动条
        vsb = ttk.Scrollbar(self.tree_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        
        # 添加水平滚动条
        hsb = ttk.Scrollbar(self.tree_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(xscrollcommand=hsb.set)
        
        # 布局树形视图和滚动条
        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        
        # 配置grid权重
        self.tree_frame.grid_columnconfigure(0, weight=1)
        self.tree_frame.grid_rowconfigure(0, weight=1)
        
        # 绑定点击事件
        self.tree.bind("<<TreeviewSelect>>", self.on_stock_selected)
    def setup_right_panel(self):
        """设置右侧面板"""
        # 创建标题标签
        self.chart_title = ttk.Label(self.right_frame, text="请选择股票查看图表", font=("Arial", 12, "bold"))
        self.chart_title.pack(pady=10)
        
        # 创建垂直分隔窗口，允许调整上下两个面板的比例
        self.right_paned = ttk.PanedWindow(self.right_frame, orient=tk.VERTICAL)
        self.right_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # 创建业绩预告信息框架 (30%)
        self.financial_frame = ttk.LabelFrame(self.right_paned, text="业绩预告信息", padding=10)
        self.financial_text = tk.Text(self.financial_frame, wrap=tk.WORD, height=8)
        self.financial_text.pack(fill=tk.BOTH, expand=True)
        self.financial_text.config(state=tk.DISABLED, font=("Arial", 10))
        self.right_paned.add(self.financial_frame, weight=30)
        
        # 创建基本面信息文本框 (70%)
        self.info_frame = ttk.LabelFrame(self.right_paned, text="基本面信息", padding=10)
        self.info_text = tk.Text(self.info_frame, wrap=tk.WORD)
        self.info_text.pack(fill=tk.BOTH, expand=True)
        self.info_text.config(state=tk.DISABLED, font=("Arial", 10))
        self.right_paned.add(self.info_frame, weight=70)
        
        # 创建HTML框架用于显示图表
        self.html_frame_container = ttk.Frame(self.right_frame)
        self.html_frame_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.html_frame = HtmlFrame(self.html_frame_container, messages_enabled=False)
        self.html_frame.pack(fill=tk.BOTH, expand=True)
        
        # 显示初始消息
        self.html_frame.load_html("<html><body><h2 style='text-align:center; margin-top:100px;'>请从左侧列表选择股票代码查看图表</h2></body></html>")
    
    def load_data(self):
        """加载数据"""
        try:
            # 读取CSV文件
            csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pattern_results.csv")
            print(f"正在读取文件: {csv_path}")  # 调试信息
            
            # 读取CSV文件并打印基本信息
            self.df = pd.read_csv(csv_path)
            print(f"CSV文件原始行数: {len(self.df)}")
            print(f"CSV文件列名: {self.df.columns.tolist()}")
            
            # 检查数据完整性
            print("\n数据样例:")
            print(self.df.head())
            
            # 按angle列降序排序
            self.df = self.df.sort_values('angle', ascending=False)
            
            # 获取唯一的股票代码
            self.unique_stocks = self.df['code'].unique()
            print(f"\n唯一股票代码数量: {len(self.unique_stocks)}")
            
            # 填充树形视图
            self.populate_tree()
            
        except pd.errors.EmptyDataError:
            print("CSV文件为空")
            tk.messagebox.showerror("错误", "CSV文件为空")
        except Exception as e:
            print(f"加载数据时出错: {str(e)}")
            tk.messagebox.showerror("错误", f"加载数据时出错: {str(e)}")
    def populate_tree(self, filter_text=None):
        """填充树形视图"""
        try:
            # 清空现有项
            for item in self.tree.get_children():
                self.tree.delete(item)
            
            # 打印调试信息
            print(f"总行数: {len(self.df)}")
            
            # 添加股票到树形视图
            count = 0
            for _, row in self.df.iterrows():
                # 确保股票代码格式正确（6位）
                code = str(row['code']).zfill(6)
                
                # 构建日期字符串
                date = f"{row['hp_start']} 至 {row['hp_end']}"
                angle = row['angle']
                
                # 如果有过滤文本，则只显示匹配的项
                if filter_text and filter_text.lower() not in code.lower():
                    continue
                
                # 插入数据到树形视图
                self.tree.insert("", tk.END, values=(code, date, angle))
                count += 1
            
            print(f"显示行数: {count}")
            
            # 调整列宽以适应内容
            for col in ("code", "date", "angle"):
                self.tree.column(col, width=None)  # 重置列宽
                # 获取最大内容宽度
                max_width = max([
                    self.tree.column(col)["width"],
                    *[len(str(self.tree.set(item, col))) * 8 for item in self.tree.get_children()]
                ])
                self.tree.column(col, width=max_width + 20)  # 添加一些padding
            
            # 按angle列降序排序
            items = [(self.tree.set(item, "angle"), item) for item in self.tree.get_children()]
            items.sort(reverse=True)  # 降序排序
            for index, (_, item) in enumerate(items):
                self.tree.move(item, "", index)
                
        except Exception as e:
            print(f"填充树形视图时出错: {str(e)}")
    
    def filter_stocks(self, *args):
        # 获取搜索文本
        search_text = self.search_var.get()
        self.populate_tree(search_text)
    
    def on_stock_selected(self, event):
        """处理股票选择事件"""
        try:
            # 获取选中的项
            selected_item = self.tree.selection()
            if not selected_item:
                return
            
            # 获取股票代码
            stock_code = self.tree.item(selected_item[0], "values")[0]
            
            # 更新标题
            self.chart_title.config(text=f"股票代码: {stock_code}")
            
            # 显示业绩预告信息
            self.show_financial_info(stock_code)
            
            # 显示基本面信息
            self.show_basic_info(stock_code)
            
            # 显示图表
            self.show_chart(stock_code)
            
        except Exception as e:
            tk.messagebox.showerror("错误", f"处理股票信息时出错: {str(e)}")
        
    def show_financial_info(self, stock_code):
        """显示业绩预告信息"""
        try:
            self.financial_text.config(state=tk.NORMAL)
            self.financial_text.delete(1.0, tk.END)
            
            # 使用绝对路径
            financial_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "financial_业绩预告.csv")
            
            if os.path.exists(financial_file):
                print(f"正在读取业绩预告文件: {financial_file}")  # 调试信息
                financial_df = pd.read_csv(financial_file)
                
                # 打印调试信息
                print(f"查找股票代码: {stock_code}")
                print(f"CSV文件中的股票代码类型: {financial_df['股票代码'].dtype}")
                print(f"CSV文件中的部分股票代码: {financial_df['股票代码'].head()}")
                
                # 统一处理股票代码格式
                financial_df['股票代码'] = financial_df['股票代码'].astype(str).str.zfill(6)
                stock_code = str(stock_code).zfill(6)
                
                # 打印处理后的信息
                print(f"处理后的查找股票代码: {stock_code}")
                print(f"处理后的CSV中的部分股票代码: {financial_df['股票代码'].head()}")
                
                # 查找匹配的记录
                stock_financial = financial_df[financial_df['股票代码'] == stock_code]
                print(f"找到的匹配记录数: {len(stock_financial)}")
                
                if not stock_financial.empty:
                    for _, row in stock_financial.iterrows():
                        self.financial_text.insert(tk.END, f"预告期间: {row.get('预告期间', 'N/A')}\n")
                        self.financial_text.insert(tk.END, f"业绩变动: {row.get('业绩变动', 'N/A')}\n")
                        self.financial_text.insert(tk.END, f"预告类型: {row.get('预告类型', 'N/A')}\n")
                        self.financial_text.insert(tk.END, f"预告内容: {row.get('预告内容', 'N/A')}\n")
                        self.financial_text.insert(tk.END, "-"*50 + "\n")
                else:
                    self.financial_text.insert(tk.END, f"未找到股票 {stock_code} 的业绩预告信息")
            else:
                print(f"业绩预告文件不存在: {financial_file}")  # 调试信息
                self.financial_text.insert(tk.END, f"业绩预告文件不存在: {financial_file}")
            
            self.financial_text.config(state=tk.DISABLED)
        except Exception as e:
            print(f"显示业绩预告信息时出错: {str(e)}")
            self.financial_text.config(state=tk.NORMAL)
            self.financial_text.delete(1.0, tk.END)
            self.financial_text.insert(tk.END, f"显示业绩预告信息时出错: {str(e)}")
            self.financial_text.config(state=tk.DISABLED)
    def show_basic_info(self, stock_code):
        """显示基本面信息"""
        try:
            self.info_text.config(state=tk.NORMAL)
            self.info_text.delete(1.0, tk.END)
            
            # 构建stock_info目录的路径
            stock_info_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stock_info")
            
            if os.path.exists(stock_info_dir):
                # 查找对应股票代码的最新信息文件
                stock_info_files = [f for f in os.listdir(stock_info_dir) 
                                if f.startswith(f"stock_info_{stock_code}_")]
                
                if stock_info_files:
                    # 获取最新的文件（按文件名排序）
                    latest_file = sorted(stock_info_files)[-1]
                    stock_info_path = os.path.join(stock_info_dir, latest_file)
                    
                    # 读取并显示文件内容
                    try:
                        with open(stock_info_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            self.info_text.insert(tk.END, content)
                    except Exception as e:
                        self.info_text.insert(tk.END, f"读取文件时出错: {str(e)}")
                else:
                    self.info_text.insert(tk.END, f"未找到股票 {stock_code} 的基本面信息文件")
            else:
                self.info_text.insert(tk.END, "stock_info 目录不存在")
            
            self.info_text.config(state=tk.DISABLED)
        
        except Exception as e:
            print(f"显示基本面信息时出错: {str(e)}")
            self.info_text.config(state=tk.NORMAL)
            self.info_text.delete(1.0, tk.END)
            self.info_text.insert(tk.END, f"显示基本面信息时出错: {str(e)}")
            self.info_text.config(state=tk.DISABLED)
    def show_chart(self, stock_code):
        """显示股票图表"""
        try:
            # 构建HTML文件路径 - 检查两个可能的位置
            html_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "matched_charts", f"{stock_code}_pattern.html")
            if not os.path.exists(html_file):
                # 如果第一个位置不存在，尝试 charts 目录
                html_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "charts", f"{stock_code}_pattern.html")
            
            if os.path.exists(html_file):
                print(f"正在加载图表文件: {html_file}")  # 调试信息
                try:
                    # 创建一个临时HTML文件，包含自动最大化的JavaScript代码
                    temp_html = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_fullscreen.html")
                    with open(html_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # 在原HTML的head部分插入最大化脚本
                    fullscreen_script = """
                    <script>
                    window.onload = function() {
                        window.moveTo(0, 0);
                        window.resizeTo(screen.availWidth, screen.availHeight);
                    };
                    </script>
                    """
                    
                    # 插入脚本到head标签后
                    if '<head>' in content:
                        content = content.replace('<head>', '<head>' + fullscreen_script)
                    else:
                        content = '<head>' + fullscreen_script + '</head>' + content
                    
                    # 写入临时文件
                    with open(temp_html, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    # 使用系统默认浏览器打开临时HTML文件
                    import webbrowser
                    absolute_path = os.path.abspath(temp_html)
                    file_url = f"file:///{absolute_path.replace(os.sep, '/')}"
                    webbrowser.open(file_url)
                    
                    # 同时在应用程序中显示原始文件
                    self.html_frame.load_url(f"file:///{html_file.replace(os.sep, '/')}")
                    self.html_frame.update()
                    print("图表加载完成")  # 调试信息
                    
                except Exception as e:
                    print(f"加载图表文件时出错: {str(e)}")
                    self.html_frame.load_html(f"<html><body><h2 style='text-align:center; margin-top:100px; color:red;'>加载图表文件时出错: {str(e)}</h2></body></html>")
            else:
                print(f"未找到图表文件: {html_file}")  # 调试信息
                self.html_frame.load_html(f"<html><body><h2 style='text-align:center; margin-top:100px; color:red;'>未找到股票 {stock_code} 的图表文件</h2></body></html>")
        except Exception as e:
            print(f"显示图表时出错: {str(e)}")
            self.html_frame.load_html(f"<html><body><h2 style='text-align:center; margin-top:100px; color:red;'>显示图表时出错: {str(e)}</h2></body></html>")
if __name__ == "__main__":
    # 检查是否安装了必要的库
    try:
        import tkinterweb
    except ImportError:
        print("正在安装必要的库...")
        import subprocess
        subprocess.check_call(["pip", "install", "tkinterweb"])
        print("安装完成，请重新运行程序")
        exit() 
    
    app = ChartViewer()
    app.mainloop()