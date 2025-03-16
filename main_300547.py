import pandas as pd
import plotly.graph_objects as go
from smc import smc
from smc_300547 import add_smc_indicators

# 新增内存优化配置
pd.options.mode.chained_assignment = None  # 禁用链式分配警告
# go.io.kaleido.scope.mathjax = None  # 禁用MathJax以减小文件体积

def main():
    try:
        # 获取数据并保存为CSV
        df = pd.read_csv(r'E:\20250305-bi\300547.csv')
        
        # 处理数据
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df#.tail(200)  # 取最近200个trade_dte交易日的数据
        
        # 创建连续索引
        continuous_index = list(range(len(df)))
        
        # 创建一个新的DataFrame，使用连续索引
        df_continuous = df.reset_index(drop=True)
        
        # 计算FVG数据并检查价格位置
        fvg_data = smc.fvg(df_continuous, join_consecutive=True)
        check_price_in_fvg(df_continuous, fvg_data)
        
        # 创建蜡烛图，使用连续索引
        # 创建蜡烛图，使用连续索引
        # 创建蜡烛图部分
        fig = go.Figure(
            data=[go.Candlestick(
                x=continuous_index,
                open=df['open'].values,
                high=df['high'].values,
                low=df['low'].values,
                close=df['close'].values,
                increasing_line_color='#ff6962',
                decreasing_line_color='#77dd76',
                text=[f"<span style='font-family:\"Microsoft YaHei\"; font-size:12px; color:#333'>" +
                      f"<b>日期</b>: {date}<br>" +
                      f"<b>index</b>: {index}<br>" +
                      f"<b>开盘</b>: {open:.2f}<br>" +
                      f"<b>最高</b>: {high:.2f}<br>" +
                      f"<b>最低</b>: {low:.2f}<br>" +
                      f"<b>收盘</b>: {close:.2f}<br>" +
                      f"<b>涨跌</b>: <span style='color:{'#ff4444' if close < open else '#22bb33'}'>" +
                      f"{((close-open)/open*100):+.2f}%</span></span>"
                      for date,index, open, high, low, close in zip(
                          df['trade_date'].dt.strftime('%Y-%m-%d'),
                          continuous_index,
                          df['open'],
                          df['high'],
                          df['low'],
                          df['close']
                      )],
                hoverlabel=dict(
                    bgcolor='rgba(255, 255, 255, 0.95)',
                    bordercolor='rgba(0, 0, 0, 0.2)',
                    font=dict(
                        family='"Microsoft YaHei", sans-serif',
                        size=13,
                        color='#333'
                    )
                ),
                hoverinfo='text',
                name='K线'
            )]
        )
        
        # 添加SMC指标到图表
        fig = add_smc_indicators_continuous(fig, df_continuous, continuous_index)
        
        # 更新图表布局
        # 在布局设置中添加全局字体配置
        fig.update_layout(
            title='300547 SMC分析',
            font=dict(  # 新增全局字体设置
                family="Microsoft YaHei",
                size=12,
                color='#333'
            ),
            xaxis_title='交易日序号',
            yaxis_title='价格',
            xaxis_rangeslider_visible=False,
            template='plotly_white',
            xaxis=dict(
                showticklabels=True,
                tickmode='array',
                tickvals=continuous_index[::10],
                ticktext=[df['trade_date'].iloc[i].strftime('%Y-%m-%d') for i in range(0, len(df), 10)],  # 修改为 trade_date
                tickangle=45,
                showgrid=True,
                gridcolor='rgba(128, 128, 128, 0.2)',
                domain=[0, 1]  # 设置x轴占据整个宽度
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(128, 128, 128, 0.2)',
                autorange=True
            ),
            height=900,
            width=2000,  # 增加整体宽度
            margin=dict(
                l=10,     # 减小左边距
                r=10,     # 减小右边距
                t=30,
                b=30,
                autoexpand=True  # 允许图表自动扩展
            ),
            legend=dict(
                x=1.02,        # 图例位置在图表右侧
                y=1,           # 顶部对齐
                xanchor='left', # 左对齐
                yanchor='top',  # 顶部对齐
                bgcolor='rgba(255, 255, 255, 0.8)',  # 半透明白色背景
                bordercolor='rgba(0, 0, 0, 0.2)',    # 浅灰色边框
                borderwidth=1,
                font=dict(size=12),  # 字体大小
                itemsizing='constant'  # 保持图例标记大小一致
            )
        )
        output_file = '300547_kline_continuous.html'
        # 保存为HTML文件


        # 删除有问题的配置行（约第7行）
        # go.io.kaleido.scope.mathjax = None  # 删除此行
        
        # 修改write_html调用（约第250行附近）
        # ... 修改write_html调用
        fig.write_html(
            output_file, 
            include_plotlyjs=True,
            full_html=True,
            config={
                'scrollZoom': True,
                'displayModeBar': True,
                'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'eraseshape'],
                'mathjax': None  # 禁用MathJax配置
            }
        )  # 移除post_script参数，简化HTML生成


        
        # 使用默认浏览器打开HTML文件
        import webbrowser
        webbrowser.open(output_file)
        
        print(f"生成了连续K线图: {output_file}")
        print(f"总共显示了 {len(df)} 个交易日的数据")
        
    except Exception as e:
        print(f"程序执行出错: {str(e)}")

def add_smc_indicators_continuous(fig, df, continuous_index):
    """添加SMC指标到图表，使用连续索引"""
    # 计算SMC指标
    fvg_data = smc.fvg(df, join_consecutive=True)
    swing_highs_lows_data = smc.swing_highs_lows(df, swing_length=5)
    bos_choch_data = smc.bos_choch(df, swing_highs_lows_data)
    ob_data = smc.ob(df, swing_highs_lows_data)

    # 添加各种指标到图表
    fig = add_FVG_continuous(fig, df, fvg_data, continuous_index)
    fig = add_swing_highs_lows_continuous(fig, df, swing_highs_lows_data, continuous_index)
    fig = add_bos_choch_continuous(fig, df, bos_choch_data, continuous_index)
    fig = add_OB_continuous(fig, df, ob_data, continuous_index)

    return fig

def add_FVG_continuous(fig, df, fvg_data, continuous_index):
    """添加FVG到图表，使用连续索引"""
    for i in range(len(df)):
        if not pd.isna(fvg_data['FVG'][i]):
            # 确定结束位置索引
            x1 = int(
                fvg_data["MitigatedIndex"][i]
                if fvg_data["MitigatedIndex"][i] != 0
                else len(df) - 1
            )
            
            fig.add_shape(
                type="rect",
                x0=continuous_index[i],
                x1=continuous_index[x1],
                y0=fvg_data['Top'][i],
                y1=fvg_data['Bottom'][i],
                fillcolor="rgba(255, 255, 0, 0.2)" if fvg_data['FVG'][i] == 1 else "rgba(255, 0, 0, 0.2)",
                line=dict(width=0),
                layer="below"
            )
            
            # 添加FVG文本标记
            mid_x = round((i + x1) / 2)
            mid_y = (fvg_data['Top'][i] + fvg_data['Bottom'][i]) / 2
            fig.add_trace(
                go.Scatter(
                    x=[continuous_index[mid_x]],
                    y=[mid_y],
                    mode="text",
                    text="FVG",
                    textposition="middle center",
                    textfont=dict(color='rgba(255, 255, 255, 0.4)', size=8),
                    showlegend=False
                )
            )
    return fig

def add_swing_highs_lows_continuous(fig, df, swing_data, continuous_index):
    """添加摇摆高低点到图表，使用连续索引"""
    high_added = False
    low_added = False
    for i in range(len(df)):
        if not pd.isna(swing_data['HighLow'][i]):
            is_high = swing_data['HighLow'][i] == 1
            fig.add_scatter(
                x=[continuous_index[i]],
                y=[swing_data['Level'][i]],
                mode='markers',
                marker=dict(
                    symbol='triangle-up' if is_high else 'triangle-down',
                    size=10,
                    color='brown' if is_high else 'red'  # 将gold改为brown，使高点更醒目
                ),
                name='Swing High' if is_high else 'Swing Low',
                showlegend=not (high_added if is_high else low_added)
            )
            if is_high:
                high_added = True
            else:
                low_added = True
    return fig  # 添加返回语句

def add_bos_choch_continuous(fig, df, bos_choch_data, continuous_index):
    """添加BOS/CHOCH到图表，使用连续索引"""
    bos_bull_added = False
    bos_bear_added = False
    choch_bull_added = False
    choch_bear_added = False
    
    for i in range(len(df)):
        if not pd.isna(bos_choch_data['BOS'][i]):
            is_bullish = bos_choch_data['BOS'][i] == 1
            fig.add_scatter(
                x=[continuous_index[i]],
                y=[bos_choch_data['Level'][i]],
                mode='markers',
                marker=dict(
                    symbol='circle',  # 圆点符号
                    size=12,
                    color='lime' if is_bullish else 'red'
                ),
                name='Break of Structure(多)' if is_bullish else 'Break of Structure(空)',  # BOS完整名称
                showlegend=not (bos_bull_added if is_bullish else bos_bear_added)
            )
            if is_bullish:
                bos_bull_added = True
            else:
                bos_bear_added = True
                
        if not pd.isna(bos_choch_data['CHOCH'][i]):
            is_bullish = bos_choch_data['CHOCH'][i] == 1
            fig.add_scatter(
                x=[continuous_index[i]],
                y=[bos_choch_data['Level'][i]],
                mode='markers',
                marker=dict(
                    symbol='star',  # 星星符号
                    size=15,
                    color='lime' if is_bullish else 'red'
                ),
                name='Change of Character(多)' if is_bullish else 'Change of Character(空)',  # CHOCH完整名称
                showlegend=not (choch_bull_added if is_bullish else choch_bear_added)
            )
            if is_bullish:
                choch_bull_added = True
            else:
                choch_bear_added = True
    return fig

def add_OB_continuous(fig, df, ob_data, continuous_index):
    """添加OB到图表，使用连续索引"""
    def format_volume(volume):
        if volume >= 1e12:
            return f"{volume / 1e12:.3f}T"
        elif volume >= 1e9:
            return f"{volume / 1e9:.3f}B"
        elif volume >= 1e6:
            return f"{volume / 1e6:.3f}M"
        elif volume >= 1e3:
            return f"{volume / 1e3:.3f}k"
        else:
            return f"{volume:.2f}"
    bull_ob_added = False  # 用于图例显示控制(看涨OB)
    bear_ob_added = False  # 用于图例显示控制(看跌OB)
    
    # 打印OB数据统计信息
    ob_data_not_na = ob_data[~ob_data['OB'].isna()]
    print("OB数据统计:")
    print(f"总OB数量: {len(ob_data_not_na)}")
    print(f"看涨OB数量: {len(ob_data_not_na[ob_data_not_na['OB'] == 1])}")
    print(f"看跌OB数量: {len(ob_data_not_na[ob_data_not_na['OB'] == -1])}")
    print("OB类型分布:", ob_data_not_na['OB'].value_counts())

    for i in range(len(ob_data["OB"])):
            if ob_data["OB"][i] == 1:
                x1 = int(
                    ob_data["MitigatedIndex"][i]
                    if ob_data["MitigatedIndex"][i] != 0
                    else len(df) - 1 )
                print('xx',x1,i)
               
                fig.add_shape(
                    type="rect",
                    x0=df.index[i],
                    y0=ob_data["Bottom"][i],
                    x1=df.index[i+ 15],
                    y1=ob_data["Top"][i],
                    line=dict(color="Gray"),
                    fillcolor=None,
                    opacity=1,
                    name="Bullish OB",
                    legendgroup="bullish ob",
                    showlegend=True,
                )

                if ob_data["MitigatedIndex"][i] > 0:
                    x_center = df.index[int(i + (ob_data["MitigatedIndex"][i] - i) / 2)]
                else:
                    x_center = df.index[int(i + (len(df) - i) / 2)]

                y_center = (ob_data["Bottom"][i] + ob_data["Top"][i]) / 2
                volume_text = format_volume(ob_data["OBVolume"][i])
                # Add annotation text
                annotation_text = f'OB: {volume_text} ({ob_data["Percentage"][i]}%)'

                fig.add_annotation(
                    x=x_center,
                    y=y_center,
                    xref="x",
                    yref="y",
                    align="center",
                    text=annotation_text,
                    font=dict(color="black", size=20),
                    showarrow=False,
                )

    for i in range(len(ob_data["OB"])):
        if ob_data["OB"][i] == -1:
            x1 = int(
                ob_data["MitigatedIndex"][i]
                if ob_data["MitigatedIndex"][i] != 0
                else len(df) - 1
            )
            fig.add_shape(
                type="rect",
                x0=df.index[i],
                y0=ob_data["Bottom"][i],
                x1=df.index[i+15],
                y1=ob_data["Top"][i],
                line=dict(color="Purple"),
                fillcolor=None,
                opacity=1,
                name="Bearish OB",
                legendgroup="bearish ob",
                showlegend=True,
            )

            if ob_data["MitigatedIndex"][i] > 0:
                x_center = df.index[int(i + (ob_data["MitigatedIndex"][i] - i) / 2)]
            else:
                x_center = df.index[int(i + (len(df) - i) / 2)]

            y_center = (ob_data["Bottom"][i] + ob_data["Top"][i]) / 2
            volume_text = format_volume(ob_data["OBVolume"][i])
            # Add annotation text
            annotation_text = f'OB: {volume_text} ({ob_data["Percentage"][i]}%)'

            fig.add_annotation(
                x=x_center,
                y=y_center,
                xref="x",
                yref="y",
                align="center",
                text=annotation_text,
                font=dict(color="rgba(255, 255, 255, 0.4)", size=8),
                showarrow=False,
            )
    return fig
    

def check_price_in_fvg(df, fvg_data):
    """检查当前价格是否在最近的FVG区域内"""
    print(df)
    current_price = df['close'].iloc[-1]
    latest_date = df['trade_date'].iloc[-1]
    fvg_date=latest_date
    # 查找当前价格下方最近的FVG区域
    latest_fvg_below = None
    for i in range(len(df)-1, -1, -1):  # 从后向前遍历
        print(df['trade_date'].iloc[i],"----",fvg_data['FVG'][i])
        if not pd.isna(fvg_data['FVG'][i]):
            # 检查FVG是否未被填补（使用MitigatedIndex）
            if fvg_data['MitigatedIndex'][i] == 0 and fvg_data['Top'][i] < current_price:
                fvg_date=df['trade_date'].iloc[i]
                latest_fvg_below = {
                    'index': i,
                    'date': df['trade_date'].iloc[i],
                    'top': fvg_data['Top'][i],
                    'bottom': fvg_data['Bottom'][i]
                }
                break
    
    # 输出分析结果
    print(f"\n=== FVG分析结果 ===")
    print(f"当前日期: {fvg_date}")  # 直接输出日期，不使用strftime
    print(f"当前价格: {current_price:.2f}")
    
    if latest_fvg_below:
        distance = current_price - latest_fvg_below['top']
        print(f"\n发现下方最近的未填补FVG区域:")
        print(f"形成日期: {latest_fvg_below['date']}")  # 直接输出日期，不使用strftime
        print(f"上边界: {latest_fvg_below['top']:.2f}")
        print(f"下边界: {latest_fvg_below['bottom']:.2f}")
        print(f"距离上边界: {distance:.2f}")
        return latest_fvg_below
    else:
        print(f"\n当前价格 {current_price:.2f} 下方没有未填补的FVG区域")
        return None

if __name__ == "__main__":
    main()