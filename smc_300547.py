from smc import smc
import pandas as pd

def add_smc_indicators(fig, df, window_df):
    # 计算SMC指标
    fvg_data = smc.fvg(window_df, join_consecutive=True)
    swing_highs_lows_data = smc.swing_highs_lows(window_df, swing_length=5)
    bos_choch_data = smc.bos_choch(window_df, swing_highs_lows_data)
    ob_data = smc.ob(window_df, swing_highs_lows_data)

    # 添加各种指标到图表
    fig = add_FVG(fig, window_df, fvg_data)
    fig = add_swing_highs_lows(fig, window_df, swing_highs_lows_data)
    fig = add_bos_choch(fig, window_df, bos_choch_data)
    fig = add_OB(fig, window_df, ob_data)

    return fig

def add_FVG(fig, df, fvg_data):
    for i in range(len(df)):
        if not pd.isna(fvg_data['FVG'][i]):
            fig.add_shape(
                type="rect",
                x0=df.index[i],
                x1=df.index[i+1],
                y0=fvg_data['Bottom'][i],
                y1=fvg_data['Top'][i],
                fillcolor="rgba(255, 255, 255, 0.2)" if fvg_data['FVG'][i] == 1 else "rgba(255, 0, 0, 0.2)",
                line=dict(width=0),
                layer="below"
            )
    return fig

def add_swing_highs_lows(fig, df, swing_data):
    for i in range(len(df)):
        if not pd.isna(swing_data['HighLow'][i]):
            fig.add_scatter(
                x=[df.index[i]],
                y=[swing_data['Level'][i]],
                mode='markers',
                marker=dict(
                    symbol='triangle-up' if swing_data['HighLow'][i] == 1 else 'triangle-down',
                    size=10,
                    color='white' if swing_data['HighLow'][i] == 1 else 'red'
                ),
                showlegend=False
            )
    return fig

def add_bos_choch(fig, df, bos_choch_data):
    for i in range(len(df)):
        if not pd.isna(bos_choch_data['BOS'][i]):
            fig.add_scatter(
                x=[df.index[i]],
                y=[bos_choch_data['Level'][i]],
                mode='markers',
                marker=dict(
                    symbol='circle',
                    size=12,
                    color='lime' if bos_choch_data['BOS'][i] == 1 else 'red'
                ),
                name='BOS',
                showlegend=False
            )
        if not pd.isna(bos_choch_data['CHOCH'][i]):
            fig.add_scatter(
                x=[df.index[i]],
                y=[bos_choch_data['Level'][i]],
                mode='markers',
                marker=dict(
                    symbol='star',
                    size=15,
                    color='lime' if bos_choch_data['CHOCH'][i] == 1 else 'red'
                ),
                name='CHOCH',
                showlegend=False
            )
    return fig

def add_OB(fig, df, ob_data):
    for i in range(len(df)):
        if not pd.isna(ob_data['OB'][i]):
            fig.add_shape(
                type="rect",
                x0=df.index[i],
                x1=df.index[i+1],
                y0=ob_data['Bottom'][i],
                y1=ob_data['Top'][i],
                fillcolor="rgba(0, 255, 0, 0.2)" if ob_data['OB'][i] == 1 else "rgba(255, 0, 0, 0.2)",
                line=dict(width=1, color="white"),
                layer="below"
            )
    return fig