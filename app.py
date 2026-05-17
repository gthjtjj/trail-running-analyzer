import streamlit as st
import gpxpy
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- 1. 配置页面 ---
st.set_page_config(page_title="越野跑赛道分析预测器 Pro+", layout="wide")

st.title("🏃‍♂️ 越野跑赛道分析与完赛时间预测 Pro+")
st.markdown("集成了 **CP分段规划**、**体能疲劳衰减模型** 以及 **CP段地形精细化拆解** 的终极越野路书工具。")

# --- 2. 核心算法逻辑 ---
def classify_slope(slope):
    if slope > 15: return '极陡坡'
    elif 8 < slope <= 15: return '陡坡'
    elif 4 < slope <= 8: return '缓坡'
    elif -4 <= slope <= 4: return '平地'
    elif -8 <= slope < -4: return '缓下坡'
    elif -15 <= slope < -8: return '陡下坡'
    else: return '极陡下坡'

SLOPE_ORDER = ['极陡坡', '陡坡', '缓坡', '平地', '缓下坡', '陡下坡', '极陡下坡']

COLOR_MAP = {
    '极陡坡': '#8B0000', '陡坡': '#FF4500', '缓坡': '#FFD700',
    '平地': '#228B22', '缓下坡': '#00FFFF', '陡下坡': '#1E90FF', '极陡下坡': '#00008B'
}

@st.cache_data
def process_gpx(file):
    gpx = gpxpy.parse(file)
    points = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                points.append([point.latitude, point.longitude, point.elevation])
    
    df = pd.DataFrame(points, columns=['lat', 'lon', 'ele'])
    df['dist_diff'] = 0.0
    for i in range(1, len(df)):
        df.loc[i, 'dist_diff'] = gpxpy.geo.distance(df.loc[i-1, 'lat'], df.loc[i-1, 'lon'], None, df.loc[i, 'lat'], df.loc[i, 'lon'], None)
    
    df['cum_dist_km'] = df['dist_diff'].cumsum() / 1000.0
    df['ele_diff'] = df['ele'].diff().fillna(0)
    df['slope'] = np.where(df['dist_diff'] > 0, (df['ele_diff'] / df['dist_diff']) * 100, 0)
    df['slope_class'] = df['slope'].apply(classify_slope)
    return df

# --- 3. 侧边栏设置 ---
st.sidebar.header("⏱️ 1. 基础配速设置 (min/km)")
paces = {
    '极陡坡': st.sidebar.number_input("极陡坡 (>15%)", value=25.0, step=0.5),
    '陡坡': st.sidebar.number_input("陡坡 (8~15%)", value=15.0, step=0.5),
    '缓坡': st.sidebar.number_input("缓坡 (4~8%)", value=8.0, step=0.5),
    '平地': st.sidebar.number_input("平地 (-4~4%)", value=5.5, step=0.1),
    '缓下坡': st.sidebar.number_input("缓下坡 (-8~-4%)", value=4.5, step=0.1),
    '陡下坡': st.sidebar.number_input("陡下坡 (-15~-8%)", value=6.0, step=0.1),
    '极陡下坡': st.sidebar.number_input("极陡下坡 (< -15%)", value=10.0, step=0.5),
}

st.sidebar.markdown("---")

st.sidebar.header("📉 2. 体能衰减模型")
fatigue_rate = st.sidebar.slider("每 10 公里配速衰减比例 (%)", min_value=0, max_value=20, value=5, step=1) / 100.0

st.sidebar.markdown("---")

st.sidebar.header("📍 3. 赛事 CP 点设置")
st.sidebar.markdown("输入每个 CP 点对应的**官方公里数**（用英文逗号分隔）")
cp_input = st.sidebar.text_input("CP点公里数列表", value="15, 30, 45")

try:
    cp_distances = [float(x.strip()) for x in cp_input.split(",") if x.strip() != ""]
    cp_distances = sorted(list(set(cp_distances)))
except ValueError:
    st.sidebar.error("⚠️ 请检查输入的公里数格式，确保只包含数字和英文逗号！")
    cp_distances = []

# --- 4. 主界面逻辑 ---
uploaded_file = st.file_uploader("第一步：选择并上传你的赛道 GPX 文件", type=["gpx"])

if uploaded_file:
    df = process_gpx(uploaded_file)
    
    total_dist = df['dist_diff'].sum() / 1000.0
    total_ascent = df[df['ele_diff'] > 0]['ele_diff'].sum()
    total_descent = abs(df[df['ele_diff'] < 0]['ele_diff'].sum())
    
    # 动态构建切分区间与标签
    break_points = [0.0] + [x for x in cp_distances if 0 < x < total_dist] + [total_dist]
    
    seg_labels = []
    for i in range(len(break_points)-1):
        if i == 0:
            seg_labels.append("起点->CP1")
        elif i == len(break_points)-2:
            seg_labels.append(f"CP{i}->终点")
        else:
            seg_labels.append(f"CP{i}->CP{i+1}")
            
    # 给每一行数据标记属于哪一个 CP 赛段
    df['cp_seg'] = pd.cut(df['cum_dist_km'], bins=break_points, labels=seg_labels, include_lowest=True)

    # 疲劳衰减计算时间
    df['fatigue_factor'] = 1.0 + (df['cum_dist_km'] // 10) * fatigue_rate
    df['pred_pace'] = df.apply(lambda row: paces.get(row['slope_class'], 6.0) * row['fatigue_factor'], axis=1)
    df['time_spent_min'] = (df['dist_diff'] / 1000.0) * df['pred_pace']
    
    total_time_min = df['time_spent_min'].sum()

    # --- 展示赛道整体核心指标 ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("总距离", f"{total_dist:.2f} km")
    col2.metric("累计爬升", f"{total_ascent:.0f} m")
    col3.metric("累计下降", f"{total_descent:.0f} m")
    hours, mins = divmod(int(total_time_min), 60)
    col4.metric("预估总用时 (含衰减)", f"{hours}h {mins}m")

    # --- 5. 交互式可视化图表 ---
    st.subheader("🌋 赛道海拔剖面图（按坡度染色）")
    fig = go.Figure()

    for cat in COLOR_MAP.keys():
        cat_df = df.copy()
        cat_df.loc[df['slope_class'] != cat, 'ele'] = None
        fig.add_trace(go.Scatter(
            x=cat_df['cum_dist_km'], 
            y=cat_df['ele'],
            mode='lines',
            line=dict(color=COLOR_MAP[cat], width=3),
            name=cat,
            hoverinfo='text',
            text=[f"距离: {d:.2f}km<br>海拔: {e:.0f}m<br>当前坡度: {s:.1f}%<br>配速: {p:.1f} min/km" 
                  for d, e, s, p in zip(df['cum_dist_km'], df['ele'], df['slope'], df['pred_pace'])]
        ))

    for cp in break_points[1:-1]:
        fig.add_vline(x=cp, line_width=1.5, line_dash="dash", line_color="gray")

    fig.update_layout(
        xaxis_title="距离 (km)", yaxis_title="海拔 (m)", legend_title="坡度分类",
        hovermode="x unified", template="plotly_white", height=500
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- 6. 核心输出 1：CP 赛段耗时表 ---
    st.subheader("📍 赛事 CP 赛段战术分解表")
    
    cp_stats = []
    cum_time_min = 0.0
    
    for i, seg_name in enumerate(seg_labels):
        seg_start = break_points[i]
        seg_end = break_points[i+1]
        
        seg_df = df[df['cp_seg'] == seg_name]
        
        if len(seg_df) > 0:
            seg_dist = seg_end - seg_start
            seg_ascent = seg_df[seg_df['ele_diff'] > 0]['ele_diff'].sum()
            seg_descent = abs(seg_df[seg_df['ele_diff'] < 0]['ele_diff'].sum())
            seg_time = seg_df['time_spent_min'].sum()
            
            cum_time_min += seg_time
            s_h, s_m = divmod(int(seg_time), 60)
            c_h, c_m = divmod(int(cum_time_min), 60)
            
            cp_stats.append({
                "赛段区间": seg_name,
                "段内距离 (km)": f"{seg_dist:.2f}",
                "段内爬升 (m)": f"+{seg_ascent:.0f}",
                "段内下降 (m)": f"-{seg_descent:.0f}",
                "本段预计耗时": f"{s_h}小时 {s_m}分钟",
                "到达该点累计时间": f"⏱️ {c_h:02d}:{c_m:02d}"
            })
            
    st.dataframe(pd.DataFrame(cp_stats), use_container_width=True, hide_index=True)


    # --- 7. 新增核心输出 2：各 CP 段地形占比分布矩阵 ---
    st.subheader("📊 各 CP 赛段详细地形分布 (单位: 公里)")
    st.markdown("该表展示了每个赛段内，不同坡度地形的具体公里数。你可以一眼看出哪个赛段“含魔量”（极陡坡）最高。")

    # 使用 pandas 的 pivot_table 交叉计算 cp_seg 和 slope_class 之间的距离总和
    # 将每步的距离从米转换为公里
    df['dist_diff_km'] = df['dist_diff'] / 1000.0
    
    pivot_df = pd.pivot_table(
        df, 
        values='dist_diff_km', 
        index='cp_seg', 
        columns='slope_class', 
        aggfunc='sum', 
        fill_value=0.0
    )
    
    # 按照合理的坡度顺序重新排列列名，并重新对齐行（保证不乱序）
    available_cols = [col for col in SLOPE_ORDER if col in pivot_df.columns]
    pivot_df = pivot_df.reindex(index=seg_labels, columns=available_cols)
    
    # 增加一行“全赛道总计”
    pivot_df.loc['全赛道总计'] = pivot_df.sum()
    
    # 格式化保留两位小数，更加易读
    pivot_df_formatted = pivot_df.round(2)
    
    # 在网页上渲染精美表格
    st.dataframe(pivot_df_formatted, use_container_width=True)


    # --- 8. 全局地形分布百分比（收纳进抽屉） ---
    with st.expander("查看全赛道整体坡度比例"):
        summary = df.groupby('slope_class')['dist_diff'].sum() / 1000.0
        stats_data = []
        for cat in SLOPE_ORDER:
            d = summary.get(cat, 0.0)
            stats_data.append({
                "地形分类": cat,
                "总距离 (km)": round(d, 2),
                "距离占比": f"{(d/total_dist*100):.1f}%" if total_dist > 0 else "0%"
            })
        st.table(pd.DataFrame(stats_data))

else:
    st.info("💡 请在上方上传 GPX 文件开始分析。在左侧栏可以随时调整配速、体能衰减率以及 CP 点。")