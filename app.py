import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import xml.etree.ElementTree as ET
from math import radians, cos, sin, asin, sqrt

# --- 1. 页面基本配置 ---
st.set_page_config(
    page_title="越野跑赛道智能分析预测器 v4", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

st.title("🏃‍♂️ 跑者硬核路书：越野跑赛道智能分析预测器 (山峰保真与山峦面积图版)")
st.markdown("本版本彻底重构了底层滤波，**废除了会削平山峰的宏观趋势平滑**，采用 100 米区间重采样机制与相邻同性质坡度合并算法，完美保留真实海拔高度。")
st.markdown("---")

# --- 2. 核心数学与地理工具函数 ---
def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371000  
    return c * r

def classify_slope(slope):
    if slope > 15: return '极陡坡'
    elif 8 < slope <= 15: return '陡坡'
    elif 4 < slope <= 8: return '缓坡'
    elif -4 <= slope <= 4: return '平地'
    elif -8 <= slope < -4: return '缓下坡'
    elif -15 <= slope < -8: return '陡下坡'
    else: return '极陡下坡'

SLOPE_ORDER = ['极陡坡', '陡坡', '缓坡', '平地', '缓下坡', '陡下坡', '极陡下坡']
# 调高了颜色饱和度和透明度配合面积填充
COLOR_MAP = {
    '极陡坡': 'rgba(139, 0, 0, 0.75)',    # 深红
    '陡坡': 'rgba(255, 69, 0, 0.75)',     # 橙红
    '缓坡': 'rgba(255, 215, 0, 0.75)',     # 金黄
    '平地': 'rgba(34, 139, 34, 0.75)',     # 森林绿
    '缓下坡': 'rgba(0, 206, 209, 0.75)',   # 闪绿
    '陡下坡': 'rgba(30, 144, 255, 0.75)',   # 道奇蓝
    '极陡下坡': 'rgba(0, 0, 139, 0.75)'     # 深蓝
}

# --- 3. 全新重构算法层：百米重采样与山峰保真处理器 ---
@st.cache_data
def process_gpx_hardcore(file, segment_size_m, vertical_threshold, gain_coef):
    try:
        tree = ET.parse(file)
        root = tree.getroot()
    except Exception:
        st.error("❌ 该文件非合法的XML格式，无法作为GPX读取！")
        return None, []

    raw_points = []
    nodes = root.findall('.//{*}trkpt')
    if not nodes: nodes = root.findall('.//{*}rtept')
    if not nodes: nodes = root.findall('.//{*}wpt')
        
    for pt in nodes:
        lat = float(pt.get('lat'))
        lon = float(pt.get('lon'))
        ele_node = pt.find('.//{*}ele')
        ele = float(ele_node.text) if ele_node is not None and ele_node.text else 0.0
        raw_points.append({'lat': lat, 'lon': lon, 'ele': ele})
        
    if len(raw_points) == 0:
        st.error("❌ 深度扫描失败：文件中未包含有效的坐标坐标！")
        return None, []

    # 1. 建立累积原始距离（无损保留山峰最高点）
    raw_df = pd.DataFrame(raw_points)
    raw_df['dist_diff'] = 0.0
    for i in range(1, len(raw_df)):
        raw_df.loc[i, 'dist_diff'] = haversine(raw_df.loc[i-1, 'lon'], raw_df.loc[i-1, 'lat'], raw_df.loc[i, 'lon'], raw_df.loc[i, 'lat'])
    raw_df['cum_dist_m'] = raw_df['dist_diff'].cumsum()
    
    total_len_m = raw_df['cum_dist_m'].iloc[-1]
    
    # 2. 100米区间锚定重采样机制（确保区间端点准确捕捉到区域内的极值）
    grid_points = []
    current_target = 0.0
    
    while current_target <= total_len_m:
        # 寻找最接近当前网格点的原始GPS点
        idx = (raw_df['cum_dist_m'] - current_target).abs().idxmin()
        matched_row = raw_df.loc[idx]
        
        # 【山峰保真窗口】在当前100米范围内扫描原始数据的最大值，防止最高海拔被稀释
        local_window = raw_df[(raw_df['cum_dist_m'] >= current_target - segment_size_m/2) & 
                              (raw_df['cum_dist_m'] <= current_target + segment_size_m/2)]
        
        ele_value = matched_row['ele']
        if not local_window.empty:
            # 如果附近有显著的高峰，优先锁死高峰海拔
            local_max = local_window['ele'].max()
            if local_max - ele_value > 5.0: 
                ele_value = local_max

        grid_points.append({
            'cum_dist_m': current_target,
            'lat': matched_row['lat'],
            'lon': matched_row['lon'],
            'ele_raw': ele_value
        })
        current_target += segment_size_m

    df_grid = pd.DataFrame(grid_points)
    n_grid = len(df_grid)
    
    # 3. 区间爬升计算与相邻性质合并
    df_grid['dist_diff'] = segment_size_m
    df_grid.loc[0, 'dist_diff'] = 0.0
    df_grid['cum_dist_km'] = df_grid['cum_dist_m'] / 1000.0
    
    ele_filtered = np.zeros(n_grid)
    ele_diff_clean = np.zeros(n_grid)
    ele_filtered[0] = df_grid['ele_raw'].iloc[0]
    
    for i in range(1, n_grid):
        h_diff = df_grid['ele_raw'].iloc[i] - df_grid['ele_raw'].iloc[i-1]
        
        # 阈值过滤卡口，但对保留下来的高度叠加放大系数
        if abs(h_diff) >= vertical_threshold:
            ele_diff_clean[i] = h_diff * gain_coef
        else:
            ele_diff_clean[i] = 0.0
            
        ele_filtered[i] = ele_filtered[i-1] + ele_diff_clean[i]
        
    df_grid['ele_filtered'] = ele_filtered
    df_grid['ele_diff_clean'] = ele_diff_clean
    
    # 计算百米均线坡度并分类
    df_grid['slope_aligned'] = np.where(df_grid['dist_diff'] > 0, (df_grid['ele_diff_clean'] / df_grid['dist_diff']) * 100, 0)
    df_grid['slope_class'] = df_grid['slope_aligned'].apply(classify_slope)

    # 4. 提取航点（CP点）
    detected_waypoints = []
    for wpt in root.findall('.//{*}wpt'):
        name_node = wpt.find('.//{*}name')
        wpt_name = name_node.text if name_node is not None and name_node.text else "未命名CP点"
        try:
            wpt_lat = float(wpt.get('lat'))
            wpt_lon = float(wpt.get('lon'))
        except (TypeError, ValueError):
            continue
            
        min_dist = float('inf')
        matched_km = 0.0
        for i in range(len(df_grid)):
            d = haversine(wpt_lon, wpt_lat, df_grid.loc[i, 'lon'], df_grid.loc[i, 'lat'])
            if d < min_dist:
                min_dist = d
                matched_km = df_grid.loc[i, 'cum_dist_km']
        if min_dist < 600: 
            detected_waypoints.append({"name": wpt_name, "km": round(matched_km, 2)})
            
    detected_waypoints = sorted(detected_waypoints, key=lambda x: x['km'])
    return df_grid, detected_waypoints

# --- 4. 侧边栏交互配置区 ---
st.sidebar.header("⏱️ 1. 基础运动配速 (min/km)")
paces = {
    '极陡坡': st.sidebar.number_input("极陡坡 (>15%)", value=25.0, step=0.5, min_value=1.0),
    '陡坡': st.sidebar.number_input("陡坡 (8~15%)", value=15.0, step=0.5, min_value=1.0),
    '缓坡': st.sidebar.number_input("缓坡 (4~8%)", value=8.0, step=0.5, min_value=1.0),
    '平地': st.sidebar.number_input("平地 (-4~4%)", value=5.5, step=0.1, min_value=1.0),
    '缓下坡': st.sidebar.number_input("缓下坡 (-8~-4%)", value=4.5, step=0.1, min_value=1.0),
    '陡下坡': st.sidebar.number_input("陡下坡 (-15~-8%)", value=6.0, step=0.1, min_value=1.0),
    '极陡下坡': st.sidebar.number_input("极陡下坡 (< -15%)", value=10.0, step=0.5, min_value=1.0),
}

st.sidebar.markdown("---")
st.sidebar.header("📉 2. 体能衰减模型")
fatigue_rate = st.sidebar.slider("每跑 10 公里，配速衰减比例 (%)", min_value=0, max_value=20, value=5, step=1) / 100.0

st.sidebar.markdown("---")
st.sidebar.header("📍 3. 备用手动 CP 点")
cp_backup_input = st.sidebar.text_input("备用手动分段公里数（逗号隔开）", value="15, 30, 45")

st.sidebar.markdown("---")
st.sidebar.header("🛡️ 4. 降噪与重采样精算调参")
# 锁定或者开放最小分析分段，默认满足用户提出的 100 米硬指标
user_segment_size = st.sidebar.slider("📐 最小爬升核算分段步长 (米)", min_value=50, max_value=500, value=100, step=50)
user_vertical_threshold = st.sidebar.slider("垂直过滤噪声门限 (米)", min_value=0.0, max_value=3.0, value=0.0, step=0.1)
user_gain_coef = st.sidebar.slider("📊 全局高程放大增益系数", min_value=1.0, max_value=1.4, value=1.08, step=0.01)

# --- 5. 主页面业务流 ---
uploaded_file = st.file_uploader("第一步：上传官方赛道或手表导出的 GPX 文件", type=["gpx"])

if uploaded_file:
    uploaded_file.seek(0)
    df, gpx_wpts = process_gpx_hardcore(uploaded_file, user_segment_size, user_vertical_threshold, user_gain_coef)
    
    if df is not None:
        total_dist = float(df['dist_diff'].sum() / 1000.0)
        total_ascent = float(df[df['ele_diff_clean'] > 0]['ele_diff_clean'].sum())
        total_descent = float(abs(df[df['ele_diff_clean'] < 0]['ele_diff_clean'].sum()))
        
        df['fatigue_factor'] = 1.0 + (df['cum_dist_km'] // 10) * fatigue_rate
        df['pred_pace'] = df.apply(lambda row: paces.get(row['slope_class'], 6.0) * row['fatigue_factor'], axis=1)
        df['time_spent_min'] = (df['dist_diff'] / 1000.0) * df['pred_pace']
        total_time_min = float(df['time_spent_min'].sum())

        # --- 仪表盘看板 ---
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("📐 赛道总里程", f"{total_dist:.2f} km")
        m_col2.metric("🔺 山峰保真总爬升", f"{total_ascent:.0f} m")
        m_col3.metric("🔻 山峰保真总下降", f"{total_descent:.0f} m")
        hours, mins = divmod(int(total_time_min), 60)
        m_col4.metric("⏱️ 智能预测总用时", f"{hours}小时 {mins}分钟")

        # 路由分段切分路由
        break_points = [0.0]
        seg_labels = []
        
        if len(gpx_wpts) > 0:
            st.success(f"🎯 成功识别到文件内置的 {len(gpx_wpts)} 个官方CP航点！")
            st.markdown(" | ".join([f"📍 **{w['name']}** ({w['km']:.1f}km)" for w in gpx_wpts]))
            for w in gpx_wpts:
                if 0 < w['km'] < total_dist:
                    break_points.append(w['km'])
            break_points.append(total_dist)
            break_points = sorted(list(set(break_points)))
            for i in range(len(break_points)-1):
                start_name = "起点" if i == 0 else gpx_wpts[i-1]['name']
                end_name = "终点" if i == len(break_points)-2 else gpx_wpts[i]['name']
                seg_labels.append(f"{start_name} ➔ {end_name}")
        else:
            try:
                manual_kms = [float(x.strip()) for x in cp_backup_input.split(",") if x.strip() != ""]
                manual_kms = sorted([x for x in manual_kms if 0 < x < total_dist])
            except ValueError:
                manual_kms = []
            break_points = [0.0] + manual_kms + [total_dist]
            for i in range(len(break_points)-1):
                if i == 0: seg_labels.append("起点 ➔ CP1")
                elif i == len(break_points)-2: seg_labels.append(f"CP{i} ➔ 终点")
                else: seg_labels.append(f"CP{i} ➔ CP{i+1}")

        df['cp_seg'] = pd.cut(df['cum_dist_km'], bins=break_points, labels=seg_labels, include_lowest=True)

        # --- 6. 全新视觉层：相邻坡度区间合并 + 山峦线下区域面积填充图 ---
        st.subheader("🌋 100米级同质坡度合并 · 山峦面积填充图")
        fig = go.Figure()

        # 算法逻辑：遍历数据，将相邻且 slope_class 相同的网格区间合并成连续的连续色块进行 Area 填充
        i = 0
        n_points = len(df)
        while i < n_points - 1:
            current_class = df.loc[i, 'slope_class']
            start_idx = i
            
            # 向后扫描直到坡度类型发生改变
            while i < n_points - 1 and df.loc[i+1, 'slope_class'] == current_class:
                i += 1
            end_idx = min(i + 1, n_points - 1) # 包含边界点使色块无缝拼接
            
            # 提取该同质区间的片段数据
            seg_chunk = df.loc[start_idx:end_idx]
            
            # 使用 fill='tozeroy' 渲染线下区域颜色
            fig.add_trace(go.Scatter(
                x=seg_chunk['cum_dist_km'], 
                y=seg_chunk['ele_filtered'],
                mode='lines',
                line=dict(width=0.5, color='rgba(0,0,0,0)'), # 隐藏线条边缘，完全靠面积展现
                fill='tozeroy',
                fillcolor=COLOR_MAP.get(current_class, 'rgba(128,128,128,0.5)'),
                name=current_class,
                legendgroup=current_class,
                showlegend=False, # 防止图例被无数的分段碎块塞满
                hoverinfo='text',
                text=[f"里程: {d:.2f}km<br>海拔: {e:.0f}m<br>类型: {c}({s:.1f}%)" 
                      for d, e, c, s in zip(seg_chunk['cum_dist_km'], seg_chunk['ele_filtered'], seg_chunk['slope_class'], seg_chunk['slope_aligned'])]
            ))
            i += 1

        # 单独为右侧图例添加 7 个虚拟样式占位，防止图例因合并算法失效
        for cat in SLOPE_ORDER:
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode='markers',
                marker=dict(size=10, color=COLOR_MAP[cat], symbol='square'),
                name=cat
            ))

        # 垂直切分线（CP分段点）
        for bp in break_points[1:-1]:
            fig.add_vline(x=bp, line_width=1.5, line_dash="dash", line_color="#4F4F4F")

        fig.update_layout(
            xaxis_title="距离里程 (km)", 
            yaxis_title="海拔高度 (m)", 
            legend_title="地形坡度分类", 
            hovermode="x unified", 
            template="plotly_white", 
            height=550
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- 赛事分段战术计划表 ---
        st.subheader("📋 赛事分段战术耗时表")
        cp_stats = []
        cum_time_min = 0.0
        
        for i, seg_name in enumerate(seg_labels):
            seg_start = break_points[i]
            seg_end = break_points[i+1]
            seg_df = df[df['cp_seg'] == seg_name]
            
            if len(seg_df) > 0:
                seg_dist = seg_end - seg_start
                seg_ascent = seg_df[seg_df['ele_diff_clean'] > 0]['ele_diff_clean'].sum()
                seg_descent = abs(seg_df[seg_df['ele_diff_clean'] < 0]['ele_diff_clean'].sum())
                seg_time = seg_df['time_spent_min'].sum()
                
                cum_time_min += seg_time
                s_h, s_m = divmod(int(seg_time), 60)
                c_h, c_m = divmod(int(cum_time_min), 60)
                
                cp_stats.append({
                    "赛段区间": seg_name,
                    "段内里程 (km)": f"{seg_dist:.2f}",
                    "本段爬升 (m)": f"+{seg_ascent:.0f}",
                    "本段下降 (m)": f"-{seg_descent:.0f}",
                    "本段预估耗时": f"{s_h}小时 {s_m}分钟",
                    "累计比赛时间": f"⏱️ {c_h:02d}:{c_m:02d}"
                })
        st.dataframe(pd.DataFrame(cp_stats), use_container_width=True, hide_index=True)