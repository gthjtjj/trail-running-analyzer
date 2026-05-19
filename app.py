import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import xml.etree.ElementTree as ET
from math import radians, cos, sin, asin, sqrt

# --- 1. 页面基本配置 ---
st.set_page_config(
    page_title="越野跑赛道智能分析预测器 Ultimate", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

st.title("🏃‍♂️ 跑者硬核路书：越野跑赛道智能分析预测器 (终极全兼容版)")
st.markdown("""
本程序采用 **底层XML高容错解析（无需时间戳）**、**空间等距抽稀（每15米采样）** 与 **物理极限坡度卡口**，
彻底根治 GPX 读取失败以及爬升膨胀问题，为你生成绝对精准、绝无滞后错位的硬核战术路书。
""")
st.markdown("---")

# --- 2. 核心数学与地理工具函数 ---
def haversine(lon1, lat1, lon2, lat2):
    """纯数学计算两点间大圆距离(米)，完全脱离外部地理库依赖，防止崩溃"""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371000 # 地球平均半径，单位为米
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
COLOR_MAP = {
    '极陡坡': '#8B0000', '陡坡': '#FF4500', '缓坡': '#FFD700',
    '平地': '#228B22', '缓下坡': '#00FFFF', '陡下坡': '#1E90FF', '极陡下坡': '#00008B'
}

# --- 3. 终极算法层：高容错、抗燥型物理 GPX 处理器 ---
@st.cache_data
def process_gpx_hardcore(file):
    try:
        # 直接使用标准的 XML 解析器，不再管 GPX 的标准版本、时间信息或扩展标签
        tree = ET.parse(file)
        root = tree.getroot()
    except Exception:
        st.error("❌ 该文件非合法的XML格式，无法作为GPX读取！请尝试从大平台重新导出。")
        return None, []

    # 自动处理 XML 命名空间问题
    ns = ''
    if root.tag.startswith('{'):
        ns = root.tag.split('}')[0] + '}'

    # A. 提取基础轨迹点（只抓取最硬核的 lat, lon, ele，无视 time）
    raw_points = []
    for trkpt in root.findall(f'.//{ns}trkpt'):
        lat = float(trkpt.get('lat'))
        lon = float(trkpt.get('lon'))
        ele_node = trkpt.find(f'{ns}ele')
        ele = float(ele_node.text) if ele_node is not None and ele_node.text else 0.0
        raw_points.append({'lat': lat, 'lon': lon, 'ele': ele})
        
    if len(raw_points) == 0:
        st.error("❌ 未在文件中找到任何轨迹坐标点 (<trkpt>)！")
        return None, []

    # B. 🚀 空间大步长抽稀滤波（Downsampling）
    # 强制让点与点之间保持至少 15 米的物理距离，直接抹杀高频密集抖动噪声
    filtered_points = [raw_points[0]]
    MIN_SAMPLING_DIST = 15.0 # 采样步长：15米
    
    for i in range(1, len(raw_points)):
        last_pt = filtered_points[-1]
        curr_pt = raw_points[i]
        d = haversine(last_pt['lon'], last_pt['lat'], curr_pt['lon'], curr_pt['lat'])
        if d >= MIN_SAMPLING_DIST or i == len(raw_points) - 1:
            filtered_points.append(curr_pt)
            
    df = pd.DataFrame(filtered_points)
    
    # 计算抽稀后的点间距与累计里程
    df['dist_diff'] = 0.0
    for i in range(1, len(df)):
        df.loc[i, 'dist_diff'] = haversine(df.loc[i-1, 'lon'], df.loc[i-1, 'lat'], df.loc[i, 'lon'], df.loc[i, 'lat'])
    df['cum_dist_km'] = df['dist_diff'].cumsum() / 1000.0

    # 🛑 核心算法 1：增强版垂直门限 + 物理极限坡度双重拦截
    ele_raw = df['ele'].values
    dist_diff = df['dist_diff'].values
    ele_clean = np.copy(ele_raw)
    
    VERTICAL_THRESHOLD = 2.0  # 提高垂直门限到 2.0 米，过滤顽固高斯噪声
    MAX_PHYSICAL_SLOPE = 65.0 # 人类越野跑物理极限坡度上限 (65%)
    
    last_valid_ele = ele_raw[0]
    for i in range(1, len(ele_raw)):
        h_diff = ele_raw[i] - last_valid_ele
        d_diff = dist_diff[i]
        
        # 计算该步长的瞬时虚假坡度
        instant_slope = (abs(h_diff) / d_diff * 100) if d_diff > 0 else 0
        
        # 只有当：超过高度门限 且 坡度没有超越人类物理极限时，才承认是一次有效爬升
        if abs(h_diff) >= VERTICAL_THRESHOLD and instant_slope <= MAX_PHYSICAL_SLOPE:
            last_valid_ele = ele_raw[i]
            ele_clean[i] = ele_raw[i]
        else:
            ele_clean[i] = last_valid_ele

    df['ele_filtered'] = ele_clean
    df['ele_diff_clean'] = df['ele_filtered'].diff().fillna(0)
    
    # 🛑 核心算法 2：窄窗口对齐坡度（保证颜色绝对不错位）
    df['slope_raw'] = np.where(df['dist_diff'] > 0, (df['ele_diff_clean'] / df['dist_diff']) * 100, 0)
    df['slope_aligned'] = df['slope_raw'].rolling(window=5, min_periods=1, center=True).mean()
    df['slope_class'] = df['slope_aligned'].apply(classify_slope)

    # C. 提取内置的 WPT 航点（同样使用原生 XML 解析，防撞车）
    detected_waypoints = []
    for wpt in root.findall(f'.//{ns}wpt'):
        name_node = wpt.find(f'{ns}name')
        wpt_name = name_node.text if name_node is not None and name_node.text else "未命名CP点"
        wpt_lat = float(wpt.get('lat'))
        wpt_lon = float(wpt.get('lon'))
        
        min_dist = float('inf')
        matched_km = 0.0
        
        # 空间最近邻扫描投影
        for i in range(len(df)):
            d = haversine(wpt_lon, wpt_lat, df.loc[i, 'lon'], df.loc[i, 'lat'])
            if d < min_dist:
                min_dist = d
                matched_km = df.loc[i, 'cum_dist_km']
        
        if min_dist < 500: # 500米半径内有效关联
            detected_waypoints.append({"name": wpt_name, "km": round(matched_km, 2)})
            
    detected_waypoints = sorted(detected_waypoints, key=lambda x: x['km'])
    return df, detected_waypoints

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

# --- 5. 主页面业务流 ---
uploaded_file = st.file_uploader("第一步：上传官方赛道或手表导出的 GPX 文件", type=["gpx"])

if uploaded_file:
    # 关键改变：每次读取重置文件指针
    uploaded_file.seek(0)
    df, gpx_wpts = process_gpx_hardcore(uploaded_file)
    
    if df is not None:
        total_dist = df['dist_diff'].sum() / 1000.0
        total_ascent = df[df['ele_diff_clean'] > 0]['ele_diff_clean'].sum()
        total_descent = abs(df[df['ele_diff_clean'] < 0]['ele_diff_clean'].sum())
        
        # 决策路由切分点
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
            st.info("ℹ️ 提示：该文件无内置航点，已启用左侧栏手动CP分段。")
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

        # 时间与衰减累加计算
        df['fatigue_factor'] = 1.0 + (df['cum_dist_km'] // 10) * fatigue_rate
        df['pred_pace'] = df.apply(lambda row: paces.get(row['slope_class'], 6.0) * row['fatigue_factor'], axis=1)
        df['time_spent_min'] = (df['dist_diff'] / 1000.0) * df['pred_pace']
        total_time_min = df['time_spent_min'].sum()

        # --- 仪表盘看板 ---
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("📐 赛道总里程", f"{total_dist:.2f} km")
        m_col2.metric("🔺 脱水总爬升", f"{total_ascent:.0f} m")
        m_col3.metric("🔻 脱水总下降", f"{total_descent:.0f} m")
        hours, mins = divmod(int(total_time_min), 60)
        m_col4.metric("⏱️ 智能预测总用时", f"{hours}小时 {mins}分钟")

        # --- 海拔剖面图 (空间精准对齐版) ---
        st.subheader("🌋 空间对齐·精细化赛道剖面图")
        fig = go.Figure()

        for cat in SLOPE_ORDER:
            if cat in COLOR_MAP:
                cat_df = df.copy()
                cat_df.loc[df['slope_class'] != cat, 'ele_filtered'] = None
                
                fig.add_trace(go.Scatter(
                    x=cat_df['cum_dist_km'], y=cat_df['ele_filtered'],
                    mode='lines', line=dict(color=COLOR_MAP[cat], width=3.5),
                    name=cat, hoverinfo='text',
                    text=[f"里程: {d:.2f}km<br>海拔: {e:.0f}m<br>坡度: {s:.1f}%<br>配速: {p:.1f} min/km" 
                          for d, e, s, p in zip(df['cum_dist_km'], df['ele_filtered'], df['slope_aligned'], df['pred_pace'])]
                ))

        for bp in break_points[1:-1]:
            fig.add_vline(x=bp, line_width=1.5, line_dash="dash", line_color="#8C8C8C")

        fig.update_layout(xaxis_title="距离里程 (km)", yaxis_title="海拔高度 (m)", legend_title="地形分类", hovermode="x unified", template="plotly_white", height=500)
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
                    "脱水爬升 (m)": f"+{seg_ascent:.0f}",
                    "脱水下降 (m)": f"-{seg_descent:.0f}",
                    "本段预估耗时": f"{s_h}小时 {s_m}分钟",
                    "累计比赛时间": f"⏱️ {c_h:02d}:{c_m:02d}"
                })
        st.dataframe(pd.DataFrame(cp_stats), use_container_width=True, hide_index=True)

        # --- 各 CP 段地形占比分布矩阵 ---
        st.subheader("📊 各 CP 赛段微观地形分布矩阵 (单位: 公里)")
        df['dist_diff_km'] = df['dist_diff'] / 1000.0
        pivot_df = pd.pivot_table(df, values='dist_diff_km', index='cp_seg', columns='slope_class', aggfunc='sum', fill_value=0.0)
        available_cols = [col for col in SLOPE_ORDER if col in pivot_df.columns]
        pivot_df = pivot_df.reindex(index=seg_labels, columns=available_cols).fillna(0.0)
        pivot_df.loc['全赛道总公里数'] = pivot_df.sum()
        st.dataframe(pivot_df.round(2), use_container_width=True)

else:
    st.info("💡 提示：请在上方上传越野赛 `.gpx` 文件。本系统已升级终极容错，无时间戳文件、手绘路网文件均可完美秒读。")