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

st.title("🏃‍♂️ 跑者硬核路书：越野跑赛道智能分析预测器 (全局独立精算版)")
st.markdown("""
本程序采用 **全局总指标独立核算机制**，彻底剥离了分段切片对大盘总爬升的影响。
无论手机/手表实测轨迹还是航点如何错位，均能保障全局总爬升的绝对精准。
""")
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
COLOR_MAP = {
    '极陡坡': '#8B0000', '陡坡': '#FF4500', '缓坡': '#FFD700',
    '平地': '#228B22', '缓下坡': '#00FFFF', '陡下坡': '#1E90FF', '极陡下坡': '#00008B'
}

# --- 3. 终极算法层：全局核算 GPX 处理器 ---
@st.cache_data
def process_gpx_hardcore(file, min_sampling_dist, vertical_threshold):
    try:
        tree = ET.parse(file)
        root = tree.getroot()
    except Exception:
        st.error("❌ 该文件非合法的XML格式，无法作为GPX读取！")
        return None, []

    raw_points = []
    nodes = root.findall('.//{*}trkpt')
    point_type = "标准轨迹点 (trkpt)"
    
    if not nodes:
        nodes = root.findall('.//{*}rtept')
        point_type = "规划航线点 (rtept)"
        
    if not nodes:
        nodes = root.findall('.//{*}wpt')
        point_type = "独立位置点 (wpt)"
        
    for pt in nodes:
        lat = float(pt.get('lat'))
        lon = float(pt.get('lon'))
        ele_node = pt.find('.//{*}ele')
        ele = float(ele_node.text) if ele_node is not None and ele_node.text else 0.0
        raw_points.append({'lat': lat, 'lon': lon, 'ele': ele})
        
    if len(raw_points) == 0:
        st.error("❌ 深度扫描失败：文件中未包含有效的坐标坐标！")
        return None, []
    
    st.toast(f"ℹ️ 成功通过 【{point_type}】 模式提取了 {len(raw_points)} 个地理坐标！")

    # 🚀 空间抽稀滤波
    filtered_points = [raw_points[0]]
    for i in range(1, len(raw_points)):
        last_pt = filtered_points[-1]
        curr_pt = raw_points[i]
        d = haversine(last_pt['lon'], last_pt['lat'], curr_pt['lon'], curr_pt['lat'])
        if d >= min_sampling_dist or i == len(raw_points) - 1:
            filtered_points.append(curr_pt)
            
    df = pd.DataFrame(filtered_points)
    
    df['dist_diff'] = 0.0
    for i in range(1, len(df)):
        df.loc[i, 'dist_diff'] = haversine(df.loc[i-1, 'lon'], df.loc[i-1, 'lat'], df.loc[i, 'lon'], df.loc[i, 'lat'])
    df['cum_dist_km'] = df['dist_diff'].cumsum() / 1000.0

    # 🛑 自适应台阶方向累加过滤算法
    ele_raw = df['ele'].values
    dist_diff = df['dist_diff'].values
    ele_clean = np.copy(ele_raw)
    
    MAX_PHYSICAL_SLOPE = 65.0 
    last_valid_ele = ele_raw[0]
    accumulated_h_diff = 0.0 
    
    for i in range(1, len(ele_raw)):
        h_diff = ele_raw[i] - last_valid_ele
        d_diff = dist_diff[i]
        
        instant_slope = (abs(h_diff) / d_diff * 100) if d_diff > 0 else 0
        if instant_slope > MAX_PHYSICAL_SLOPE:
            ele_clean[i] = last_valid_ele
            continue
            
        if abs(h_diff) >= vertical_threshold:
            last_valid_ele = ele_raw[i]
            ele_clean[i] = ele_raw[i]
            accumulated_h_diff = 0.0 
        else:
            accumulated_h_diff += h_diff
            if abs(accumulated_h_diff) >= vertical_threshold:
                last_valid_ele = ele_raw[i]
                ele_clean[i] = ele_raw[i]
                accumulated_h_diff = 0.0
            else:
                ele_clean[i] = last_valid_ele

    df['ele_filtered'] = ele_clean
    df['ele_diff_clean'] = df['ele_filtered'].diff().fillna(0)
    
    # 坡度平滑
    df['slope_raw'] = np.where(df['dist_diff'] > 0, (df['ele_diff_clean'] / df['dist_diff']) * 100, 0)
    df['slope_aligned'] = df['slope_raw'].rolling(window=5, min_periods=1, center=True).mean()
    df['slope_class'] = df['slope_aligned'].apply(classify_slope)

    # 航点提取逻辑（修正拼写错误并加入类型防御）
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
        for i in range(len(df)):
            d = haversine(wpt_lon, wpt_lat, df.loc[i, 'lon'], df.loc[i, 'lat'])
            if d < min_dist:
                min_dist = d
                matched_km = df.loc[i, 'cum_dist_km']
        if min_dist < 500: 
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

st.sidebar.markdown("---")
st.sidebar.header("🛡️ 4. 降噪与爬升对齐核心调参")
user_sampling_dist = st.sidebar.slider("空间抽稀采样步长 (米)", min_value=1, max_value=30, value=2, step=1)
user_vertical_threshold = st.sidebar.slider("垂直过滤门限 (米)", min_value=0.1, max_value=5.0, value=0.5, step=0.1)

# --- 5. 主页面业务流 ---
uploaded_file = st.file_uploader("第一步：上传官方赛道或手表导出的 GPX 文件", type=["gpx"])

if uploaded_file:
    uploaded_file.seek(0)
    df, gpx_wpts = process_gpx_hardcore(uploaded_file, user_sampling_dist, user_vertical_threshold)
    
    if df is not None:
        # 🛑 【核心修正】：全局指标一锅出，完全不受任何后续分段切片逻辑的干扰！
        total_dist = float(df['dist_diff'].sum() / 1000.0)
        total_ascent = float(df[df['ele_diff_clean'] > 0]['ele_diff_clean'].sum())
        total_descent = float(abs(df[df['ele_diff_clean'] < 0]['ele_diff_clean'].sum()))
        
        # 基础耗时重算
        df['fatigue_factor'] = 1.0 + (df['cum_dist_km'] // 10) * fatigue_rate
        df['pred_pace'] = df.apply(lambda row: paces.get(row['slope_class'], 6.0) * row['fatigue_factor'], axis=1)
        df['time_spent_min'] = (df['dist_diff'] / 1000.0) * df['pred_pace']
        total_time_min = float(df['time_spent_min'].sum())

        # --- 仪表盘看板 ---
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("📐 赛道总里程", f"{total_dist:.2f} km")
        m_col2.metric("🔺 全局独立总爬升", f"{total_ascent:.0f} m")
        m_col3.metric("🔻 全局独立总下降", f"{total_descent:.0f} m")
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

        # 使用对边界更有耐受力的分段方式
        df['cp_seg'] = pd.cut(df['cum_dist_km'], bins=break_points, labels=seg_labels, include_lowest=True)

        # --- 海拔剖面图 ---
        st.subheader("🌋 空间对齐·精细化赛道地形剖面图")
        fig = go.Figure()
        for cat in SLOPE_ORDER:
            if cat in COLOR_MAP:
                cat_df = df.copy()
                cat_df.loc[df['slope_class'] != cat, 'ele_filtered'] = None
                fig.add_trace(go.Scatter(
                    x=cat_df['cum_dist_km'], y=cat_df['ele_filtered'],
                    mode='lines', line=dict(color=COLOR_MAP[cat], width=3.5),
                    name=cat, hoverinfo='text',
                    text=[f"里程: {d:.2f}km<br>海拔: {e:.0f}m<br>坡度: {s:.1f}%" 
                          for d, e, s in zip(df['cum_dist_km'], df['ele_filtered'], df['slope_aligned'])]
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
                    "本段爬升 (m)": f"+{seg_ascent:.0f}",
                    "本段下降 (m)": f"-{seg_descent:.0f}",
                    "本段预估耗时": f"{s_h}小时 {s_m}分钟",
                    "累计比赛时间": f"⏱️ {c_h:02d}:{c_m:02d}"
                })
        st.dataframe(pd.DataFrame(cp_stats), use_container_width=True, hide_index=True)