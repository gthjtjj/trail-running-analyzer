import streamlit as st
import gpxpy
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- 1. 页面基本配置 ---
st.set_page_config(
    page_title="越野跑赛道智能分析预测器 Pro", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

st.title("🏃‍♂️ 跑者硬核路书：越野跑赛道智能分析预测器")
st.markdown("""
本程序采用 **自适应垂直门限算法** 与 **空间坐标投影技术**，
完美对齐赛道海拔与坡度颜色，精准剔除 GPS 锯齿噪声，并优先自动提取 GPX 内置的官方 CP 点信息。
""")
st.markdown("---")

# --- 2. 常量与色彩映射定义 ---
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
    '极陡坡': '#8B0000',      # 深红
    '陡坡': '#FF4500',        # 橙红
    '缓坡': '#FFD700',        # 金黄
    '平地': '#228B22',        # 森林绿
    '缓下坡': '#00FFFF',      # 青色
    '陡下坡': '#1E90FF',      # 道奇蓝
    '极陡下坡': '#00008B'     # 深蓝
}

# --- 3. 核心算法层：增强版 GPX 处理器 ---
@st.cache_data
def process_gpx_ultimate(file):
    try:
        gpx_content = file.read().decode("utf-8")
        gpx = gpxpy.parse(gpx_content)
    except Exception:
        try:
            file.seek(0)
            gpx_content = file.read().decode("gbk")
            gpx = gpxpy.parse(gpx_content)
        except Exception:
            st.error("❌ GPX文件编码或结构解析失败！请确保它是标准合法的GPX文件。")
            return None, []

    # A. 提取基础轨迹点
    points = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                ele = point.elevation if point.elevation is not None else 0.0
                points.append([point.latitude, point.longitude, ele])
    
    if len(points) == 0:
        st.error("❌ 未在GPX文件中找到有效的轨迹坐标点！")
        return None, []
        
    df = pd.DataFrame(points, columns=['lat', 'lon', 'ele'])
    
    # 逐点计算水平距离 (米)
    df['dist_diff'] = 0.0
    for i in range(1, len(df)):
        df.loc[i, 'dist_diff'] = gpxpy.geo.distance(
            df.loc[i-1, 'lat'], df.loc[i-1, 'lon'], None, 
            df.loc[i, 'lat'], df.loc[i, 'lon'], None
        )
    df['cum_dist_km'] = df['dist_diff'].cumsum() / 1000.0

    # 🛑 核心算法 1：垂直门限滤波 (Vertical Threshold Filter)
    # 彻底杜绝“海岸线悖论”导致的爬升值虚高
    ele_raw = df['ele'].values
    ele_clean = np.copy(ele_raw)
    THRESHOLD = 1.5  # 物理海拔过滤门限（米），抹平高频高斯噪声
    
    last_valid_ele = ele_raw[0]
    for i in range(1, len(ele_raw)):
        if abs(ele_raw[i] - last_valid_ele) >= THRESHOLD:
            last_valid_ele = ele_raw[i]
            ele_clean[i] = ele_raw[i]
        else:
            ele_clean[i] = last_valid_ele

    df['ele_filtered'] = ele_clean
    df['ele_diff_clean'] = df['ele_filtered'].diff().fillna(0)
    
    # 🛑 核心算法 2：窄窗口空间对齐坡度 (No Phase Lag)
    # 杜绝相位滞后带来的“大陡坡变绿色平路”失真
    df['slope_raw'] = np.where(df['dist_diff'] > 0, (df['ele_diff_clean'] / df['dist_diff']) * 100, 0)
    df['slope_aligned'] = df['slope_raw'].rolling(window=9, min_periods=1, center=True).mean()
    df['slope_class'] = df['slope_aligned'].apply(classify_slope)

    # B. 核心算法 3：空间最近邻航点匹配
    detected_waypoints = []
    for wpt in gpx.waypoints:
        wpt_name = wpt.name if wpt.name else "未命名CP点"
        min_dist = float('inf')
        matched_km = 0.0
        
        # 将航点垂直投影匹配到最近的轨迹线上
        for i in range(0, len(df), max(1, len(df)//2000)):  # 步长优化防卡死
            d = gpxpy.geo.distance(wpt.latitude, wpt.longitude, None, df.loc[i, 'lat'], df.loc[i, 'lon'], None)
            if d < min_dist:
                min_dist = d
                matched_km = df.loc[i, 'cum_dist_km']
        
        # 过滤掉离赛道过远的无关航点（阈值：500米）
        if min_dist < 500:
            detected_waypoints.append({
                "name": wpt_name,
                "km": round(matched_km, 2)
            })
            
    detected_waypoints = sorted(detected_waypoints, key=lambda x: x['km'])
    return df, detected_waypoints

# --- 4. 侧边栏用户交互配置 ---
st.sidebar.header("⏱️ 1. 基础运动配速 (min/km)")
st.sidebar.markdown("请根据个人体能填入各坡度下的**纯平路/无衰减配速**：")
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
st.sidebar.header("📉 2. 核心体能衰减模型")
fatigue_rate = st.sidebar.slider("每跑 10 公里，配速衰减比例 (%)", min_value=0, max_value=20, value=5, step=1) / 100.0

st.sidebar.markdown("---")
st.sidebar.header("📍 3. 备用手动 CP 点")
st.sidebar.markdown("💡 *仅当上传的GPX文件没有内置官方航点时，才会启用此手动公里数切分：*")
cp_backup_input = st.sidebar.text_input("备用手动分段公里数（逗号隔开）", value="15, 30, 45")


# --- 5. 主页面业务流控制 ---
uploaded_file = st.file_uploader("第一步：上传官方赛道或手表导出的 GPX 文件", type=["gpx"])

if uploaded_file:
    df, gpx_wpts = process_gpx_ultimate(uploaded_file)
    
    if df is not None:
        # 计算大盘全局指标
        total_dist = df['dist_diff'].sum() / 1000.0
        total_ascent = df[df['ele_diff_clean'] > 0]['ele_diff_clean'].sum()
        total_descent = abs(df[df['ele_diff_clean'] < 0]['ele_diff_clean'].sum())
        
        # 动态切分点决策系统
        break_points = [0.0]
        seg_labels = []
        
        if len(gpx_wpts) > 0:
            st.success(f"🎯 成功识别到该赛事GPX文件内置的 {len(gpx_wpts)} 个官方CP航点！已自动关联：")
            # 优雅展示检测到的航点路牌
            wpt_info_html = " | ".join([f"📍 **{w['name']}** ({w['km']:.1f}km)" for w in gpx_wpts])
            st.markdown(wpt_info_html)
            
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
            st.info("ℹ️ 提示：该GPX文件未包含内置航点，系统已自动切换至侧边栏的备用手动CP分段。")
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

        # 将全量点归类到对应的赛段区间中
        df['cp_seg'] = pd.cut(df['cum_dist_km'], bins=break_points, labels=seg_labels, include_lowest=True)

        # 积分计算动态体能衰减配速与用时
        df['fatigue_factor'] = 1.0 + (df['cum_dist_km'] // 10) * fatigue_rate
        df['pred_pace'] = df.apply(lambda row: paces.get(row['slope_class'], 6.0) * row['fatigue_factor'], axis=1)
        df['time_spent_min'] = (df['dist_diff'] / 1000.0) * df['pred_pace']
        total_time_min = df['time_spent_min'].sum()

        # --- 数据可视化仪表盘 ---
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("📐 赛道总里程", f"{total_dist:.2f} km")
        m_col2.metric("🔺 净化累计爬升", f"{total_ascent:.0f} m")
        m_col3.metric("🔻 净化累计下降", f"{total_descent:.0f} m")
        hours, mins = divmod(int(total_time_min), 60)
        m_col4.metric("⏱️ 智能预测总用时", f"{hours}小时 {mins}分钟")

        # --- 海拔剖面图 (空间精准对齐版) ---
        st.subheader("🌋 空间对齐·精细化赛道剖面图")
        fig = go.Figure()

        # 逐色彩/坡度级别绘制曲线，确保互斥不重合
        for cat in SLOPE_ORDER:
            if cat in COLOR_MAP:
                cat_df = df.copy()
                # 核心改动：不仅坡度分类要对应，且 Y 轴数据必须统一指向净化后的 ele_filtered
                cat_df.loc[df['slope_class'] != cat, 'ele_filtered'] = None
                
                fig.add_trace(go.Scatter(
                    x=cat_df['cum_dist_km'], y=cat_df['ele_filtered'],
                    mode='lines', line=dict(color=COLOR_MAP[cat], width=3.5),
                    name=cat, hoverinfo='text',
                    text=[f"里程: {d:.2f}km<br>去噪海拔: {e:.0f}m<br>即时坡度: {s:.1f}%<br>动态配速: {p:.1f} min/km" 
                          for d, e, s, p in zip(df['cum_dist_km'], df['ele_filtered'], df['slope_aligned'], df['pred_pace'])]
                ))

        # 在画布上绘制 CP 航点垂直切分虚线
        for bp in break_points[1:-1]:
            fig.add_vline(x=bp, line_width=1.5, line_dash="dash", line_color="#8C8C8C")

        fig.update_layout(
            xaxis_title="距离里程 (km)", 
            yaxis_title="海拔高度 (m)", 
            legend_title="地形与坡度分类", 
            hovermode="x unified", 
            template="plotly_white", 
            height=520
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- 赛事分段战术计划表 ---
        st.subheader("📋 赛事官方航点分段战术耗时表")
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
                    "净化爬升 (m)": f"+{seg_ascent:.0f}",
                    "净化下降 (m)": f"-{seg_descent:.0f}",
                    "本段预估耗时": f"{s_h}小时 {s_m}分钟",
                    "赛道累计时间": f"⏱️ {c_h:02d}:{c_m:02d}"
                })
        st.dataframe(pd.DataFrame(cp_stats), use_container_width=True, hide_index=True)

        # --- 各 CP 段地形占比分布矩阵 ---
        st.subheader("📊 各 CP 赛段微观地形分布矩阵 (单位: 公里)")
        df['dist_diff_km'] = df['dist_diff'] / 1000.0
        pivot_df = pd.pivot_table(df, values='dist_diff_km', index='cp_seg', columns='slope_class', aggfunc='sum', fill_value=0.0)
        available_cols = [col for col in SLOPE_ORDER if col in pivot_df.columns]
        pivot_df = pivot_df.reindex(index=seg_labels, columns=available_cols).fillna(0.0)
        
        # 追加汇总行
        pivot_df.loc['全赛道总公里数'] = pivot_df.sum()
        st.dataframe(pivot_df.round(2), use_container_width=True)

else:
    st.info("💡 提示：请在上方上传越野赛 `.gpx` 轨迹文件，即可启动全自动无失真专业路书分析。")