import streamlit as st
import gpxpy
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- 1. 配置页面 ---
st.set_page_config(page_title="越野跑赛道分析预测器 Pro++", layout="wide")

st.title("跑者硬核路书：越野跑赛道智能分析预测器")
st.markdown("支持 **自动解析GPX内置CP航点**、**颂拓级数据去噪** 与 **全赛道战术拆解**。")

# 坡度分类函数
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

# --- 2. 增强版 GPX 解析逻辑 ---
@st.cache_data
def process_gpx_advanced(file):
    try:
        gpx_content = file.read().decode("utf-8")
        gpx = gpxpy.parse(gpx_content)
    except Exception:
        try:
            file.seek(0)
            gpx_content = file.read().decode("gbk")
            gpx = gpxpy.parse(gpx_content)
        except Exception:
            st.error("❌ GPX文件编码或格式解析失败！")
            return None, []

    # A. 提取赛道轨迹点
    points = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                ele = point.elevation if point.elevation is not None else 0.0
                points.append([point.latitude, point.longitude, ele])
    
    if len(points) == 0:
        st.error("❌ 未找到有效轨迹坐标点！")
        return None, []
        
    df = pd.DataFrame(points, columns=['lat', 'lon', 'ele'])

    # 双轨制数据去噪：计算层精细，画图层整洁
    df['ele_calc'] = df['ele'].rolling(window=3, min_periods=1, center=True).mean()
    df['ele_diff_clean'] = df['ele_calc'].diff().fillna(0)
    
    df['dist_diff'] = 0.0
    for i in range(1, len(df)):
        df.loc[i, 'dist_diff'] = gpxpy.geo.distance(df.loc[i-1, 'lat'], df.loc[i-1, 'lon'], None, df.loc[i, 'lat'], df.loc[i, 'lon'], None)
    
    df['cum_dist_km'] = df['dist_diff'].cumsum() / 1000.0
    
    # 颂拓风格强力色彩平滑（画图专用）
    df['slope_raw'] = np.where(df['dist_diff'] > 0, (df['ele_diff_clean'] / df['dist_diff']) * 100, 0)
    df['slope_suunto'] = df['slope_raw'].rolling(window=15, min_periods=1, center=True).median()
    df['slope_suunto'] = df['slope_suunto'].rolling(window=25, min_periods=1, center=True).mean()
    df['slope_class'] = df['slope_suunto'].apply(classify_slope)

    # B. 核心更新：提取 GPX 内置的 WPT 航点
    detected_waypoints = []
    for wpt in gpx.waypoints:
        wpt_name = wpt.name if wpt.name else "未命名CP点"
        
        # 在轨迹线中寻找离该航点空间距离最近的轨迹点
        min_dist = float('inf')
        matched_km = 0.0
        
        # 优化匹配性能：抽样或全量匹配
        for i in range(len(df)):
            d = gpxpy.geo.distance(wpt.latitude, wpt.longitude, None, df.loc[i, 'lat'], df.loc[i, 'lon'], None)
            if d < min_dist:
                min_dist = d
                matched_km = df.loc[i, 'cum_dist_km']
        
        # 如果航点距离赛道线太远（比如超过 500 米），可能是无关航点，排除掉
        if min_dist < 500:
            detected_waypoints.append({
                "name": wpt_name,
                "km": round(matched_km, 2)
            })
            
    # 按公里数从小到大重新排序航点
    detected_waypoints = sorted(detected_waypoints, key=lambda x: x['km'])
    
    return df, detected_waypoints

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
st.sidebar.header("📍 3. 备用手动 CP 点设置")
st.sidebar.markdown("💡 当GPX文件**没有自带航点**时，才会启用下方的手动设置：")
cp_backup_input = st.sidebar.text_input("备用手动公里数（逗号隔开）", value="15, 30, 45")

# --- 4. 主界面逻辑 ---
uploaded_file = st.file_uploader("第一步：选择并上传你的赛道 GPX 文件", type=["gpx"])

if uploaded_file:
    df, gpx_wpts = process_gpx_advanced(uploaded_file)
    
    if df is not None:
        total_dist = df['dist_diff'].sum() / 1000.0
        total_ascent = df[df['ele_diff_clean'] > 0]['ele_diff_clean'].sum()
        total_descent = abs(df[df['ele_diff_clean'] < 0]['ele_diff_clean'].sum())
        
        # === 核心切分点逻辑判断 ===
        break_points = [0.0]
        seg_labels = []
        
        if len(gpx_wpts) > 0:
            # 💡 优先：使用 GPX 文件自带的航点信息
            st.success(f"自动识别成功！从您的GPX文件中检测到 {len(gpx_wpts)} 个官方内置CP航点：")
            
            # 展示识别到的内置航点让用户心里有数
            wpt_info_str = " | ".join([f"📍 **{w['name']}** ({w['km']}km)" for w in gpx_wpts])
            st.markdown(wpt_info_str)
            
            # 构建切分点
            for w in gpx_wpts:
                if 0 < w['km'] < total_dist:
                    break_points.append(w['km'])
            break_points.append(total_dist)
            break_points = sorted(list(set(break_points))) # 确保递增不重复
            
            # 构建具有可读性的赛段标签
            for i in range(len(break_points)-1):
                start_name = "起点" if i == 0 else gpx_wpts[i-1]['name']
                end_name = "终点" if i == len(break_points)-2 else gpx_wpts[i]['name']
                seg_labels.append(f"{start_name} -> {end_name}")
                
        else:
            # 🪵 兜底：如果文件没有航点，则解析侧边栏手动输入的数据
            st.info("ℹ️ 提示：该GPX文件未包含内置航点，已自动启用您在左侧栏设置的手动CP点。")
            try:
                manual_kms = [float(x.strip()) for x in cp_backup_input.split(",") if x.strip() != ""]
                manual_kms = sorted([x for x in manual_kms if 0 < x < total_dist])
            except ValueError:
                manual_kms = []
            
            break_points = [0.0] + manual_kms + [total_dist]
            for i in range(len(break_points)-1):
                if i == 0: seg_labels.append("起点 -> CP1")
                elif i == len(break_points)-2: seg_labels.append(f"CP{i} -> 终点")
                else: seg_labels.append(f"CP{i} -> CP{i+1}")

        # 给数据打上赛段标签
        df['cp_seg'] = pd.cut(df['cum_dist_km'], bins=break_points, labels=seg_labels, include_lowest=True)

        # 时间预测计算
        df['fatigue_factor'] = 1.0 + (df['cum_dist_km'] // 10) * fatigue_rate
        df['pred_pace'] = df.apply(lambda row: paces.get(row['slope_class'], 6.0) * row['fatigue_factor'], axis=1)
        df['time_spent_min'] = (df['dist_diff'] / 1000.0) * df['pred_pace']
        total_time_min = df['time_spent_min'].sum()

        # --- 展示大盘指标 ---
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("总距离", f"{total_dist:.2f} km")
        col2.metric("累计爬升", f"{total_ascent:.0f} m")
        col3.metric("累计下降", f"{total_descent:.0f} m")
        hours, mins = divmod(int(total_time_min), 60)
        col4.metric("预估总用时 (含体能衰减)", f"{hours}h {mins}m")

        # --- 5. 海拔剖面图 ---
        st.subheader("🌋 颂拓风格·精细化赛道剖面图")
        fig = go.Figure()

        for cat in COLOR_MAP.keys():
            cat_df = df.copy()
            cat_df.loc[df['slope_class'] != cat, 'ele'] = None
            fig.add_trace(go.Scatter(
                x=cat_df['cum_dist_km'], y=cat_df['ele'],
                mode='lines', line=dict(color=COLOR_MAP[cat], width=3),
                name=cat, hoverinfo='text',
                text=[f"距离: {d:.2f}km<br>海拔: {e:.0f}m<br>坡度: {s:.1f}%<br>配速: {p:.1f} min/km" 
                      for d, e, s, p in zip(df['cum_dist_km'], df['ele'], df['slope_suunto'], df['pred_pace'])]
            ))

        # 在图表上为检测到的 CP 点画垂直虚线
        for bp in break_points[1:-1]:
            fig.add_vline(x=bp, line_width=1.5, line_dash="dash", line_color="gray")

        fig.update_layout(xaxis_title="距离 (km)", yaxis_title="海拔 (m)", legend_title="坡度分类", hovermode="x unified", template="plotly_white", height=500)
        st.plotly_chart(fig, use_container_width=True)

        # --- 6. 赛事 CP 赛段分段耗时表 ---
        st.subheader("📍 官方航点分段战术耗时表")
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
                    "段内距离 (km)": f"{seg_dist:.2f}",
                    "段内爬升 (m)": f"+{seg_ascent:.0f}",
                    "段内下降 (m)": f"-{seg_descent:.0f}",
                    "本段预计耗时": f"{s_h}小时 {s_m}分钟",
                    "预计累计耗时": f"⏱️ {c_h:02d}:{c_m:02d}"
                })
        st.dataframe(pd.DataFrame(cp_stats), use_container_width=True, hide_index=True)

        # --- 7. 各 CP 段地形占比分布矩阵 ---
        st.subheader("📊 各 CP 赛段详细地形分布 (单位: 公里)")
        df['dist_diff_km'] = df['dist_diff'] / 1000.0
        pivot_df = pd.pivot_table(df, values='dist_diff_km', index='cp_seg', columns='slope_class', aggfunc='sum', fill_value=0.0)
        available_cols = [col for col in SLOPE_ORDER if col in pivot_df.columns]
        pivot_df = pivot_df.reindex(index=seg_labels, columns=available_cols).fillna(0.0)
        pivot_df.loc['全赛道总计'] = pivot_df.sum()
        st.dataframe(pivot_df.round(2), use_container_width=True)

else:
    st.info("💡 请在上方上传包含或不包含航点的 GPX 文件。系统会自动判断并优先使用官方内置CP。")