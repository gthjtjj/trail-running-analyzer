import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import xml.etree.ElementTree as ET
from math import radians, cos, sin, asin, sqrt
import datetime

# --- 1. 页面基本配置 ---
st.set_page_config(
    page_title="越野跑赛道智能分析预测器 v9", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

st.title("🏃‍♂️ 越野跑者硬核路书：越野跑个性化完赛时间预测")
st.markdown("说明：输入各种坡路的配速，及速度衰减比例。配速输入进化为「分:秒」格式；支持绝对起跑时间推算；融合生成一体化实战数据大表；新增便携卡片打印支持。")
st.markdown("---")

# --- 2. 工具函数：配速时间格式转换与数学核算 ---
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

def parse_pace_str(pace_str, default_val):
    """将 '5:30' 或 '5.5' 转换为纯分钟数(float)"""
    try:
        pace_str = pace_str.strip()
        if ":" in pace_str:
            parts = pace_str.split(":")
            mins = int(parts[0])
            secs = int(parts[1]) if len(parts) > 1 else 0
            return mins + (secs / 60.0)
        else:
            return float(pace_str)
    except Exception:
        return default_val

def format_pace_out(minutes_float):
    """将纯分钟数(float)转换为 '5:30' 格式的字符串"""
    mins = int(minutes_float)
    secs = int(round((minutes_float - mins) * 60))
    if secs == 60:
        mins += 1
        secs = 0
    return f"{mins}:{secs:02d}"

def format_time_duration(minutes_float):
    """格式化比赛耗时"""
    hours = int(minutes_float // 60)
    mins = int(minutes_float % 60)
    return f"{hours}h {mins:02d}m"

# 线性视觉顺序及颜色映射
SLOPE_ORDER = ['极陡坡', '陡坡', '缓坡', '平地', '缓下坡', '陡下坡', '极陡下坡']
COLOR_MAP = {
    '极陡坡': 'rgba(139, 0, 0, 0.85)',    
    '陡坡': 'rgba(255, 69, 0, 0.85)',     
    '缓坡': 'rgba(255, 215, 0, 0.85)',     
    '平地': 'rgba(34, 139, 34, 0.85)',     
    '缓下坡': 'rgba(0, 206, 209, 0.85)',   
    '陡下坡': 'rgba(30, 144, 255, 0.85)',   
    '极陡下坡': 'rgba(0, 0, 139, 0.85)'     
}

# --- 3. 算法层：百米重采样与山峰保真处理器 ---
@st.cache_data
def process_gpx_hardcore(file, segment_size_m, vertical_threshold):
    try:
        tree = ET.parse(file)
        root = tree.getroot()
    except Exception:
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
        return None, []

    raw_df = pd.DataFrame(raw_points)
    raw_df['dist_diff'] = 0.0
    for i in range(1, len(raw_df)):
        raw_df.loc[i, 'dist_diff'] = haversine(raw_df.loc[i-1, 'lon'], raw_df.loc[i-1, 'lat'], raw_df.loc[i, 'lon'], raw_df.loc[i, 'lat'])
    raw_df['cum_dist_m'] = raw_df['dist_diff'].cumsum()
    
    total_len_m = raw_df['cum_dist_m'].iloc[-1]
    
    grid_points = []
    current_target = 0.0
    
    while current_target <= total_len_m:
        idx = (raw_df['cum_dist_m'] - current_target).abs().idxmin()
        matched_row = raw_df.loc[idx]
        
        local_window = raw_df[(raw_df['cum_dist_m'] >= current_target - segment_size_m/2) & 
                              (raw_df['cum_dist_m'] <= current_target + segment_size_m/2)]
        
        ele_value = matched_row['ele']
        if not local_window.empty:
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
    
    df_grid['dist_diff'] = segment_size_m
    df_grid.loc[0, 'dist_diff'] = 0.0
    df_grid['cum_dist_km'] = df_grid['cum_dist_m'] / 1000.0
    
    ele_filtered = np.zeros(n_grid)
    ele_diff_clean = np.zeros(n_grid)
    ele_filtered[0] = df_grid['ele_raw'].iloc[0]
    
    for i in range(1, n_grid):
        h_diff = df_grid['ele_raw'].iloc[i] - df_grid['ele_raw'].iloc[i-1]
        if abs(h_diff) >= vertical_threshold:
            ele_diff_clean[i] = h_diff
        else:
            ele_diff_clean[i] = 0.0
        ele_filtered[i] = ele_filtered[i-1] + ele_diff_clean[i]
        
    df_grid['ele_filtered'] = ele_filtered
    df_grid['ele_diff_clean'] = ele_diff_clean
    df_grid['slope_aligned'] = np.where(df_grid['dist_diff'] > 0, (df_grid['ele_diff_clean'] / df_grid['dist_diff']) * 100, 0)
    df_grid['slope_class'] = df_grid['slope_aligned'].apply(classify_slope)

    detected_waypoints = []
    for wpt in root.findall('.//{*}wpt'):
        name_node = wpt.find('.//{*}name')
        wpt_name = name_node.text if name_node is not None and name_node.text else "未命名CP"
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
st.sidebar.header("⏱️ 1. 运动配速设定 (分:秒)")
st.sidebar.markdown("<small>直接输入您习惯的配速格式，例如 5:30</small>", unsafe_allow_html=True)

# 需求优化一：将配速数字改为“分:秒”格式输入
raw_input_paces = {
    '极陡坡': st.sidebar.text_input("极陡坡 (>15%)", value="25:00"),
    '陡坡': st.sidebar.text_input("陡坡 (8~15%)", value="15:00"),
    '缓坡': st.sidebar.text_input("缓坡 (4~8%)", value="8:00"),
    '平地': st.sidebar.text_input("平地 (-4~4%)", value="5:30"),
    '缓下坡': st.sidebar.text_input("缓下坡 (-8~-4%)", value="4:30"),
    '陡下坡': st.sidebar.text_input("陡下坡 (-15~-8%)", value="6:00"),
    '极陡下坡': st.sidebar.text_input("极陡下坡 (< -15%)", value="10:00"),
}

paces = {k: parse_pace_str(v, 6.0) for k, v in raw_input_paces.items()}

st.sidebar.markdown("---")
st.sidebar.header("⏰ 2. 赛事时间规则")
# 需求优化三：加入起跑时间选项
start_time = st.sidebar.time_input("设定赛事起跑时间", datetime.time(6, 0))
fatigue_rate = st.sidebar.slider("每跑 10 公里，配速衰减比例 (%)", min_value=0, max_value=20, value=5, step=1) / 100.0

st.sidebar.markdown("---")
st.sidebar.header("📍 3. 备用手动 CP 点")
cp_backup_input = st.sidebar.text_input("备用手动分段公里数（逗号隔开）", value="15, 30, 45")

st.sidebar.markdown("---")
st.sidebar.header("🎨 4. 高级算法调参")
user_visual_window = st.sidebar.slider("🌍 图像坡度趋势平滑窗口 (米)", min_value=200, max_value=5000, value=2000, step=200)
user_segment_size = st.sidebar.slider("📐 基础精细核算步长 (米)", min_value=50, max_value=200, value=100, step=50)
user_vertical_threshold = st.sidebar.slider("垂直噪声过滤门限 (米)", min_value=0.0, max_value=3.0, value=0.0, step=0.1)

# --- 5. 主页面业务流 ---
uploaded_file = st.file_uploader("第一步：上传官方赛道或手表导出的 GPX 文件", type=["gpx"])

if uploaded_file:
    uploaded_file.seek(0)
    df, gpx_wpts = process_gpx_hardcore(uploaded_file, user_segment_size, user_vertical_threshold)
    
    if df is not None:
        total_dist = float(df['dist_diff'].sum() / 1000.0)
        total_ascent = float(df[df['ele_diff_clean'] > 0]['ele_diff_clean'].sum())
        total_descent = float(abs(df[df['ele_diff_clean'] < 0]['ele_diff_clean'].sum()))
        
        df['fatigue_factor'] = 1.0 + (df['cum_dist_km'] // 10) * fatigue_rate
        df['pred_pace'] = df.apply(lambda row: paces.get(row['slope_class'], 6.0) * row['fatigue_factor'], axis=1)
        df['time_spent_min'] = (df['dist_diff'] / 1000.0) * df['pred_pace']
        
        # 建立全局累计时间轴（分钟数）
        df['cum_time_min'] = df['time_spent_min'].cumsum()
        total_time_min = float(df['time_spent_min'].sum())

        # 【自适应地形边界保护平滑处理器】
        window_points = max(1, int(user_visual_window / user_segment_size))
        half_w = window_points // 2
        raw_slopes = df['slope_aligned'].to_numpy()
        raw_eles = df['ele_filtered'].to_numpy()
        smooth_slopes = np.zeros_like(raw_slopes)
        
        n_p = len(df)
        for idx in range(n_p):
            st_idx = max(0, idx - half_w)
            ed_idx = min(n_p, idx + half_w + 1)
            local_eles = raw_eles[st_idx:ed_idx]
            local_max_i = st_idx + np.argmax(local_eles)
            local_min_i = st_idx + np.argmin(local_eles)
            
            barrier_idx = None
            if idx < local_max_i < ed_idx and local_max_i != idx: barrier_idx = local_max_i
            if idx < local_min_i < ed_idx and local_min_i != idx:
                if barrier_idx is None or local_min_i < barrier_idx: barrier_idx = local_min_i
            if st_idx < local_max_i < idx and local_max_i != idx: barrier_idx = local_max_i
            if st_idx < local_min_i < idx and local_min_i != idx:
                if barrier_idx is None or local_min_i > barrier_idx: barrier_idx = local_min_i
            
            if barrier_idx is not None:
                sub_slopes = raw_slopes[st_idx:barrier_idx] if barrier_idx > idx else raw_slopes[barrier_idx:ed_idx]
                smooth_slopes[idx] = np.mean(sub_slopes) if len(sub_slopes) > 0 else raw_slopes[idx]
            else:
                smooth_slopes[idx] = np.mean(raw_slopes[st_idx:ed_idx])
                
        df['slope_display_smooth'] = smooth_slopes
        df['slope_class_display'] = df['slope_display_smooth'].apply(classify_slope)

        # 仪表盘看板
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("📐 赛道总里程", f"{total_dist:.2f} km")
        m_col2.metric("🔺 真实原生总爬升", f"{total_ascent:.0f} m")
        m_col3.metric("🔻 真实原生总下降", f"{total_descent:.0f} m")
        hours, mins = divmod(int(total_time_min), 60)
        m_col4.metric("⏱️ 智能预测总用时", f"{hours}小时 {mins}分钟")

        # CP分段解析
        valid_wpts = []
        if len(gpx_wpts) > 0:
            st.success(f"🎯 成功识别到文件内置的 {len(gpx_wpts)} 个官方CP航点！")
            for w in gpx_wpts:
                if 0.1 < w['km'] < total_dist - 0.1:
                    if not valid_wpts or (w['km'] - valid_wpts[-1]['km']) > 0.1:
                        valid_wpts.append(w)
        else:
            try:
                manual_kms = [float(x.strip()) for x in cp_backup_input.split(",") if x.strip() != ""]
                manual_kms = sorted([x for x in manual_kms if 0.1 < x < total_dist - 0.1])
                for mk in manual_kms:
                    valid_wpts.append({"name": f"CP{len(valid_wpts)+1}({mk}km)", "km": mk})
            except ValueError:
                valid_wpts = []

        break_points = [0.0] + [w['km'] for w in valid_wpts] + [total_dist]
        
        # 构造节点综合名称数组
        node_names = ["起点"] + [w['name'] for w in valid_wpts] + ["终点"]
        seg_labels = [f"{node_names[i]} ➔ {node_names[i+1]}" for i in range(len(break_points)-1)]
        df['cp_seg'] = pd.cut(df['cum_dist_km'], bins=break_points, labels=seg_labels, include_lowest=True)

        # --- 6. 视觉层：大趋势连续地形图 ---
        st.subheader(f"🌋 地形大趋势线 · {user_visual_window}米自适应色块聚合图")
        fig = go.Figure()

        i = 0
        n_points = len(df)
        while i < n_points - 1:
            current_class = df.loc[i, 'slope_class_display']
            start_idx = i
            while i < n_points - 1 and df.loc[i+1, 'slope_class_display'] == current_class:
                i += 1
            end_idx = min(i + 1, n_points - 1)
            
            seg_chunk = df.loc[start_idx:end_idx]
            fig.add_trace(go.Scatter(
                x=seg_chunk['cum_dist_km'], y=seg_chunk['ele_filtered'], mode='lines',
                line=dict(width=0.5, color='rgba(0,0,0,0)'), fill='tozeroy',
                fillcolor=COLOR_MAP.get(current_class, 'rgba(128,128,128,0.5)'),
                name=current_class, legendgroup=current_class, showlegend=False,
                hoverinfo='text', 
                text=[f"里程: {d:.2f}km<br>实际海拔: {e:.0f}m<br>趋向地形: {c}(趋势坡度:{s:.1f}%)" 
                      for d, e, c, s in zip(seg_chunk['cum_dist_km'], seg_chunk['ele_filtered'], seg_chunk['slope_class_display'], seg_chunk['slope_display_smooth'])]
            ))
            i += 1

        for cat in SLOPE_ORDER:
            fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=10, color=COLOR_MAP[cat], symbol='square'), name=cat))

        for bp in break_points[1:-1]:
            fig.add_vline(x=bp, line_width=1.5, line_dash="dash", line_color="rgba(79, 79, 79, 0.6)")
            
        max_y = df['ele_filtered'].max()
        for idx, wp in enumerate(valid_wpts):
            y_pos = max_y * 0.92 if idx % 2 == 0 else max_y * 0.75
            fig.add_annotation(
                x=wp['km'], y=y_pos, text=f" 📍 {wp['name']} ", showarrow=False, textangle=-90, 
                font=dict(color="#111111", size=11, family="Arial Bold"), align="center",
                bordercolor="#4F4F4F", borderwidth=1, borderpad=3, bgcolor="rgba(255, 255, 255, 0.93)"
            )

        fig.update_layout(xaxis_title="距离里程 (km)", yaxis_title="海拔高度 (m)", hovermode="x unified", template="plotly_white", height=400)
        st.plotly_chart(fig, use_container_width=True)
 # --- 7. 数据汇总与精算核心层 ---
        plot_segments = []
        base_start_datetime = datetime.datetime.combine(datetime.date.today(), start_time)

        for idx, seg_name in enumerate(seg_labels):
            seg_df = df[df['cp_seg'] == seg_name]
            if len(seg_df) == 0: continue
            
            seg_dist = seg_df['dist_diff'].sum() / 1000.0
            seg_ascent = seg_df[seg_df['ele_diff_clean'] > 0]['ele_diff_clean'].sum()
            seg_descent = abs(seg_df[seg_df['ele_diff_clean'] < 0]['ele_diff_clean'].sum())
            seg_time = seg_df['time_spent_min'].sum()
            
            # 本区间终点所在的全局累计分钟数
            end_cum_minutes = seg_df['cum_time_min'].iloc[-1]
            start_cum_minutes = seg_df['cum_time_min'].iloc[0] - seg_df['time_spent_min'].iloc[0]
            
            # 计算绝对到达钟点时间
            abs_arrival_dt = base_start_datetime + datetime.timedelta(minutes=end_cum_minutes)
            abs_arrival_str = abs_arrival_dt.strftime("%H:%M")
            
            ratios = {}
            for cat in SLOPE_ORDER:
                cat_dist = seg_df[seg_df['slope_class_display'] == cat]['dist_diff'].sum() / 1000.0
                ratios[cat] = (cat_dist / seg_dist) * 100 if seg_dist > 0 else 0
                
            plot_segments.append({
                'id': idx + 1,
                'name': seg_name, 
                'target_node': node_names[idx+1],
                'node_km': break_points[idx+1],
                'ratios': ratios, 
                'dist': seg_dist,
                'ascent': seg_ascent, 
                'descent': seg_descent, 
                'time': seg_time,
                'abs_arrival': abs_arrival_str,
                'cum_time': end_cum_minutes
            })

        # --- 8. 视觉层：镜像正负地形堆叠柱状图 ---
        st.subheader("📊 赛段路况立体化拆解 · 正负地形大趋势构成图")
        fig_bar = go.Figure()
        x_ticks = [s['name'] for s in plot_segments]

        up_cats_reversed = ['极陡坡', '陡坡', '缓坡']
        for i, cat in enumerate(up_cats_reversed):
            current_up_base = np.array([s['ratios']['平地'] / 2.0 for s in plot_segments])
            for prev_cat in up_cats_reversed[i+1:]:
                current_up_base += np.array([s['ratios'][prev_cat] for s in plot_segments])
            y_vals = np.array([s['ratios'][cat] for s in plot_segments])
            fig_bar.add_trace(go.Bar(
                name=cat, x=x_ticks, y=y_vals, base=current_up_base, marker_color=COLOR_MAP[cat],
                hovertemplate=f"%{{x}}<br>{cat}占比: %{{y:.1f}}%<extra></extra>"
            ))

        flat_vals = np.array([s['ratios']['平地'] for s in plot_segments])
        flat_base = - (flat_vals / 2.0)
        fig_bar.add_trace(go.Bar(
            name='平地', x=x_ticks, y=flat_vals, base=flat_base, marker_color=COLOR_MAP['平地'],
            hovertemplate="%{x}<br>平地占比: %{y:.1f}%<extra></extra>"
        ))

        down_cats = ['缓下坡', '陡下坡', '极陡下坡']
        current_down_base = np.array([-s['ratios']['平地'] / 2.0 for s in plot_segments])
        for cat in down_cats:
            y_vals = np.array([-s['ratios'][cat] for s in plot_segments])
            fig_bar.add_trace(go.Bar(
                name=cat, x=x_ticks, y=y_vals, base=current_down_base, marker_color=COLOR_MAP[cat],
                hovertemplate=f"%{{x}}<br>{cat}占比: %{{customdata:.1f}}%<extra></extra>",
                customdata=np.array([s['ratios'][cat] for s in plot_segments])
            ))
            current_down_base += y_vals

        fig_bar.update_layout(
            barmode='group', yaxis_title="◀ 下坡占比 (%)  │  上坡占比 (%) ▶", template="plotly_white", height=380,
            yaxis=dict(tickmode='linear', tick0=-100, dtick=20, ticktext=[str(abs(x)) for x in range(-100, 101, 20)], tickvals=list(range(-100, 101, 20))),
            margin=dict(t=10, b=10)
        )
        fig_bar.add_hline(y=0, line_width=2, line_color="#222222")
        st.plotly_chart(fig_bar, use_container_width=True)

        # --- 9. 【终极进化】：数据一体化大综合表格（包含地形拆解与绝对时间） ---
        st.markdown("---")
        st.subheader("📋 赛事全功能一体化核心数据战术表")
        st.markdown("本表格融合了**地理区间划分、大趋势数据累加、绝对抵达钟点、以及详细的连续地形占比特征**。")

        full_table_data = []
        for s in plot_segments:
            # 特征提取整合：找出各赛段占比最高的两项地形作为战术特征提示
            sorted_ratios = sorted(s['ratios'].items(), key=lambda x: x[1], reverse=True)
            feature_desc = f"{sorted_ratios[0][0]}({sorted_ratios[0][1]:.0f}%) + {sorted_ratios[1][0]}({sorted_ratios[1][1]:.0f}%)"
            
            full_table_data.append({
                "序号": s['id'],
                "赛段区间": s['name'],
                "区间距离 (km)": round(s['dist'], 2),
                "累计里程 (km)": round(s['node_km'], 2),
                "赛段爬升 (m)": f"+{s['ascent']:.0f}",
                "赛段下降 (m)": f"-{s['descent']:.0f}",
                "预估耗时": format_time_duration(s['time']),
                "累计用时": format_time_duration(s['cum_time']),
                "🎯 到达时间点": f"⏰ {s['abs_arrival']}",
                "主导地形路况特征": feature_desc
            })
            
        df_full_report = pd.DataFrame(full_table_data)
        st.dataframe(df_full_report, use_container_width=True, hide_index=True)
        
        # 提供标准 Excel/CSV 的物理下载链接
        csv_buffer = df_full_report.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="💾 下载该一体化战术表格 (CSV格式)",
            data=csv_buffer,
            file_name=f"Race_Pacing_Card_{datetime.date.today()}.csv",
            mime="text/csv"
        )

        # --- 10. 【新增功能】：便携式防水路书贴纸卡片 (支持打印) ---
        st.markdown("---")
        st.subheader("🖨️ 选手专属便携式手腕路书贴纸 / 打印卡片")
        st.markdown("💡 **使用指南：** 下方卡片针对越野跑实战场景设计（高对比度、大字号）。您可以直接点击下方的 **「打印路书卡片」** 按钮，或者在浏览器直接按 `Ctrl + P` (Mac上为 `Cmd + P`)，页面会自动过滤杂项，将此核心卡片完美排版输出为纸张或PDF，裁剪后可塑封贴在手臂或水袋包上！")

        # 构建高密度卡片 HTML 实体
        card_rows_html = ""
        for s in plot_segments:
            card_rows_html += f"""
            <tr>
                <td style="font-weight:bold; border-bottom:1px solid #333; font-size:15px;">{s['target_node']}</td>
                <td style="border-bottom:1px solid #333;">{s['node_km']:.1f}k</td>
                <td style="color:#b30000; font-weight:bold; border-bottom:1px solid #333;">+{s['ascent']:.0f}m</td>
                <td style="color:#004d99; border-bottom:1px solid #333;">-{s['descent']:.0f}m</td>
                <td style="border-bottom:1px solid #333;">{format_time_duration(s['time'])}</td>
                <td style="background-color:#f2f2f2; font-weight:black; font-size:16px; border-bottom:1px solid #333; text-align:center;">{s['abs_arrival']}</td>
            </tr>
            """

        html_pacing_card = f"""
        <div id="pacing-card-container" class="printable-card" style="
            max-width: 480px; 
            border: 3px solid #111; 
            padding: 15px; 
            background-color: #fff; 
            font-family: 'Arial', sans-serif;
            color: #000;
            box-shadow: 5px 5px 0px #888888;
        ">
            <!-- 打印媒体控制样式 -->
            <style>
                @media print {{
                    body * {{ visibility: hidden; }}
                    #pacing-card-container, #pacing-card-container * {{ visibility: visible; }}
                    #pacing-card-container {{ position: absolute; left: 0; top: 0; width: 100%; border:2px solid #000; box-shadow:none; }}
                }}
            </style>
            
            <div style="text-align: center; border-bottom: 2px solid #111; padding-bottom: 5px; margin-bottom: 10px;">
                <h3 style="margin: 0; font-size: 18px; letter-spacing:1px;">RACE PACING STRAP</h3>
                <span style="font-size: 11px; color: #555;">起跑时间：{start_time.strftime("%H:%M")} | 衰减率：{fatigue_rate*100:.0f}%</span>
            </div>
            
            <table style="width: 100%; border-collapse: collapse; font-size: 13px; text-align: left;">
                <thead>
                    <tr style="background-color: #111; color: #fff; text-align:center;">
                        <th style="padding:4px; text-align:left;">点位</th>
                        <th style="padding:4px;">累计</th>
                        <th style="padding:4px;">爬升</th>
                        <th style="padding:4px;">下降</th>
                        <th style="padding:4px;">单段</th>
                        <th style="padding:4px; background-color:#d9d9d9; color:#000;">🕒 到达</th>
                    </tr>
                </thead>
                <tbody>
                    {card_rows_html}
                </tbody>
            </table>
            
            <div style="margin-top: 10px; font-size: 10px; text-align: center; color: #444; border-top: 1px dashed #666; padding-top: 5px;">
                赛道总长: {total_dist:.2f}km | 累计总爬升: {total_ascent:.0f}m | 终点预估: {hours}小时{mins}分
            </div>
        </div>
        """
        
        # 在 Streamlit 中渲染预览
        st.components.v1.html(html_pacing_card, height=450, scrolling=True)
        
        # 触发浏览器打印流的便捷按钮
        st.markdown(
            '<button onclick="window.print()" style="padding: 10px 20px; background-color: #222; color: #fff; border: none; border-radius: 4px; cursor: pointer; font-weight: bold;">🖨️ 唤醒浏览器面板：直接打印该路书贴纸</button>',
            unsafe_allow_html=True
        )
