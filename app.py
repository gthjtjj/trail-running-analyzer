import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import xml.etree.ElementTree as ET
from math import radians, cos, sin, asin, sqrt
import datetime
import matplotlib.pyplot as plt
import io
import base64

# --- 1. 页面基本配置 ---
st.set_page_config(
    page_title="越野跑赛道智能分析预测器 v12", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

st.title("🏃‍♂️ 跑者硬核路书：越野跑赛道智能分析预测器 (全景+战术融合版)")
st.markdown("进化说明：**完美保留前序版本全局大趋势图**，并在此基础上附加各区间独立微型地形切片图嵌入战术表格与手腕卡片。")
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

def format_time_duration(minutes_float):
    hours = int(minutes_float // 60)
    mins = int(minutes_float % 60)
    return f"{hours}h {mins:02d}m"

# 坡度分类颜色映射
COLOR_MAP = {
    '极陡坡': '#8B0000', '陡坡': '#FF4500', '缓坡': '#FFD700', '平地': '#228B22',
    '缓下坡': '#00CED1', '陡下坡': '#1E94FF', '极陡下坡': '#00008B'
}

# 后台微型地形图生成器 (Sparkline)
def generate_sparkline_base64(seg_df, global_min_ele, global_max_ele):
    if len(seg_df) < 2:
        return ""
    
    fig, ax = plt.subplots(figsize=(3, 0.6), dpi=120)
    x = seg_df['cum_dist_km'].values
    y = seg_df['ele_filtered'].values
    
    ax.plot(x, y, color='#222222', linewidth=1.8)
    ax.fill_between(x, y, global_min_ele, color='#e5e5e5', alpha=0.7)
    
    ax.plot(x[0], y[0], 'go', markersize=3.5) 
    ax.plot(x[-1], y[-1], 'ro', markersize=3.5) 
    
    ax.set_ylim(global_min_ele - 20, global_max_ele + 20)
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return f"data:image/png;base64,{img_base64}"

# --- 3. 算法层：网格化重采样处理器 ---
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
        
    if len(raw_points) == 0: return None, []

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
        local_window = raw_df[(raw_df['cum_dist_m'] >= current_target - segment_size_m/2) & (raw_df['cum_dist_m'] <= current_target + segment_size_m/2)]
        
        ele_value = matched_row['ele']
        if not local_window.empty:
            local_max = local_window['ele'].max()
            if local_max - ele_value > 5.0: ele_value = local_max

        grid_points.append({'cum_dist_m': current_target, 'lat': matched_row['lat'], 'lon': matched_row['lon'], 'ele_raw': ele_value})
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
        ele_diff_clean[i] = h_diff if abs(h_diff) >= vertical_threshold else 0.0
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
        except (TypeError, ValueError): continue
            
        min_dist = float('inf')
        matched_km = 0.0
        for i in range(len(df_grid)):
            d = haversine(wpt_lon, wpt_lat, df_grid.loc[i, 'lon'], df_grid.loc[i, 'lat'])
            if d < min_dist:
                min_dist = d
                matched_km = df_grid.loc[i, 'cum_dist_km']
        if min_dist < 600: detected_waypoints.append({"name": wpt_name, "km": round(matched_km, 2)})
            
    return df_grid, sorted(detected_waypoints, key=lambda x: x['km'])

# --- 4. 侧边栏配置区 ---
st.sidebar.header("⏱️ 1. 运动配速设定 (分:秒)")
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
start_time = st.sidebar.time_input("设定赛事起跑时间", datetime.time(6, 0))
fatigue_rate = st.sidebar.slider("每 10 公里，配速衰减比例 (%)", min_value=0, max_value=20, value=5, step=1) / 100.0

st.sidebar.markdown("---")
st.sidebar.header("📍 3. 备用手动 CP 点")
cp_backup_input = st.sidebar.text_input("备用手动分段公里数（逗号隔开）", value="15, 30, 45")

st.sidebar.markdown("---")
st.sidebar.header("🎨 4. 高级算法调参")
user_visual_window = st.sidebar.slider("🌍 图像坡度趋势平滑窗口 (米)", min_value=200, max_value=5000, value=2000, step=200)
user_segment_size = st.sidebar.slider("📐 基础精细核算步长 (米)", min_value=10, max_value=200, value=50, step=10)
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
        
        # 融入衰减和时钟轴核算
        df['fatigue_factor'] = 1.0 + (df['cum_dist_km'] // 10) * fatigue_rate
        df['pred_pace'] = df.apply(lambda row: paces.get(row['slope_class'], 6.0) * row['fatigue_factor'], axis=1)
        df['time_spent_min'] = (df['dist_diff'] / 1000.0) * df['pred_pace']
        df['cum_time_min'] = df['time_spent_min'].cumsum()
        total_time_min = float(df['time_spent_min'].sum())

        global_min_ele = df['ele_filtered'].min()
        global_max_ele = df['ele_filtered'].max()

        # 核心仪表盘看板
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("📐 赛道总里程", f"{total_dist:.2f} km")
        m_col2.metric("🔺 真实原生总爬升", f"{total_ascent:.0f} m")
        m_col3.metric("🔻 真实原生总下降", f"{total_descent:.0f} m")
        hours, mins = divmod(int(total_time_min), 60)
        m_col4.metric("⏱️ 智能预测总用时", f"{hours}小时 {mins}分钟")

        # 动态解析分段锚点
        valid_wpts = []
        if len(gpx_wpts) > 0:
            for w in gpx_wpts:
                if 0.1 < w['km'] < total_dist - 0.1:
                    if not valid_wpts or (w['km'] - valid_wpts[-1]['km']) > 0.1: valid_wpts.append(w)
        else:
            try:
                manual_kms = [float(x.strip()) for x in cp_backup_input.split(",") if x.strip() != ""]
                manual_kms = sorted([x for x in manual_kms if 0.1 < x < total_dist - 0.1])
                for mk in manual_kms: valid_wpts.append({"name": f"CP{len(valid_wpts)+1}", "km": mk})
            except ValueError: pass

        break_points = [0.0] + [w['km'] for w in valid_wpts] + [total_dist]
        node_names = ["起点"] + [w['name'] for w in valid_wpts] + ["终点"]
        seg_labels = [f"{node_names[i]} ➔ {node_names[i+1]}" for i in range(len(break_points)-1)]
        df['cp_seg'] = pd.cut(df['cum_dist_km'], bins=break_points, labels=seg_labels, include_lowest=True)

        # =========================================================================
        # 🔥【前序版本功能回归】：绘制全局大趋势全景图（Plotly 交互式大图）
        # =========================================================================
        st.markdown("---")
        st.subheader("📊 赛道全局宏观地形大趋势图 (彩色时钟时序全景轴)")
        st.markdown("移动鼠标可实时捕获赛道任意一点的**累计里程、实时海拔、精细坡度分类**及**预测到达的绝对时钟时间**。")
        
        # 为大图计算时间悬停
        base_start_datetime = datetime.datetime.combine(datetime.date.today(), start_time)
        df['hover_time_str'] = df['cum_time_min'].apply(lambda m: (base_start_datetime + datetime.timedelta(minutes=m)).strftime("%H:%M"))
        
        # 坡度平滑窗口计算（用于大图可视化渲染，不破坏底层10米步长的精细数据）
        window_size_points = max(2, int(user_visual_window / user_segment_size))
        df['slope_smoothed'] = df['slope_aligned'].rolling(window=window_size_points, center=True, min_periods=1).mean()
        df['visual_slope_class'] = df['slope_smoothed'].apply(classify_slope)

        fig_global = go.Figure()
 # 分色彩条渲染全局大图
        for s_class, group in df.groupby('visual_slope_class'):
            fig_global.add_trace(go.Scatter(
                x=group['cum_dist_km'],
                y=group['ele_filtered'],
                mode='markers+lines',
                marker=dict(size=2, color=COLOR_MAP.get(s_class, '#777')),
                line=dict(color=COLOR_MAP.get(s_class, '#777'), width=1.5),
                name=s_class,
                customdata=np.stack((group['slope_aligned'], group['hover_time_str']), axis=-1),
                hovertemplate="<b>里程:</b> %{x:.2f} km<br><b>海拔:</b> %{y:.0f} m<br><b>精细坡度:</b> %{customdata[0]:.1f}%<br><b>🕒 预估到达:</b> %{customdata[1]}<extra></extra>"
            ))

        # 覆盖 CP 垂直分割参考线与标记
        for wpt in valid_wpts:
            # 找到CP点附近的实际海拔作为打点垂直高度
            matched_idx = (df['cum_dist_km'] - wpt['km']).abs().idxmin()
            wpt_ele = df.loc[matched_idx, 'ele_filtered']
            wpt_time = df.loc[matched_idx, 'hover_time_str']
            
            # 垂直虚线
            fig_global.add_vline(x=wpt['km'], line_width=1, line_dash="dash", line_color="#444")
            # 地图打点
            fig_global.add_trace(go.Scatter(
                x=[wpt['km']], y=[wpt_ele],
                mode="markers+text",
                marker=dict(color="black", size=8, symbol="diamond"),
                text=[f"📍 {wpt['name']}<br>{wpt['km']}k ({wpt_time})"],
                textposition="top center",
                showlegend=False
            ))

        fig_global.update_layout(
            hovermode="x unified",
            xaxis=dict(title="累计里程 (km)", showgrid=True, zeroline=False),
            yaxis=dict(title="海拔高度 (m)", showgrid=True),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=30, b=40),
            height=420
        )
        st.plotly_chart(fig_global, use_container_width=True)

        # =========================================================================
        # 📋【附加战术处理】：各CP段独立切碎微型图核算与一体化数据大表
        # =========================================================================
        plot_segments = []

        with st.spinner("正在切分赛段地形微趋势，并无缝注入战术表格中..."):
            for idx, seg_name in enumerate(seg_labels):
                seg_df = df[df['cp_seg'] == seg_name]
                if len(seg_df) == 0: continue
                
                seg_dist = seg_df['dist_diff'].sum() / 1000.0
                seg_ascent = seg_df[seg_df['ele_diff_clean'] > 0]['ele_diff_clean'].sum()
                seg_descent = abs(seg_df[seg_df['ele_diff_clean'] < 0]['ele_diff_clean'].sum())
                seg_time = seg_df['time_spent_min'].sum()
                end_cum_minutes = seg_df['cum_time_min'].iloc[-1]
                
                abs_arrival_dt = base_start_datetime + datetime.timedelta(minutes=end_cum_minutes)
                abs_arrival_str = abs_arrival_dt.strftime("%H:%M")
                
                # 独立生成当前CP段的地形微型切片图（高归一化对照）
                sparkline_b64 = generate_sparkline_base64(seg_df, global_min_ele, global_max_ele)
                
                plot_segments.append({
                    'id': idx + 1,
                    'name': seg_name, 
                    'target_node': node_names[idx+1],
                    'node_km': break_points[idx+1],
                    'dist': seg_dist,
                    'ascent': seg_ascent, 
                    'descent': seg_descent, 
                    'time': seg_time,
                    'abs_arrival': abs_arrival_str,
                    'cum_time': end_cum_minutes,
                    'sparkline': sparkline_b64
                })

        st.markdown("---")
        st.subheader("📋 赛事核心战术一体化超级大表 (独立趋势图融合版)")
        st.markdown("上方大趋势图掌控**全局战术策略**，下方拆解切片图监控**各赛段局部地形和时钟节奏**。")

        # 拼装安全的独立 HTML 全功能大表页面
        table_rows_html = ""
        for s in plot_segments:
            table_rows_html += f"""
                <tr>
                    <td><b>{s['id']}</b></td>
                    <td style="text-align:left; font-weight:500;">{s['name']}</td>
                    <td>{s['dist']:.2f} km</td>
                    <td>{s['node_km']:.2f} km</td>
                    <td style="color:#b30000; font-weight:600;">+{s['ascent']:.0f} m</td>
                    <td style="color:#004d99;">-{s['descent']:.0f} m</td>
                    <!-- 附加要求：无缝切碎并嵌入表格的独立区间地形趋势线 -->
                    <td>< img style="display:block; margin:0 auto; max-height:42px; width:auto;" src="{s['sparkline']}" /></td>
                    <td>{format_time_duration(s['time'])}</td>
                    <td>{format_time_duration(s['cum_time'])}</td>
                    <td style="background-color:#fff3cd; font-weight:bold; color:#856404;">⏰ {s['abs_arrival']}</td>
                </tr>
            """

        complete_table_page = f"""
        <html>
        <head>
            <style>
                .t-table {{ width: 100%; border-collapse: collapse; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; font-size: 14px; color:#333; }}
                .t-table th {{ background-color: #1a1a1a; color: white; padding: 12px 8px; text-align: center; font-weight: 600; }}
                .t-table td {{ padding: 10px 8px; border-bottom: 1px solid #eeeeee; text-align: center; vertical-align: middle; }}
                .t-table tr:hover {{ background-color: #f9f9f9; }}
            </style>
        </head>
        <body style="margin:0; padding:0; background:transparent;">
            <table class="t-table">
                <thead>
                    <tr>
                        <th>序号</th>
                        <th>赛段区间</th>
                        <th>区间距离</th>
                        <th>累计里程</th>
                        <th>赛段爬升</th>
                        <th>赛段下降</th>
                        <th>赛段趋势地形切片</th>
                        <th>预估耗时</th>
                        <th>累计用时</th>
                        <th>🎯 到达时间点</th>
                    </tr>
                </thead>
                <tbody>
                    {table_rows_html}
                </tbody>
            </table>
        </body>
        </html>
        """
        
        calculated_table_height = 80 + len(plot_segments) * 62
        st.components.v1.html(complete_table_page, height=max(calculated_table_height, 250), scrolling=False)

        # 提供基础文本 CSV 下载
        raw_download_rows = []
        for s in plot_segments:
            raw_download_rows.append({
                "序号": s['id'], "赛段区间": s['name'], "区间距离(km)": round(s['dist'], 2),
                "累计里程(km)": round(s['node_km'], 2), "赛段爬升(m)": int(s['ascent']), "赛段下降(m)": int(s['descent']),
                "预估耗时": format_time_duration(s['time']), "累计用时": format_time_duration(s['cum_time']), "到达具体时间点": s['abs_arrival']
            })
        csv_buffer = pd.DataFrame(raw_download_rows).to_csv(index=False).encode('utf-8-sig')
        st.download_button(label="💾 下载该一体化战术表格数据 (CSV纯文本格式)", data=csv_buffer, file_name="Race_Tactical_Data.csv", mime="text/csv")

        # --- 7. 便携式手腕路书贴纸卡片 (附加项：保留打印切片) ---
        st.markdown("---")
        st.subheader("🖨️ 选手专属便携式手腕路书贴纸 / 打印卡片")
        
        card_rows_html = ""
        for s in plot_segments:
            card_rows_html += f"""
            <tr>
                <td style="font-weight:bold; border-bottom:1px solid #222; font-size:15px; padding:6px 2px; text-align:left;">{s['target_node']}</td>
                <td style="border-bottom:1px solid #222;">{s['node_km']:.1f}k</td>
                <td style="color:#b30000; font-weight:bold; border-bottom:1px solid #222;">+{s['ascent']:.0f}</td>
                <td style="border-bottom:1px solid #222; padding:2px;">< img style="height:25px; width:auto; display:block; margin:0 auto;" src="{s['sparkline']}" /></td>
                <td style="background-color:#e6e6e6; font-weight:900; font-size:16px; border-bottom:1px solid #222; text-align:center;">{s['abs_arrival']}</td>
            </tr>
            """

        html_pacing_card_page = f"""
        <html>
        <head>
            <style>
                @media print {{
                    body * {{ visibility: hidden; }}
                    #card-root, #card-root * {{ visibility: visible; }}
                    #card-root {{ position: absolute; left: 0; top: 0; width: 100%; border:2px solid #000; box-shadow:none; }}
                }}
            </style>
        </head>
        <body style="margin:0; padding:10px; background:transparent;">
            <div id="card-root" style="max-width: 360px; border: 3px solid #111; padding: 12px; background-color: #fff; color: #000; font-family: sans-serif; box-shadow: 4px 4px 0px #888;">
                <div style="text-align: center; border-bottom: 2px solid #111; padding-bottom: 5px; margin-bottom: 8px;">
                    <h3 style="margin: 0; font-size: 16px; letter-spacing:1px;">🏃‍♂️ 越野赛时钟轴硬核路书</h3>
                    <span style="font-size: 11px; color: #555;">起跑时间：{start_time.strftime("%H:%M")} | 步长：{user_segment_size}米</span>
                </div>
                <table style="width: 100%; border-collapse: collapse; font-size: 13px; text-align: center;">
                    <thead>
                        <tr style="background-color: #111; color: #fff;">
                            <th style="padding:5px; text-align:left;">点位</th>
                            <th style="padding:5px;">里程</th>
                            <th style="padding:5px; color:#ff6666;">爬升</th>
                            <th style="padding:5px;">赛段地形</th>
                            <th style="padding:5px; background-color:#cccccc; color:#000;">🕒 到达</th>
                        </tr>
                    </thead>
                    <tbody>{card_rows_html}</tbody>
                </table>
                <div style="margin-top: 8px; font-size: 10px; text-align: center; color: #444; border-top: 1px dashed #666; padding-top: 5px;">
                    里程: {total_dist:.2f}km | 总用时: {hours}h {mins}m | 祝完赛！
                </div>
            </div>
            
            <div style="margin-top:15px;" class="no-print">
                <button onclick="window.print()" style="padding: 10px 18px; background-color: #222; color: #fff; border: none; border-radius: 4px; cursor: pointer; font-weight: bold; font-size:13px;">🖨️ 启动打印机调配面板：点击直接打印</button>
            </div>
        </body>
        </html>
        """
        
        calculated_card_height = 140 + len(plot_segments) * 45
        st.components.v1.html(html_pacing_card_page, height=max(calculated_card_height, 300), scrolling=False)