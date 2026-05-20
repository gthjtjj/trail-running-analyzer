import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import xml.etree.ElementTree as ET
from math import radians, cos, sin, asin, sqrt

# --- 1. 页面基本配置 ---
st.set_page_config(
   page_title="越野跑赛道智能分析预测器 v7",
   layout="wide",
   initial_sidebar_state="expanded"
)

st.title("🏃‍♂️ 跑者硬核路书：越野跑赛道智能分析预测器 (正负地形镜像版)")
st.markdown("终极打磨：下方的赛段拆解图已重构为**「正负对称地形柱状图」**。上坡在轴上，下坡在轴下，平路跨轴对称。")
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
   '极陡坡': 'rgba(139, 0, 0, 0.75)',    # 深红
   '陡坡': 'rgba(255, 69, 0, 0.75)',     # 橙红
   '缓坡': 'rgba(255, 215, 0, 0.75)',     # 金黄
   '平地': 'rgba(34, 139, 34, 0.75)',     # 森林绿
   '缓下坡': 'rgba(0, 206, 209, 0.75)',   # 闪绿
   '陡下坡': 'rgba(30, 144, 255, 0.75)',   # 道奇蓝
   '极陡下坡': 'rgba(0, 0, 139, 0.75)'     # 深蓝
}

# --- 3. 算法层：百米重采样与山峰保真处理器 ---
@st.cache_data
def process_gpx_hardcore(file, segment_size_m, vertical_threshold):
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
st.sidebar.header("🎨 4. 图像渲染宏观化专属调参")
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
       total_time_min = float(df['time_spent_min'].sum())

       # 宏观平滑窗口计算
       window_points = max(1, int(user_visual_window / user_segment_size))
       df['slope_display_smooth'] = df['slope_aligned'].rolling(window=window_points, center=True, min_periods=1).mean()
       df['slope_class_display'] = df['slope_display_smooth'].apply(classify_slope)

       # 仪表盘看板
       m_col1, m_col2, m_col3, m_col4 = st.columns(4)
       m_col1.metric("📐 赛道总里程", f"{total_dist:.2f} km")
       m_col2.metric("🔺 真实原生总爬升", f"{total_ascent:.0f} m")
       m_col3.metric("🔻 真实原生总下降", f"{total_descent:.0f} m")
       hours, mins = divmod(int(total_time_min), 60)
       m_col4.metric("⏱️ 智能预测总用时", f"{hours}小时 {mins} minutes")

       # CP分段解析
       valid_wpts = []
       if len(gpx_wpts) > 0:
           st.success(f"🎯 成功识别到文件内置的 {len(gpx_wpts)} 个官方CP航点！")
           st.markdown(" | ".join([f"📍 **{w['name']}** ({w['km']:.1f}km)" for w in gpx_wpts]))
           for w in gpx_wpts:
               if 0.1 < w['km'] < total_dist - 0.1:
                   if not valid_wpts or (w['km'] - valid_wpts[-1]['km']) > 0.1:
                       valid_wpts.append(w)
       else:
           try:
               manual_kms = [float(x.strip()) for x in cp_backup_input.split(",") if x.strip() != ""]
               manual_kms = sorted([x for x in manual_kms if 0.1 < x < total_dist - 0.1])
               for mk in manual_kms:
                   valid_wpts.append({"name": f"手动CP({mk}km)", "km": mk})
           except ValueError:
               valid_wpts = []

       break_points = [0.0] + [w['km'] for w in valid_wpts] + [total_dist]
       seg_labels = []
       for i in range(len(break_points)-1):
           start_name = "起点" if i == 0 else valid_wpts[i-1]['name']
           end_name = "终点" if i == len(valid_wpts) else valid_wpts[i]['name']
           seg_labels.append(f"{start_name} ➔ {end_name}")

       df['cp_seg'] = pd.cut(df['cum_dist_km'], bins=break_points, labels=seg_labels, include_lowest=True)
# --- 6. 视觉层：大趋势连续地形图 ---
       st.subheader(f"🌋 地形大趋势线 · {user_visual_window}米空间色块聚合图")
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
           fig.add_vline(x=bp, line_width=1.5, line_dash="dash", line_color="#4F4F4F")
           
       for wp in valid_wpts:
           fig.add_annotation(x=wp['km'], y=df['ele_filtered'].max() * 0.95, text=wp['name'], showarrow=False, textangle=-90, font=dict(color="#4F4F4F", size=10))

       fig.update_layout(xaxis_title="距离里程 (km)", yaxis_title="海拔高度 (m)", hovermode="x unified", template="plotly_white", height=450)
       st.plotly_chart(fig, use_container_width=True)

       # --- 7. 数据分析层：重构为镜像正负地形堆叠柱状图 ---
       st.subheader("📊 赛段路况立体化拆解 · 正负地形大趋势构成图")
       
       cp_stats = []
       
       # 构建图表底盘容器
       plot_segments = []
       for seg_name in seg_labels:
           seg_df = df[df['cp_seg'] == seg_name]
           if len(seg_df) == 0: continue
           
           seg_dist = seg_df['dist_diff'].sum() / 1000.0
           
           # 计算无损底层数据
           seg_ascent = seg_df[seg_df['ele_diff_clean'] > 0]['ele_diff_clean'].sum()
           seg_descent = abs(seg_df[seg_df['ele_diff_clean'] < 0]['ele_diff_clean'].sum())
           seg_time = seg_df['time_spent_min'].sum()
           
           # 精算视觉大趋势下的各类地形占比
           ratios = {}
           for cat in SLOPE_ORDER:
               cat_dist = seg_df[seg_df['slope_class_display'] == cat]['dist_diff'].sum() / 1000.0
               ratios[cat] = (cat_dist / seg_dist) * 100 if seg_dist > 0 else 0
               
           plot_segments.append({
               'name': seg_name,
               'ratios': ratios,
               'dist': seg_dist,
               'ascent': seg_ascent,
               'descent': seg_descent,
               'time': seg_time
           })

       # 分层绘图：动态控制基础值（Base）来实现不规则正负堆叠
       fig_bar = go.Figure()
       n_segs = len(plot_segments)
       x_ticks = [s['name'] for s in plot_segments]

       # 核心算法一：上坡流 (从平地上半层开始向上叠加)
       up_cats = ['缓坡', '陡坡', '极陡坡']
       current_up_base = np.array([s['ratios']['平地'] / 2.0 for s in plot_segments])
       
       for cat in up_cats:
           y_vals = np.array([s['ratios'][cat] for s in plot_segments])
           fig_bar.add_trace(go.Bar(
               name=cat, x=x_ticks, y=y_vals,
               base=current_up_base,
               marker_color=COLOR_MAP[cat],
               hovertemplate=f"%{{x}}<br>{cat}占比: %{{y:.1f}}%<extra></extra>"
           ))
           current_up_base += y_vals

       # 核心算法二：平地对称流 (完美跨越0轴线)
       flat_vals = np.array([s['ratios']['平地'] for s in plot_segments])
       flat_base = - (flat_vals / 2.0) # 往负半轴沉下一半
       fig_bar.add_trace(go.Bar(
           name='平地', x=x_ticks, y=flat_vals,
           base=flat_base,
           marker_color=COLOR_MAP['平地'],
           hovertemplate="%{x}<br>平地占比: %{y:.1f}%<extra></extra>"
       ))

       # 核心算法三：下坡流 (从平地下半层开始向下蔓延)
       down_cats = ['缓下坡', '陡下坡', '极陡下坡']
       current_down_base = np.array([-s['ratios']['平地'] / 2.0 for s in plot_segments])
       
       for cat in down_cats:
           # 物理上往下长，所以 y_vals 赋值为负数
           y_vals = np.array([-s['ratios'][cat] for s in plot_segments])
           fig_bar.add_trace(go.Bar(
               name=cat, x=x_ticks, y=y_vals,
               base=current_down_base,
               marker_color=COLOR_MAP[cat],
               hovertemplate=f"%{{x}}<br>{cat}占比: %{{customdata:.1f}}%<extra></extra>",
               customdata=np.array([s['ratios'][cat] for s in plot_segments]) # 悬浮提示显示正数占比
           ))
           current_down_base += y_vals

       fig_bar.update_layout(
           barmode='group', # 通过显式声明 base 自主控制堆叠，避开系统预设
           yaxis_title="◀ 下坡占比 (%)  │  上坡占比 (%) ▶",
           xaxis_title="赛道切分区间",
           template="plotly_white",
           height=400,
           showlegend=True,
           # 将Y轴刻度线标签变成易读的绝对值格式
           yaxis=dict(
               tickmode='linear', tick0=-100, dtick=20,
               ticktext=[str(abs(x)) for x in range(-100, 101, 20)],
               tickvals=list(range(-100, 101, 20))
           ),
           margin=dict(t=10, b=10)
       )
       # 补一条清晰的 0 轴地平线
       fig_bar.add_hline(y=0, line_width=2, line_color="#222222")
       st.plotly_chart(fig_bar, use_container_width=True)

       # 综合文字数据表
       st.subheader("📋 赛事分段精准战术表")
       cum_time_min = 0.0
       for s in plot_segments:
           cum_time_min += s['time']
           s_h, s_m = divmod(int(s['time']), 60)
           c_h, c_m = divmod(int(cum_time_min), 60)
           cp_stats.append({
               "赛段区间": s['name'],
               "里程 (km)": f"{s['dist']:.2f}",
               "精算爬升 (m)": f"+{s['ascent']:.0f}",
               "精算下降 (m)": f"-{s['descent']:.0f}",
               "分段预估耗时": f"{s_h}小时 {s_m}分钟",
               "总累计比赛时间": f"⏱ *{c_h:02d}:{c_m:02d}*"
           })
       st.dataframe(pd.DataFrame(cp_stats), use_container_width=True, hide_index=True)