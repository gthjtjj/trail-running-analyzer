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