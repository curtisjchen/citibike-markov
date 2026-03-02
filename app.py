import streamlit as st
import folium
from streamlit_folium import st_folium
import numpy as np
import json
import pandas as pd
import plotly.graph_objects as go

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Citibike Explorer", page_icon="🚲", layout="wide")

# ── styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    .main { background-color: #0d0f14; }
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
    h1, h2, h3, h4 { font-family: 'IBM Plex Mono', monospace !important; color: #ffffff !important; letter-spacing: 1px; }
    .metric-box { background: #161920; border-radius: 6px; padding: 14px 18px; border: 1px solid #2a2d3a; border-left: 3px solid #00d4aa; }
    .metric-label { color: #6b7280; font-size: 10px; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 4px; font-family: 'IBM Plex Mono', monospace; }
    .metric-value { color: #ffffff; font-size: 18px; font-weight: 600; font-family: 'IBM Plex Mono', monospace; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .section-header { font-family: 'IBM Plex Mono', monospace; font-size: 11px; text-transform: uppercase; letter-spacing: 3px; color: #6b7280; margin-bottom: 10px; padding-bottom: 6px; border-bottom: 1px solid #2a2d3a; }
    .hint { font-size: 11px; color: #6b7280; font-family: 'IBM Plex Mono', monospace; margin-top: 6px; }
    .selected-banner { background: #0d1f1a; border: 1px solid #00d4aa33; border-left: 3px solid #00d4aa; border-radius: 6px; padding: 10px 16px; margin-bottom: 10px; }
    .selected-name { color: #00d4aa; font-family: 'IBM Plex Mono', monospace; font-size: 14px; font-weight: 600; }
    .selected-sub { color: #6b7280; font-size: 11px; font-family: 'IBM Plex Mono', monospace; margin-top: 2px; }
    div[data-baseweb="slider"] > div > div { background: #2a2d3a !important; }
    div[data-baseweb="slider"] > div > div > div { background: #00d4aa !important; }
</style>
""", unsafe_allow_html=True)

# ── load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    pi_matrix         = np.load('outputs/pi_by_hour.npy')
    pi_day_hour        = np.load('outputs/pi_by_day_hour.npy')
    flow_ratio         = np.load('outputs/flow_ratio.npy')
    flow_ratio_by_day_hour = np.load('outputs/flow_ratio_by_day_hour.npy')
    with open('outputs/stations.json', 'r') as f:
        station_info = json.load(f)
    station_info = {int(k): v for k, v in station_info.items()}
    return pi_matrix, pi_day_hour, flow_ratio, flow_ratio_by_day_hour, station_info

pi_matrix, pi_day_hour, flow_ratio, flow_ratio_by_day_hour, station_info = load_data()
n_stations   = pi_matrix.shape[1]
uniform      = 1 / n_stations
name_to_idx  = {info['name']: idx for idx, info in station_info.items() if info.get('name')}
sorted_names = sorted(name_to_idx.keys())

# ── cache pre-computed station data per hour (no lambdas, pickle-safe) ──────────
@st.cache_data
def compute_station_data(day, hour):
    _pi  = pi_day_hour[day, hour]
    _min = _pi.min()
    _max = _pi.max()
    rows = []
    for idx, info in station_info.items():
        if info['lat'] is None or info['lng'] is None:
            continue
        pi_i = float(_pi[idx])
        norm = (pi_i - _min) / (_max - _min + 1e-10)
        rows.append({
            'lat':    info['lat'],
            'lng':    info['lng'],
            'name':   info['name'],
            'pi':     pi_i,
            'color':  f'#{int(255*(1-norm)):02x}{int(200*norm):02x}{int(80*norm):02x}',
            'radius': 4 + norm * 9,
        })
    return rows

# ── build folium map from cached data ────────────────────────────────────────
def build_map(day, hour, selected_name=None):
    rows = compute_station_data(day, hour)
    m = folium.Map(location=[40.738, -73.99], zoom_start=12, tiles='CartoDB dark_matter')
    selected_row = None
    for row in rows:
        is_sel = row['name'] == selected_name
        if is_sel:
            selected_row = row
        folium.CircleMarker(
            location     = [row['lat'], row['lng']],
            radius       = row['radius'],
            color        = row['color'],
            fill         = True,
            fill_color   = row['color'],
            fill_opacity = 0.85,
            weight       = 0,
            tooltip      = row['name'],
            popup        = folium.Popup(
                f"<b style='font-family:monospace'>{row['name']}</b><br>"
                f"π = {row['pi']:.5f}<br>{row['pi']/uniform:.1f}× average",
                max_width=200
            )
        ).add_to(m)
    # draw selected station on top as a bright white ring
    if selected_row:
        folium.CircleMarker(
            location     = [selected_row['lat'], selected_row['lng']],
            radius       = selected_row['radius'] + 5,
            color        = '#ffffff',
            fill         = True,
            fill_color   = selected_row['color'],
            fill_opacity = 1.0,
            weight       = 3,
            tooltip      = selected_row['name'],
        ).add_to(m)
    return m

# ── warm cache for all 24 hours on first load ─────────────────────────────────
if 'cache_warmed' not in st.session_state:
    with st.spinner("Loading map data for all hours and days..."):
        for d in range(7):
            for h in range(24):
                compute_station_data(d, h)
    st.session_state.cache_warmed = True

# ── session state ─────────────────────────────────────────────────────────────
if 'selected_idx' not in st.session_state:
    st.session_state.selected_idx  = None
if 'selected_name' not in st.session_state:
    st.session_state.selected_name = None

# ── title ─────────────────────────────────────────────────────────────────────
st.markdown("# 🚲 CITIBIKE EXPLORER")

# ── controls ──────────────────────────────────────────────────────────────────
DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

ctrl_day, ctrl_left, ctrl_mid = st.columns([2, 4, 1])
with ctrl_day:
    day = st.select_slider("Day", options=list(range(7)),
                           format_func=lambda d: DAYS[d],
                           value=0, label_visibility="collapsed")
with ctrl_left:
    hour = st.slider("Hour of day", 0, 23, 8, label_visibility="collapsed")
with ctrl_mid:
    period = "AM" if hour < 12 else "PM"
    dh = hour % 12 or 12
    st.markdown(f"""
    <div style="padding-top:6px; text-align:center; line-height:1.2">
        <span style="font-family:'IBM Plex Mono',monospace; font-size:11px; color:#6b7280; letter-spacing:2px; text-transform:uppercase; display:block">{DAYS[day][:3]}</span>
        <span style="font-family:'IBM Plex Mono',monospace; font-size:26px; color:#00d4aa; font-weight:600">{dh}</span><span style="font-family:'IBM Plex Mono',monospace; font-size:13px; color:#6b7280; margin-left:2px">{period}</span>
    </div>""", unsafe_allow_html=True)
top_n = 30

# ── pi and flow score for this hour + day ────────────────────────────────────
pi         = pi_day_hour[day, hour]
flow_score = (flow_ratio_by_day_hour[day, hour] - 1) * pi * n_stations

top_idx    = int(np.argsort(pi)[-1])
bot_idx    = int(np.argsort(pi)[0])
accum_idx  = int(np.argsort(flow_score)[-1])   # highest urgency accumulator right now
drain_idx  = int(np.argsort(flow_score)[0])     # highest urgency drainer right now

# ── metrics ───────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
for col, label, value in [
    (c1, "Most accumulating now",  station_info.get(accum_idx, {}).get('name', '—')),
    (c2, "Most draining now",      station_info.get(drain_idx, {}).get('name', '—')),
    (c3, "Highest activity",       station_info.get(top_idx,   {}).get('name', '—')),
    (c4, "Selected station",       st.session_state.selected_name or "Click map or search →"),
]:
    with col:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)

# ── map ───────────────────────────────────────────────────────────────────────
map_col, dive_col = st.columns(2)

with map_col:
    st.markdown('<div class="section-header">Station map — π_i at selected hour</div>', unsafe_allow_html=True)

    with st.spinner("Loading map..."):
        result = st_folium(build_map(day, hour, selected_name=st.session_state.selected_name), width=None, height=650,
                           returned_objects=["last_object_clicked_tooltip"])

    clicked = result.get("last_object_clicked_tooltip")
    if clicked and clicked in name_to_idx:
        new_idx = name_to_idx[clicked]
        if new_idx != st.session_state.selected_idx:
            st.session_state.selected_idx  = new_idx
            st.session_state.selected_name = clicked
            st.rerun()

    st.markdown('<div class="hint">💡 Click any station or search on the right to see its 24-hour profile</div>', unsafe_allow_html=True)

# ── chart (fragment — only reruns on search change) ───────────────────────────
with dive_col:
    @st.fragment
    def render_chart():
        st.markdown('<div class="section-header">24-hour profile</div>', unsafe_allow_html=True)

        options     = [""] + sorted_names
        current     = st.session_state.selected_name
        default_idx = options.index(current) if current in options else 0

        search = st.selectbox(
            "Search station",
            options=options,
            index=default_idx,
            label_visibility="collapsed",
            placeholder="🔍  Search for a station...",
        )

        if search and search != st.session_state.selected_name:
            st.session_state.selected_idx  = name_to_idx[search]
            st.session_state.selected_name = search
            st.rerun()

        if st.session_state.selected_idx is not None:
            sel_idx    = st.session_state.selected_idx
            sel_name   = st.session_state.selected_name
            sel_pi     = float(pi[sel_idx])
            sel_fr     = float(flow_ratio_by_day_hour[day, hour][sel_idx])
            sel_fs     = float((flow_ratio_by_day_hour[day, hour][sel_idx] - 1) * pi[sel_idx] * n_stations)
            hourly_pi  = [float(pi_day_hour[day, h][sel_idx]) for h in range(24)]
            hourly_fs  = [float((flow_ratio_by_day_hour[day, h][sel_idx] - 1) * pi_day_hour[day, h][sel_idx] * n_stations) for h in range(24)]
            peak_h    = int(np.argmax(hourly_pi))
            peak_dh   = peak_h % 12 or 12
            peak_per  = "AM" if peak_h < 12 else "PM"

            st.markdown(f"""
            <div class="selected-banner">
                <div class="selected-name">{sel_name}</div>
                <div class="selected-sub">π = {sel_pi:.5f} &nbsp;·&nbsp; {sel_pi/uniform:.1f}× average at {DAYS[day]} {dh}{period} &nbsp;·&nbsp; flow ratio = {sel_fr:.2f} &nbsp;·&nbsp; score = {sel_fs:.3f}</div>
            </div>""", unsafe_allow_html=True)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(24)), y=hourly_pi,
                mode='lines+markers',
                line=dict(color='#00d4aa', width=2),
                marker=dict(size=5, color='#00d4aa'),
                fill='tozeroy', fillcolor='rgba(0,212,170,0.08)',
                name='π_i', hovertemplate='<b>%{x}:00</b><br>π = %{y:.5f}<extra></extra>'
            ))
            fig.add_trace(go.Scatter(
                x=list(range(24)), y=hourly_fs,
                mode='lines',
                line=dict(color='#ff6b6b', width=1.5, dash='dot'),
                name='flow score',
                yaxis='y2',
                hovertemplate='<b>%{x}:00</b><br>flow score = %{y:.5f}<extra></extra>'
            ))
            fig.add_vline(x=hour, line_dash="dash", line_color="rgba(255,255,255,0.2)", line_width=1)
            fig.add_trace(go.Scatter(
                x=[hour], y=[hourly_pi[hour]], mode='markers',
                marker=dict(size=10, color='#ffffff'), showlegend=False,
                name='', hovertemplate='<b>%{x}:00</b><br>π = %{y:.5f}<extra></extra>'
            ))
            fig.update_layout(
                xaxis=dict(title="Hour", tickmode='linear', dtick=2,
                           gridcolor='#1e2130', tickfont=dict(family='IBM Plex Mono', size=10),
                           range=[-0.5, 23.5]),
                yaxis=dict(title="π_i", gridcolor='#1e2130',
                           tickfont=dict(family='IBM Plex Mono', size=10)),
                yaxis2=dict(title="flow score", overlaying='y', side='right',
                            tickfont=dict(family='IBM Plex Mono', size=10),
                            showgrid=False, zeroline=False),
                paper_bgcolor='#161920', plot_bgcolor='#161920',
                font=dict(color='#aaaaaa', family='IBM Plex Mono'),
                margin=dict(t=10, b=40, l=60, r=20),
                height=430, showlegend=True,
                shapes=[dict(
                    type='line', xref='x', yref='y2',
                    x0=-0.5, x1=23.5, y0=0, y1=0,
                    line=dict(color='#6b7280', width=1, dash='dot')
                )],
                legend=dict(font=dict(family='IBM Plex Mono', size=10),
                            bgcolor='rgba(22,25,32,0.8)',
                            x=0.01, y=0.99, xanchor='left', yanchor='top')
            )
            st.plotly_chart(fig, width="stretch")

            mc1, mc2, mc3 = st.columns(3)
            for col, label, val in [
                (mc1, "Peak hour",   f"{DAYS[day][:3]} {peak_dh}{peak_per}"),
                (mc2, "Peak π",      f"{max(hourly_pi):.5f}"),
                (mc3, "Peak vs avg", f"{max(hourly_pi)/uniform:.1f}×"),
            ]:
                with col:
                    st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-label">{label}</div>
                        <div class="metric-value">{val}</div>
                    </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="height:460px; display:flex; align-items:center; justify-content:center;
                        background:#161920; border-radius:6px; border:1px dashed #2a2d3a;">
                <div style="text-align:center; color:#6b7280; font-family:'IBM Plex Mono',monospace;">
                    <div style="font-size:40px; margin-bottom:16px">🗺</div>
                    <div style="font-size:11px; letter-spacing:2px; text-transform:uppercase; line-height:2">
                        Click a station on the map<br>or search above
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

    render_chart()

# ── rankings ──────────────────────────────────────────────────────────────────
st.markdown("<div style='margin-top:24px'></div>", unsafe_allow_html=True)
st.markdown('<div class="section-header">Station rankings</div>', unsafe_allow_html=True)

def build_table(indices):
    rows = []
    for rank, idx in enumerate(indices, 1):
        info = station_info.get(int(idx), {})
        fr   = float(flow_ratio_by_day_hour[day, hour][idx])
        fs   = float((flow_ratio_by_day_hour[day, hour][idx] - 1) * pi[idx] * n_stations)
        rows.append({
            "#":            rank,
            "Station":      info.get('name', 'Unknown'),
            "π_i":          f"{pi[idx]:.5f}",
            "vs avg":       f"{pi[idx]/uniform:.1f}×",
            "flow ratio":   round(fr, 2),
            "flow score":   round(fs, 3),
            "direction":    "↑ accumulating" if fr > 1 else "↓ draining",
        })
    return pd.DataFrame(rows)

# sort by hourly flow score so direction matches sort order
hourly_flow_score = (flow_ratio_by_day_hour[day, hour] - 1) * pi * n_stations

tc, bc = st.columns(2)
with tc:
    st.markdown(f"**Top {top_n} — most accumulating now**")
    st.dataframe(build_table(np.argsort(hourly_flow_score)[-top_n:][::-1]),
                 hide_index=True, width="stretch", height=350)
with bc:
    st.markdown(f"**Top {top_n} — most draining now**")
    st.dataframe(build_table(np.argsort(hourly_flow_score)[:top_n]),
                 hide_index=True, width="stretch", height=350)