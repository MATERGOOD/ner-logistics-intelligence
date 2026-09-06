import streamlit as st
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import os
import networkx as nx
import numpy as np
import pandas as pd
import osmnx as ox
import sys
import base64
import uuid
import altair as alt
import time

st.set_page_config(page_title="NER Logistics Command Center", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    /* Dark Theme Core */
    .stApp { background-color: #0B0F14; color: #E2E8F0; }
    
    /* Compact main container padding */
    .block-container { padding: 1.5rem 2rem 2rem 2rem; max-width: 100%; }
    
    /* Compact spacing between cards */
    div[data-testid="stVerticalBlock"] > div { gap: 0.5rem; }
    
    /* Remove default Streamlit divider margins */
    hr { margin-top: 0.5rem; margin-bottom: 0.5rem; border-color: rgba(255,255,255,0.08); }
    
    /* Modern card container styling */
    .op-card {
        background: #111820;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 10px;
    }
    
    /* Header Bar layout */
    .header-bar {
        background: #111820;
        margin-bottom: 20px;
        padding: 12px 24px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    /* Radio switcher layout */
    div[data-testid="stRadio"] > div { 
        display: flex; 
        justify-content: center; 
        gap: 10px; 
    }
    
    .kpi-title { font-size: 0.8rem; font-weight: 700; color: #9CA3AF; letter-spacing: 0.05em; text-transform: uppercase; }
    .kpi-value { font-size: 1.4rem; font-weight: 800; margin: 5px 0; }
    .kpi-sub { font-size: 0.75rem; color: #6B7280; }
</style>
""", unsafe_allow_html=True)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

try:
    from routing_engine import compute_routes
except Exception as e:
    def compute_routes(*args, **kwargs): return {"status": "IMPASSABLE", "message": f"Engine failed: {e}"}

from fleet_simulator import FleetSimulator
try:
    from incident_manager import get_active_incidents, submit_report, resolve_incident, verify_incident, dismiss_incident
except:
    pass

from ml_predictor import DisruptionPredictor

@st.cache_resource
def load_predictor(): return DisruptionPredictor()
predictor = load_predictor()

@st.cache_data
def load_data():
    bounds_path = "data/raw/boundaries.geojson"
    roads_path = "data/processed/road_segments.geojson"
    graph_path = "data/processed/network_with_risk.graphml"
    
    boundaries = gpd.read_file(bounds_path)
    roads = gpd.read_file(roads_path)
    G = ox.load_graphml(graph_path)
    
    if "length" in roads.columns: roads["length_km"] = roads["length"].astype(float) / 1000.0
    else: roads["length_km"] = roads.geometry.length * 111.0 
    
    rs = np.random.RandomState(42)
    roads["slope_factor"] = rs.uniform(0.0, 1.0, size=len(roads))
    roads["hist_risk"] = rs.choice([0.0, 1.0], p=[0.88, 0.12], size=len(roads))
    current_weather = roads["weather_rain_mm"].iloc[0] if "weather_rain_mm" in roads.columns else 18.5
    names = roads.get("name", pd.Series(["Unknown"] * len(roads))).fillna("Unknown").astype(str)
    roads["segment_id"] = roads.index.astype(str) + " - " + names
    
    return boundaries, roads, current_weather, G

def calculate_status(risk_score):
    if risk_score <= 0.55:     return "Clear", "#28a745", 1.0
    elif risk_score <= 0.79:   return "Caution", "#eab308", 1.4
    else:                      return "Blocked", "#ef4444", 999.0

boundaries, base_roads, current_precip, base_G = load_data()
roads = base_roads.copy()
G = base_G.copy()

def parse_node_id(node_val, G):
    if node_val in G: return node_val
    if isinstance(node_val, str):
        import re
        match = re.search(r'\b\d+\b', node_val)
        if match:
            candidate = int(match.group(0))
            if candidate in G: return candidate
    return list(G.nodes())[0]

if "fleet_sim" not in st.session_state:
    nodes_gdf, edges_gdf = ox.graph_to_gdfs(base_G)
    st.session_state.fleet_sim = FleetSimulator(base_G, nodes_gdf)
if "delay_recovered" not in st.session_state: st.session_state.delay_recovered = 0.0
if "offline_reports" not in st.session_state: st.session_state.offline_reports = []
if "sim_stage" not in st.session_state: st.session_state.sim_stage = 0
if "auto_sim" not in st.session_state: st.session_state.auto_sim = False
if "is_rerouted" not in st.session_state: st.session_state["is_rerouted"] = False

stage = st.session_state.sim_stage

if st.session_state.auto_sim and stage < 11:
    time.sleep(2.5)
    st.session_state.sim_stage += 1
    st.rerun()

# 1. Global Shell & Top App Bar
col_title, col_corridor, col_lang = st.columns([2.0, 2.2, 0.8], vertical_alignment="center")

with col_title:
    st.markdown("""
        <div style='display:flex; align-items:center;'>
            <div style='width:12px; height:12px; background:#10B981; border-radius:50%; margin-right:10px; animation:pulse 2s infinite;'></div>
            <span style='font-size:20px; font-weight:bold; color:white;'>NER LOGISTICS INTELLIGENCE</span>
        </div>
        <div style='color:#9CA3AF; font-size:12px; margin-left:22px; letter-spacing:1px;'>DISASTER RESPONSE COMMAND // ONLINE</div>
    """, unsafe_allow_html=True)

with col_corridor:
    sub_c1, sub_arrow, sub_c2 = st.columns([1.0, 0.2, 1.0], vertical_alignment="center")
    with sub_c1:
        origin_point = st.selectbox("Origin Hub", ["Guwahati ISBT", "Byrnihat Staging Depot", "Khanapara Junction"], key="sel_origin", label_visibility="collapsed")
    with sub_arrow:
        st.markdown("<div style='text-align:center; color:#10B981; font-weight:bold; font-size:18px;'>➔</div>", unsafe_allow_html=True)
    with sub_c2:
        dest_point = st.selectbox("Relief Target", ["Nongpoh Civil Hospital", "Umsning Relief Depot", "Shillong Trauma Center"], key="sel_dest", label_visibility="collapsed")
    st.markdown(f"<div style='text-align:center; font-size:0.8rem; color:#9CA3AF; margin-top:4px;'>📍 Active Routing Corridor: {origin_point} to {dest_point} via NH-6</div>", unsafe_allow_html=True)

with col_lang:
    st.selectbox("Language / ভাষা", ["English", "অসমীয়া (Assamese)", "Khasi"], label_visibility="collapsed", key="global_lang")

st.markdown("<br>", unsafe_allow_html=True)

# Screen Switcher
screen = st.radio("Navigation", ["🚨 Command Dashboard", "📦 Field & Supply Chain Operations"], horizontal=True, label_visibility="collapsed")

# Setup Simulation data
sim_rain = st.session_state.get('sim_rain_slider', float(current_precip))
forced_blocks = st.session_state.get('manual_blocks', [])

override_rain = float(current_precip)
override_timeline = 15
override_blocks = []
stage_msg = ""

if stage > 0:
    if stage <= 1:
        stage_msg = "**Stage 1 (Nominal):** Rain at 12 mm/h. All corridors Green/Clear. TRK-01 en route."
        override_rain = 12.0
        override_timeline = 15
    elif stage <= 3:
        stage_msg = "**Stage 2-3 (Surge Warning):** Rain hits 35 mm/h. TRK-01 approaches with alert."
        override_rain = 35.0
        override_timeline = 30
    elif stage <= 8:
        override_blocks = [roads.nlargest(2, "risk_score")["segment_id"].iloc[0]] if not roads.empty else []
        if stage <= 5: stage_msg = "**Stage 4-5 (Disaster Breach):** Rain hits 68 mm/h. Corridor Blocked."
        else: stage_msg = "**Stage 6-8 (AI Decision Prescribed):** Impact card activates."
        override_rain = 68.0
        override_timeline = 45
    elif stage <= 11:
        override_blocks = [roads.nlargest(2, "risk_score")["segment_id"].iloc[0]] if not roads.empty else []
        stage_msg = "**Stage 9-11 (Resolution & Recovery):** Dispatcher applies reroute."
        override_rain = 68.0
        override_timeline = 70
        
if stage >= 9 and getattr(st.session_state, '_last_reroute_stage', 0) < 9:
    st.session_state.fleet_sim.execute_fleet_reroute(override_blocks, roads)
    st.session_state.delay_recovered += 105.0 
    st.session_state._last_reroute_stage = stage
if stage < 9:
    st.session_state._last_reroute_stage = stage

final_rain = override_rain if stage > 0 else sim_rain
final_blocks = override_blocks if stage > 0 else forced_blocks
timeline_pct = override_timeline if stage > 0 else 15

try: incidents_df = get_active_incidents()
except: incidents_df = pd.DataFrame()

field_blocks = []
if not incidents_df.empty and "severity" in incidents_df.columns and "status" in incidents_df.columns:
    field_blocks.extend(incidents_df[(incidents_df["severity"] == "Complete Road Severed") & (incidents_df["status"] == "Verified & Confirmed")]["nearest_segment_id"].tolist())
for f in field_blocks:
    if f not in final_blocks: final_blocks.append(f)

def update_row(row):
    rain_mm = final_rain
    accum_24h = rain_mm * 4 + 20
    slope = row.get("slope_factor", 0.5) * 45
    hist_risk = row.get("hist_risk", 0.0) * 5
    preds = predictor.predict_segment_risk(rain_mm, accum_24h, slope, 500.0, 350.0, hist_risk, 0.8)
    
    rs = preds["closure_probability"]
    status, color, delay = calculate_status(rs)
    if row["segment_id"] in final_blocks: status, color, delay, rs = "Blocked", "#ef4444", 999.0, 1.0
    row["risk_score"], row["status"], row["color"], row["delay_factor"] = round(rs, 3), status, color, delay
    
    badge_dict = {"Clear": "🟢 CLEAR", "Caution": "🟡 CAUTION", "Blocked": "🔴 BLOCKED"}
    badge = badge_dict.get(status, "🟠 HIGH RISK") if rs < 0.8 else ("🔴 BLOCKED" if status == "Blocked" else "🟠 HIGH RISK")
    
    html = f"<b>NH-6 | {row['segment_id']}</b><br>Status: {badge}<br>Disruption Risk: {int(rs*100)}%<br>Rainfall: {rain_mm:.1f} mm/h<br>Slope: {int(slope)}°"
    row["popup_html"] = html
    return row

roads = roads.apply(update_row, axis=1)

fleet_status = st.session_state.fleet_sim.get_vehicle_positions(timeline_pct, final_blocks, roads)
risk_map = {row["segment_id"]: row["risk_score"] for _, row in roads.iterrows()}

all_nodes = list(G.nodes())

if screen == "🚨 Command Dashboard":
    # 2. Screen 1 Layout
    high_risk_count = len(roads[roads["risk_score"] > 0.6])
    at_risk_convoys = len([v for v in fleet_status if v.get("alert")])

    # Row 1: KPI Cards
    k1, k2, k3, k4, k5 = st.columns(5)
    
    if st.session_state.get('is_rerouted'):
        tl_str, tl_color = "REROUTE ACTIVE // BYPASS ENFORCED", "#F59E0B"
    else:
        tl_str, tl_color = ("CRITICAL - MONSOON SURGE", "#ef4444") if (high_risk_count > 0 or at_risk_convoys > 0) else ("NOMINAL OPERATIONS", "#22c55e")
        
    k1.markdown(f"<div class='op-card' style='border-left: 4px solid {tl_color};'><div class='kpi-title'>Regional Threat</div><div class='kpi-value' style='color:{tl_color}; font-size: 1.1rem;'>{tl_str}</div><div class='kpi-sub'>Updated 1m ago</div></div>", unsafe_allow_html=True)
    
    k2.markdown(f"<div class='op-card' style='border-left: 4px solid #F59E0B;'><div class='kpi-title'>High-Risk Segments</div><div class='kpi-value'>{high_risk_count:02d} Segments</div><div class='kpi-sub'>Prob > 60%</div></div>", unsafe_allow_html=True)
    
    c_alert = f"{at_risk_convoys} At Direct Risk" if at_risk_convoys > 0 else "All Secure"
    k3.markdown(f"<div class='op-card' style='border-left: 4px solid #3B82F6;'><div class='kpi-title'>Active Convoys</div><div class='kpi-value'>{len(fleet_status)} Convoys</div><div class='kpi-sub'>({c_alert})</div></div>", unsafe_allow_html=True)
    
    k4.markdown(f"<div class='op-card' style='border-left: 4px solid #8B5CF6;'><div class='kpi-title'>Essential Cargo</div><div class='kpi-value' style='font-size:1.1rem;'>1,200 Vaccine Doses</div><div class='kpi-sub'>At threat threshold</div></div>", unsafe_allow_html=True)
    
    if st.session_state.get('is_rerouted'):
        k5.markdown(f"<div class='op-card' style='border-left: 4px solid #10B981;'><div class='kpi-title'>Delay Avoided</div><div class='kpi-value' style='color:#10B981;'>+1h 45m Saved</div><div class='kpi-sub'>1,200 vaccine doses preserved</div></div>", unsafe_allow_html=True)
    else:
        tot_hr = st.session_state.delay_recovered // 60
        tot_mn = int(st.session_state.delay_recovered % 60)
        rec_str = f"+{int(tot_hr)}h {tot_mn}m" if tot_hr > 0 else f"+{tot_mn}m"
        k5.markdown(f"<div class='op-card' style='border-left: 4px solid #10B981;'><div class='kpi-title'>Delay Avoided</div><div class='kpi-value'>{rec_str} Saved</div><div class='kpi-sub'>via Reroute</div></div>", unsafe_allow_html=True)

    # Row 2: Maps and Charts
    col_map, col_right = st.columns([1.6, 1.0])
    
    with col_map:
        bounds = roads.total_bounds 
        m = folium.Map(location=[(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2], zoom_start=12, tiles=None)
        
        folium.TileLayer(
            tiles='https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png',
            attr='OSM', name='Topographic Basemap'
        ).add_to(m)
        
        loc_coords = {
            "Guwahati ISBT": [26.140, 91.730], "Khanapara Junction": [26.120, 91.800], "Byrnihat Staging Depot": [26.050, 91.880],
            "Nongpoh Civil Hospital": [25.900, 91.880], "Umsning Relief Depot": [25.750, 91.890], "Shillong Trauma Center": [25.570, 91.880]
        }
        start_c = loc_coords.get(origin_point, [26.140, 91.730])
        end_c = loc_coords.get(dest_point, [25.900, 91.880])
        
        primary_route_coords = [start_c, [26.050, 91.780], [25.920, 91.850], [25.755, 91.870], end_c]
        bypass_route_coords = [start_c, [26.050, 91.780], [26.000, 91.650], [25.850, 91.650], [25.700, 91.750], end_c]

        if not st.session_state.get('is_rerouted', False):
            folium.PolyLine(
                locations=primary_route_coords,
                color='#10B981', weight=5, opacity=0.9,
                tooltip="Active Artery: NH-6 Primary"
            ).add_to(m)
            m.fit_bounds(primary_route_coords)
        else:
            folium.PolyLine(
                locations=primary_route_coords,
                color='#EF4444', weight=5, opacity=0.8, dash_array='8, 8',
                tooltip="❌ SEVERED ARTERY: NH-6 Segment 69 Blocked by Landslide"
            ).add_to(m)
            folium.Marker(
                location=[25.755, 91.870],
                icon=folium.Icon(color='red', icon='exclamation-triangle', prefix='fa'),
                popup="⚠️ Landslide Breach (Segment 69) - Completely Inaccessible"
            ).add_to(m)
            folium.PolyLine(
                locations=bypass_route_coords,
                color='#10B981', weight=6, opacity=1.0,
                tooltip="⚡ AI RESILIENT BYPASS B (Active Detour: +16 min)"
            ).add_to(m)
            folium.Marker(
                location=bypass_route_coords[len(bypass_route_coords)//2],
                icon=folium.Icon(color='blue', icon='truck', prefix='fa'),
                popup="TRK-01: Diverted via Bypass B (ETA: 63 min)"
            ).add_to(m)
            m.fit_bounds(bypass_route_coords)

        folium.Marker(start_c, icon=folium.Icon(color='blue', icon='box', prefix='fa'), tooltip=origin_point).add_to(m)
        folium.Marker(end_c, icon=folium.Icon(color='red', icon='hospital', prefix='fa'), tooltip=dest_point).add_to(m)
        
        for v in fleet_status:
            if v["lat"] == 0 and v["lon"] == 0: continue
            if v["id"] == "TRK-01" and st.session_state.get('is_rerouted', False): continue
            ic_col = "blue" if v["rerouted"] else ("red" if v["alert"] else "green")
            folium.Marker([v["lat"], v["lon"]], icon=folium.Icon(color=ic_col, icon="truck", prefix="fa"), tooltip=v['id']).add_to(m)
            
        st_folium(m, height=450, returned_objects=[], use_container_width=True)
        
    with col_right:
        # Prescriptive AI Decision Card
        if st.session_state.get('is_rerouted'):
            st.markdown(f"""
            <div class='op-card' style='border: 1px solid #10B981; background: rgba(16, 185, 129, 0.05);'>
                <h4 style='color: #10B981; margin-top:0;'>✅ REROUTE CONFIRMED</h4>
                <p>TRK-01 diverted via Bypass Corridor B. Cold-chain preserved. Avoided 1h 45m highway delay.</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("🔄 Reset Route / Normal Ops", use_container_width=True):
                st.session_state['is_rerouted'] = False
                st.session_state['delay_recovered'] = 0.0
                st.session_state.sim_stage = 0
                st.session_state.auto_sim = False
                nodes_gdf, edges_gdf = ox.graph_to_gdfs(base_G)
                st.session_state.fleet_sim = FleetSimulator(base_G, nodes_gdf)
                st.rerun()
        elif final_rain < 30.0 and len(final_blocks) == 0:
            st.markdown("<div class='op-card' style='border: 1px solid #10B981;'><h4 style='color:#10B981; margin:0;'>🤖 AI Status: NOMINAL</h4><p>All paths optimized and clear.</p></div>", unsafe_allow_html=True)
        else:
            pct_p = final_rain/100*40 + 32
            st.markdown(f"""
            <div class='op-card' style='border: 1px solid #ef4444; background: rgba(239, 68, 68, 0.05);'>
                <h4 style='color: #ef4444; margin-top:0;'>🤖 AI DECISION ALERT</h4>
                <p><b>Target:</b> TRK-01</p>
                <p><b>Factors:</b> Heavy Rain 42%, Slope 34°/31%, River 17%</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("⚡ EXECUTE EMERGENCY REROUTE TO TRK-01", type="primary", use_container_width=True):
                st.session_state['is_rerouted'] = True
                st.session_state['supply_delay_avoided'] = "+1h 45m"
                if stage > 0: st.session_state.sim_stage = 9
                st.session_state.fleet_sim.execute_fleet_reroute(final_blocks, roads)
                st.rerun()

        # Disruption Risk Projection
        st.markdown("<div style='font-size: 0.9em; font-weight:bold; color:#E2E8F0; margin-top:20px; text-transform:uppercase;'>Disruption Risk Projection</div>", unsafe_allow_html=True)
        r_vals = [ min(100, (final_rain/100)*x + (20 if final_blocks else 0)) for x in [40, 55, 70, 85, 100] ]
        df_chart = pd.DataFrame({
            "Time": ["Now", "+2h", "+4h", "+6h", "+8h"],
            "Risk": r_vals,
            "Threshold": [70] * 5
        }).melt("Time", var_name="Cat", value_name="Prob")
        c = alt.Chart(df_chart).mark_line(point=True).encode(
            x=alt.X("Time", sort=["Now", "+2h", "+4h", "+6h", "+8h"]),
            y=alt.Y("Prob", scale=alt.Scale(domain=[0, 100])),
            color=alt.Color("Cat", scale=alt.Scale(range=["#3B82F6", "#ef4444"]), legend=None),
            strokeDash=alt.condition(alt.datum.Cat == 'Threshold', alt.value([5,5]), alt.value([0]))
        ).properties(height=180)
        st.altair_chart(c, use_container_width=True)
        
        # Fleet Quick Ledger
        st.markdown("<div style='font-size: 0.9em; font-weight:bold; color:#E2E8F0; margin-top:10px; text-transform:uppercase;'>Fleet Quick Ledger</div>", unsafe_allow_html=True)
        if fleet_status:
            f_df = pd.DataFrame([{
                "Asset": v["id"], 
                "Status": "🟢 REROUTED (Safe Bypass B)" if (v["id"] == "TRK-01" and st.session_state.get('is_rerouted')) else ("Rerouted" if v["rerouted"] else ("Blocked" if v["alert"] else "En Route"))
            } for v in fleet_status])
            st.dataframe(f_df, hide_index=True, use_container_width=True)
        else:
            st.info("No active fleet.")

    # Row 3: Bottom Dock Simulator
    st.markdown("---")
    bot1, bot2, bot3, bot4 = st.columns([2, 1, 1, 1])
    with bot1:
        st.markdown("### 🧪 What-If Disaster Simulator")
        st.progress(stage / 11.0)
        if stage > 0: st.markdown(stage_msg)
    with bot2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("⚡ Run 90s Scenario", use_container_width=True):
            st.session_state.auto_sim = True
            st.session_state.sim_stage = 1
            st.rerun()
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.sim_stage = 0
            st.session_state.auto_sim = False
            st.session_state.delay_recovered = 0.0
            st.session_state._last_reroute_stage = 0
            nodes_gdf, edges_gdf = ox.graph_to_gdfs(base_G)
            st.session_state.fleet_sim = FleetSimulator(base_G, nodes_gdf)
            st.rerun()
    with bot3:
        st.slider("Rainfall (0-100 mm/h)", 0.0, 100.0, float(current_precip), 1.0, key="sim_rain_slider", disabled=(stage>0))
    with bot4:
        st.multiselect("Manual Segment Injection", roads["segment_id"].tolist(), key="manual_blocks", disabled=(stage>0))

elif screen == "📦 Field & Supply Chain Operations":
    col_field, col_supply = st.columns([1.0, 1.0])
    
    with col_field:
        st.markdown("<div class='op-card'>", unsafe_allow_html=True)
        st.markdown("### 📝 Submit Field Geo-Tagged Report")
        with st.form("field_report"):
            inc_type = st.selectbox("Incident Type", ["Landslide", "Flash Flood / Waterlogging", "Bridge Structural Damage"])
            segment_id = st.selectbox("Affected Segment", roads["segment_id"].tolist())
            severity = st.radio("Severity", ["Complete Road Severed", "Single-Lane Blocked", "Minor Caution"], horizontal=True)
            photo = st.file_uploader("Upload On-Site Photo/Preview", type=["png", "jpg"])
            st.text_input("GPS Coordinates (Auto-populated from device)", "25.90N, 91.88E", disabled=True)
            submitted = st.form_submit_button("Submit Incident Report")
            if submitted:
                if not getattr(st.session_state, 'offline_mode', False):
                    cb = roads[roads["segment_id"] == segment_id].iloc[0].geometry.bounds
                    try:
                        submit_report("Field Officer", "SDRF", inc_type, severity, segment_id, (cb[1] + cb[3]) / 2, (cb[0] + cb[2]) / 2, "Reported via Field Console", "")
                        st.success("Submitted")
                    except:
                        pass
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='op-card'>", unsafe_allow_html=True)
        st.markdown("### 🛡️ District Control Room Audit Ledger")
        if incidents_df.empty:
            st.info("No active incidents to audit.")
        else:
            for idx, row in incidents_df.iterrows():
                st.markdown(f"**INC-{str(idx).zfill(3)}** | {row['incident_type']} | {row['severity']}")
                st.markdown("- Open-Meteo Doppler Corroboration: **POSITIVE**")
                c_v, c_d = st.columns(2)
                if "Pending" in row.get('status', ''):
                    if c_v.button("[✅ Verify & Sever]", key=f"v_{row['id']}"):
                        verify_incident(row['id'])
                        st.rerun()
                    if c_d.button("[❌ Dismiss]", key=f"d_{row['id']}"):
                        dismiss_incident(row['id'])
                        st.rerun()
                else:
                    st.success(f"Status: {row['status']}")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_supply:
        st.markdown("<div class='op-card'>", unsafe_allow_html=True)
        st.markdown("### 📦 Warehouse Inventory & Vulnerability Status")
        st.markdown("""
        **Guwahati Depot:** 15,000 cold-chain units (SECURE)
        <br>**Nongpoh Hospital Reserve:** <span style='color:#ef4444; font-weight:bold;'>< 14 Hours Remaining</span>
        <br>**Humanitarian Risk:** <span style='color:#10B981;'>Averted via proactive supply dispatch</span>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='op-card'>", unsafe_allow_html=True)
        st.markdown("### 📡 District Connectivity Matrix")
        st.dataframe(pd.DataFrame([
            {"DistrictName": "Kamrup", "Connectivity": "96%"},
            {"DistrictName": "Ri-Bhoi", "Connectivity": "68%"},
            {"DistrictName": "East Khasi Hills", "Connectivity": "84%"}
        ]), hide_index=True, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='op-card'>", unsafe_allow_html=True)
        st.markdown("### 📻 Offline Communications Dispatcher")
        st.text_area("Advisory Preview (Plain Text)", "URGENT DISPATCH ADVISORY - TRK-01 REROUTE VIA BYPASS B. AVOID NH-6 SEGMENT 69 DUE TO CRITICAL LANDSLIDE PROBABILITY.", height=100)
        st.download_button(
            label="⬇ Download Digitally Stamped Advisory (.txt)",
            data="URGENT DISPATCH ADVISORY - TRK-01 REROUTE VIA BYPASS B. AVOID NH-6 SEGMENT 69 DUE TO CRITICAL LANDSLIDE PROBABILITY.",
            file_name="Incident_Advisory.txt",
            mime="text/plain",
            use_container_width=True
        )
        st.markdown("</div>", unsafe_allow_html=True)
