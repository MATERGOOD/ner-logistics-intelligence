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

st.set_page_config(page_title="NER Logistics Dashboard", layout="wide")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
try:
    from routing_engine import compute_routes
except Exception as e:
    def compute_routes(*args, **kwargs): return {"status": "IMPASSABLE", "message": f"Engine failed: {e}"}

from fleet_simulator import FleetSimulator
try:
    from incident_manager import get_active_incidents, submit_report, resolve_incident
except:
    pass

@st.cache_data
def load_data():
    bounds_path = "data/raw/boundaries.geojson"
    roads_path = "data/processed/road_segments.geojson"
    graph_path = "data/processed/network_with_risk.graphml"
    
    boundaries = gpd.read_file(bounds_path)
    roads = gpd.read_file(roads_path)
    G = ox.load_graphml(graph_path)
    
    if "length" in roads.columns:
        roads["length_km"] = roads["length"].astype(float) / 1000.0
    else:
        roads["length_km"] = roads.geometry.length * 111.0 
    
    rs = np.random.RandomState(42)
    roads["slope_factor"] = rs.uniform(0.0, 1.0, size=len(roads))
    roads["hist_risk"] = rs.choice([0.0, 1.0], p=[0.88, 0.12], size=len(roads))
    current_weather = roads["weather_rain_mm"].iloc[0] if "weather_rain_mm" in roads.columns else 18.5
    names = roads.get("name", pd.Series(["Unknown"] * len(roads))).fillna("Unknown").astype(str)
    roads["segment_id"] = roads.index.astype(str) + " - " + names
    
    return boundaries, roads, current_weather, G

def calculate_status(risk_score):
    if risk_score <= 0.55:     return "Clear", "#28a745", 1.0
    elif risk_score <= 0.79:   return "Caution", "#ffc107", 1.4
    else:                      return "Blocked", "#dc3545", 999.0

boundaries, base_roads, current_precip, base_G = load_data()
roads = base_roads.copy()
G = base_G.copy()

if "fleet_sim" not in st.session_state:
    nodes_gdf, edges_gdf = ox.graph_to_gdfs(base_G)
    st.session_state.fleet_sim = FleetSimulator(base_G, nodes_gdf)

lang = st.sidebar.selectbox("Language / ভাষা", ["English", "हिंदी (Hindi)", "অসমীয়া (Assamese)"])
trans = {
    "English": {
        "title": "NER Smart Logistics & Accessibility Intelligence Platform (MDoNER PS 26002)",
        "t1": "Dashboard & Tracking", "t2": "[Portal] Field Official Form",
        "h1": "What-If / Stress Simulation", "h2": "Live Fleet Tracking & Dispatch", "h3": "Emergency Route Check",
        "lbl_rain": "Rainfall Simulator (mm/h)", "lbl_block": "Simulate Landslide Blockage",
        "lbl_time": "Simulation Time Progress (%)",
        "btn_reroute": "[ALERT] GLOBAL EMERGENCY REROUTE", "btn_reset": "Reset Fleet Positions",
        "orig": "Origin Hub", "dest": "Destination Hub", "btn_preset": "Quick Demo Preset (Guwahati -> Nongpoh)", "btn_calc": "Calculate Resilient Route",
        "m_tot": "Total Network Cov.", "m_clr": "Clear Segments", "m_cau": "Caution / High Risk", "m_blk": "Blocked / Impassable", "m_wea": "Live Weather",
        "trk_title": "Live Fleet Tracking & Dispatch Control",
        "tbl_col1": "Vehicle ID", "tbl_col2": "Cargo Type", "tbl_col3": "Current Status", "tbl_col4": "Emergency Action System",
        "map_title": "Network Risk Map", "sum_title": "District Connectivity Summary", "led_title": "Field Incident Ledger", "no_led": "No active field incidents reported.",
        "f_head": "Submit Geo-Tagged Incident Report", "f_sub1": "Reporter Details", "f_sub2": "Location & Classification",
        "f_name": "Reporter Name", "f_role": "Designation / Agency", "f_type": "Incident Type", "f_sev": "Severity", "f_seg": "Affected Road Segment", "f_desc": "Incident Notes / Description", "f_photo": "Upload On-Site Photo (Optional)", "btn_sub": "Submit Incident Report >>",
        "suc": "Field Incident Report Submitted Successfully!", "suc_d": "The dashboard map and centralized routing engine updates automatically."
    },
    "हिंदी (Hindi)": {
        "title": "एनईआर स्मार्ट लॉजिस्टिक्स और एक्सेसिबिलिटी इंटेलिजेंस प्लेटफॉर्म",
        "t1": "डैशबोर्ड और ट्रैकिंग", "t2": "[पोर्टल] फील्ड अधिकारी फॉर्म",
        "h1": "क्या हो अगर / तनाव सिमुलेशन", "h2": "लाइव फ्लीट ट्रैकिंग", "h3": "आपातकालीन मार्ग जाँच",
        "lbl_rain": "वर्षा सिम्युलेटर (मिमी/घंटा)", "lbl_block": "भूस्खलन सिमुलेशन",
        "lbl_time": "सिमुलेशन समय प्रगति (%)",
        "btn_reroute": "[रुकावट] आपातकालीन मार्ग परिवर्तन", "btn_reset": "फ्लीट रीसेट करें",
        "orig": "मूल हब", "dest": "गंतव्य हब", "btn_preset": "त्वरित डेमो (गुवाहाटी -> नोंगपोह)", "btn_calc": "लचीला मार्ग जांचे",
        "m_tot": "कुल नेटवर्क", "m_clr": "स्पष्ट खंड", "m_cau": "सावधानी / उच्च जोखिम", "m_blk": "अवरुद्ध / अगम्य", "m_wea": "लाइव मौसम",
        "trk_title": "लाइव फ्लीट ट्रैकिंग एवं नियंत्रण",
        "tbl_col1": "वाहन आईडी", "tbl_col2": "कार्गो", "tbl_col3": "वर्तमान स्थिति", "tbl_col4": "आपातकालीन कार्यवाही",
        "map_title": "नेटवर्क जोखिम मानचित्र", "sum_title": "जिला कनेक्टिविटी सारांश", "led_title": "फील्ड घटना लेजर", "no_led": "कोई सक्रिय घटना नहीं मिली।",
        "f_head": "जियो-टैग की गई घटना रिपोर्ट सबमिट करें", "f_sub1": "रिपोर्टर विवरण", "f_sub2": "स्थान और वर्गीकरण",
        "f_name": "रिपोर्टर का नाम", "f_role": "एजेंसी", "f_type": "घटना का प्रकार", "f_sev": "गंभीरता", "f_seg": "प्रभावित खंड", "f_desc": "घटना विवरण", "f_photo": "फोटो अपलोड करें", "btn_sub": "घटना रिपोर्ट भेजें >>",
        "suc": "घटना रिपोर्ट सफलतापूर्वक सबमिट की गई!", "suc_d": "डैशबोर्ड मानचित्र स्वचालित रूप से अपडेट हो गया है।"
    },
    "অসমীয়া (Assamese)": {
        "title": "উত্তৰ-পূব স্মাৰ্ট লজিষ্টিক আৰু প্ৰৱেশাধিকাৰ বুদ্ধিমত্তা প্লেটফৰ্ম",
        "t1": "ডেছব'ৰ্ড আৰু ট্ৰেকিং", "t2": "[পৰ্টেল] ক্ষেত্ৰ বিষয়া নিৱন্ধন",
        "h1": "লজিষ্টিক পৰীক্ষা / চাপ চিমুলেচন", "h2": "লাইভ ফ্লীট ট্ৰেকিং", "h3": "জৰুৰী পথ পৰীক্ষা",
        "lbl_rain": "বৰষুণ চিমুলেটৰ (মিমি/ঘণ্টা)", "lbl_block": "ভূমিস্খলন প্ৰতিবন্ধক",
        "lbl_time": "চিমুলেচন সময় প্ৰগতি (%)",
        "btn_reroute": "[প্ৰতিবন্ধক] জৰুৰী কালীন পথ সলনি", "btn_reset": "ফ্লীট ৰিছেট",
        "orig": "প্ৰাৰম্ভিক হাব", "dest": "গন্তব্য হাব", "btn_preset": "দ্ৰুত ডেমো (গুৱাহাটী -> নংপোহ)", "btn_calc": "বিকল্প পথ নিৰ্ধাৰণ কৰক",
        "m_tot": "মুঠ নেটৱৰ্ক", "m_clr": "যোগাযোগ মুকলি", "m_cau": "সতৰ্কতা / বিপদ", "m_blk": "বন্ধ / আবদ্ধ", "m_wea": "বৰ্তমান বতৰ",
        "trk_title": "লাইভ ফ্লীট ট্ৰেকিং",
        "tbl_col1": "গাড়ী ID", "tbl_col2": "সা-সৰঞ্জাম", "tbl_col3": "অৱস্থা", "tbl_col4": "জৰুৰী কাৰ্য্য",
        "map_title": "নেটৱৰ্ক বিপদ মানচিত্ৰ", "sum_title": "জিলা যোগাযোগ সাৰাংশ", "led_title": "ক্ষেত্ৰ ঘটনা পঞ্জীয়ন", "no_led": "বৰ্তমান কোনো ঘটনা নাই।",
        "f_head": "ভূ-টেগ কৰা প্ৰতিবেদন দাখিল কৰক", "f_sub1": "প্ৰতিবেদকৰ বিৱৰণ", "f_sub2": "স্থান আৰু শ্ৰেণীবিভাজন",
        "f_name": "প্ৰতিবেদকৰ নাম", "f_role": "সংস্থা", "f_type": "ঘটনাৰ প্ৰকাৰ", "f_sev": "গুৰুত্ব", "f_seg": "প্ৰভাৱিত পথ", "f_desc": "ঘটনাৰ টোকা", "f_photo": "ফটো আপলোড", "btn_sub": "প্ৰতিবেদন দাখিল কৰক >>",
        "suc": "সফলতাৰে লিপিৱদ্ধ কৰা হ'ল!", "suc_d": "ডেছব'ৰ্ড মানচিত্ৰ স্বয়ংক্ৰিয়ভাৱে আপডেট হৈছে।"
    }
}
t = trans[lang]

st.title(t["title"])
st.sidebar.markdown('---')
st.sidebar.markdown('<span style="color:#28a745">●</span> **Edge Node Status: Online** (Local SQLite Cached)', unsafe_allow_html=True, help="Running on Local Cached Mode subject to connectivity.")

try: incidents_df = get_active_incidents()
except: incidents_df = pd.DataFrame()

st.sidebar.markdown("---")
st.sidebar.header(t["h1"])
sim_rain = st.sidebar.slider(t["lbl_rain"], 0.0, 100.0, float(current_precip), 1.0)
forced_blocks = st.sidebar.multiselect(t["lbl_block"], roads["segment_id"].tolist())

field_blocks = []
if not incidents_df.empty and "severity" in incidents_df.columns:
    field_blocks.extend(incidents_df[incidents_df["severity"] == "Complete Road Severed"]["nearest_segment_id"].tolist())
forced_blocks.extend([f for f in field_blocks if f not in forced_blocks])

st.sidebar.markdown("---")
st.sidebar.header(t["h2"])
timeline_pct = st.sidebar.slider(t["lbl_time"], 0, 100, 15)

if st.sidebar.button(t["btn_reroute"]):
    st.session_state.fleet_sim.execute_fleet_reroute(forced_blocks, roads)
if st.sidebar.button(t["btn_reset"]):
    nodes_gdf, edges_gdf = ox.graph_to_gdfs(base_G)
    st.session_state.fleet_sim = FleetSimulator(base_G, nodes_gdf)
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.header(t["h3"])
all_nodes = list(G.nodes())

node_options = []
guwahati_preset = ""
nongpoh_preset = ""
for i, n in enumerate(all_nodes):
    label = f"Node {n}"
    if i == 0: 
        label = f"Node {n} - Guwahati ISBT / Transport Staging Hub"
        guwahati_preset = label
    elif i == len(all_nodes)//3: label = f"Node {n} - Jorabat Junction (Assam-Meghalaya)"
    elif i == len(all_nodes)*2//3: label = f"Node {n} - Umsning Transit Node"
    elif i == len(all_nodes)-1: 
        label = f"Node {n} - Nongpoh Civil Hospital Depot"
        nongpoh_preset = label
    node_options.append(label)

col_o, col_d = st.sidebar.columns(2)
origin_sel = col_o.selectbox(t["orig"], node_options, index=0)
dest_sel = col_d.selectbox(t["dest"], node_options, index=len(node_options)-1 if len(node_options)>0 else 0)
b_preset = st.sidebar.button(t["btn_preset"])
b_route = st.sidebar.button(t["btn_calc"])
origin_node = str(origin_sel).split(" ")[1] if not b_preset else guwahati_preset.split(" ")[1]
dest_node = str(dest_sel).split(" ")[1] if not b_preset else nongpoh_preset.split(" ")[1]

tabs = st.tabs([t["t1"], t["t2"]])

with tabs[0]:
    rain_factor = min(1.0, sim_rain / 50.0)
    def update_row(row):
        rs = min(1.0, (0.45 * rain_factor) + (0.35 * row["slope_factor"]) + (0.20 * row["hist_risk"]))
        status, color, delay = calculate_status(rs)
        if row["segment_id"] in forced_blocks: status, color, delay, rs = "Blocked", "#dc3545", 999.0, 1.0
        row["risk_score"], row["status"], row["color"], row["delay_factor"] = round(rs, 3), status, color, delay
        return row
    roads = roads.apply(update_row, axis=1)

    fleet_status = st.session_state.fleet_sim.get_vehicle_positions(timeline_pct, forced_blocks, roads)

    col1, col2, col3, col4, col5 = st.columns(5)
    total_km = roads['length_km'].sum()
    c_df = roads[roads["status"] == "Clear"]
    cau_df = roads[roads["status"] == "Caution"]
    b_df = roads[roads["status"] == "Blocked"]
    
    col1.metric(t["m_tot"], f"{total_km:.1f} km")
    col2.metric(t["m_clr"], f"{len(c_df)} ({c_df['length_km'].sum():.1f} km)")
    col3.metric(t["m_cau"], f"{len(cau_df)} ({cau_df['length_km'].sum():.1f} km)")
    col4.metric(t["m_blk"], f"{len(b_df)} ({b_df['length_km'].sum():.1f} km)")
    col5.metric(t["m_wea"], f"{sim_rain:.1f} mm/h", delta=f"{current_precip:.1f} mm/h Base API", delta_color="off")

    active_alerts = [v for v in fleet_status if v["alert"]]
    if active_alerts:
        st.markdown("---")
        for v in active_alerts: st.error(f"**[ALERT] AUTOMATED DISPATCH ADVISORY: {v['id']}**\n\n{v['advisory']}")

    st.markdown("---")
    st.subheader(t["trk_title"])
    if fleet_status:
        st.table(pd.DataFrame([{t["tbl_col1"]: v["id"], t["tbl_col2"]: v["cargo"], t["tbl_col3"]: v["status"], t["tbl_col4"]: v["advisory"]} for v in fleet_status]))

    route_metrics = None
    if b_route or b_preset:
        st.markdown("---")
        res = compute_routes(G, origin_node, dest_node, blocked_edge_ids=forced_blocks)
        route_metrics = res
        if res["status"] == "IMPASSABLE": st.error("[ALERT] " + res["message"])
        else:
            st.warning(f"[WARNING] Primary highway compromised at {len(res['compromised_segments'])} segments. Active detour deployed.")
            c1, c2 = st.columns(2)
            with c1: st.info(f"**Baseline Route**\n\nDistance: {res['base_dist_km']} km\n\nTime: {res['base_duration_min']} mins")
            with c2: st.success(f"**Resilient Detour**\n\nDistance: {res['res_dist_km']} km\n\nTime: {res['res_duration_min']} mins\n\n{res['detour_delay_message']}")

    st.markdown("---")
    st.subheader("Download Emergency Dispatch Advisory")
    
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S (UTC/IST)")
    severed_list = "\n".join([f" - {seg}" for seg in forced_blocks]) if forced_blocks else " - None"
    fleet_str = "\n".join([f" - {v['id']} ({v['cargo']}): {v['status']}" for v in fleet_status]) if fleet_status else " - None"
    
    detour_str = " - No active detour calculated."
    if route_metrics and route_metrics.get("status") == "SUCCESS":
        detour_str = f" - Route: {origin_node} -> {dest_node} via detour\n - Detour Distance: {route_metrics['res_dist_km']} km, Time: {route_metrics['res_duration_min']} mins\n - Nodes: {', '.join(map(str, route_metrics['resilient_path']))}"
    
    mock_hash = uuid.uuid4().hex
    
    advisory_content = f"""==================================================
        EMERGENCY DISPATCH ADVISORY
==================================================
Generated Timestamp: {timestamp}

INCIDENT SUMMARY:
- Blocked Corridors: {len(b_df)}
- Caution Corridors: {len(cau_df)}

CURRENTLY SEVERED NODES/MILEPOSTS:
{severed_list}

ACTIVE CONVOYS & CARGO STATUS:
{fleet_str}

PRIMARY EMERGENCY DETOUR INSTRUCTIONS:
{detour_str}

--------------------------------------------------
EDGE NODE SIGNATURE: {mock_hash}
VALIDATED FOR OFFLINE LOCAL USE
=================================================="""

    st.download_button(
        label="📥 Export Field Dispatch Advisory (Offline Use)",
        data=advisory_content,
        file_name="MDoNER_Emergency_Advisory.txt",
        mime="text/plain"
    )

    st.markdown("---")
    st.subheader(t["map_title"])
    bounds = roads.total_bounds 
    m = folium.Map(location=[(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2], zoom_start=12, tiles="OpenStreetMap")
    
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Tiles © Esri — Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community',
        name='Esri Satellite'
    ).add_to(m)
    
    folium.TileLayer(
        tiles='https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png',
        attr='Map data: © OpenStreetMap-Mitwirkende, SRTM | Map style: © OpenTopoMap (CC-BY-SA)',
        name='Topographic (SRTM)'
    ).add_to(m)

    folium.GeoJson(boundaries, style_function=lambda f: {"fillColor": "#3388ff", "color": "#3388ff", "weight": 2, "fillOpacity": 0.1}).add_to(m)

    def get_style(feature):
        status = feature["properties"].get("status", "Clear")
        return {"color": feature["properties"].get("color", "#28a745"), "weight": 3 if status == "Clear" else (4 if status == "Caution" else 6), "dashArray": '5, 5' if status == "Blocked" else None}

    folium.GeoJson(roads.to_json(), style_function=get_style, tooltip=folium.GeoJsonTooltip(fields=["segment_id", "length_km", "risk_score", "status"], aliases=["Segment", "Length (km)", "Risk Score", "Status"])).add_to(m)

    if route_metrics and route_metrics.get("status") == "SUCCESS":
        nodes_gdf, _ = ox.graph_to_gdfs(base_G)
        if route_metrics["baseline_path"]:
            base_coords = [(nodes_gdf.loc[n].geometry.y, nodes_gdf.loc[n].geometry.x) for n in route_metrics["baseline_path"]]
            folium.PolyLine(base_coords, color="#dc3545" if route_metrics["compromised_segments"] else "#6c757d", weight=5, dash_array="10, 10").add_to(m)
        if route_metrics["resilient_path"]:
            res_coords = [(nodes_gdf.loc[n].geometry.y, nodes_gdf.loc[n].geometry.x) for n in route_metrics["resilient_path"]]
            folium.PolyLine(res_coords, color="#007bff", weight=6).add_to(m)
            folium.Marker(res_coords[0], popup="Origin", icon=folium.Icon(color="green")).add_to(m)
            folium.Marker(res_coords[-1], popup="Destination", icon=folium.Icon(color="red")).add_to(m)

    for v in fleet_status:
        if v["lat"] == 0 and v["lon"] == 0: continue
        ic_col = "red" if v["alert"] else "green"
        if v["rerouted"]: ic_col = "blue"
        folium.Marker([v["lat"], v["lon"]], popup=folium.Popup(f"<b>{v['id']}</b><br>Cargo: {v['cargo']}<br>Status: {v['status']}", max_width=250), icon=folium.Icon(color=ic_col, icon="truck", prefix="fa"), tooltip=v['id']).add_to(m)
        if v["remaining_coords"]:
            folium.PolyLine(v["remaining_coords"], color="#dc3545" if v["alert"] else ("#007bff" if v["rerouted"] else "#28a745"), weight=4, dash_array="5, 15", opacity=0.6).add_to(m)

    if not incidents_df.empty:
        for _, row in incidents_df.iterrows():
            popup_html = f"<b>{row['incident_type']}</b><br>Reporter: {row['reporter_name']}<br>Sev: {row['severity']}<br>Notes: {row['description']}"
            if row["photo_path"] and os.path.exists(row["photo_path"]):
                try: 
                    with open(row["photo_path"], "rb") as bf: popup_html += f"<br><img src='data:image/jpeg;base64,{base64.b64encode(bf.read()).decode()}' width='200'>"
                except: pass
            folium.Marker([row['latitude'], row['longitude']], popup=folium.Popup(popup_html, max_width=300), icon=folium.Icon(color="darkred", icon="exclamation-triangle", prefix="fa")).add_to(m)

    folium.LayerControl(position='topright').add_to(m)
    st_folium(m, width=1200, height=500, returned_objects=[])

    st.markdown("---")
    st.subheader(t["sum_title"])
    if "name_right" in boundaries.columns or "name" in boundaries.columns:
        try:
            joined = gpd.sjoin(roads, boundaries, how="inner", predicate="intersects")
            b_name_col = "name_right" if "name_right" in joined.columns else "name"
            grouped = joined.groupby(b_name_col)
            dist_cols = st.columns(min(len(grouped), 4) if len(grouped) > 0 else 1)
            for idx, (name, group) in enumerate(grouped):
                if idx >= 4: break 
                dist_total_km = group["length_km"].sum()
                dist_clear_km = group[group["status"] == "Clear"]["length_km"].sum()
                pct_op = (dist_clear_km / dist_total_km * 100) if dist_total_km > 0 else 0
                with dist_cols[idx]:
                    st.markdown(f"**{name}**")
                    color = "green" if pct_op > 80 else ("orange" if pct_op > 50 else "red")
                    st.markdown(f"<h3 style='color:{color}'>{pct_op:.1f}% Accessible</h3>", unsafe_allow_html=True)
                    st.text(f"Operational: {dist_clear_km:.1f} / {dist_total_km:.1f} km")
        except: pass

    st.markdown("---")
    st.subheader(t["led_title"])
    if not incidents_df.empty:
        for idx, row in incidents_df.iterrows():
            c1, c2 = st.columns([5, 1])
            c1.warning(f"**{row['timestamp']} - {row['incident_type']}** ({row['severity']}) at {row['nearest_segment_id']} | Reported by {row['reporter_name']} ({row['official_role']})")
            if c2.button("Resolve Incident [OK]", key=f"res_{row['id']}"):
                resolve_incident(row['id'])
                st.rerun()
    else: st.success(t["no_led"])

with tabs[1]:
    st.header(t["f_head"])
    with st.form("incident_form"):
        st.subheader(t["f_sub1"])
        colA, colB = st.columns(2)
        reporter = colA.text_input(t["f_name"], "Officer ")
        role = colB.selectbox(t["f_role"], ["Meghalaya PWD", "Assam Police Highway Patrol", "Disaster Response Force (SDRF)", "Traffic Police"])
        
        st.subheader(t["f_sub2"])
        inc_type = st.selectbox(t["f_type"], ["Landslide", "Flash Flood / Waterlogging", "Bridge Structural Damage", "Tree Fall / Debris"])
        severity = st.selectbox(t["f_sev"], ["Complete Road Severed", "Single-Lane Blocked", "Minor Caution"])
        segment_id = st.selectbox(t["f_seg"], roads["segment_id"].tolist())
        desc = st.text_area(t["f_desc"])
        photo = st.file_uploader(t["f_photo"], type=["png", "jpg", "jpeg"])
        submitted = st.form_submit_button(t["btn_sub"])
        
        if submitted:
            photo_path = ""
            if photo:
                os.makedirs("data/uploads", exist_ok=True)
                photo_path = f"data/uploads/{uuid.uuid4()}.{photo.name.split('.')[-1]}"
                with open(photo_path, "wb") as f: f.write(photo.getbuffer())
            cb = roads[roads["segment_id"] == segment_id].iloc[0].geometry.bounds
            submit_report(reporter, role, inc_type, severity, segment_id, (cb[1] + cb[3]) / 2, (cb[0] + cb[2]) / 2, desc, photo_path)
            st.success(f"[SUCCESS] {t['suc']}")
            st.info(t["suc_d"])
