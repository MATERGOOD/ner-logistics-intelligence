import os
import random
import logging
import requests
import geopandas as gpd
import networkx as nx
import osmnx as ox

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def get_weather_data(lat, lon):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=precipitation,rain,weather_code"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        precip = data.get("current", {}).get("precipitation", 0.0)
        logging.info(f"Fetched weather from Open-Meteo: {precip} mm/h precipitation")
        return precip
    except Exception as e:
        logging.warning(f"Weather API failed or timed out: {e}. Using fallback 18.5 mm/h.")
        return 18.5

def run_risk_engine():
    data_dir_raw = os.path.join("data", "raw")
    data_dir_processed = os.path.join("data", "processed")
    os.makedirs(data_dir_processed, exist_ok=True)
    
    graph_path = os.path.join(data_dir_raw, "network_combined.graphml")
    
    if not os.path.exists(graph_path):
        logging.error(f"Graph file not found at {graph_path}. Exiting.")
        return
        
    logging.info("Loading network graph...")
    G = ox.load_graphml(graph_path)
    
    # Calculate graph center
    nodes, edges = ox.graph_to_gdfs(G)
    center_lat = nodes.geometry.y.mean()
    center_lon = nodes.geometry.x.mean()
    
    logging.info(f"Graph center calculated at Lat: {center_lat:.4f}, Lon: {center_lon:.4f}")
    precip_mm = get_weather_data(center_lat, center_lon)
    rain_factor = min(1.0, precip_mm / 50.0)
    
    counts = {"Clear": 0, "Caution": 0, "Blocked / Critical Risk": 0}
    random.seed(42) # Reproducible randomness for heuristics
    
    # Process edges
    for u, v, key, data in G.edges(keys=True, data=True):
        # 10-15% chance of being a historical landslide hotspot
        hist_risk = 1.0 if random.random() < 0.12 else 0.0
        
        # generate normalized slope factor (0.0 to 1.0)
        slope_factor = random.uniform(0.0, 1.0)
        
        # Composite risk score formula
        risk_score = min(1.0, (0.45 * rain_factor) + (0.35 * slope_factor) + (0.20 * hist_risk))
        
        if risk_score <= 0.39:
            status = "Clear"
            color = "Green"
            delay_factor = 1.0
        elif risk_score <= 0.69:
            status = "Caution"
            color = "Orange"
            delay_factor = 1.4
        else:
            status = "Blocked / Critical Risk"
            color = "Red"
            delay_factor = 999.0
            
        counts[status] += 1
        
        # Edge Enrichment
        data["weather_rain_mm"] = float(precip_mm)
        data["risk_score"] = float(risk_score)
        data["status"] = status
        data["color"] = color
        data["delay_factor"] = float(delay_factor)
        
        length = float(data.get("length", 0.0))
        maxspeed = data.get("maxspeed", 50)
        if isinstance(maxspeed, list): maxspeed = maxspeed[0]
        try:
            speed_kmh = float(maxspeed)
        except:
            speed_kmh = 50.0
            
        base_time = (length / 1000.0) / speed_kmh * 60 # minutes
        data["travel_time_adjusted"] = base_time * delay_factor

    logging.info("Saving updated graph with risk metadata...")
    ox.save_graphml(G, os.path.join(data_dir_processed, "network_with_risk.graphml"))
    
    # Convert and export to GeoPandas
    logging.info("Exporting road segments to GeoJSON...")
    _, edges_gdf = ox.graph_to_gdfs(G)
    
    # GeoJSON doesn't support complex list/dict types. Stringify them to prevent Fiona schema failures.
    for col in edges_gdf.columns:
        if edges_gdf[col].dtype == object or isinstance(edges_gdf[col].iloc[0], list):
            edges_gdf[col] = edges_gdf[col].apply(lambda x: str(x) if x is not None else "")
            
    geojson_path = os.path.join(data_dir_processed, "road_segments.geojson")
    edges_gdf.to_file(geojson_path, driver="GeoJSON")
    
    logging.info("--- Execution Summary ---")
    logging.info(f"Total Edges Processed: {len(edges_gdf)}")
    logging.info(f"Status Breakdown: {counts}")
    logging.info("Saved files:")
    logging.info(f"- {os.path.join(data_dir_processed, 'network_with_risk.graphml')}")
    logging.info(f"- {geojson_path}")
    logging.info("Phase 2 Complete.")

if __name__ == "__main__":
    run_risk_engine()
