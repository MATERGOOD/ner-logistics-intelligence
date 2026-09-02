import os
import logging
import pandas as pd
import geopandas as gpd
import osmnx as ox
from shapely.geometry import box

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def fetch_data():
    data_dir = os.path.join("data", "raw")
    os.makedirs(data_dir, exist_ok=True)
    
    # Configure robust OSMnx settings at the top of the file
    ox.settings.requests_timeout = 180
    ox.settings.use_cache = True
    ox.settings.log_console = True
    ox.settings.overpass_url = "https://lz4.overpass-api.de/api"
    ox.settings.http_accept_language = "en-US,en;q=0.9"
    
    # User's specified bbox (min_lat=25.85, max_lat=26.15, min_lon=91.80, max_lon=91.95)
    # OSMnx > 2.0 requires (left, bottom, right, top) => (min_lon, min_lat, max_lon, max_lat)
    bbox_env = (91.80, 25.85, 91.95, 26.15)
    
    # Target districts for geocoding
    district_queries = [
        "Kamrup Metropolitan, Assam, India",
        "Ri-Bhoi, Meghalaya, India"
    ]
    
    # Step 1: Manage Boundaries
    boundaries_gdf = None
    try:
        logging.info("Attempting structured geocoding for district boundaries...")
        gdfs = []
        for q in district_queries:
            try:
                gdf = ox.geocode_to_gdf(q)
                gdfs.append(gdf)
            except Exception as e:
                logging.warning(f"Failed to geocode {q}: {e}")
                
        if len(gdfs) == len(district_queries):
            boundaries_gdf = pd.concat(gdfs, ignore_index=True)
            logging.info("Successfully fetched all requested districts via structured geocoding.")
        else:
            raise ValueError(f"Could not retrieve all districts (Expected: {len(district_queries)}, found: {len(gdfs)}). Proceeding to synthetic fallback.")
            
    except Exception as e:
        logging.warning(f"Geocoding failed or timed out: {e}")
        logging.info("Creating synthetic boundary from fallback bounding box...")
        # Fallback to BBox polygon
        boundary_poly = box(*bbox_env)
        boundaries_gdf = gpd.GeoDataFrame([
            {"geometry": boundary_poly, "name": "Synthetic Fallback Boundary (Guwahati to Nongpoh)", "state": "Assam/Meghalaya"}
        ], crs="EPSG:4326")
        
    logging.info("Saving district boundaries to data/raw/boundaries.geojson")
    boundaries_gdf.to_file(os.path.join(data_dir, "boundaries.geojson"), driver="GeoJSON")
    
    # Step 2: Fetch major transit highways using custom filter
    c_filter = '["highway"~"primary|secondary|trunk|motorway"]'
    logging.info(f"Fetching road network for BBox {bbox_env} with custom filter: {c_filter}")
    
    try:
        G = ox.graph_from_bbox(bbox=bbox_env, custom_filter=c_filter, retain_all=True)
        
        logging.info("Saving network graph to data/raw/network_combined.graphml")
        ox.save_graphml(G, os.path.join(data_dir, "network_combined.graphml"))
        
        nodes, edges = ox.graph_to_gdfs(G)
        
        logging.info("--- Execution Summary ---")
        logging.info(f"Total nodes: {len(nodes)}")
        logging.info(f"Total edges: {len(edges)}")
        logging.info(f"Bounding box limits (West, South, East, North): {list(boundaries_gdf.total_bounds)}")
        logging.info("Saved files:")
        logging.info(f"- {os.path.join(data_dir, 'boundaries.geojson')}")
        logging.info(f"- {os.path.join(data_dir, 'network_combined.graphml')}")
        logging.info("Pipeline Complete.")
    except Exception as e:
        logging.error(f"Failed to fetch network graph: {e}")

if __name__ == "__main__":
    fetch_data()
