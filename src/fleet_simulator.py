import networkx as nx
from shapely.geometry import LineString
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class FleetSimulator:
    def __init__(self, G, nodes_gdf):
        self.G = G
        self.nodes_gdf = nodes_gdf
        
        # Attempt to grab distinct nodes spread across the graph to make real routes
        nodes_list = list(G.nodes())
        if len(nodes_list) < 10:
            self.fleet = {}
            return
            
        n1, n2 = nodes_list[0], nodes_list[-1]
        n3, n4 = nodes_list[len(nodes_list)//4], nodes_list[len(nodes_list)//2]
        n5, n6 = nodes_list[len(nodes_list)//3], nodes_list[int(len(nodes_list)*0.75)]
            
        self.fleet = {
            "TRK-01": {
                "cargo": "Medical & Vaccine Supplies",
                "origin": n1,
                "dest": n2,
                "driver_contact": "Convoy Alpha"
            },
            "TRK-02": {
                "cargo": "Grain & Rice Ration Delivery",
                "origin": n3,
                "dest": n4,
                "driver_contact": "Convoy Bravo"
            },
            "TRK-03": {
                "cargo": "Emergency Fuel Tanker",
                "origin": n5,
                "dest": n6,
                "driver_contact": "Convoy Gamma"
            }
        }
        
        # Precalculate baseline paths
        for v_id, v in self.fleet.items():
            self._set_vehicle_path(v_id, v["origin"], v["dest"])
            v["rerouted"] = False
            
    def _set_vehicle_path(self, v_id, origin, dest, weight="length"):
        v = self.fleet[v_id]
        try:
            path = nx.shortest_path(self.G, origin, dest, weight=weight)
            v["path_nodes"] = path
            
            coords = []
            for n in path:
                # Retrieve coordinates safely from nodes_gdf (GeoDataFrame of graph nodes)
                # GeoDataFrame y is Lat, x is Lon (we return lat, lon pair)
                coords.append((self.nodes_gdf.loc[n].geometry.y, self.nodes_gdf.loc[n].geometry.x))
            v["path_coords"] = coords
            
            if len(coords) > 1:
                v["linestring"] = LineString(coords)
                v["total_length"] = v["linestring"].length # geometric length
            else:
                v["linestring"] = None
                v["total_length"] = 0
                
        except nx.NetworkXNoPath:
            v["path_nodes"] = []
            v["path_coords"] = []
            v["linestring"] = None
            v["total_length"] = 0

    def get_vehicle_positions(self, progress_pct, blocked_edge_ids, roads_df):
        """
        progress_pct: 0 to 100
        roads_df: the dynamically evaluated GeoDataFrame containing updated "status" per row
        """
        results = []
        
        # Build an easy lookup map for dynamically blocked segments from roads_df
        # All edges with "status == 'Blocked'"
        live_blocked_segments = set(roads_df[roads_df["status"] == "Blocked"]["segment_id"].tolist())
        
        # Combine explicit dropdown simulation bounds with active environmental blocks
        blocked_set = live_blocked_segments.union(set(blocked_edge_ids))
        
        for v_id, v in self.fleet.items():
            if not v["path_nodes"]: continue
            
            # 1. Calculate Spatial Position via interpolation
            fract = progress_pct / 100.0
            if v["linestring"]:
                pt = v["linestring"].interpolate(v["total_length"] * fract)
                current_lat, current_lon = pt.x, pt.y # pt is technically (lat, lon) because we built LineString from coords list which is Y/X natively reversed if we used normal logic, but wait: LineString(coords) has X=Lat, Y=Lon because coords = [(lat,lon)]. 
                # Yes: pt.x is Lat, pt.y is Lon.
            else:
                current_lat, current_lon = 0, 0
                
            # 2. Check Upcoming Path for Blockages
            # Approximate current node
            current_idx = int((len(v["path_nodes"]) - 1) * fract)
            remaining_nodes = v["path_nodes"][current_idx:]
            
            status = "🟢 En Route - Clear"
            advisory = "None"
            alert = False
            
            if not v["rerouted"]:
                # scan upcoming path
                for i in range(len(remaining_nodes)-1):
                    n1, n2 = remaining_nodes[i], remaining_nodes[i+1]
                    edge_data = self.G.get_edge_data(n1, n2)
                    if edge_data:
                        d = edge_data[0]
                        seg_name = d.get("name", "Unknown")
                        if isinstance(seg_name, list): seg_name = seg_name[0]
                        exact_id = f"{(n1, n2, 0)} - {seg_name}"
                        
                        # does this exactly strictly match a blocked row? Look at explicit or live subsets
                        # We also check base G status if synced just in case
                        if exact_id in blocked_set or "Blocked" in exact_id:
                            status = "🚨 ALERT: ROUTE SEVERED"
                            advisory = f"Convoy {v_id} carrying '{v['cargo']}' approaching Landslide hazard at {seg_name}. Emergency advisory: Deploy alternate detour immediately (+45 min delay penalty)."
                            alert = True
                            break
            else:
                status = "🔵 Secure Reroute - Diverted"
                advisory = "Successfully diverted away from active hazard zone."
                
            results.append({
                "id": v_id,
                "cargo": v["cargo"],
                "origin": v["origin"],
                "dest": v["dest"],
                "lat": current_lat,
                "lon": current_lon,
                "status": status,
                "advisory": advisory,
                "alert": alert,
                "remaining_coords": v["path_coords"][current_idx:],
                "rerouted": v["rerouted"]
            })
            
        return results
        
    def execute_fleet_reroute(self, blocked_edge_ids, roads_df):
        from routing_engine import compute_routes
        
        live_blocked_segments = set(roads_df[roads_df["status"] == "Blocked"]["segment_id"].tolist())
        complete_block_set = list(live_blocked_segments.union(set(blocked_edge_ids)))
        
        success = 0
        for v_id, v in self.fleet.items():
            if v["rerouted"]: continue
            # Attempt to reroute
            res = compute_routes(self.G, v["origin"], v["dest"], blocked_edge_ids=complete_block_set)
            
            if res.get("status") == "SUCCESS" and res.get("resilient_path"):
                v["path_nodes"] = res["resilient_path"]
                coords = [(self.nodes_gdf.loc[n].geometry.y, self.nodes_gdf.loc[n].geometry.x) for n in v["path_nodes"]]
                v["path_coords"] = coords
                if len(coords) > 1:
                    v["linestring"] = LineString(coords)
                    v["total_length"] = v["linestring"].length
                v["rerouted"] = True
                success += 1
                
        return success
