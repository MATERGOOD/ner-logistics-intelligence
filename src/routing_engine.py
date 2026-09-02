import networkx as nx
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def compute_routes(G, origin_node, dest_node, blocked_edge_ids=[]):
    """
    Computes both baseline and resilient shortest routes accounting for blocked edges.
    """
    # Create working copies
    G_baseline = G.copy()
    G_resilient = G.copy()
    
    # 1. Baseline Route Simulation (ignoring dynamic weather/blockages)
    # Just use 'length' or generic 'travel_time_base_min'
    # By default, use 'travel_time_base_min' if available, else 'length'
    has_base_time = any("travel_time_base_min" in data for _, _, data in G.edges(data=True))
    base_weight = "travel_time_base_min" if has_base_time else "length"
    
    try:
        baseline_path = nx.shortest_path(G_baseline, origin_node, dest_node, weight=base_weight)
        
        # Calculate baseline metrics
        base_dist_m = sum(G_baseline[u][v][0].get("length", 0) for u, v in zip(baseline_path[:-1], baseline_path[1:]))
        base_dist_km = base_dist_m / 1000.0
        
        # Compute how the baseline route holds up under *current* dynamic conditions
        # (It will traverse blocked edges anyway, so we sum actual travel times and catch impassable)
        compromised_segments = []
        base_actual_t_min = 0.0
        
        for u, v in zip(baseline_path[:-1], baseline_path[1:]):
            edge_data = G_baseline[u][v][0]
            # Identify segment_id string based on index and name, or just fallback to u-v
            seg_name = edge_data.get("name", "Unknown")
            if isinstance(seg_name, list): seg_name = seg_name[0]
            # Edge index or segment ID string matching
            seg_id = f"{(u, v, 0)} - {seg_name}" 
            
            # Check dynamic blockages
            st = edge_data.get("status", "Clear")
            # Override if in explicit blocked_edge_ids
            # blocked_edge_ids will loosely match edge u,v or string id
            if any(str(b_id) in str(edge_data) or str(b_id) in seg_id for b_id in blocked_edge_ids):
                st = "Blocked"
                
            if st == "Blocked":
                compromised_segments.append(seg_name)
                # Impassable
                base_actual_t_min += float("inf")
            elif st == "Caution":
                base_actual_t_min += edge_data.get(base_weight, 1.0) * 1.5
            else:
                base_actual_t_min += edge_data.get(base_weight, 1.0)
                
    except nx.NetworkXNoPath:
        baseline_path = None
        base_dist_km = 0
        base_actual_t_min = float("inf")
        compromised_segments = []
        
    # 2. Resilient Route (avoiding Blocked, penalizing Caution)
    # Prune Blocked edges and penalize Caution
    edges_to_remove = []
    for u, v, k, data in G_resilient.edges(keys=True, data=True):
        st = data.get("status", "Clear")
        seg_name = data.get("name", "Unknown")
        if isinstance(seg_name, list): seg_name = seg_name[0]
        
        # Explicit block check
        if any(str(b_id) in str(data) or str(b_id) in f"{(u, v, k)} - {seg_name}" for b_id in blocked_edge_ids):
            st = "Blocked"
            
        if st == "Blocked":
            edges_to_remove.append((u, v, k))
        elif st == "Caution":
            data["resilient_weight"] = data.get(base_weight, 1.0) * 1.5
        else:
            data["resilient_weight"] = data.get(base_weight, 1.0)
            
    G_resilient.remove_edges_from(edges_to_remove)
    
    try:
        resilient_path = nx.shortest_path(G_resilient, origin_node, dest_node, weight="resilient_weight")
        
        # Compute resilient path metrics
        res_dist_m = sum(G_resilient[u][v][0].get("length", 0) for u, v in zip(resilient_path[:-1], resilient_path[1:]))
        res_dist_km = res_dist_m / 1000.0
        
        res_t_min = sum(G_resilient[u][v][0].get("resilient_weight", 0) for u, v in zip(resilient_path[:-1], resilient_path[1:]))
        status_msg = "SUCCESS"
        
    except nx.NetworkXNoPath:
        return {
            "status": "IMPASSABLE",
            "message": "All road corridors blocked. Recommend staging at nearest hub or air-relief.",
            "baseline_path": baseline_path,
            "resilient_path": None
        }

    return {
        "status": status_msg,
        "message": "Alternate route successfully plotted.",
        "baseline_path": baseline_path,
        "base_dist_km": round(base_dist_km, 2),
        "base_duration_min": round(base_actual_t_min, 2) if base_actual_t_min != float("inf") else "Indefinite (Blocked)",
        "compromised_segments": list(set(compromised_segments)),
        "resilient_path": resilient_path,
        "res_dist_km": round(res_dist_km, 2),
        "res_duration_min": round(res_t_min, 2),
        "detour_delay_message": f"+{round(res_dist_km - base_dist_km, 2)} km detour"
    }
