# transit_network_full.py (Main route + zonal feeder clustering)

import geopandas as gpd
import pandas as pd
import numpy as np
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
from shapely import wkt
from shapely.geometry import LineString
from shapely.ops import linemerge
from sklearn.cluster import KMeans

# --- Config ---
CITY_NAME = "General Santos, Philippines"
STOP_FILE = "transit_stops.geojson"
GRID_FILE = "population_density_data.csv"
BARANGAY_FILE = "barangays-municity-ph126303000.0.1.json"
PROJECTED_CRS = "EPSG:32651"

N_MAIN_STOPS = 5
MAIN_BUFFER_METERS = 800
NETWORK_TYPE = "drive"
N_ZONES = 5

# --- Load stops ---
print("Loading stop candidates...")
stops = gpd.read_file(STOP_FILE).to_crs(PROJECTED_CRS)
stops["centroid"] = stops.geometry.centroid

# --- Load road network ---
print("Downloading and preparing road network...")
G = ox.graph_from_place(CITY_NAME, network_type=NETWORK_TYPE, simplify=True)
G = ox.project_graph(G, to_crs=PROJECTED_CRS)

# --- Add custom weights favoring major roads ---
print("Applying custom road weights...")
road_priority = {
    "motorway": 1.0, "trunk": 1.0, "primary": 1.2,
    "secondary": 1.4, "tertiary": 1.6, "residential": 2.5,
    "unclassified": 3.0, "service": 4.0, "track": 5.0
}
for u, v, k, d in G.edges(keys=True, data=True):
    hwy = d.get("highway")
    if isinstance(hwy, list): hwy = hwy[0]
    factor = road_priority.get(hwy, 5.0)
    d["custom_weight"] = d.get("length", 1) * factor

# --- Convert major roads for background map ---
edges = ox.graph_to_gdfs(G, nodes=False)
road_classes = ["motorway", "trunk", "primary", "secondary", "tertiary"]
major_roads = edges[edges["highway"].isin(road_classes)]

# --- Main route using top-scoring stops (MST logic) ---
print("Routing top stops as main corridor...")
stops = stops.sort_values("weighted_score", ascending=False).reset_index(drop=True)
top_stops = stops.head(N_MAIN_STOPS).copy()
top_stops["node"] = [ox.distance.nearest_nodes(G, pt.x, pt.y) for pt in top_stops["centroid"]]

G_complete = nx.Graph()
paths = {}
for i in range(len(top_stops)):
    for j in range(i+1, len(top_stops)):
        try:
            path = nx.shortest_path(G, top_stops.loc[i, "node"], top_stops.loc[j, "node"], weight="custom_weight")
            cost = sum(G[u][v][0]["custom_weight"] for u, v in zip(path[:-1], path[1:]))
            G_complete.add_edge(i, j, weight=cost)
            paths[(i, j)] = path
        except Exception as e:
            print(f"⚠️ Skipping ({i}, {j}): {e}")

mst_edges = list(nx.minimum_spanning_edges(G_complete, data=True))
main_routes = []
for i, j, data in mst_edges:
    path = paths.get((i, j)) or paths.get((j, i))
    edge_gdf = ox.graph_to_gdfs(G.subgraph(path), nodes=False)
    merged = linemerge(list(edge_gdf.geometry))
    main_routes.append({"geometry": merged, "route": 0})

main_route_gdf = gpd.GeoDataFrame(main_routes, crs=PROJECTED_CRS)

# --- Buffer and assign stops ---
main_buffer = main_route_gdf.unary_union.buffer(MAIN_BUFFER_METERS)
stops["served_by_main"] = stops["centroid"].within(main_buffer)
served_stops = stops[stops["served_by_main"]].copy()
unserved_stops = stops[~stops["served_by_main"]].copy()

# --- Zonal clustering on unserved stops ---
print("Clustering unserved stops into zones...")
coords = np.array([(pt.x, pt.y) for pt in unserved_stops["centroid"]])
unserved_stops["zone"] = KMeans(n_clusters=N_ZONES).fit_predict(coords)

zone_anchors = (
    unserved_stops
    .sort_values("weighted_score", ascending=False)
    .groupby("zone")
    .head(1)
).copy()

main_nodes = [ox.distance.nearest_nodes(G, pt.x, pt.y) for pt in top_stops["centroid"]]
zone_anchors["node"] = [ox.distance.nearest_nodes(G, pt.x, pt.y) for pt in zone_anchors["centroid"]]

# --- Route feeders from zone anchors to nearest main stop ---
print("Routing feeders from zones...")
feeder_routes = []
for i, row in zone_anchors.iterrows():
    try:
        distances = [nx.shortest_path_length(G, row["node"], mn, weight="custom_weight") for mn in main_nodes]
        nearest_main_node = main_nodes[np.argmin(distances)]
        path = nx.shortest_path(G, row["node"], nearest_main_node, weight="custom_weight")
        edge_gdf = ox.graph_to_gdfs(G.subgraph(path), nodes=False)
        merged = linemerge(list(edge_gdf.geometry))
        feeder_routes.append({"geometry": merged, "route": row["zone"] + 1})
    except Exception as e:
        print(f"⚠️ Routing failed for zone {row['zone']}: {e}")

feeder_gdf = gpd.GeoDataFrame(feeder_routes, crs=PROJECTED_CRS)

# --- Load background ---
print("Loading map background...")
barangays = gpd.read_file(BARANGAY_FILE).to_crs(PROJECTED_CRS)
grid = pd.read_csv(GRID_FILE)
grid["geometry"] = grid["geometry"].apply(wkt.loads)
gdf = gpd.GeoDataFrame(grid, geometry="geometry", crs="EPSG:4326").to_crs(PROJECTED_CRS)

# --- Plot ---
print("Plotting transit network...")
fig, ax = plt.subplots(1, 1, figsize=(16, 12))

gdf[gdf["pop_density"] > 0].plot(ax=ax, column="pop_density", cmap="YlGnBu", alpha=0.5, linewidth=0)
barangays.boundary.plot(ax=ax, color="gray", linewidth=0.5)
major_roads.plot(ax=ax, color="lightgray", linewidth=0.5)

main_route_gdf.plot(ax=ax, color="blue", linewidth=2, label="Main Route")
feeder_gdf.plot(ax=ax, column="route", cmap="tab10", linewidth=1, legend=True)

zone_anchors.set_geometry("centroid").plot(ax=ax, color="green", markersize=25, label="Zone Anchors")
top_stops.set_geometry("centroid").plot(ax=ax, color="red", markersize=30, label="Top Main Stops")
served_stops[~served_stops.index.isin(top_stops.index)].set_geometry("centroid").plot(ax=ax, color="yellow", markersize=20, label="Other Served Stops")
unserved_stops.set_geometry("centroid").plot(ax=ax, color="orange", markersize=20, label="Unassigned Stops")

ax.set_title("Transit Network with Main Route and Zonal Feeders", fontsize=14)
ax.axis("off")
plt.legend()
plt.tight_layout()
plt.show()

# --- Save (optional) ---
# main_route_gdf.to_file("main_transit_route.geojson", driver="GeoJSON")
# feeder_gdf.to_file("feeder_routes.geojson", driver="GeoJSON")