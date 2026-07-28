# transit_stops.py

import geopandas as gpd
import pandas as pd
import numpy as np
import osmnx as ox
import matplotlib.pyplot as plt
import networkx as nx
from shapely import wkt
from shapely.geometry import Point
from shapely.strtree import STRtree
from sklearn.preprocessing import MinMaxScaler
import matplotlib.patches as mpatches

# Config
CITY_NAME = "General Santos, Philippines"
CSV_PATH = "population_density_data.csv"
BARANGAY_JSON = "barangays-municity-ph126303000.0.1.json"
PROJECTED_CRS = "EPSG:32651"  # UTM Zone 51N
DIST_THRESHOLD = 1000
TOP_N_CANDIDATES = 200

# Load grid
print("Loading population-weighted grid")
grid = pd.read_csv(CSV_PATH)
grid["geometry"] = grid["geometry"].apply(wkt.loads)
grid = gpd.GeoDataFrame(grid, geometry="geometry", crs="EPSG:4326").to_crs(PROJECTED_CRS)

# Load road network + node degree
print("Downloading road network")
G = ox.graph_from_place(CITY_NAME, network_type="all", simplify=True)
G_proj = ox.project_graph(G, to_crs=PROJECTED_CRS)
nodes = ox.graph_to_gdfs(G_proj, edges=False)
nodes["degree"] = [val for _, val in dict(G_proj.degree()).items()]
nodes = nodes.to_crs(grid.crs)

print("Calculating max node degree per cell")
joined = gpd.sjoin(grid, nodes[["geometry", "degree"]], how="left", predicate="contains")
joined["grid_index"] = joined.index
max_degree = joined.groupby("grid_index")["degree"].max()
grid["max_node_degree"] = grid.index.map(max_degree).fillna(0)

# Load POIs + score
print("Downloading POIs from OSM")
tags = {
    "amenity": True,
    "office": True,
    "building": True,
    "shop": True,
    "landuse": True,
    "man_made": True
}
pois = ox.features_from_place(CITY_NAME, tags=tags).to_crs(grid.crs)

poi_weights = {
    'school': 1.5,
    'college': 1.8,
    'university': 2.0,
    'hospital': 2.5,
    'clinic': 1.5,
    'townhall': 2.0,
    'government': 2.0,
    'mall': 2.5,
    'supermarket': 1.5,
    'industrial': 1.5  # For landuse/building fallback
}

def compute_poi_score(row):
    score = 0

    # Standard amenity/shop/office-based POIs
    if pd.notnull(row.get("amenity")) and row["amenity"] in poi_weights:
        score = poi_weights[row["amenity"]]
    elif pd.notnull(row.get("office")) and row["office"] in poi_weights:
        score = poi_weights[row["office"]]
    elif pd.notnull(row.get("shop")) and row["shop"] in poi_weights:
        score = poi_weights[row["shop"]]

    # Building fallback for schools/public buildings
    if pd.notnull(row.get("building")) and row["building"] in ["public", "college", "university"]:
        score = max(score, 1.5)

    # Mall detection: co-located commercial building + fast food/bank
    is_mall_like = (
        row.get("building") in ["retail", "commercial"] and
        row.get("amenity") in ["fast_food", "bank", "restaurant"]
    )
    if is_mall_like:
        score = max(score, 2.5)

    # Industrial detection
    if row.get("landuse") == "industrial" or row.get("building") == "industrial":
        score = max(score, 1.5)
    if row.get("man_made") == "works":
        score = max(score, 1.5)

    return score

pois["poi_score"] = pois.apply(compute_poi_score, axis=1)
pois = pois[pois["poi_score"] > 0]

# Aggregate POI score per grid cell
print("Scoring POIs per grid cell")
poi_joined = gpd.sjoin(grid, pois[["geometry", "poi_score"]], how="left", predicate="contains")
poi_sums = poi_joined.groupby(poi_joined.index)["poi_score"].sum()
grid["poi_score"] = grid.index.map(poi_sums).fillna(0)

# Normalize + scoring
print("Calculating weighted scores")
scaler = MinMaxScaler()
grid[["pop_norm", "bldg_norm", "node_norm", "poi_norm"]] = scaler.fit_transform(
    grid[["pop_density", "building_count", "max_node_degree", "poi_score"]]
)

grid["weighted_score"] = (
    0.4 * grid["pop_norm"] +
    0.2 * grid["bldg_norm"] +
    0.1 * grid["node_norm"] +
    0.3 * grid["poi_norm"]
)

# Select spaced top cells
print("Filtering spaced top cells...")
grid["centroid"] = grid.geometry.centroid
top_candidates = grid.sort_values("weighted_score", ascending=False).head(TOP_N_CANDIDATES)

selected = []
for _, row in top_candidates.iterrows():
    pt = row["centroid"]
    if all(pt.distance(other["centroid"]) > DIST_THRESHOLD for other in selected):
        selected.append(row)

final_stops = gpd.GeoDataFrame(selected, crs=grid.crs)

# Snap to nearest major/semi-major intersections
print("Snapping selected stops to intersections...")

# Get high-degree intersections from major/semi-major roads
edges = ox.graph_to_gdfs(G_proj, nodes=False)
major_classes = ["motorway", "trunk", "primary", "secondary", "tertiary"]
major_edges = edges[edges["highway"].isin(major_classes)]

major_nodes = ox.graph_to_gdfs(G_proj, edges=False)
major_nodes["degree"] = [val for _, val in dict(G_proj.degree()).items()]
major_nodes = major_nodes[major_nodes["degree"] >= 3].to_crs(grid.crs)

print("Snapping selected stops to intersections...")

# Get major/semi-major road intersections (degree >= 3)
major_nodes = ox.graph_to_gdfs(G_proj, edges=False)
major_nodes["degree"] = [val for _, val in dict(G_proj.degree()).items()]
major_nodes = major_nodes[major_nodes["degree"] >= 3].to_crs(grid.crs)

# Build spatial index of intersection points
geoms = list(major_nodes.geometry)
tree = STRtree(geoms)

# Snap each centroid to nearest geometry in STRtree
snapped = []
max_snap_dist = 500  # maximum distance in meters for snapping

for pt in final_stops["centroid"]:
    try:
        idx = tree.nearest(pt)
        nearest_geom = geoms[idx]
        if pt.distance(nearest_geom) <= max_snap_dist:
            snapped.append(nearest_geom)
        else:
            snapped.append(None)  # too far, discard
    except Exception:
        snapped.append(None)

# Assign snapped points as final geometry (safe method)
final_stops = final_stops.copy()
final_stops["snapped"] = snapped
final_stops = final_stops[final_stops["snapped"].notnull()].copy()
# Drop old geometry column first (if needed), then set snapped as geometry
final_stops = final_stops.drop(columns=["geometry"], errors="ignore")
final_stops = final_stops.set_geometry("snapped")
final_stops = final_stops.set_crs(grid.crs)
final_stops = final_stops.rename_geometry("geometry")
final_stops = final_stops.drop(columns=["centroid"], errors="ignore")

# Plot final snapped stop locations
print("Plotting final snapped transit stop candidates...")

fig, ax = plt.subplots(1, 1, figsize=(16, 12))

# Background layers
grid.plot(column="weighted_score", cmap="viridis", ax=ax, legend=True)
barangays = gpd.read_file(BARANGAY_JSON).to_crs(grid.crs)
barangays.boundary.plot(ax=ax, color="black", linewidth=0.5)

# Snapped stop markers
final_stops.plot(ax=ax, color="green", markersize=30, label="Snapped Stops")

ax.set_title("Final Transit Stop Candidates", fontsize=14)
ax.axis("off")
ax.legend()
plt.tight_layout()
plt.show()

# --- Step 8: Save outputs ---
print("Saving final stop candidates...")
final_stops.to_file("transit_stops.geojson", driver="GeoJSON")
final_stops.to_csv("final_transit_stops.csv", index=False)
