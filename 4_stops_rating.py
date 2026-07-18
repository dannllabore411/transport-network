import geopandas as gpd
import matplotlib.pyplot as plt
import osmnx as ox

# --- Config ---
STOP_FILE = "transit_stops.geojson"
BARANGAY_FILE = "barangays-municity-ph126303000.0.1.json"
CITY_NAME = "General Santos, Philippines"
CRS = "EPSG:32651"
NETWORK_TYPE = "drive"

# --- Load data ---
print("Loading data...")
stops = gpd.read_file(STOP_FILE).to_crs(CRS)
barangays = gpd.read_file(BARANGAY_FILE).to_crs(CRS)

# --- Load major roads ---
print("Loading major/semi-major roads...")
G = ox.graph_from_place(CITY_NAME, network_type=NETWORK_TYPE)
G_proj = ox.project_graph(G, to_crs=CRS)
edges = ox.graph_to_gdfs(G_proj, nodes=False)

road_classes = ["motorway", "trunk", "primary", "secondary", "tertiary"]
major_roads = edges[edges["highway"].isin(road_classes)]

# --- Plot ---
print("Plotting stop scores...")
fig, ax = plt.subplots(figsize=(14, 12))

barangays.boundary.plot(ax=ax, color="gray", linewidth=0.5)
major_roads.plot(ax=ax, color="lightgray", linewidth=0.5)
stops.plot(ax=ax, column="weighted_score", cmap="YlOrRd", markersize=40, legend=True)

# Optional: annotate with score or index
for i, row in stops.iterrows():
    ax.annotate(text=f"{row['weighted_score']:.2f}", xy=(row.geometry.centroid.x, row.geometry.centroid.y),
                fontsize=7, ha="center", va="center", color="black")

ax.set_title("Transit Stop Candidates by Weighted Score", fontsize=14)
ax.axis("off")
plt.tight_layout()
plt.show()