# key_graphs.py

import geopandas as gpd
import pandas as pd
import numpy as np
import osmnx as ox
import matplotlib.pyplot as plt
import networkx as nx
from shapely import wkt

# --- Step 1: Load population grid data from CSV ---
print("Loading grid from CSV...")
grid = pd.read_csv("population_density_data.csv")

# Convert geometry column from WKT string to actual geometry
grid['geometry'] = grid['geometry'].apply(wkt.loads)
grid = gpd.GeoDataFrame(grid, geometry='geometry', crs="EPSG:4326")  # assuming WGS84

# Project to UTM for accurate distance-based calculations
projected_crs = "EPSG:32651"  # UTM Zone 51N for General Santos
grid = grid.to_crs(projected_crs)

# --- Step 2: Download and process road network ---
print("Downloading road network from OSM...")
G = ox.graph_from_place("General Santos, Philippines", network_type="all", simplify=True)
G_proj = ox.project_graph(G, to_crs=projected_crs)

# --- Step 3: Compute max node degree per grid cell ---
print("Computing max node degree...")
nodes = ox.graph_to_gdfs(G_proj, edges=False)
nodes["degree"] = [val for _, val in dict(G_proj.degree()).items()]

# Spatial join: find highest-degree node per cell
joined = gpd.sjoin(grid, nodes[["geometry", "degree"]], how="left", predicate="contains")
# Create a temporary column for group assignment
joined['grid_index'] = joined.index

# Aggregate max node degree per original grid index
max_degree = joined.groupby('grid_index')["degree"].max()

# Assign back to grid
grid["max_node_degree"] = grid.index.map(max_degree).fillna(0)

# --- Step 4: Plot features ---
print("Plotting...")
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

grid.plot(column="pop_density", cmap="viridis", ax=axs[0], legend=True)
axs[0].set_title("Population Density (people/km²)")
axs[0].axis("off")

grid.plot(column="building_count", cmap="plasma", ax=axs[1], legend=True)
axs[1].set_title("Building Count per Grid Cell")
axs[1].axis("off")

grid.plot(column="max_node_degree", cmap="inferno", ax=axs[2], legend=True)
axs[2].set_title("Max Node Degree")
axs[2].axis("off")

plt.tight_layout()
plt.show()

# --- Optional: Save enriched grid ---
# grid.to_file("transit_feature_grid.geojson", driver="GeoJSON")
grid[['pop_density', 'building_count', 'max_node_degree']].to_csv("transit_features.csv", index=False)
