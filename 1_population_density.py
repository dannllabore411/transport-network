import geopandas as gpd
import pandas as pd
import numpy as np
import osmnx as ox
import matplotlib.pyplot as plt
from shapely.geometry import box

# Config
CITY_NAME = "General Santos, Philippines"
GRID_SIZE = 0.002  # ~200m resolution in degrees

# City Limits
city_boundary = ox.geocode_to_gdf(CITY_NAME)
city_polygon = city_boundary.geometry.union_all()

# Load barangay boundaries
barangays = gpd.read_file("barangays-municity-ph126303000.0.1.json")
barangays = barangays.to_crs(city_boundary.crs)

# Scrape population data
url = "https://www.citypopulation.de/en/philippines/generalsantos/"
tables = pd.read_html(url)
pop_df = tables[0]

# Fix formats
pop_df = pop_df.rename(columns={"Name": "ADM4_EN", "Population Census 2020-05-01": "population"})
pop_df["ADM4_EN"] = pop_df["ADM4_EN"].str.strip()
rename_map = {
    "Dadiangas East": "Dadiangas East (Pob.)",
}
pop_df["ADM4_EN"] = pop_df["ADM4_EN"].replace(rename_map)

# Merge with GeoDataFrame
barangays = barangays.merge(pop_df[["ADM4_EN", "population"]], on="ADM4_EN", how="left")
barangays["population"] = barangays["population"].fillna(0)

# Generate grid over city area
minx, miny, maxx, maxy = city_polygon.bounds
cols = np.arange(minx, maxx, GRID_SIZE)
rows = np.arange(miny, maxy, GRID_SIZE)
grid_cells = [box(x, y, x + GRID_SIZE, y + GRID_SIZE) for x in cols for y in rows]
grid = gpd.GeoDataFrame(geometry=grid_cells, crs=city_boundary.crs)
grid = grid[grid.geometry.intersects(city_polygon)].reset_index(drop=True)
# Join grid with barangays
grid = gpd.sjoin(grid, barangays[["geometry", "population"]], how="left", predicate="intersects")
grid["population"] = grid["population"].fillna(0)

# Count buildings in each grid cell
print("Downloading building footprints...")
tags = {'building': True}
buildings = ox.features_from_place(CITY_NAME, tags=tags)
buildings = buildings.to_crs(city_boundary.crs)

grid["building_count"] = 0
for i, cell in grid.iterrows():
    clipped = buildings[buildings.intersects(cell.geometry)]
    grid.at[i, "building_count"] = len(clipped)

# Redistribute population by building count
grid["weighted_pop"] = 0.0
for b_id, group in grid.groupby("index_right"):
    total_score = group["building_count"].sum()
    if total_score == 0:
        continue
    total_pop = barangays.loc[b_id, "population"]
    for i, row in group.iterrows():
        grid.at[i, "weighted_pop"] = total_pop * (row["building_count"] / total_score)

# Compute population density (people per km²)
grid_area_km2 = (GRID_SIZE * 111) ** 2  # degrees to km²
grid["pop_density"] = grid["weighted_pop"] / grid_area_km2
grid.to_csv('population_density_data.csv')

# Plot
fig, ax = plt.subplots(figsize=(10, 10))
grid.plot(column="pop_density", cmap="viridis", linewidth=0, ax=ax, legend=True)
city_boundary.boundary.plot(ax=ax, color="black", linewidth=1)
plt.title(f"Estimated Population Density Grid ({CITY_NAME}) - by Building Count", fontsize=14)
plt.axis("off")
plt.show()