# transport-network
This project builds a data-driven prototype for a public transit network in General Santos City, Philippines. Using open geospatial data, it identifies high-demand locations through gridded population density and road accessibility, computes a main route connecting the top-scoring stops via major roads, and visualizes remaining stops for manual feeder planning. The workflow combines population estimation, custom route scoring, and cost-optimized pathfinding, all within a reproducible Python pipeline.

## 1. Data Collection & Cleaning
* Scraped barangay-level population data from CityPopulation.de
* Loaded and aligned spatial boundaries (GeoJSON) and OpenStreetMap road data (via osmnx)
* Standardized barangay names and resolved spatial mismatches

## 2. Geospatial Feature Engineering
* Generated a 200m-resolution grid over the city area
* Distributed barangay population across grid cells using total road length per cell as a weighting factor
* Computed:
Road access (sum of road lengths)
Connectivity (maximum node degree from nearby OSM intersections)

## 3. Scoring & Stop Selection
* Combined metrics into a composite weighted stop score
* Selected top 5 scoring cells as main stops
* Included additional stops within an 800m buffer for full coverage along the main corridor

## 4. Routing & Optimization
* Constructed a road network graph (networkx) with edge weights defined as:
weighted_cost = road_length × road_type_factor
where road_type_factor penalizes minor roads and favors motorway, primary, etc.
* Applied Dijkstra’s algorithm to connect main stops via cost-optimized paths

## 5. Visualization
* Mapped:
Population density heatmap
Major/semi-major roads (for routing context)
Main route (blue), top stops (red), and remaining scored stops (orange)
Ensured that all served stops within route buffers are displayed, even if not explicitly routed
