import pandas as pd
import geopandas as gpd
from shapely import wkt
import matplotlib.pyplot as plt
# 1. Load data
csv_path = "mds_V2_5.372k.csv"
df = pd.read_csv(csv_path)
df['geom'] = df['geom'].apply(wkt.loads)
gdf = gpd.GeoDataFrame(df, geometry='geom')
# 2. Isolate a plan
sample_plan_id = 7988
apartment_gdf = gdf[gdf['plan_id'] == sample_plan_id]
rooms_gdf = apartment_gdf[apartment_gdf['entity_type'] == 'area']
# Merge the rooms to create the outline
# 1. Buffer outward by 30 centimeters (MSD uses meters as units)
# 2. Cleanly unify them into one solid geometry
# 3. Buffer back inward by 30 centimeters to restore the original scale
wall_bridge_distance = 0.3
solid_outline_geom = rooms_gdf.geometry.buffer(wall_bridge_distance).unary_union.buffer(-wall_bridge_distance)
# Convert the resulting Shapely geometry back into a GeoDataFrame for plotting
outline_gdf = gpd.GeoDataFrame(geometry=[solid_outline_geom], crs=rooms_gdf.crs)
# -------------------------------------
# 3. Plot side-by-side again
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
# Left Plot: Condition
outline_gdf.plot(ax=ax1, facecolor='#f4f4f4', edgecolor='black', linewidth=3)
ax1.set_title("Input: Clean Apartment Outline", fontsize=14, fontweight='bold')
ax1.axis('equal')
ax1.axis('off')
# Right Plot: Target
rooms_gdf.plot(ax=ax2, cmap='Set3', edgecolor='white', linewidth=1.5)
outline_gdf.plot(ax=ax2, facecolor='none', edgecolor='black', linewidth=2, alpha=0.4)
ax2.set_title("Target: Generated Rooms", fontsize=14, fontweight='bold')
ax2.axis('equal')
ax2.axis('off')
plt.suptitle(f"Generative Task Data Pairing (Plan ID: {sample_plan_id})", fontsize=16)
plt.tight_layout()
plt.show()