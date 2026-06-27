
# EDA report -- mds_V2_5.372k.csv (MSD / Swiss Dwellings)


## 1. Dataset overview

- File: `mds_V2_5.372k.csv`  (401.4 MB on disk)
- Rows analysed: **1,086,846**  (1 row = 1 geometric entity)
- Structural columns: **14**  (`geom` excluded from this load)
- In-memory size of loaded columns: 456.2 MB

|  | dtype | n_unique | n_missing | pct_missing |
| --- | --- | --- | --- | --- |
| apartment_id | object | 18269 | 193367 | 17.790 |
| site_id | int64 | 1130 | 0 | 0.000 |
| building_id | int64 | 2550 | 0 | 0.000 |
| plan_id | int64 | 5372 | 0 | 0.000 |
| floor_id | int64 | 5372 | 0 | 0.000 |
| unit_id | float64 | 18902 | 193367 | 17.790 |
| area_id | float64 | 203330 | 883516 | 81.290 |
| unit_usage | object | 2 | 0 | 0.000 |
| entity_type | object | 3 | 0 | 0.000 |
| entity_subtype | object | 22 | 0 | 0.000 |
| elevation | float64 | 163 | 0 | 0.000 |
| height | float64 | 80 | 0 | 0.000 |
| zoning | object | 8 | 0 | 0.000 |
| roomtype | object | 13 | 0 | 0.000 |


## 2. Missing values are structural

Missingness in this table is not noise -- it encodes the schema:

**`area_id` is NULL fraction by `entity_type`** (=> set only for rooms/areas):
| entity_type | frac_area_id_null |
| --- | --- |
| area | 0.000 |
| opening | 1.000 |
| separator | 1.000 |

**`apartment_id` presence vs `unit_usage`** (=> PUBLIC space has no dwelling id):
| unit_usage | apartment_id present | apartment_id NULL |
| --- | --- | --- |
| PUBLIC | 0 | 193367 |
| RESIDENTIAL | 893479 | 0 |


## 3. Categorical columns

**`unit_usage`** (residential dwelling vs shared/public space):
| unit_usage | count | pct |
| --- | --- | --- |
| RESIDENTIAL | 893479 | 82.210 |
| PUBLIC | 193367 | 17.790 |

**`entity_type`** (what each row geometrically is):
| entity_type | count | pct |
| --- | --- | --- |
| separator | 602196 | 55.410 |
| opening | 281320 | 25.880 |
| area | 203330 | 18.710 |

**`entity_subtype` per `entity_type`** (room vocabulary vs wall/opening vocabulary):
- `area`: SHAFT (37,989), ROOM (34,379), BATHROOM (28,276), CORRIDOR (25,701), BALCONY (19,484), KITCHEN (17,991), LIVING_DINING (10,382), BEDROOM (8,205), STOREROOM (7,367), STAIRCASE (4,847), ELEVATOR (4,194), LIVING_ROOM (3,110), DINING (540), VOID (520), TERRACE (299), CORRIDORS_AND_HALLS (37), KITCHEN_DINING (9)
- `separator`: WALL (594,554), COLUMN (7,642)
- `opening`: DOOR (135,619), WINDOW (103,173), ENTRANCE_DOOR (42,528)

**`roomtype`** (human label; non-area rows carry Structure/Door/Window):
| roomtype | count | pct |
| --- | --- | --- |
| Structure | 640185 | 58.900 |
| Door | 135619 | 12.480 |
| Window | 103173 | 9.490 |
| Bedroom | 42584 | 3.920 |
| Entrance Door | 42528 | 3.910 |
| Bathroom | 28276 | 2.600 |
| Corridor | 25738 | 2.370 |
| Balcony | 19783 | 1.820 |
| Kitchen | 18000 | 1.660 |
| Livingroom | 13492 | 1.240 |
| Stairs | 9561 | 0.880 |
| Storeroom | 7367 | 0.680 |
| Dining | 540 | 0.050 |

**`zoning`** (private zones Zone1-4 for rooms; Structure/Door/Window else):
| zoning | count | pct |
| --- | --- | --- |
| Structure | 640185 | 58.900 |
| Door | 135619 | 12.480 |
| Window | 103173 | 9.490 |
| Zone2 | 57770 | 5.320 |
| Zone3 | 45204 | 4.160 |
| Zone1 | 42584 | 3.920 |
| Entrance Door | 42528 | 3.910 |
| Zone4 | 19783 | 1.820 |


## 4. ID hierarchy and nesting

**Cardinality of each level** (coarse -> fine):
|  | n_unique |
| --- | --- |
| site_id | 1130 |
| building_id | 2550 |
| plan_id | 5372 |
| floor_id | 5372 |
| apartment_id | 18269 |
| unit_id | 18902 |
| area_id | 203330 |

**Branching factor** -- distinct children per parent (mean / median / max):
| relation | mean | median | max |
| --- | --- | --- | --- |
| site_id -> building_id | 2.260 | 1 | 25 |
| building_id -> plan_id | 2.110 | 2 | 10 |
| plan_id -> floor_id | 1.000 | 1 | 1 |
| plan_id -> apartment_id | 3.520 | 3 | 92 |
| apartment_id -> unit_id | 1.030 | 1 | 9 |
| apartment_id -> area_id | 9.630 | 9 | 69 |

Notes: `plan_id -> floor_id` is 1:1 (a plan == a floor); `apartment_id -> unit_id` ~1:1 (treat apartment as the dwelling); `plan_id -> apartment_id` shows multiple dwellings share one floor plan.

## 5. Per-apartment composition (the training sample)

**Entities per apartment** (rooms + walls + openings):
|  | value |
| --- | --- |
| count | 18269.000 |
| mean | 48.907 |
| std | 17.331 |
| min | 7.000 |
| 25% | 38.000 |
| 50% | 47.000 |
| 75% | 58.000 |
| max | 349.000 |

**Rooms (areas) per apartment** -- target token-set size:
|  | value |
| --- | --- |
| count | 18269.000 |
| mean | 9.630 |
| std | 3.288 |
| min | 1.000 |
| 25% | 8.000 |
| 50% | 9.000 |
| 75% | 11.000 |
| max | 69.000 |

**Most common room types** (`entity_subtype` for areas):
| entity_subtype | count | pct |
| --- | --- | --- |
| ROOM | 33719 | 19.170 |
| SHAFT | 27827 | 15.820 |
| BATHROOM | 27745 | 15.770 |
| CORRIDOR | 21513 | 12.230 |
| BALCONY | 19182 | 10.900 |
| KITCHEN | 17800 | 10.120 |
| LIVING_DINING | 10279 | 5.840 |
| BEDROOM | 8175 | 4.650 |
| STOREROOM | 5567 | 3.160 |
| LIVING_ROOM | 3102 | 1.760 |
| DINING | 535 | 0.300 |
| TERRACE | 294 | 0.170 |
| VOID | 151 | 0.090 |
| STAIRCASE | 25 | 0.010 |
| KITCHEN_DINING | 8 | 0.000 |


## 6. Room geometry (sampled WKT polygons)

Parsing 40,000 area polygons (of 203,330 in the loaded rows).
**Per-room geometry summary:**
|  | area_m2 | width_m | height_m | aspect | n_vertices |
| --- | --- | --- | --- | --- | --- |
| count | 40000.000 | 40000.000 | 40000.000 | 40000.000 | 40000.000 |
| mean | 9.390 | 3.610 | 3.610 | 1.550 | 7.990 |
| std | 9.620 | 2.290 | 2.270 | 1.340 | 5.850 |
| min | 0.010 | 0.070 | 0.070 | 1.000 | 3.000 |
| 25% | 2.480 | 2.000 | 1.980 | 1.080 | 4.000 |
| 50% | 7.150 | 3.370 | 3.410 | 1.220 | 6.000 |
| 75% | 13.820 | 4.930 | 4.910 | 1.510 | 10.000 |
| max | 208.180 | 47.250 | 50.150 | 27.320 | 134.000 |

**Median area (m^2) by room type** (largest first):
| subtype | count | median | mean |
| --- | --- | --- | --- |
| TERRACE | 37 | 34.500 | 50.540 |
| LIVING_DINING | 2205 | 30.040 | 30.240 |
| CORRIDORS_AND_HALLS | 4 | 28.620 | 23.850 |
| KITCHEN_DINING | 7 | 23.670 | 32.530 |
| LIVING_ROOM | 540 | 20.020 | 22.500 |
| ROOM | 6134 | 14.350 | 15.030 |
| BEDROOM | 2173 | 14.150 | 14.320 |
| STAIRCASE | 1047 | 12.340 | 13.560 |
| DINING | 115 | 10.560 | 12.170 |
| KITCHEN | 3426 | 8.380 | 8.940 |
| BALCONY | 4050 | 8.050 | 10.490 |
| CORRIDOR | 4983 | 7.740 | 9.560 |
| BATHROOM | 5607 | 3.910 | 3.910 |
| ELEVATOR | 918 | 2.570 | 2.460 |
| STOREROOM | 1523 | 2.550 | 4.890 |
| VOID | 93 | 1.120 | 6.490 |
| SHAFT | 7138 | 0.240 | 0.360 |

Geometry types present: ['Polygon'].  Rooms are not all axis-aligned rectangles -- median vertex count 6, so the decoder must tolerate polygons, not just boxes.

## 7. Vertical attributes (elevation / ceiling height)

`elevation` = base height of the entity above ground (proxy for floor level); `height` = ceiling/entity height in metres.
|  | elevation | height |
| --- | --- | --- |
| count | 1086846.000 | 1086846.000 |
| mean | 5.500 | 2.430 |
| std | 6.080 | 0.280 |
| min | -2.900 | 0.550 |
| 25% | 0.500 | 2.000 |
| 50% | 2.900 | 2.600 |
| 75% | 8.700 | 2.600 |
| max | 57.740 | 3.130 |


## 8. Figures

Saved figures:
- `eda/fig_entity_type.png`
- `eda/fig_rooms_per_apartment.png`
- `eda/fig_room_types.png`
- `eda/fig_geometry.png`

## 9. Takeaways for OutlineFlow

1. **Filter early.** Keep `entity_type == 'area'` for rooms; walls
   (`separator`) and openings (`opening`) are optional structure that
   can be ignored for the MVP token set.
2. **Group by `apartment_id`** to form one training sample; drop rows
   where it is NULL (those are PUBLIC/shared space, not a dwelling).
3. **Outline = union of the apartment's area polygons**
   (`shapely.unary_union`); rooms = the individual `area_id` polygons
   with their `entity_subtype`/`roomtype` label.
4. **Token-set size is small and bounded** (~10 rooms median, ~30 max)
   -- a set-Transformer over <=~32 tokens is sufficient.
5. **Rooms are polygons, not always rectangles** -- the geometry head /
   decoder must handle non-axis-aligned, multi-vertex shapes.
6. **Normalise per apartment** (translate to centroid, scale by bbox)
   before training; raw coordinates span tens of metres with arbitrary
   offsets.
