This repository accompanies the paper "A location-based model for forecasting hierarchical road network growth", which forecasts road length by road class from socioeconomic trajectories and allocates the forecast roads spatially through radial pattern growth, hierarchical upgrading and linear growth, with case studies of Bangladesh, Brazil and Nigeria under SSP1, SSP3 and SSP5 for 2030, 2040 and 2050. It includes the MATLAB code of the spatial allocation model with the road demand of every forecast run, the Python scripts for the spatial validation against OpenStreetMap, the network reliability metrics, the growth rates and the sensitivity analyses, and the scripts that draw the figures of the paper and its Supplementary Information, with the figures they draw.

The forecast networks, the 2015 road network and the input rasters of every model run are on Zenodo, [Zenodo DOI].

## Spatial allocation model

The folder spatial_distribution holds the MATLAB code of the spatial allocation model. run_country.m runs the model for one country, any of SSP1, SSP3 and SSP5 and any of 2030, 2040 and 2050, with the road demand in data/road_demand_published.csv (road length added by class in each period, in km) and the input rasters from Zenodo.

1. Download model_inputs_<ISO3>.zip (for Brazil, model_inputs_BRA_SSP1.zip, model_inputs_BRA_SSP3.zip and model_inputs_BRA_SSP5.zip) from Zenodo and unzip it into data, which gives data/model_inputs/<ISO3>/SSP<s>/<year>/.
2. In MATLAB, change to spatial_distribution and run

        run_country('BGD')

   for all three scenarios and years, or for example run_country('NGA', 1, 2050) for one. The arguments after the country are the scenarios, the years, the inputs folder, the demand file and the output folder.

Each run starts from the network at the start of its period, which the input folder holds as <ISO3>_road_types.tif (the 2015 network for 2030, the forecast of the previous period for 2040 and 2050). The settings are those of the forecasts in the paper: distance-based selection ('dist_only'), radial growth on, points of interest off and random seed 42. The outputs are written to spatial_distribution/output_data/<ISO3>/SSP<s>/<year>/, with the forecast network in <ISO3>_all_final_roads.tif. The model needs MATLAB with the Mapping Toolbox (geotiffread, geotiffwrite) and the Statistics and Machine Learning Toolbox (dbscan).

The forecasts on Zenodo were produced by a Python implementation of this model with the same inputs, settings and seed. Because the two implementations draw different random sequences, a MATLAB run allocates the same road demand but may place some roads in different cells.

## Figures

The folder figures/ holds the script behind each figure of the paper and its Supplementary Information, one folder per figure, with the figure it draws as PNG (600 dpi) and PDF. The folder data/figures holds the inputs the scripts read. Run a script from the repository folder without arguments, for example

    python figures/Fig4_upgrade_locations/make_fig4_upgrade_locations.py

Each script writes the figure as PNG, PDF and SVG to its own folder, or to the folder given by --out. The input arguments, listed by --help, point a script to other copies of the inputs. The scripts of Figure 5 and Supplementary Fig. 6 draw SSP1 by default and take --ssp 3 or --ssp 5 for the other scenarios. They do not overwrite an existing figure and add _r1, _r2 and so on to the name of a new one. figures/Fig5_road_growth/test_road_lines.py tests the tracing of road lines from the forecast rasters on synthetic roads. Supplementary Figs 1 to 3 are not script-made.

| Figure | Content | Script (in figures/) | Inputs (in data/figures unless noted) |
|---|---|---|---|
| Figure 1 | Road growth modeling framework | Fig1_framework/make_fig1_framework.py | None |
| Figure 2 | Spatial allocation steps | Fig2_allocation_steps/make_fig2_allocation_steps.py | figures/Fig2_allocation_steps/thumbnails, BGD_drivers_osm15, roads |
| Figure 3 | Radial pattern detection and growth | Fig3_pattern_radial_schematic/make_fig3_pattern_radial_schematic.py | None |
| Figure 4 | Upgrade location and length, Nigeria, 2050, SSP1 | Fig4_upgrade_locations/make_fig4_upgrade_locations.py | NGA_drivers, NGA_forecast_SSP1_2050, roads, naturalearth |
| Figure 5 | New and upgraded roads to 2050, Bangladesh, Nigeria and Brazil, SSP1 | Fig5_road_growth/make_fig5_road_growth.py | fig5_forecast, fig5_roads_2015, roads, naturalearth |
| Supplementary Fig. 4 | Predicted road length and residuals | SFig04_residuals/make_sfig04_residuals.py | figures/RoadForecast_urban.xlsx |
| Supplementary Fig. 5 | Observed and predicted road length, five panels in one row | SFig05_obs_vs_pred/make_sfig05_regression_one_row.py | figures/RoadForecast_urban.xlsx |
| Supplementary Fig. 5 | Observed and predicted road length, larger panels with country labels | SFig05_obs_vs_pred/make_sfig05_regression_country_labels.py | figures/RoadForecast_urban.xlsx |
| Supplementary Fig. 6 | New roads to 2050 by class, Bangladesh, Nigeria and Brazil, SSP1 | SFig06_new_roads_by_class/make_sfig06_new_roads_by_class.py | fig5_forecast, fig5_roads_2015, roads, naturalearth |
| Supplementary Fig. 7 | Link consistency, Dhaka, Chattogram and Abuja | SFig07_link_consistency/make_sfig07_link_consistency.py | SFig07_district_extracts, roads |
| Supplementary Fig. 8 | OpenStreetMap baseline coverage, Bangladesh | SFig08_baseline_coverage/make_sfig08_baseline_coverage.py | roads, osm, naturalearth |
| Supplementary Fig. 9 | Precision, recall and F1 | SFig09_precision_recall_F1/make_sfig09_precision_recall_f1.py | validation/metrics |
| Supplementary Fig. 10 | Spatial classification of all roads, Bangladesh | SFig10_BGD_spatial_classification/make_sfig10_allroads_spatial.py | roads, osm, BGD_forecast_osm15_baseline, naturalearth |
| Supplementary Fig. 11 | Commission and omission case studies | SFig11_commission_omission/make_sfig11_case_studies.py | validation |
| Supplementary Fig. 12 | Padma Bridge | SFig12_Padma_Bridge/make_sfig12_padma.py | roads, validation, osm |
| Supplementary Fig. 13 | Upgrade-location sensitivity | SFig13_upgrade_sensitivity/make_sfig13_upgrade_sensitivity.py | sensitivity/upgrade_locations, roads, naturalearth |
| Supplementary Fig. 14 | Urban-cluster sensitivity | SFig14_urban_cluster_sensitivity/make_sfig14_urban_cluster_sensitivity.py | sensitivity/urban_clusters, roads, naturalearth |

The scripts were run with Python 3.13, numpy 2.4, scipy 1.18, pandas 2.3, geopandas 1.2, pyogrio 0.13, shapely 2.1, pyproj 3.8, rasterio 1.5, matplotlib 3.10 and scikit-image 0.26. requirements.txt lists the packages.

## Figure inputs

| Folder in data/figures | Contents |
|---|---|
| roads | District boundaries of Bangladesh and Nigeria (bgd_dist, nga_dist). The 2030 SSP1 forecast raster of Bangladesh, which the scripts use as the 300 m reference grid. The GRIP 2015 road network of Bangladesh (bgd_roads_2015, with the road-type attribute GP_RTP only). |
| fig5_forecast | New and upgraded roads from 2015 to 2050 of Bangladesh, Nigeria and Brazil under SSP1, SSP3 and SSP5 (<ISO3>_SSP<s>_since2015_2050_new.tif and _upgraded.tif, band 5 holds the road class), as on Zenodo. |
| fig5_roads_2015 | GRIP 2015 road network of Bangladesh, Nigeria and Brazil in the map projection of each country, with the model road class k (1 main, 2 secondary, 3 tertiary, 4 local), as zipped FlatGeobuf parts (<ISO3>_roads_2015.part<n>.fgb.zip). |
| SFig07_district_extracts | The features of the GRIP 2015 network and the 2050 forecast networks (SSP1, SSP3 and SSP5) of Bangladesh and Nigeria that intersect Dhaka, Chattogram and Abuja Municipal. These are the features the script keeps before it clips the networks to the districts. |
| osm | Geofabrik extracts of Bangladesh of 1 January 2015 and 6 June 2026. Roads of the classes used in the validation, with the class attribute (fclass) only, and the national boundary from the 2026 administrative areas. |
| naturalearth | Natural Earth 1:10m admin-0 countries, Bangladesh and Nigeria (used by Figure 4 and Supplementary Figs 8, 10 and 13) and Bangladesh, Nigeria and Brazil (used by Figure 5 and Supplementary Fig. 6), to trim the district boundaries to the coastline and to outline Brazil. |
| BGD_drivers_osm15 | Driver rasters of Bangladesh used in the run from the OSM 2015 baseline: urban extent, population density, road density, distance to roads, distance to urban areas and building height. |
| BGD_forecast_osm15_baseline | Forecast network of Bangladesh from the OSM 2015 baseline. |
| NGA_drivers | Driver rasters of Nigeria: urban extent, road density and distance to urban areas, and the 2020 SSP1 population density. |
| NGA_forecast_SSP1_2050 | Forecast network of Nigeria for 2050 under SSP1, all roads and upgraded roads (band 5 holds the road class). |
| validation | Classification raster of Bangladesh at 300 m (true positives, false positives and false negatives), and the precision, recall and F1 tables of Bangladesh, Brazil and Nigeria (metrics). |
| sensitivity | Upgrade locations found at 100, 300 and 1,000 m (upgrade_locations), and urban clusters at 30, 100, 300, 500 and 1,000 m with their metrics and agreement tables (urban_clusters). |

Large vector layers are stored as zipped files. The GRIP 2015 networks and the 2026 OpenStreetMap roads are split by rows into parts below 25 MB (<name>.part1.shp.zip, <name>.part2.shp.zip, ... and <ISO3>_roads_2015.part<n>.fgb.zip), and the scripts read the parts in order and join them. Only the attributes the scripts use are kept. Geometries and feature order are unchanged, and the scripts reproduce the published figures from these files.

## Analyses

| Folder | Contents |
|---|---|
| validation | Spatial validation of the forecast networks against OpenStreetMap for Bangladesh, Brazil and Nigeria (scripts 01 to 19): precision, recall and F1 from two baselines, locational accuracy, convergence points, the no-growth benchmarks and the exports of the validation layers. |
| reliability | Network reliability metrics of the observed and forecast networks and their comparison. |
| growth_rates | Annual growth rates and class shares of road length by country, SSP and year, from a table of road lengths by class. |
| sensitivity | Sensitivity of the upgrade locations and of the urban clusters to the grid resolution. |

The analysis scripts read their data from the folder data, or from the folder named by the environment variable ROAD_DATA, and their arguments point them to other copies.

## Sources

OpenStreetMap data © OpenStreetMap contributors, available under the Open Database License (ODbL) 1.0, from Geofabrik (https://download.geofabrik.de). GRIP road network from the Global Roads Inventory Project, GRIP4 (Meijer et al. 2018, Environmental Research Letters 13, 064006, https://www.globio.info/download-grip-dataset), available under the Creative Commons Attribution 4.0 International License (CC BY 4.0). Country boundaries from Natural Earth (https://www.naturalearthdata.com), which is in the public domain. The data in data/figures keep the licenses of their sources.
