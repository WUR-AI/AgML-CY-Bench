# Data sources selection
## Weather variables and moisture indicators
| Source | Considered variables | Temporal frequency | Spatial resolution | Caveats |
|--------|--------------------|------------------|------------------|---------|
| **[AgERA5](https://cds.climate.copernicus.eu/cdsapp#!/dataset/sis-agrometeorological-indicators?tab=overview)** | temp_2m, prec, rad, wspd_10m, relh_2m, et0, vpd | Daily | 0.1° | — |
| **[GLDAS](https://ldas.gsfc.nasa.gov/gldas/model-output)** | ssm, rsm | Daily | 0.25° | Latency of 2–6 months |
| [FAO-AQUASTAT](https://data.apps.fao.org/static/data/index.html?prefix=static%2Fdata%2Fc3s%2FMAPSET%2FAGERA5-ET0-D) | et0 | Daily | 0.1° | Not bias-corrected |
| [ERA5](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels) | temp_2m, prec, rad, wspd_10m, relh_2m | Hourly | 0.25° | Not bias-corrected |
| [ERA5-Land](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-land?tab=overview) | temp_2m, prec, rad, wspd_10m, relh_2m | Hourly | 0.1° | Not bias-corrected |
| [W5E5](https://www.isimip.org/gettingstarted/input-data-bias-adjustment/details/114/) | temp_2m, prec, rad, wspd_10m, relh_2m | Daily | 0.5° | Low spatial resolution; not operational |
| [CPC](https://psl.noaa.gov/data/gridded/data.cpc.globaltemp.html) | prec, temp | Daily | 0.5° | Limited variables; low spatial resolution |
| [GLEAM](https://www.gleam.eu/) | et0, rsm | Daily | 0.1° | Latency up to 1 year |
| [MSWEP](https://www.gloh2o.org/mswep/) | prec | 3-hourly | 0.1° | Single-variable product |
| [ESA CCI SM](https://cds.climate.copernicus.eu/portfolio/dataset/satellite-soil-moisture) | ssm | Daily | 0.25° | Single-variable product |



We selected data from the [AgERA5](https://cds.climate.copernicus.eu/cdsapp#!/dataset/sis-agrometeorological-indicators?tab=overview) project, which provides daily weather variables at a 0.1° spatial resolution, directly relevant to crop yields, including temperature, precipitation, and solar radiation flux. AgERA5 offers agrometeorological indicators from 1979 to the present, derived from ERA5 reanalysis and tailored for agricultural studies. Its key advantage is the high-quality, operational input for numerous variables essential for crop yield forecasting, and it is freely available and well-documented on the Copernicus Climate Data Store [CDS](https://cds.climate.copernicus.eu/#!/home). 

Other datasets have limitations such as fewer variables, lower spatial resolution, or shorter temporal coverage. For instance, [MSWEP](https://www.gloh2o.org/mswep/) is a high-quality dataset but only includes precipitation. [CPC](https://psl.noaa.gov/data/gridded/data.cpc.globaltemp.html) relies on station observations but has coarser spatial resolution. [W5E5](https://www.isimip.org/gettingstarted/input-data-bias-adjustment/details/114/) is bias-adjusted based on ERA5 but lacks real-time temporal coverage. 

For soil moisture, the only relevant weather variable not covered by AgERA5, we use the [GLADS](https://ldas.gsfc.nasa.gov/gldas/model-output) dataset, which is available from 2003 to present, and can be freely downloaded [here](https://disc.gsfc.nasa.gov/datasets?keywords=GLDAS). This dataset represents gridded and global soil moisture data developed by integrating satellite- and ground-based observational data products, using advanced land surface modeling and data assimilation techniques. Another option is to consider the [ESA CCI SM]( https://cds.climate.copernicus.eu/portfolio/dataset/satellite-soil-moisture) dataset for soil moisture although the latter only provides surface estimates. Finally, the [GLEAM](https://www.gleam.eu/) dataset does provide root zone soil moisture, but the data is currently available only up to December 2022.

### Contact
Raed Hamed

## Remote sensing indicators
Various remote sensing indicators of biomass status and health exists at the global scale. These include various vegetation indexes (VIs; e.g. NDVI, EVI, etc.) and various biophysical indicators (e.g. FPAR, LAI). Sub-national yield forecasting necessitates long-term time series (>20 years) of these indicators, coupled with frequent satellite revisits to ensure cloud-free imagery. These requirements practically limit options to the coarse-resolution missions MODIS and its successor, VIIRS.

While MODIS data can be directly downloaded from NASA LPDACC, the raw VIs and biophysical variables are often of low quality due to issues like cloud cover. These limitations necessitate screening, gap-filling, and corrections. The additional processing, which includes utilizing quality flags and applying temporal smoothing procedures, is both time-consuming and complex. These challenges become even more pronounced when processing near-real-time data, which is essential for operational yield forecasting.

In view of an operational deployment of sub-national yield forecasting, we selected two analysis-ready operational products representing crop biomass and health: FPAR and NDVI. 
FPAR (Fraction of Absorbed Photosynthetically Active Radiation) is provided as dekadal (10-day) data with a 500-meter spatial resolution, utilizing gap-filled and smoothed MODIS and VIIRS datasets. This data is sourced from the European Commission's Joint Research Centre (JRC) from the following [url](https://agricultural-production-hotspots.ec.europa.eu/data/indicators_fpar/). The quality of this near-real-time product has been assessed in [Meroni et al. (2019)](https://doi.org/10.1016/j.rse.2018.11.041).

NDVI (Normalized Difference Vegetation Index), a key indicator of vegetation greenness, is derived from MOD09CMG, available from NASA LDPACC. The data is prepared as an 8-day composite with a spatial resolution of 0.05 degree, selecting the pixel with the highest quality for each composite period.

### Contact
FPAR: Michele Meroni
NDVI: Ritvik Sahajpal

## Soil
We selected data from the [WISE project](global_soil_WISE/README.md) for soil properties. Another choice is [SoilGrids](https://www.isric.org/explore/soilgrids).

WISE soil data is a better choice than SoilGrids for the following reasons:
* agronomic interpretation is easier for WISE compared to SoilGrids.
* SoilGrids did not provide soil rootable depth and water holding capacity (this may have changed recently)
* WISE is based on soil maps whose properties are estimated by expert knowledge collected over the years. For example, in the Netherlands, we see that the performance of [BOFEK](https://doi.org/10.1016/j.geoderma.2022.116123) (similar to WISE) is still considerably better than SoilGrids.

### Contact
Allard de Wit

## Crop masks
With the aim of making available global maize and winter-spring cereals map we sourced the only crop-type specific maps currently available: maize and winter-spring cereals high spatial resolution (10 m) maps from ESA WorldCereal (https://esa-worldcereal.org/en). Alternatives sources of crop masks include Anomaly Hotspots of Agricultural Production (ASAP) from the European Commission's Joint Research Centre and IIASA (JRC-IIASA) and Global Best Available Crop Specific Masks (GEOGLAM-BACS) from the Group on Earth Observations Global Agriculture Monitoring (GEOGLAM). ESA WorldCereal is a better choice than the generic cropland layer from JRC-IIASA because of the availability of crop type maps for maize and wheat (spring and winter cereals). Although GEOGLAM-BACS provides crop type maps for maize and wheat (spring and winter cereals), their spatial resolution (0.05 degree) is lower compared to ESA WorldCereal (500m = 0.0045 degree).

### Contact
Francesco Collivignarelli, Michele Meroni, Ritvik Sahajpal

## Crop calendars

| Source | Area/crops | Phases reported | Caveat | Open and Free? |
|------|------|------|------|:---:|
| [FAO-GIEWS](https://www.fao.org/giews/countrybrief/country.jsp?lang=en&code=DZA) | Country-level, main crops (wheat and maize covered) | Sowing-growing-harvest | It covers food insecure countries (e.g. no US or EU) | Y |
| [FAO-Crop Calendars](https://cropcalendar.apps.fao.org/#/home) | Many crops, sub-national, sometimes agro-eco zones | Sowing, harvesting | no detailed description, not all countries, sometimes seems not correct | Y |
| [GEOGLAM](https://cropmonitor.org/index.php/eodatatools/baseline-data/) |  | Planting through Early Vegetative*<br>Vegetative through Reproductive<br>Ripening through Harvest<br>Harvest (End of Season)<br>Out of Season | Not global cover (on public website) | Y |
| [WorldCereal](https://github.com/ucg-uv/research_products/tree/main/cropcalendars) - Selected for benchmark because rainfed and irrigated are aggregated making it consistent with the yield data | Merging of different sources, raster maps for wheat and maize (res 0.5°). Maize calendars available only for the first maize season. | SOS, EOS [temporal resolution: day] | Aggregation to admin level needed | Y |
| GGCMI: [Data](https://zenodo.org/records/5062513), [Paper](https://www.nature.com/articles/s43016-021-00400-y), [Supplementary material](https://static-content.springer.com/esm/art%3A10.1038%2Fs43016-021-00400-y/MediaObjects/43016_2021_400_MOESM1_ESM.pdf), [ISIMIP link](https://www.isimip.org/gettingstarted/input-data-bias-adjustment/details/115/)  | Gridded 0.5. Differentiate between irrigated and rainfed, winter and spring wheat. Only one season for maize. | planting and maturity dates, growing season length [temporal resolution: day] | Aggregation to admin level needed | Y |
| MIRCA: [Data](https://zenodo.org/records/7422506), [Documentation](https://www.uni-frankfurt.de/45218023/MIRCA) | 5 arc-minutes x 5 arc-minutes = ~ 9.2 km x 9.2 km at the equator<br>all major food crops including regionally important ones. Differentiate between irrigated and rainfed. Crops can have multiple growing seasons | the month in which the growing period starts and the month in which the growing period ends. [temporal resolution: month] | Based on the period 1998-2002<br>For 402 spatial units (e.g. California is one unit) | Y |
| SAGE: [Data](https://sage.nelson.wisc.edu/data-and-models/datasets/crop-calendar-dataset/netcdf-5-min/), [Paper](https://onlinelibrary.wiley.com/doi/10.1111/j.1466-8238.2010.00551.x) | Gridded global crop calendars plus other crop climate parameters, for 19 crops. Resolution for both: 5 minute & 0.5 Degree. | Planting & harvesting dates, growing season length, Summaries of other crop-climate characteristics | Global cover except USA & Canada | Y |
| [USDA](https://ipad.fas.usda.gov/ogamaps/cropcalendar.aspx) | Qualitative global crop calendars. Few windows per country, mostly determined by the number of rainfall seasons. | Planting and harvesting. | Global cover but the planting and harvesting dates have to be manually approximated. Useful for general information purposes only. | Y |

### Contact
Pratishtha Poudel, Michele Meroni

## Crop yield DE
We selected data from [Duden et al. (2023)](crop_statistics_DE/README.md) for crop (Grain Maize and Wheat) yield. Another choice is [EU-EUROSTAT](crop_statistics_EU). crop_statistics_DE is a better choice because of the below mentioned reasons:
- Duden et al. (2023) data source for both crops have a longer and harmonized time series (spanning from 1979-2021) available as opposed to dataset from crop_statistics_EU (which is available from 1999-2020).
- For Grain Maize, Duden et al. (2023) has data at NUTS level 3 (administrative units = 138), compared to the one from EU-EUROSTAT which is at NUTS level 1.
- Data for DE from EU-EUROSTAT also does not meet some consistency checks done by Ronchetti et al. (2024), e.g. yield = production/area.

### Contact
Rahel Laudien

### References
Duden, C., Nacke, C. & Offermann, F. Crop yields and area in Germany from 1979 to 2021 at a harmonized district-level. OpenAgrar https://doi.org/10.3220/DATA20231117103252-0 (2023).

Ronchetti, G., Nisini Scacchiafichi, L., Seguini, L., Cerrani, I., and van der Velde, M.: Harmonized European Union subnational crop statistics can reveal climate impacts and crop cultivation shifts, Earth Syst. Sci. Data, 16, 1623–1649, https://doi.org/10.5194/essd-16-1623-2024, 2024.

## Crop yield ML
We selected Compagnie Malienne pour le Developpement des Textiles (CMDT) dataset that provides unparalleled granularity for Mali, offering arrondissement-level ((administrative level 3 equivalent within Mali) insights across five regions and 28 sectors from 1990-2017, making it optimal for localized maize production studies. Conversely, there is FEWSNET data that presents a broader regional perspective, covering multiple African countries including Mali with subnational data at administrative levels 1 and 2, which enables more comprehensive cross-country agricultural comparisons. Ultimately, the choice between these datasets depends on the specific research objectives: CMDT was ideal for granular, Mali-centric investigations.

### Contact
Janet Mutuku, PCS Traore, Celeste Tchampi 
