# Morocco National Crop Statistics

## Short description
This dataset provides national-level crop production, yield, and harvested area statistics for Morocco, covering the period 1961–2023. The data originate from FAOSTAT’s **Production – Crops and Livestock Products (QCL)** domain and were adapted to follow the standardized **AgML CY-Bench** format for yield data integration and benchmarking.

## Link
[FAOSTAT – Production: Crops and Livestock Products (QCL)](https://www.fao.org/faostat/en/#data/QCL)

## Publisher
Food and Agriculture Organization of the United Nations (FAO)

## Dataset owner
Statistics Division (ESS), Food and Agriculture Organization of the United Nations (FAO)

## Data card author
Abdelghani Belgaid — Mohammed VI Polytechnic University (UM6P)

## Dataset overview
**Crops**: Wheat, Barley, Maize (corn), Green corn (maize), Rice, Soya beans, Sugar cane  
**Variables [unit]**:  
- Yield [t/ha]  
- Production [t]  
- Harvested area [ha]  
**Temporal coverage**: 1961 – 2023  
**Temporal resolution**: Annual  
**Spatial resolution**: National (adm_id = MA-NAT)  
**Date Published**: 2025-11-08  
**Data Modality**: Tabular (CSV)

### Summary
The dataset harmonizes FAOSTAT national statistics with the CY-Bench schema:  
`crop_name, country_code, adm_id, season_name, planting_year, harvest_year, planting_date, harvest_date, yield, production, planted_area, harvest_area`.  
All yield values are converted to tonnes per hectare (t/ha) when required.  

### Upcoming tasks
- Extend coverage to include additional crops available in FAOSTAT at the national level.  
- Integrate ADM1-level (regional) statistics for higher granularity (e.g., Casablanca-Settat, Fès-Meknès, Marrakech-Safi, etc).  
- Validate consistency across FAO sub-national series and CY-Bench data ingestion pipeline.

## Data access API
Data can be accessed programmatically using the [FAOSTAT FENIX API](https://fenixservices.fao.org/faostat/api/v1/en/data/QCL)  
or via the official [faostat](https://pypi.org/project/faostat/) Python package.

## Provenance 
**Source:** FAOSTAT (downloaded on 2025-11-08 from the QCL domain).  
**Processing script:** `faostat_qcl_morocco_to_cybench.py` — to be included in the repository under `/scripts/`.  
Dataset adapted for CY-Bench ingestion by aligning variable names, units, and structure.

# License 
FAOSTAT datasets are distributed under the [CC BY-NC-SA 3.0 IGO License](https://creativecommons.org/licenses/by-nc-sa/3.0/igo/).  
This repository redistributes only the derived, reformatted dataset for research and non-commercial use with proper attribution.

## How to cite
Food and Agriculture Organization of the United Nations (FAO).  
*FAOSTAT – Production: Crops and Livestock Products (QCL).*  
Rome, Italy. Available at: https://www.fao.org/faostat/en/#data/QCL  
Adapted for AgML CY-Bench format (Belgaid, A. 2025).

## Additional information
The dataset serves as a reference baseline for the AgML CY-Bench initiative, facilitating cross-country benchmarking of yield-model performance and data harmonization.  
Future releases will include ADM1 and ADM2 granularity to support regional model evaluation.

## References
- FAO. *FAOSTAT – Production: Crops and Livestock Products (QCL).* Rome, FAO. https://www.fao.org/faostat/en/#data/QCL  
- AgML CY-Bench Repository. https://github.com/WUR-AI/AgML-CY-Bench
