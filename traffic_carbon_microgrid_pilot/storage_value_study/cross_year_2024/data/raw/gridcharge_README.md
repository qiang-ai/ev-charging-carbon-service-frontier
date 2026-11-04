---

pretty_name: GridCharge
dataset_name: GridCharge
library_name: GridCharge

language:
- en

license: cc-by-4.0

size_categories:
- 1M<n<10M

tags:
- energy
- time-series
- ev-charging
- carbon-intensity
- electric-vehicles
- charging-stations
- tabular

pretty_visual: assets/car.png

---

# ⚡🔌 GridCharge Dataset: Capturing Carbon Intensity of 5M+ Public EV Charging Sessions in Scotland

<img src="assets/car.png" width="500" />

## Description

The GridCharge dataset provides a comprehensive view of over 5 million public Electric Vehicle (EV) charging sessions across Scotland, enriched with granular carbon intensity data from the UK National Grid. This dataset combines charging session records from ChargePlace Scotland with carbon intensity, energy consumption, and occupancy information at a 30-minute granularity. It is designed to support research in areas such as smart charging, grid load analysis, carbon-aware energy consumption, and the impact of weather on charging behavior.

## Code Availability

The code used to generate the GridCharge dataset is publicly available on Github: [https://github.com/djordjebatic/GridCharge](https://github.com/djordjebatic/GridCharge)

## Dataset Structure

The dataset is organized hierarchically by geographical region in Scotland. Each regional directory contains a file detailing the charging infrastructure, a folder containing the time-series data for each individual charger, and supplementary tariff information.

```
/sessions
|-- april-2023.csv
|-- april-2024.csv
|-- april-2025.csv
|-- august-2023.csv
|-- ...
/carbon_mix
|-- Aberdeenshire
|   |-- charging_infrastructure.csv
|   |-- sessions_mix/
|       |-- 50656_1.csv
|       |-- 50656_2.csv
|       |-- 50675_1.csv
|       |-- ...
|-- City of Edinburgh
|   |-- charging_infrastructure.csv
|   |-- sessions_mix/
|       |-- ...
|-- Clackmannanshire
|   |-- charging_infrastructure.csv
|   |-- sessions_mix/
|       |-- ...
|-- Glasgow City
|   |-- charging_infrastructure.csv
|   |-- sessions_mix/
|       |-- ...
|-- Highland
|   |-- charging_infrastructure.csv
|   |-- sessions_mix/
|       |-- ...
|-- ...
/tariff_information
|   |-- tariff.csv
|   |-- parsed_tariffs_gemini_results.csv

```

`charging_infrastructure.csv`

This file contains static information about each charging station within a specific region.

**Fields:**

| Field | Description |
|---|---|
| Latitude | Latitude of the charging station. |
| Longitude | Longitude of the charging station. |
| CP ID | The unique identifier for the charging point. |
| Connector | The identifier for a specific connector on a charging point. |
| Nominal Power (kW) | The nominal power capacity of the charger in kW. |
| Connector Type | The type of connector (e.g., "Type 2"). |
| Tariff | Tariff information for the charger. |
| Connection Fee | The fee for initiating a charging session. |
| Address | The descriptive address of the charging station location. |
| Postcode | The postcode of the charging station. |
| Local Authority | The local authority responsible for the region. |
| Region ID | A numerical identifier for the grid region. |

**Example Data** (`charging_infrastructure.csv` for **Glasgow City**):

| Latitude | Longitude | CP ID | Connector | Nominal Power (kW) | Connector Type | Tariff | Connection Fee | Address | Postcode | Local Authority | Region ID |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 55.853724 | -4.261707 | 52117 | 1 | 7.0 | Type 2 | 0.40 | NaN | Centre Street, Glasgow | G5 | Glasgow City | 2 |
| 55.853724 | -4.261707 | 52117 | 2 | 22.0 | Type 2 | 0.40 | NaN | Centre Street, Glasgow | G5 | Glasgow City | 2 |
| 55.853724 | -4.261707 | 52118 | 1 | 7.0 | Type 2 | 0.40 | NaN | Centre Street, Glasgow | G5 | Glasgow City | 2 |

`sessions_mix/{CP ID}_{Connector}.csv`

Each file in this directory corresponds to a single charger connector and contains time-series data for its charging sessions and the associated grid carbon mix. The data for each charger starts on the day of its first recorded session and ends on the day of its last recorded session.

**Fields:**

| Field | Description |
|---|---|
| Timestamp | The start of the 30-minute interval in UTC. |
| Consumed | The total energy consumed (in kWh) during the interval, estimated from the session duration and charger's nominal power. |
| Occupied | A binary flag (0 or 1) indicating if the charger was occupied during the interval. |
| Biomass | gCO2/kWh from biomass. |
| Coal | gCO2/kWh from coal. |
| Gas | gCO2/kWh from gas. |
| Nuclear | gCO2/kWh from nuclear. |
| Hydro | gCO2/kWh from hydro. |
| Solar | gCO2/kWh from solar. |
| Wind | gCO2/kWh from wind. |
| Imports | gCO2/kWh from imports. |
| Other | gCO2/kWh from other sources. |
| Forecast | The forecasted carbon intensity (gCO2/kWh) for the interval. |
| Carbon Index | The carbon intensity index level (e.g., "very low", "moderate", "high"). |
| Region ID | The numerical identifier for the grid region. |

**Example Data** (`sessions_mix/52117_1.csv`):

| Timestamp | Consumed | Occupied | Biomass | Coal | Imports | Gas | Nuclear | Other | Hydro | Solar | Wind | Forecast | Carbon Index | Region ID |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2025-07-16 10:00:00+00:00 | 2.917 | 1 | 0.104 | 0.0 | 0.164 | 0.005 | 0.473 | 0.0 | 0.0 | 0.107 | 0.146 | 15 | very low | 2 |
| 2025-07-16 10:30:00+00:00 | 3.500 | 1 | 0.097 | 0.0 | 0.158 | 0.003 | 0.469 | 0.0 | 0.0 | 0.126 | 0.146 | 13 | very low | 2 |
| 2025-07-16 11:00:00+00:00 | 3.500 | 1 | 0.093 | 0.0 | 0.155 | 0.005 | 0.456 | 0.0 | 0.0 | 0.150 | 0.141 | 13 | very low | 2 |


`tariff_info/parsed_tariffs_gemini_results.csv`

The structured tariff data in `parsed_tariffs_gemini_results.csv` was generated by parsing the unstructured text from the `Tariff Description` column in the original `tariff.csv` file. This process was automated using a Large Language Model (Gemini 2 Flash). The model was tasked with extracting specific, structured details from each description. The implementation, including the API call, can be found in the `process.py` script.

**Fields:**

| Field | Description |
|---|---|
| CP ID | The unique identifier for the charging point. |
| Connector | The identifier for a specific connector on a charging point. |
| Tariff | Tariff information for the charger. |
| Connection Fee | The fee for initiating a charging session. |
| Tariff Description | The original textual description of the tariff structure. |
| Minimum Fee (gbp) | Any minimum fee for a charging session. |
| Rate (p/kWh) | The primary rate in pence per kilowatt-hour. |
| Flat | Details of any flat-rate or tiered pricing. |
| Overstay Fee (gbp) | Details of any overstay fees in GBP. |

**Example Data** (`tariff_info/parsed_tariffs_gemini_results.csv`):

| CP ID | Connector | Tariff | Connection Fee | Tariff Description | Minimum Fee (gbp) | Rate (p/kWh) | Flat | Overstay Fee (gbp) |
|---|---|---|---|---|---|---|---|---|
| 53512 | 3 | 0.7 | N\A  | 70p/kWh. Overstay fee of GBP30.00 applied after 60 minutes. | N\A | 70.0 | Yes | GBP30.00 after 60 minutes |
| 52301 | 3 | 0.7 | 1.0 | 170p/kWh for the first kWh and 70p/kWh thereafter. Minimum charge of GBP1.50. | 1.50 | 170.0 | First kWh at 170p, then 70p | N\A |
| 52944 | 1 | 0.47 | 0.0 | 47p/kWh. Minimum charge GBP1.00. An overstay fee of GBP10.00 is applied after 4 hours. | 1.00 | 47.0 | Yes | GBP10.00 after 4 hours |


## Data Creation Methodology

The dataset was created by processing and merging two primary data sources:

### **ChargePlace Scotland Session Data**: 
The raw EV charging session data was sourced from a `sessions` directory, containing monthly CSV files (e.g., `january-2025.csv`, `february-2025.csv`). These files detail individual charging events. An example of the data format is shown below:

**Example Data** (`sessions/april-2023.csv`):

| Start | Duration | Consumed(kWh) | Paid(gbp) | CP ID | Connector |
|---|---|---|---|---|---|
| 2023-04-01 01:00:07 | 08:14:11 | 21.67 | 5.85 | 60021 | 1 |
| 2023-04-01 01:00:29 | 06:48:05 | 32.37 | 0.0 | APT50662 | 1 |

### **UK National Grid Carbon Intensity API**: 
Provided historical carbon intensity data for different UK regions. The data were merged based on the postcode and region id information.

The raw session data was processed to align with 30-minute intervals. The energy drawn within each interval was estimated by assuming a uniform discharge rate based on the charger's nominal power. The grid carbon mix data was merged with the session data based on the charging station's postcode and its corresponding `Region ID`.

### Data Cleaning

The dataset has been cleaned to ensure quality:

- Overstay periods longer than one day have been removed (accounting for <1% of the original data).
- Rows with NaN values have been removed.
- All timestamps are standardized to UTC.

### Tariff Information Processing

The following prompt was used to instruct the model to extract the relevant fields into a JSON format:

```python
prompt = f"""From the following tariff description, extract these specific details:

Description: "{description}"

Extract these exact fields:
- "minimum_fee": The minimum charge for a session (e.g., "2.00" for £2.00). If not mentioned, return null.
- "rate": The cost per kWh as a number (e.g., 39 for "39p/kWh"). If not mentioned, return null.
- "flat": If there's a single flat rate, return "Yes". If the rate structure changes, describe the structure.
- "overstay_fee": The fee for staying too long after charging. If mentioned, describe it. If not, return null.

Return only valid JSON with these exact keys and no additional text."""
```

### Notes

- Exact carbon intensity of imports can potentially be inferred through additional data collection, for more details visit the official Carbon Intensity API documentation.

## Citation

If you use this dataset in your research, please cite it as follows:

```bibtex
@dataset{gridcharge2025,
    title  = {GridCharge: Capturing Carbon Intensity of 5M+ Public EV Charging Sessions in Scotland},
    author = {Djordje Batic},
    year   = {2025}
    url    = {https://huggingface.co/datasets/djordjebatic/GridCharge}
}
```


