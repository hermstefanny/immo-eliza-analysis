# Real Estate Data Cleaning and Preprocessing Pipeline

A comprehensive Python pipeline for cleaning, preprocessing, and encoding Belgian real estate data for machine learning applications.

## Overview

This script processes raw Belgian real estate data (from Immoweb) and transforms it into a clean, ML-ready dataset. It handles missing values, removes outliers, encodes categorical variables, and prepares the data for machine learning models.

## Features

- **Data Cleaning**: Removes duplicates, handles missing values, and filters unrealistic data
- **Feature Engineering**: Merges related columns and creates new features
- **Categorical Encoding**: Converts text categories to numerical representations
- **Boolean Handling**: Standardizes True/False values across features
- **Missing Value Imputation**: Smart handling of missing data based on feature context
- **ML-Ready Output**: Produces a dataset ready for machine learning algorithms

## Requirements

```python
pandas
numpy
scikit-learn
matplotlib 
seaborn
scipy 
requests 
```

Install with:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy requests 
```

## Input Data Format

The script expects a CSV file with Belgian real estate data containing columns such as:
- `price` (target variable)
- `type`, `subtype` (property types)
- `province`, `locality` (location data)
- `bedroomCount`, `bathroomCount` (numeric features)
- Various boolean features (`hasGarden`, `hasTerrace`, etc.)
- `epcScore` (energy performance rating)

## Usage

### Basic Usage

```python
import pandas as pd
from clean_immo_datasetV2 import clean_real_estate_data, full_encoding_pipeline

# Load your data
df = pd.read_csv("your_real_estate_data.csv")

# Clean the data
df_clean = clean_real_estate_data(df)

# Apply encoding pipeline
df_encoded, encoding_mappings = full_encoding_pipeline(df_clean)

# Save results
df_clean.to_csv("cleaned_data.csv", index=False)
df_encoded.to_csv("ml_ready_data.csv", index=False)
```

### Command Line Usage

```bash
python clean_immo_datasetV2.py
```

Make sure your input file is at `data/immoweb-dataset.csv` or modify the path in the script.

## Processing Steps

### 1. Data Cleaning (`clean_real_estate_data`)

- **Empty Data Removal**: Removes completely empty rows and columns
- **Column Cleanup**: Drops unnecessary columns (URL, unnamed indices)
- **Feature Removal**: Removes low-value columns like `heatingType`, `monthlyCost`
- **Parking Merge**: Combines indoor and outdoor parking counts
- **Duplicate Removal**: Removes duplicate properties based on ID
- **Price Filtering**: Removes unrealistic prices (< €50k or > €5M)
- **Garden Imputation**: Sets garden surface to 0 where `hasGarden = False`
- **Numeric Validation**: Handles negative values and unrealistic counts
- **Text Standardization**: Removes spaces and standardizes text format

### 2. Missing Value Preprocessing (`preprocess_missing_values`)

- Fills boolean NaN values with `False`
- Handles missing EPC scores appropriately

### 3. Categorical Encoding (`encode_categorical_features`)

#### Province Encoding (Ordinal)
```python
Brussels: 1, Luxembourg: 2, Antwerp: 3, ...
```

#### Property Type Encoding
```python
APARTMENT: 1, HOUSE: 2
```

#### Property Subtype Encoding
```python
APARTMENT: 1, HOUSE: 2, FLAT_STUDIO: 3, DUPLEX: 4, ...
```

#### EPC Score Encoding (Energy Performance)
```python
A+: 8, A: 7, B: 6, C: 5, D: 4, E: 3, F: 2, G: 1
```

#### Boolean Features
All boolean features are converted to 1/0 encoding:
- `hasGarden_encoded`, `hasTerrace_encoded`, etc.

### 4. Final Dataset Creation (`create_final_ml_dataset`)

- Selects relevant features for ML
- Fills remaining numeric NaN values with median
- Creates the final ML-ready dataset

## Output Files

The script generates two main outputs:

1. **`cleaned_real_estate_data.csv`**: Clean dataset with original column names
2. **`ml_ready_real_estate_data.csv`**: Encoded dataset ready for ML algorithms

## Key Features Retained

### Numeric Features
- `bedroomCount`, `bathroomCount`, `habitableSurface`
- `toiletCount`, `terraceSurface`, `postCode`
- `gardenSurface`, `totalParkingCount`

### Encoded Categorical Features
- `province_encoded`, `type_encoded`, `subtype_encoded`
- `locality_encoded`, `epcScore_encoded`

### Boolean Features (encoded as 0/1)
- Property amenities: `hasGarden`, `hasTerrace`, `hasSwimmingPool`
- Building features: `hasLift`, `hasBasement`, `hasAttic`
- Technology: `hasAirConditioning`, `hasHeatPump`, `hasPhotovoltaicPanels`
- Security: `hasArmoredDoor`, `hasVisiophone`

## Data Quality Measures

- **Price Range**: €50,000 - €5,000,000
- **Realistic Counts**: Max 10 bedrooms/bathrooms, 5 toilets, 10 parking spaces
- **Missing Value Threshold**: Columns with >80% missing values are removed
- **Duplicate Handling**: Properties with same ID are deduplicated

## Customization

### Modify Price Range
```python
df = df[(df["price"] >= YOUR_MIN) & (df["price"] <= YOUR_MAX)]
```

### Add Custom Encoding
```python
your_mapping = {"VALUE1": 1, "VALUE2": 2}
df["your_column_encoded"] = df["your_column"].map(your_mapping)
```

### Change Missing Value Threshold
```python
threshold = 0.5  # Remove columns with >50% missing
```
## Error Handling

The script includes robust error handling:
- Checks for column existence before processing
- Handles various boolean representations (`True`, `"True"`, etc.)
- Manages missing values appropriately for each data type
- Provides detailed logging of all processing steps


# Data Analysis

The goal of this analysis is to perform an exploratory data analysis in the _'ml_ready_real_estate_data.csv'_ file

The Jupyter Notebook _eda_draft.ipynb_ contains the following cleaning steps:

## Price Analysis

### Relevant functions

- _remove_outliers_iqr(df, col)_: Pass a dataframe and a column to obtain the dataframe without outliers
- identify_outliers_iqr(df, col): Pass a dataframe and a column to obtain the rows with outliers

### Price distribution analyis

The following features were defined to study the price feature:

- Description tables for price with outliers and price without outliers
- Histogram with price outliers
- Histogram without price outliers
- Histogram without price outliers and density function

## Correlation Analysis

### Correlation for Continuous Variables

The following features were defined to study correlation for the continuous variables:

- Correlation matrix with Pearson correlation
- Heatmap correlation matrix

### Correlation for Boolean Variables

The following features were defined to study correlation for the boolean variables:

- Correlation study with Point-Serial correlation
- Barplot with boolean features
