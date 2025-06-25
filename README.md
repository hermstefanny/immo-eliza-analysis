# immo-eliza-analysis

## Data Analysis

The goal of this analysis is to perform an exploratory data analysis in the _'ml_ready_real_estate_data.csv'_ file

The Jupyter Notebook _eda_draft.ipynb_ contains the following cleaning steps:

### Price Analysis

#### Relevant functions

- _remove_outliers_iqr(df, col)_: Pass a dataframe and a column to obtain the dataframe without outliers
- identify_outliers_iqr(df, col): Pass a dataframe and a column to obtain the rows with outliers

#### Price distribution analyis

The following features were defined to study the price feature:

- Description tables for price with outliers and price without outliers
- Histogram with price outliers
- Histogram without price outliers
- Histogram without price outliers and density function

### Correlation Analysis

#### Correlation for Continuous Variables

The following features were defined to study correlation for the continuous variables:

- Correlation matrix with Pearson correlation
- Heatmap correlation matrix

#### Correlation for Boolean Variables

The following features were defined to study correlation for the boolean variables:

- Correlation study with Point-Serial correlation
- Barplot with boolean features
