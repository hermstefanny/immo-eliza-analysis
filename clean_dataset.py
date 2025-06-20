import pandas as pd

def import_csv_data(file_location):
    """Import CSV file and convert it to a pandas DataFrame structure."""
    return pd.read_csv(file_location)

def process_and_sanitize_data(dataframe):
    """Process raw data by eliminating duplicates, managing null values, and optimizing data types."""
    # Remove duplicate entries based on address-related columns
    dataframe.drop_duplicates(subset=['postal_code', 'street', 'number', 'box'], inplace=True)
    
    # Eliminate rows that are completely empty
    dataframe.dropna(how='all', inplace=True)
    
    # Transform specific columns to categorical format for better memory usage
    dataframe['property_type'] = dataframe['property_type'].astype('category')
    dataframe['property_subtype'] = dataframe['property_subtype'].astype('category')
    dataframe['locality'] = dataframe['locality'].astype('category')
    dataframe['buildingState'] = dataframe['buildingState'].astype('category')
    
    # Handle missing values by replacing with None
    dataframe.map(lambda x: None if pd.isna(x) else x)
    
    
    return dataframe

def export_processed_data(dataframe, destination_path):
    """Export the processed dataset to a CSV file with proper indexing."""
    # Configure 'house_index' as the primary index
    dataframe.set_index('house_index', drop=True, inplace=True)
    
    # Organize the dataset by index values
    dataframe.sort_index(inplace=True)
    
    # Export the processed dataset
    dataframe.to_csv(destination_path)

def execute_data_pipeline():
    input_file = "Data/all_properties_output.csv"
    output_file = "Data/cleaned_dataset.csv"
    
    raw_data = import_csv_data(input_file)
    processed_data = process_and_sanitize_data(raw_data)
    export_processed_data(processed_data, output_file)

if __name__ == "__main__":
    execute_data_pipeline()