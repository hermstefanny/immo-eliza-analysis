import pandas as pd
import numpy as np

def clean_real_estate_data(df):
    print(" Starting data cleaning...")
    print(f"Original data shape: {df.shape}")
    
    # Step 1: Remove completely empty rows and columns
    print("\n Step 1: Removing empty rows and columns...")
    
    # Remove rows where all values are NaN
    df = df.dropna(how='all')
    
    # Remove columns where all values are NaN
    df = df.dropna(axis=1, how='all')
    
    print(f"After removing empty: {df.shape}")
    
    # Step 2: Handle the unnamed index column
    print("\n Step 2: Cleaning column names...")
    
    # Drop the unnamed index column if it exists
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
        print("Removed 'Unnamed: 0' column")
    
    # Step 3: Remove duplicates based on ID
    print("\n Step 3: Removing duplicate properties...")
    
    if 'id' in df.columns:
        before_dup = len(df)
        df = df.drop_duplicates(subset=['id'], keep='first')
        after_dup = len(df)
        print(f"Removed {before_dup - after_dup} duplicate properties")
    
    # Step 4: Clean price column (main target variable)
    print("\n Step 4: Cleaning price data...")
    
    if 'price' in df.columns:
        # Remove rows with missing prices (can't train without target)
        before_price = len(df)
        df = df.dropna(subset=['price'])
        after_price = len(df)
        print(f"Removed {before_price - after_price} rows with missing prices")
        
        # Remove unrealistic prices (too low or too high)
        # Assuming Belgian real estate: min 50k, max 5M euros
        df = df[(df['price'] >= 50000) & (df['price'] <= 5000000)]
        print(f"Kept prices between 50k and 5M euros: {len(df)} properties")
    
    # Step 5: Clean numeric columns
    print("\n Step 5: Cleaning numeric features...")
    
    numeric_cols = ['bedroomCount', 'bathroomCount', 'toiletCount', 'terraceSurface']
    
    for col in numeric_cols:
        if col in df.columns:
            # Replace negative values with NaN
            df.loc[df[col] < 0, col] = np.nan
            
            # Replace unrealistic high values
            if col in ['bedroomCount', 'bathroomCount']:
                df.loc[df[col] > 10, col] = np.nan  # Max 10 bedrooms/bathrooms
            elif col == 'toiletCount':
                df.loc[df[col] > 5, col] = np.nan   # Max 5 toilets
            
            print(f"Cleaned {col}: {df[col].notna().sum()} valid values")
    
    # Step 6: Clean categorical columns
    print("\n Step 6: Cleaning categorical features...")
    
    # Clean property type and subtype - REMOVE ALL SPACES
    if 'type' in df.columns:
        df['type'] = df['type'].str.upper().str.strip().str.replace(' ', '')
        print(f"Property types: {df['type'].value_counts().to_dict()}")
    
    if 'subtype' in df.columns:
        df['subtype'] = df['subtype'].str.upper().str.strip().str.replace(' ', '')
    
    # Clean location data - REMOVE ALL SPACES
    location_cols = ['province', 'locality']
    for col in location_cols:
        if col in df.columns:
            df[col] = df[col].str.strip().str.title().str.replace(' ', '')
            print(f"Unique {col}s: {df[col].nunique()}")
    
    # Clean ALL text columns - remove spaces everywhere
    text_cols = df.select_dtypes(include=['object']).columns
    for col in text_cols:
        if col not in ['url']:  # Keep URLs as they are
            df[col] = df[col].astype(str).str.strip().str.replace(' ', '')
            print(f"Removed spaces from {col}")
    
    # Step 7: Handle boolean columns
    print("\n Step 7: Cleaning boolean features...")
    
    boolean_cols = ['hasOffice', 'hasSwimmingPool', 'hasFireplace', 'hasTerrace', 
                   'accessibleDisabledPeople']
    
    for col in boolean_cols:
        if col in df.columns:
            # Convert to proper boolean (True/False/NaN)
            df[col] = df[col].map({True: True, False: False, 'True': True, 'False': False})
            print(f"{col}: {df[col].value_counts(dropna=False).to_dict()}")
    
    # Step 8: Clean energy performance (EPC)
    print("\n Step 8: Cleaning energy performance...")
    
    if 'epcScore' in df.columns:
        # Keep only valid EPC scores (A, B, C, D, E, F, G)
        valid_epc = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
        df.loc[~df['epcScore'].isin(valid_epc), 'epcScore'] = np.nan
        print(f"EPC distribution: {df['epcScore'].value_counts().to_dict()}")
    
    # Step 9: Remove columns with too many missing values
    print("\n Step 9: Removing columns with too many missing values...")
    
    # Remove columns where more than 80% of values are missing
    threshold = 0.8
    missing_pct = df.isnull().sum() / len(df)
    cols_to_drop = missing_pct[missing_pct > threshold].index.tolist()
    
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        print(f"Removed columns with >80% missing: {cols_to_drop}")
    
    # Step 10: Final summary
    print("\n✅ Data cleaning completed!")
    print(f"Final data shape: {df.shape}")
    print(f"Columns kept: {list(df.columns)}")
    
    # Show missing values summary
    print("\n Missing values summary:")
    missing_summary = df.isnull().sum()
    missing_summary = missing_summary[missing_summary > 0].sort_values(ascending=False)
    for col, missing_count in missing_summary.items():
        pct = (missing_count / len(df)) * 100
        print(f"  {col}: {missing_count} ({pct:.1f}%)")
    
    return df


def prepare_for_ai_training(df):
    print("\n Preparing data for AI training...")
    
    # Create dummy variables for categorical columns
    categorical_cols = ['type', 'subtype', 'province', 'locality', 'epcScore']
    
    df_final = df.copy()
    
    for col in categorical_cols:
        if col in df_final.columns and df_final[col].dtype == 'object':
            # Create dummy variables
            dummies = pd.get_dummies(df_final[col], prefix=col, dummy_na=True)
            df_final = pd.concat([df_final, dummies], axis=1)
            df_final = df_final.drop(col, axis=1)
            print(f"Created dummy variables for {col}")
    
    # Fill remaining numeric NaN values with median
    numeric_cols = df_final.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df_final[col].isnull().sum() > 0:
            median_val = df_final[col].median()
            df_final[col] = df_final[col].fillna(median_val)
            print(f"Filled {col} NaN values with median: {median_val}")
    
    print(f"Final AI-ready data shape: {df_final.shape}")
    return df_final


#1. Load your data:
df = pd.read_csv('data/immoweb-dataset.csv')

#2. Clean the data:
df_clean = clean_real_estate_data(df)

#3. Prepare for AI training:
df_ai_ready = prepare_for_ai_training(df_clean)

#4. Save the cleaned data:
df_ai_ready.to_csv('cleaned_real_estate_data.csv', index=False)
