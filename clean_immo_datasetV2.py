import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder


def clean_real_estate_data(df):
    print(" Starting data cleaning...")
    print(f"Original data shape: {df.shape}")

    # Step 1: Remove completely empty rows and columns
    print("\n Step 1: Removing empty rows and columns...")

    # Remove  Url row
    df = df.drop("url", axis="columns")
    # Remove rows where all values are NaN
    df = df.dropna(how="all")

    # Remove columns where all values are NaN
    df = df.dropna(axis=1, how="all")

    print(f"After removing empty: {df.shape}")

    # Step 2: Handle the unnamed index column
    print("\n Step 2: Cleaning column names...")

    # Drop the unnamed index column if it exists
    if "Unnamed: 0" in df.columns:
        df = df.drop("Unnamed: 0", axis=1)
        print("Removed 'Unnamed: 0' column")

    # Step 3: Remove duplicates based on ID
    print("\n Step 3: Removing duplicate properties...")

    if "id" in df.columns:
        before_dup = len(df)
        df = df.drop_duplicates(subset=["id"], keep="first")
        after_dup = len(df)
        print(f"Removed {before_dup - after_dup} duplicate properties")

    # Step 4: Clean price column (main target variable)
    print("\n Step 4: Cleaning price data...")

    if "price" in df.columns:
        # Remove rows with missing prices (can't train without target)
        before_price = len(df)
        df = df.dropna(subset=["price"])
        after_price = len(df)
        print(f"Removed {before_price - after_price} rows with missing prices")

        # Remove unrealistic prices (too low or too high)
        # Assuming Belgian real estate: min 50k, max 5M euros
        df = df[(df["price"] >= 50000) & (df["price"] <= 5000000)]
        print(f"Kept prices between 50k and 5M euros: {len(df)} properties")

    # Step 5: Clean numeric columns
    print("\n Step 5: Cleaning numeric features...")

    numeric_cols = ["bedroomCount", "bathroomCount", "toiletCount", "terraceSurface"]

    for col in numeric_cols:
        if col in df.columns:
            # Replace negative values with NaN
            df.loc[df[col] < 0, col] = np.nan

            # Replace unrealistic high values
            if col in ["bedroomCount", "bathroomCount"]:
                df.loc[df[col] > 10, col] = np.nan  # Max 10 bedrooms/bathrooms
            elif col == "toiletCount":
                df.loc[df[col] > 5, col] = np.nan  # Max 5 toilets

            print(f"Cleaned {col}: {df[col].notna().sum()} valid values")

    # Step 6: Clean categorical columns
    print("\n Step 6: Cleaning categorical features...")

    # Clean property type and subtype - REMOVE ALL SPACES
    if "type" in df.columns:
        df["type"] = df["type"].str.upper().str.strip().str.replace(" ", "")
        print(f"Property types: {df['type'].value_counts().to_dict()}")

    if "subtype" in df.columns:
        df["subtype"] = df["subtype"].str.upper().str.strip().str.replace(" ", "")

    # Clean location data - REMOVE ALL SPACES
    location_cols = ["province", "locality"]
    for col in location_cols:
        if col in df.columns:
            df[col] = df[col].str.strip().str.title().str.replace(" ", "")
            print(f"Unique {col}s: {df[col].nunique()}")

    # Clean ALL text columns - remove spaces everywhere
    text_cols = df.select_dtypes(include=["object"]).columns
    for col in text_cols:
        if col not in ["url"]:  # Keep URLs as they are
            df[col] = df[col].astype(str).str.strip().str.replace(" ", "")
            print(f"Removed spaces from {col}")

    # Step 7: Handle boolean columns
    print("\n Step 7: Cleaning boolean features...")

    boolean_cols = [
        "hasAttic",
        "hasBasement",
        "hasDressingRoom",
        "hasDiningRoom",
        "hasLift",
        "hasHeatPump",
        "hasPhotovoltaicPanels",
        "hasThermicPanels",
        "hasBalcony",
        "hasGarden",
        "hasAirConditioning",
        "hasArmoredDoor",
        "hasVisiophone",
        "hasOffice",
        "hasSwimmingPool",
        "hasFireplace",
        "hasLivingRoom",
        "accessibleDisabledPeople",
    ]

    for col in boolean_cols:
        if col in df.columns:
            # Convert to proper boolean (True/False/NaN)
            df[col] = df[col].map(
                {True: True, False: False, "True": True, "False": False}
            )
            df[col] = df[col].fillna(False)  # Convert any remaining NaN to False
            print(f"{col}: {df[col].value_counts(dropna=False).to_dict()}")

    # Step 8: Clean energy performance (EPC)
    print("\n Step 8: Cleaning energy performance...")

    if "epcScore" in df.columns:
        # Keep only valid EPC scores (A, B, C, D, E, F, G)
        valid_epc = ["A", "B", "C", "D", "E", "F", "G"]
        df.loc[~df["epcScore"].isin(valid_epc), "epcScore"] = np.nan
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
    print("\n Data cleaning completed!")
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


def filling_data_with_median(df):
    print("\n Preparing data for AI training...")

    # Create dummy variables for categorical columns
    # categorical_cols = ['type', 'subtype', 'province', 'locality', 'epcScore']

    df_final = df.copy()

    # for col in categorical_cols:
    # if col in df_final.columns and df_final[col].dtype == 'object':
    # Create dummy variables
    # dummies = pd.get_dummies(df_final[col], prefix=col, dummy_na=True)
    # df_final = pd.concat([df_final, dummies], axis=1)
    # df_final = df_final.drop(col, axis=1)
    # print(f"Created dummy variables for {col}")

    # Fill remaining numeric NaN values with median
    numeric_cols = df_final.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df_final[col].isnull().sum() > 0:
            median_val = df_final[col].median()
            df_final[col] = df_final[col].fillna(median_val)
            print(f"Filled {col} NaN values with median: {median_val}")

    print(f"Datashape filled with median ready: {df_final.shape}")
    return df_final


def fill_garden_surface(df):
    df["gardenSurface"][~df["hasGarden"]] = 0


# NEW CATEGORICAL ENCODING FUNCTIONS


def encode_categorical_features(df):
    """
    Encode all categorical features in the real estate dataset using ordinal/label encoding
    """
    print("\n Starting categorical encoding...")
    df_encoded = df.copy()

    # 1. PROVINCE ENCODING (Ordinal based on specified order)
    print("\n Encoding provinces...")
    province_mapping = {
        "Brussels": 1,
        "Luxembourg": 2,
        "Antwerp": 3,
        "FlemishBrabant": 4,  # Note: spaces removed in cleaning
        "EastFlanders": 5,
        "WestFlanders": 6,
        "Liège": 7,
        "WalloonBrabant": 8,
        "Limburg": 9,
        "Namur": 10,
        "Hainaut": 11,
    }
    df_encoded["province_encoded"] = df_encoded["province"].map(province_mapping)
    print(
        f"Province encoding: {df_encoded['province_encoded'].value_counts().sort_index().to_dict()}"
    )

    # 2. TYPE ENCODING (Simple ordinal)
    print("\n Encoding property types...")
    type_mapping = {"APARTMENT": 1, "HOUSE": 2}
    df_encoded["type_encoded"] = df_encoded["type"].map(type_mapping)
    print(f"Type encoding: {df_encoded['type_encoded'].value_counts().to_dict()}")

    # 3. SUBTYPE ENCODING
    print("\n Encoding property subtypes...")
    # 3. SUBTYPE ENCODING
    print("\n Encoding property subtypes...")
    subtype_mapping = {
        "APARTMENT": 1,
        "HOUSE": 2,
        "FLAT_STUDIO": 3,
        "FLATSTUDIO": 3,  # Handle version without underscore
        "DUPLEX": 4,
        "PENTHOUSE": 5,
        "GROUND_FLOOR": 6,
        "GROUNDFLOOR": 6,  # Handle version without underscore
        "APARTMENT_BLOCK": 7,
        "APARTMENTBLOCK": 7,  # Handle version without underscore
        "MANSION": 8,
        "EXCEPTIONAL_PROPERTY": 9,
        "EXCEPTIONALPROPERTY": 9,  # Handle version without underscore
        "MIXED_USE_BUILDING": 10,
        "MIXEDUSEBUILDING": 10,  # Handle version without underscore
        "TRIPLEX": 11,
        "LOFT": 12,
        "VILLA": 13,
        "TOWN_HOUSE": 14,
        "TOWNHOUSE": 14,  # Handle version without underscore
        "CHALET": 15,
        "MANOR_HOUSE": 16,
        "MANORHOUSE": 16,  # Handle version without underscore
        "SERVICE_FLAT": 17,
        "SERVICEFLAT": 17,  # Handle version without underscore
        "KOT": 18,
        "FARMHOUSE": 19,
        "BUNGALOW": 20,
        "COUNTRY_COTTAGE": 21,
        "COUNTRYCOTTAGE": 21,  # Handle version without underscore
        "OTHER_PROPERTY": 22,
        "OTHERPROPERTY": 22,  # Handle version without underscore
        "CASTLE": 23,
        "PAVILION": 24,
    }
    df_encoded["subtype_encoded"] = df_encoded["subtype"].map(subtype_mapping)
    print(f"Subtype encoding: {df_encoded['subtype_encoded'].value_counts().to_dict()}")

    # 4. LOCALITY ENCODING (Label encoding - too many unique values for ordinal)
    print("\n Encoding localities...")
    if "locality" in df_encoded.columns and df_encoded["locality"].notna().sum() > 0:
        le_locality = LabelEncoder()
        # Only encode non-null values
        mask = df_encoded["locality"].notna()
        df_encoded.loc[mask, "locality_encoded"] = le_locality.fit_transform(
            df_encoded.loc[mask, "locality"]
        )
        print(
            f"Locality encoded: {df_encoded['locality_encoded'].notna().sum()} localities"
        )
    else:
        le_locality = None
        print("No locality column found or all values are null")

    # 5. GARDEN ORIENTATION ENCODING (if exists)
    print("\n Encoding garden orientation...")
    if "gardenOrientation" in df_encoded.columns:
        garden_orientation_mapping = {
            "SOUTH": 4,
            "SOUTH_WEST": 3,
            "SOUTHWEST": 3,
            "SOUTH_EAST": 3,
            "SOUTHEAST": 3,
            "WEST": 2,
            "EAST": 2,
            "NORTH_WEST": 1,
            "NORTHWEST": 1,
            "NORTH_EAST": 1,
            "NORTHEAST": 1,
            "NORTH": 5,
            "UNKNOWN": 0,
        }
        df_encoded["gardenOrientation_encoded"] = df_encoded["gardenOrientation"].map(
            garden_orientation_mapping
        )
        print(
            f"Garden orientation encoded: {df_encoded['gardenOrientation_encoded'].value_counts().to_dict()}"
        )

    # 6. TERRACE ORIENTATION ENCODING (if exists)
    print("\n Encoding terrace orientation...")
    if "terraceOrientation" in df_encoded.columns:
        df_encoded["terraceOrientation_encoded"] = df_encoded["terraceOrientation"].map(
            garden_orientation_mapping
        )
        print(
            f"Terrace orientation encoded: {df_encoded['terraceOrientation_encoded'].value_counts().to_dict()}"
        )

    # 7. EPC SCORE ENCODING (Energy Performance Certificate - ordinal)
    print("\n Encoding EPC scores...")
    if "epcScore" in df_encoded.columns:
        epc_mapping = {"A+": 8, "A": 7, "B": 6, "C": 5, "D": 4, "E": 3, "F": 2, "G": 1}
        df_encoded["epcScore_encoded"] = df_encoded["epcScore"].map(epc_mapping)
        print(
            f"EPC encoding: {df_encoded['epcScore_encoded'].value_counts().sort_index().to_dict()}"
        )

    # 8. BOOLEAN FEATURES - Convert True/False to 1/0
    print("\n Encoding boolean features...")
    boolean_columns = [
        "hasAttic",
        "hasGarden",
        "hasAirConditioning",
        "hasArmoredDoor",
        "hasVisiophone",
        "hasTerrace",
        "hasOffice",
        "hasSwimmingPool",
        "hasFireplace",
        "accessibleDisabledPeople",
    ]

    for col in boolean_columns:
        if col in df_encoded.columns:
            df_encoded[f"{col}_encoded"] = df_encoded[col].map(
                {True: 1, False: 0, np.nan: 0}
            )
            print(
                f"{col} encoded: {df_encoded[f'{col}_encoded'].value_counts(dropna=False).to_dict()}"
            )

    return df_encoded, {
        "province_mapping": province_mapping,
        "type_mapping": type_mapping,
        "subtype_mapping": subtype_mapping,
        "locality_encoder": le_locality,
        "garden_orientation_mapping": (
            garden_orientation_mapping
            if "gardenOrientation" in df_encoded.columns
            else None
        ),
        "epc_mapping": epc_mapping if "epcScore" in df_encoded.columns else None,
    }


def preprocess_missing_values(df):
    """
    Handle missing values before encoding
    """
    print("\n Preprocessing missing values...")
    df_processed = df.copy()

    # Fill missing orientations with 'UNKNOWN'
    if "gardenOrientation" in df_processed.columns:
        df_processed["gardenOrientation"] = df_processed["gardenOrientation"].fillna(
            "UNKNOWN"
        )
        print("Filled missing garden orientations with 'UNKNOWN'")

    if "terraceOrientation" in df_processed.columns:
        df_processed["terraceOrientation"] = df_processed["terraceOrientation"].fillna(
            "UNKNOWN"
        )
        print("Filled missing terrace orientations with 'UNKNOWN'")

    # Fill missing EPC scores with 'UNKNOWN' (will be encoded as NaN)
    if "epcScore" in df_processed.columns:
        missing_epc = df_processed["epcScore"].isna().sum()
        print(f"EPC scores missing: {missing_epc}")

    # Fill boolean NaN with False (assuming missing means feature not present)
    boolean_cols = [
        "hasAttic",
        "hasGarden",
        "hasAirConditioning",
        "hasArmoredDoor",
        "hasVisiophone",
        "hasTerrace",
        "hasOffice",
        "hasSwimmingPool",
        "hasFireplace",
        "accessibleDisabledPeople",
    ]

    for col in boolean_cols:
        if col in df_processed.columns:
            missing_before = df_processed[col].isna().sum()
            df_processed[col] = df_processed[col].fillna(False)
            print(f"Filled {missing_before} missing {col} values with False")

    return df_processed


def create_final_ml_dataset(df_encoded):
    """
    Select and prepare final features for machine learning
    """
    print("\n Creating final ML dataset...")

    # Define potential feature columns
    numeric_features = [
        "bedroomCount",
        "bathroomCount",
        "habitableSurface",
        "toiletCount",
        "terraceSurface",
        "postCode",
    ]

    encoded_features = [
        "province_encoded",
        "type_encoded",
        "subtype_encoded",
        "locality_encoded",
        "gardenOrientation_encoded",
        "terraceOrientation_encoded",
        "epcScore_encoded",
    ]

    boolean_features = [
        "hasAttic_encoded",
        "hasGarden_encoded",
        "hasAirConditioning_encoded",
        "hasArmoredDoor_encoded",
        "hasVisiophone_encoded",
        "hasTerrace_encoded",
        "hasOffice_encoded",
        "hasSwimmingPool_encoded",
        "hasFireplace_encoded",
        "accessibleDisabledPeople_encoded",
    ]

    # Combine all potential features
    all_features = numeric_features + encoded_features + boolean_features + ["price"]

    # Filter to only include columns that exist in the dataset
    available_features = [col for col in all_features if col in df_encoded.columns]

    df_final = df_encoded[available_features].copy()

    # Fill remaining numeric NaN values with median
    numeric_cols = df_final.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != "price"]  # Don't fill price

    for col in numeric_cols:
        if df_final[col].isnull().sum() > 0:
            median_val = df_final[col].median()
            df_final[col] = df_final[col].fillna(median_val)
            print(f"Filled {col} NaN values with median: {median_val}")

    print(f"Final ML dataset shape: {df_final.shape}")
    print(f"Final features: {[col for col in df_final.columns if col != 'price']}")

    return df_final


def full_encoding_pipeline(df):
    """
    Complete preprocessing and encoding pipeline
    """
    print("\n=== STARTING FULL ENCODING PIPELINE ===")

    # 1. Handle missing values
    df_processed = preprocess_missing_values(df)

    # 2. Encode categorical features
    df_encoded, encoders = encode_categorical_features(df_processed)

    # 3. Create final ML dataset
    df_final = create_final_ml_dataset(df_encoded)

    print("\n=== ENCODING PIPELINE COMPLETED ===")
    return df_final, encoders


# MAIN EXECUTION
if __name__ == "__main__":

    # 1. Load your data:
    df = pd.read_csv("data/immoweb-dataset.csv")

    # 2. Clean the data:
    df_clean = clean_real_estate_data(df)

    # 3. Apply categorical encoding:
    df_encoded, encoding_mappings = full_encoding_pipeline(df_clean)

    # 4. Save the results:
    df_clean.to_csv("cleaned_real_estate_data.csv", index=False)
    df_encoded.to_csv("ml_ready_real_estate_data.csv", index=False)

    print(f"\n=== FINAL RESULTS ===")
    print(f"Cleaned data saved: cleaned_real_estate_data.csv ({df_clean.shape})")
    print(f"ML-ready data saved: ml_ready_real_estate_data.csv ({df_encoded.shape})")
    print(f"Encoding mappings: {list(encoding_mappings.keys())}")
