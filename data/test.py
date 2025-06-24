import pandas as pd 

df = pd.read_csv("data/immoweb-dataset.csv")



missing_summary = df.isnull().sum()
missing_summary = missing_summary[missing_summary > 0].sort_values(ascending=False)
for col, missing_count in missing_summary.items():
    pct = (missing_count / len(df)) * 100
    print(f"  {col}: {missing_count} ({pct:.1f}%)")
    
