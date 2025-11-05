# Grouping and aggregation
# aka Split-Apply-Combine

import pandas as pd

data = {
    'Region': ['East', 'West', 'East', 'West', 'East', 'West', 'East', 'East'],
    'Product': ['A', 'A', 'B', 'B', 'A', 'B', 'A', 'B'],
    'Sales': [100, 150, 50, 200, 120, 180, 110, 60],
    'Quantity': [10, 15, 5, 20, 12, 18, 11, 6]
}
df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)
print("-" * 40)

# Grouping by a single column and aggregating
regional_sales = df.groupby('Region')['Sales'].sum()
print("Total Sales per Region:")
print(regional_sales)
print("-" * 40)

# Grouping by multiple columns
multi_level_summary = df.groupby(['Region', 'Product'])['Quantity'].mean()
print("Average Quantity per Region and Product:")
print(multi_level_summary)
print("-" * 40)

# Applying multiple aggregation functions to multiple columns
regional_summary = df.groupby('Region').agg(
    Total_Sales=('Sales', 'sum'),        # Rename output column to 'Total_Sales' and apply 'sum' to 'Sales'
    Average_Quantity=('Quantity', 'mean') # Rename output column and apply 'mean' to 'Quantity'
)
print("Summary using .agg():")
print(regional_summary)
