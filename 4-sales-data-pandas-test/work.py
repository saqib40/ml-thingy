import pandas as pd

# load the data
try:
    data = pd.read_csv("./data/sales_data_sample.csv", encoding='latin1')
except UnicodeDecodeError:
    data = pd.read_csv("./data/sales_data_sample.csv", encoding='windows-1252')

# info
print(data.info())
print("*" * 20)

# clean the data

# 1- fix the ORDERDATE
data["ORDERDATE"] = pd.to_datetime(data["ORDERDATE"])
# 2- drop ADDRESSLINE2, only 302 non-null values
# 3- STATE??
# 4- POSTALCODE??
# 5- TERRITORY??
# Dropping them all since i won't need them for analysis
columns_to_drop = ['ADDRESSLINE2', 'STATE', 'POSTALCODE', 'TERRITORY']
data.drop(columns=columns_to_drop, inplace=True)

# info again
print(data.info())
print("*" * 20)

# analysis

# What was the total sales per month?
print("Total sales per month: ")
print(data.groupby('MONTH_ID')['SALES'].sum())

# What are your top 5 best-selling products?
print("Top 5 best-selling products: ")
top_products = data.groupby("PRODUCTLINE")["SALES"].sum().sort_values(ascending=False)
print(top_products.head(5))

# What city ('CITY') had the highest sales in 2004?
print("City with highest sales in 2004: ")
data_2004 = data[data['YEAR_ID'] == 2004]
city_sales_2004 = data_2004.groupby("CITY")["SALES"].sum().sort_values(ascending=False)
print(city_sales_2004.head(1))

# Who are your top 5 customers?
print("Top 5 customers: ") # wrt sales
top_customers = data.groupby('CUSTOMERNAME')['SALES'].sum().sort_values(ascending=False)
print(top_customers.head(5))