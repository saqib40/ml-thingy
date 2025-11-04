import pandas as pd

df = pd.read_csv('./data/data.csv')

# print(df.to_string()) # -> prints all rows
print(df) # -> prints first 5 rows and last 5 rows
print(pd.options.display.max_rows)
# prints 60 =>  if the DataFrame contains more than 60 rows, 
# the print(df) statement will return only the headers and 
# the first and last 5 rows. you can update it as well