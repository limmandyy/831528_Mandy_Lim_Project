import requests
import json
from datetime import datetime
from pprint import pprint
import pandas as pd

# Get the current date
current_date = datetime.now()

iyear = int(input('Year:'))

# Define the endpoint URL with the start year and month and the current year and month
start_year = iyear
start_month = 1
current_year = current_date.year
current_month = current_date.month

# Create a list to store the data for each month
data_list = []

# Loop through the months from the start date to the current date
while start_year <= current_year:
    # Define the endpoint URL for the current month
    endpoint_url = f"https://api.bnm.gov.my/public/msb/1.10/year/{start_year}/"

    # Define the header parameters
    headers = {
        "Accept": "application/vnd.BNM.API.v1+json"
    }

    # Make a GET request to the API
    response = requests.get(endpoint_url, headers=headers)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        data = json.loads(response.text)

        # Append the data to the list
        data_list.append(data)
    else:
        print(f"Failed to retrieve data for year {start_year}, month {start_month}. HTTP Status Code: {response.status_code}")

    # Move to the next month
    if start_month == 12:
        start_year += 1
        start_month = 1
    else:
        start_month += 1

# Extracted Json data 
pprint(data_list)
json_data = data_list

# Extract the 'data' part for time series
data_list = [item['data'] for item in json_data]

# Flatten the list of dictionaries
flat_data_list = [entry for sublist in data_list for entry in sublist]

# Create a DataFrame
df = pd.DataFrame(flat_data_list)

# Convert 'date' column to datetime type
df['year_dt'] = pd.to_datetime(df['year_dt'])

# Sort the DataFrame by 'date' in ascending order
df.sort_values(by='year_dt', inplace=True)

# Set 'date' column as the index
df.set_index('year_dt', inplace=True)

# Filter the DataFrame to include only rows where 'pur' is 'Jumlah / Total'
filtered_df = df[df['pur'] == 'Jumlah / Total']

# Sorting the DataFrame according to the month
filtered_df = filtered_df.sort_values(by='month_dt')

# Convert 'tot_loa_app' values to numeric (float) and then divide by 1000
filtered_df['tot_loa_app'] = pd.to_numeric(filtered_df['tot_loa_app']) / 1000

# Save the DataFrame to a CSV file
filtered_df.to_csv('loan_data.csv')