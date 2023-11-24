import pandas as pd
import dash
from dash import dcc, html, Input, Output
import base64
from io import BytesIO
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import calendar
import plotly.express as px

# Function to generate SARIMA forecast chart
def generate_sarima_chart():
    # Load the previously saved CSV file (Replace 'loan_data.csv' with the path to your CSV file)
    df = pd.read_csv('loan_data.csv')

    # Convert 'year_dt' and 'month_dt' columns to datetime
    df['year_dt'] = df['year_dt'].astype(str)
    df['month_dt'] = df['month_dt'].astype(str)
    df['date'] = pd.to_datetime(df['year_dt'] + '-' + df['month_dt'], format='%Y-%m')

    # Set 'date' column as the index
    df.set_index('date', inplace=True)

    # Filter the data from 2019 to the latest
    start_date = '2019-01-01'
    end_date = df.index[-1]  # Assuming the latest date is the last entry in the DataFrame

    filtered_df = df[start_date:end_date]

    # SARIMA Model
    order = (1, 1, 1)  # (p, d, q)
    seasonal_order = (1, 1, 1, 12)  # (P, D, Q, S)
    model = SARIMAX(filtered_df['tot_loa_app'], order=order, seasonal_order=seasonal_order)
    sarima_results = model.fit()

    # Forecasting
    forecast_steps = 12  # Forecasting for the next 12 steps (months)
    forecast = sarima_results.get_forecast(steps=forecast_steps)

    # Getting forecast values and confidence intervals
    forecast_values = forecast.predicted_mean
    confidence_intervals = forecast.conf_int()

    # Plotting the forecasted data
    plt.figure(figsize=(12, 6))
    plt.plot(filtered_df.index, filtered_df['tot_loa_app'], marker='o', linestyle='-', label='Actual Data')
    plt.plot(forecast_values.index, forecast_values, label='SARIMA Forecast', linestyle='--', color='red')
    plt.fill_between(confidence_intervals.index, confidence_intervals.iloc[:, 0], confidence_intervals.iloc[:, 1], color='pink', alpha=0.3)
    plt.xlabel('Date')
    plt.ylabel('Total (RM billion)')
    plt.title('SARIMA Forecast of Loan Applications')
    plt.legend()
    plt.grid(True)
    
    # Connecting the last point of the original plot with the first point of the forecasted plot
    plt.plot([filtered_df.index[-1], forecast_values.index[0]],
             [filtered_df['tot_loa_app'].iloc[-1], forecast_values.iloc[0]],
             linestyle='--', color='red')

    # Save the plot to a buffer
    img_buf = BytesIO()
    plt.savefig(img_buf, format='png')
    plt.close()
    
    img_buf.seek(0)
    return img_buf

# Function to encode the generated SARIMA chart image to base64
def encode_image(image):
    encoded = base64.b64encode(image.getvalue()).decode('utf-8')
    return f'data:image/png;base64,{encoded}'

# Load your data from the CSV file
data_path = 'loan_data.csv'
df = pd.read_csv(data_path)

# Purpose labels
purpose_labels = {
    'pur_sec': 'Purchase of Security',
    'pur_tra_veh': 'Purchase of Transport Vehicles',
    'pur_res_pro': 'Purchase of Residential Property',
    'pur_non_res_pro': 'Purchase of Non-residential Property',
    'pur_fix_ass_oth_lan_and_bui': 'Purchase of Fixed Assets other than Land and Building',
    'per_use': 'Personal uses',
    'cre_car': 'Credit Card',
    'pur_con_goo': 'Purchase of Consumer Durable Goods',
    'con': 'Construction',
    'wor_cap': 'Working Capital',
    'oth_pur': 'Other Purposes'
}

# Get unique years and months from the dataset
unique_years = df['year_dt'].unique()
unique_months = df['month_dt'].unique()

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Loan Analysis Dashboard"),
    html.P("Please select a year and month:", style={'font-size': '20px', 'font-family': 'Times New Roman', 'font-weight': 'bold'}),
    html.Div([
        dcc.Dropdown(
            id='year-dropdown',
            options=[{'label': year, 'value': year} for year in unique_years],
            value=unique_years.min(),
        ),
        dcc.Dropdown(
            id='month-dropdown',
            options=[{'label': calendar.month_name[int(month)], 'value': month} for month in unique_months],
            value=unique_months.min(),
        ),
        html.Div([
            html.Img(id='sarima-chart')
        ], style={'margin': 'auto'}),
        html.Div([
            dcc.Graph(id='bar-chart', config={'displayModeBar': False, 'staticPlot': False}),
        ], style={'margin': 'auto'})
    ], style={'text-align': 'center'})
])

loan_purpose_columns = [
    'pur_sec', 'pur_tra_veh', 'pur_res_pro', 'pur_non_res_pro',
    'pur_fix_ass_oth_lan_and_bui', 'per_use', 'cre_car',
    'pur_con_goo', 'con', 'wor_cap', 'oth_pur'
]

def update_charts(selected_year, selected_month):
    filtered_df = df[(df['year_dt'] == selected_year) & (df['month_dt'] == selected_month)]

    purpose_sums = filtered_df[loan_purpose_columns].sum()
    purpose_sums = purpose_sums / 1000

    purpose_sums_df = pd.DataFrame({
        'Loan Purpose': [purpose_labels[column] for column in purpose_sums.index],
        'Total Value (Billion)': purpose_sums.values
    })

    bar_fig = px.bar(
        purpose_sums_df,
        x='Loan Purpose',
        y='Total Value (Billion)',
        labels={'Total Value (Billion)': 'Total (RM Billion)', 'Loan Purpose': 'Loan Purpose'},
        title=f'Loan Purpose Distribution in {calendar.month_name[int(selected_month)]}, {selected_year}',
        text=purpose_sums.values.round(2)
    )

    chart_width = 1250
    chart_height = 700
    bar_fig.update_layout(width=chart_width, height=chart_height)

    return bar_fig

@app.callback(
    Output('sarima-chart', 'src'),
    [Input('year-dropdown', 'value'), Input('month-dropdown', 'value')]
)
def update_sarima_chart(selected_year, selected_month):
    sarima_image = generate_sarima_chart()
    encoded_image = encode_image(sarima_image)
    return encoded_image

@app.callback(
    Output('bar-chart', 'figure'),
    [Input('year-dropdown', 'value'), Input('month-dropdown', 'value')]
)
def update_bar_chart(selected_year, selected_month):
    return update_charts(selected_year, selected_month)

if __name__ == '__main__':
    app.run_server(debug=True)
