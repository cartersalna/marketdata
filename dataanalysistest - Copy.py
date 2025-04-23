import pandas as pd
from datetime import datetime

# Load Excel file
file_path = "2024 Postings.xlsx"
df = pd.read_excel(file_path, sheet_name=1, skiprows=2)

# Standardize columns
df.columns = [col.strip().upper() for col in df.columns]

# Check necessary columns
if 'DATE' not in df.columns or 'RACK LOW' not in df.columns:
    raise ValueError("Required columns 'DATE' and 'RACK LOW' are missing.")

df = df[['DATE', 'RACK LOW']]
df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
df = df.dropna(subset=['DATE'])

# Add helpers
df['YEAR'] = df['DATE'].dt.year
df['MONTH'] = df['DATE'].dt.month

# Filter up to today
today = pd.to_datetime(datetime.today().date())
df = df[df['DATE'] <= today].sort_values('DATE').reset_index(drop=True)

# Monthly averages (only for last date of each month)
monthly_avg_df = df.groupby(['YEAR', 'MONTH'])['RACK LOW'].mean().round(4).reset_index()
monthly_avg_df.rename(columns={'RACK LOW': 'MONTHLY_AVG'}, inplace=True)

# Merge monthly avg to all rows
df = df.merge(monthly_avg_df, on=['YEAR', 'MONTH'], how='left')

# Set MONTHLY_AVG only on last date of each month
df['IS_MONTH_END'] = df.groupby(['YEAR', 'MONTH'])['DATE'].transform('max') == df['DATE']
df['MONTHLY_AVG'] = df['MONTHLY_AVG'].where(df['IS_MONTH_END'])
df.drop(columns='IS_MONTH_END', inplace=True)

# Year-to-date average (only on first date)
ytd_avg = df['RACK LOW'].mean().round(4)
first_date = df['DATE'].min()
df['YTD_AVG'] = df['DATE'].apply(lambda x: ytd_avg if x == first_date else None)

# Final formatting
df['DATE'] = df['DATE'].dt.date
df = df[['DATE', 'RACK LOW', 'MONTHLY_AVG', 'YTD_AVG']]

# Save to Excel and add chart
output_file = "2024_Postings_with_Averages.xlsx"
with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
    df.to_excel(writer, sheet_name='Averages', index=False)
    workbook = writer.book
    worksheet = writer.sheets['Averages']

    # Format columns and widths
    date_format = workbook.add_format({'num_format': 'yyyy-mm-dd'})
    worksheet.set_column('A:A', 15, date_format)
    worksheet.set_column('B:D', 14)

    # Chart data prep
    chart_data = monthly_avg_df.copy()
    chart_data['LABEL'] = chart_data.apply(lambda row: f"{int(row['MONTH']):02}-{int(row['YEAR'])}", axis=1)


    # Write chart data offscreen (to avoid clutter)
    chart_data_col = 6  # Column G
    chart_data_row = 0
    worksheet.write(chart_data_row, chart_data_col, 'MONTH')
    worksheet.write(chart_data_row, chart_data_col + 1, 'AVG')
    for i, row in chart_data.iterrows():
        worksheet.write(chart_data_row + 1 + i, chart_data_col, row['LABEL'])
        worksheet.write(chart_data_row + 1 + i, chart_data_col + 1, row['MONTHLY_AVG'])

    # Create line chart
    chart = workbook.add_chart({'type': 'line'})
    chart.add_series({
        'name': 'Monthly Rack Low Avg',
        'categories': ['Averages', chart_data_row + 1, chart_data_col,
                       chart_data_row + len(chart_data), chart_data_col],
        'values': ['Averages', chart_data_row + 1, chart_data_col + 1,
                   chart_data_row + len(chart_data), chart_data_col + 1],
        'line': {'color': 'blue'},
        'marker': {'type': 'circle', 'size': 5},
    })
    chart.set_title({'name': 'Monthly Rack Low Average'})
    chart.set_x_axis({'name': 'Month'})
    chart.set_y_axis({'name': 'Rack Low ($)'})
    chart.set_style(10)

    # Insert chart to the right of the YTD column (which is column D → E is good)
    worksheet.insert_chart('E2', chart)

print(f"✅ Excel saved with chart placed beside yearly average: {output_file}")
