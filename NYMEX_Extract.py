import yfinance as yf
import pandas as pd

ulsd_ticker = "HO=F"
start_date = "2024-01-01"
end_date = "2024-12-31"

ulsd_data = yf.download(ulsd_ticker, start=start_date, end=end_date, auto_adjust=True)

# Flatten MultiIndex columns if needed
if isinstance(ulsd_data.columns, pd.MultiIndex):
    ulsd_data.columns = ['_'.join(col).strip() for col in ulsd_data.columns.values]

# Print column names to confirm what we have
print("Columns after fetch:", ulsd_data.columns)

# Try to find the Close price column dynamically
close_col = [col for col in ulsd_data.columns if 'close' in col.lower()]
if not close_col:
    raise ValueError("No 'Close' column found in data!")

# Use the first match as the Close column
ulsd_data = ulsd_data[[close_col[0]]].reset_index()
ulsd_data.rename(columns={'Date': 'DATE', close_col[0]: 'PRICE'}, inplace=True)

# Format date
ulsd_data['DATE'] = ulsd_data['DATE'].dt.strftime('%d-%b')

# Save to CSV
ulsd_data.to_csv("nymex_ulsd_jan_to_mar_2024.csv", columns=['DATE', 'PRICE'], index=False)

# Show a preview
print(ulsd_data.head())
