import pandas as pd
import matplotlib.pyplot as plt

# --- Load and clean Rack data ---
rack_df = pd.read_csv("Book1.csv", skiprows=2)
rack_df = rack_df[rack_df['DATE'].notna() & (rack_df['DATE'].str.strip() != '')]
rack_df['DATE'] = pd.to_datetime(rack_df['DATE'] + '-2024', format='%d-%b-%Y', errors='coerce')

# Convert Rack Average columns from cents to dollars
rack_df['RACK AVERAGE'] = rack_df['RACK AVERAGE'] * 0.01
rack_df['RACK AVERAGE.1'] = rack_df['RACK AVERAGE.1'] * 0.01

# --- Load NYMEX data ---
nymex_df = pd.read_csv("nymex_ulsd_jan_to_mar_2024.csv")
nymex_df['DATE'] = pd.to_datetime(nymex_df['DATE'] + '-2024', format='%d-%b-%Y', errors='coerce')
nymex_df = nymex_df.dropna(subset=['DATE'])

# --- Merge datasets on DATE ---
merged_df = pd.merge(rack_df, nymex_df, on='DATE', how='inner')

# --- Plotting ---
plt.figure(figsize=(12, 6))
plt.plot(merged_df['DATE'], merged_df['RACK AVERAGE'], label='Gross - Rack Average ($)', marker='o', color='blue')
plt.plot(merged_df['DATE'], merged_df['RACK AVERAGE.1'], label='Net - Rack Average ($)', marker='x', color='green')
plt.plot(merged_df['DATE'], merged_df['PRICE'], label='NYMEX ULSD ($)', marker='s', color='red')

# Formatting
plt.title("Rack Averages vs NYMEX ULSD - Jan to Dec 2024")
plt.xlabel("Date")
plt.ylabel("Price (USD per gallon)")
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.tight_layout()

import mplcursors
import matplotlib.dates as mdates

# Get all line objects (Gross, Net, NYMEX)
lines = plt.gca().lines

# Hover mode: follow cursor, disappears automatically
cursor = mplcursors.cursor(lines, hover=True)

@cursor.connect("add")
def on_add(sel):
    x = mdates.num2date(sel.target[0])
    y = sel.target[1]
    date_str = x.strftime('%A, %Y-%m-%d')  # Day of the week + date
    label = sel.artist.get_label()

    # Get NYMEX price for that date
    nymex_price = merged_df.loc[merged_df['DATE'] == pd.Timestamp(x.date()), 'PRICE']
    spread_text = ""

    if "Rack Average" in label and not nymex_price.empty:
        spread = y - nymex_price.values[0]
        spread_text = f"\nSpread vs NYMEX: ${spread:.3f}"

    sel.annotation.set(text=f"{label}\n{date_str}\n${y:.3f}{spread_text}")


def on_move(event):
    if event.inaxes != plt.gca():
        for sel in cursor.selections:
            sel.annotation.set_visible(False)
        plt.draw()


# --- And connect it here ---
plt.gcf().canvas.mpl_connect("motion_notify_event", on_move)

plt.show()

