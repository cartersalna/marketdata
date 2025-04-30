import streamlit as st
import pandas as pd
import plotly.express as px
from dateutil import parser
import io
from io import StringIO
import yfinance as yf
import os
import numpy as np

st.set_page_config(page_title="Rack vs NYMEX vs Platts Viewer", layout="wide")
st.title("📄 View Rack, NYMEX, and Platts Data")

st.markdown("Upload **Rack Pricing CSV**, **NYMEX CSV**, and **Platts CSV** to view the raw data.")

def unify_date_format(date_string, output_format='%m-%d-%y'):
    try:
        parsed_date = parser.parse(date_string)
        return parsed_date.strftime(output_format)
    except Exception:
        return None

def find_skip_count(file_obj, header_column="DATE"):
    """
    Function to find the number of rows to skip in the CSV file
    based on the presence of the header column.
    """
    file_obj.seek(0)  # Ensure we're at the start of the file
    lines = file_obj.readlines()  # Read the entire file into lines

    skip_count = 0
    for line in lines:
        skip_count += 1  # Start counting rows
        if header_column in line:  # If the "DATE" column is found
            break  # Stop as soon as we find it

    return skip_count - 1  # Subtract 1 because we should skip the lines before the header row

rack_df = None
nymex_df = None
platts_df = None

if "merged_sheet_df" in st.session_state:
    merged_sheet_df = st.session_state["merged_sheet_df"]
else:
    merged_sheet_df = None
    
if 'merged_df' not in st.session_state:
    st.session_state.merged_df = None

rack_file = st.file_uploader("Upload .xlsx or .csv file:", type=["csv", "xls", "xlsx"])

selected_df = None

if rack_file is not None:
    file_name = rack_file.name.lower()

    if file_name.endswith(".csv"):
        # CSV file: call find_skip_count directly
        skip_rows = find_skip_count(rack_file, header_column="DATE")
        rack_file.seek(0)
        selected_df = pd.read_csv(rack_file, skiprows=skip_rows)

        # Prefix column names with the filename (without extension)
        prefix = os.path.splitext(rack_file.name)[0]  # "OPIS ULSC2.csv" -> "OPIS ULSC2"
        selected_df.columns = [f"{prefix} {col}" if col != "DATE" else "DATE" for col in selected_df.columns]

        st.success("✅ CSV uploaded and parsed successfully.")

    elif file_name.endswith((".xls", ".xlsx")):
        xl = pd.ExcelFile(rack_file)
        sheet_choices = st.multiselect("Select a sheet to treat as CSV", xl.sheet_names)

        dataframes_to_merge = []
        if sheet_choices:
            for sheet_choice in sheet_choices:
                df_sheet = xl.parse(sheet_name=sheet_choice, header=None)
               
                # Find the first row that contains the header (row with 'DATE')
                header_row_idx = df_sheet[df_sheet.apply(lambda row: row.astype(str).str.contains("DATE", case=False).any(), axis=1)].index
                if len(header_row_idx) == 0:
                    st.error(f"❌ No 'DATE' header found in sheet '{sheet_choice}'.")
                    continue

                skip_rows = header_row_idx[0]
                df_sheet = xl.parse(sheet_name=sheet_choice, skiprows=skip_rows)
                # Rename columns: prefix except 'DATE'
                df_sheet.columns = [f"{sheet_choice} {col}" if col != "DATE" else "DATE" for col in df_sheet.columns]

                dataframes_to_merge.append(df_sheet)
                st.success(f"✅ Excel sheet '{sheet_choice}' parsed successfully.")

            if dataframes_to_merge:
                if st.button("🔗 Merge Selected Sheets"):
                    from functools import reduce

                    for i in range(len(dataframes_to_merge)):
                        dataframes_to_merge[i]['DATE'] = pd.to_datetime(dataframes_to_merge[i]['DATE'], errors='coerce')

                    merged_df = dataframes_to_merge[0]

                    # Merge subsequent dataframes one by one
                    for df in dataframes_to_merge[1:]:
                        merged_df = pd.merge(merged_df, df, on="DATE", how="outer")  # Use 'outer' join to keep all dates

                    # Sort by DATE after merging
                    merged_df = merged_df.sort_values(by="DATE")

                    # Replace 0s with NaN so we can clean rows and columns effectively
                    merged_df.replace(0, np.nan, inplace=True)

                    # Drop rows where all values are NaN (i.e., originally empty or all zeros)
                    merged_df = merged_df.dropna(how="all")

                    # Drop columns where all values are NaN (i.e., originally empty or all zeros)
                    merged_df = merged_df.dropna(axis=1, how="all")

                    # Store merged result in session state
                    st.session_state["merged_sheet_df"] = merged_df

                    st.success(f"✅ Merged {len(dataframes_to_merge)} sheets successfully.")
                    st.write("📊 Merged Data Preview:")
                    st.dataframe(st.session_state["merged_sheet_df"])


if selected_df is not None:
    st.write("📊 Preview of uploaded data:")
    st.dataframe(selected_df)

##nymex_file = st.file_uploader("Upload NYMEX Data", type=["csv"])
platts_file = st.file_uploader("Upload Platts Data", type=["csv"])

col1, col2 = st.columns([2, 1])

with col1:
    nymex_file = st.file_uploader("Upload NYMEX Data", type=["csv"], key="nymex")

with col2:
    st.write("")
    st.write("")
    st.write("")
    pull_data = st.button("📥 Pull NYMEX Data")

nymex_df = None  # Initialize

# Assuming you have the file and using this function
if selected_df is not None:
    st.write("📊 Preview of uploaded data:")
    with st.expander("View OPIS Rack Data 🔽"):
        st.dataframe(selected_df)

    if 'DATE' in selected_df.columns:
        selected_df['DATE'] = selected_df['DATE'].astype(str).apply(unify_date_format)
        selected_df = selected_df.dropna(subset=['DATE'])
    else:
        st.error("Uploaded data missing 'DATE' column.")
        st.stop()

if platts_file:
    st.subheader("📈 Platts Data")

    # Convert the uploaded file to a StringIO object (in-memory file object)
    file_content = platts_file.getvalue().decode("utf-8")
    file_obj = StringIO(file_content)

    # Use the function to determine how many rows to skip
    skip_count = find_skip_count(file_obj)

    # Now read the CSV with the skiprows value calculated above
    file_obj.seek(0)  # Reset file pointer
    platts_df = pd.read_csv(file_obj, skiprows=skip_count)

    # Strip column names of extra spaces
    platts_df.columns = platts_df.columns.str.strip()

    # Find the column that contains the date
    date_col = next((col for col in platts_df.columns if 'date' in col.lower()), None)

    if date_col:
        platts_df[date_col] = platts_df[date_col].astype(str).apply(unify_date_format)
        platts_df = platts_df.dropna(subset=[date_col])
        platts_df = platts_df.rename(columns={date_col: 'DATE'})
    else:
        st.error("Could not find a 'DATE' column in the Platts data.")
        st.stop()

    # Display the dataframe
    with st.expander("View Platts Data 🔽"):
        st.dataframe(platts_df)


if nymex_file:
    file_content = nymex_file.getvalue().decode("utf-8")
    file_obj = StringIO(file_content)
    skip_count = find_skip_count(file_obj)
    file_obj.seek(0)
    nymex_df = pd.read_csv(file_obj, skiprows=skip_count)
    nymex_df.columns = nymex_df.columns.str.strip()

    date_col = next((col for col in nymex_df.columns if 'date' in col.lower()), None)

    if date_col:
        nymex_df[date_col] = nymex_df[date_col].astype(str).apply(unify_date_format)
        nymex_df = nymex_df.dropna(subset=[date_col])
        nymex_df = nymex_df.rename(columns={date_col: 'DATE'})
    else:
        st.error("Could not find a 'DATE' column in the NYMEX data.")
        st.stop()

    st.success("✅ NYMEX CSV Uploaded Successfully")
    with st.expander("View NYMEX Data 🔽 "):
        st.dataframe(nymex_df)


elif pull_data or "pull_nymex" in st.session_state:
    st.session_state["pull_nymex"] = True

    st.markdown("### 📆 Select Date Range for NYMEX ULSD Data")

    if "nymex_dates" not in st.session_state:
        today = pd.Timestamp.today()
        st.session_state["nymex_dates"] = [today - pd.Timedelta(days=30), today]

    st.session_state["nymex_dates"] = st.date_input(
        "Choose start and end date",
        value=st.session_state["nymex_dates"],
        key="nymex_date_input"
    )

    if len(st.session_state["nymex_dates"]) == 2:
        start_date, end_date = st.session_state["nymex_dates"]
        if st.button("Fetch NYMEX ULSD from Yahoo Finance"):
            with st.spinner("Fetching data from Yahoo Finance..."):
                try:
                    ulsd = yf.download("HO=F", start=start_date, end=end_date)

                    if ulsd is not None and not ulsd.empty:
                        ulsd.reset_index(inplace=True)
                        ulsd['DATE'] = ulsd['Date'].dt.strftime('%m-%d-%y')
                        ulsd = ulsd[['DATE', 'Close']].rename(columns={'Close': 'NYMEX_ULSD'})

                        ulsd.columns = ['_'.join(col) for col in ulsd.columns]
                        ulsd = ulsd.rename(columns={'DATE_': 'DATE'})
                        st.session_state["nymex_df"] = ulsd

                        st.success("✅ Fetched NYMEX ULSD prices from Yahoo Finance.")
                        st.dataframe(st.session_state["nymex_df"])
                        
                    else:
                        st.warning("No data returned for the selected date range.")
                except Exception as e:
                    st.error(f"Failed to fetch data: {e}")

    else:
        st.info("Please select both start and end dates.")


if 'merged_df' not in st.session_state:
    st.session_state.merged_df = None

generate_button = st.button("Merge Data (by date) & Generate Graph ->", key="generate_button")
nymex_df = st.session_state.get("nymex_df")
if merged_sheet_df is not None:
    rack_df = merged_sheet_df

if generate_button and any([rack_df is not None, nymex_df is not None, platts_df is not None]):
    dfs = []
    
    if rack_df is not None:
        rack_df = rack_df.dropna(subset=['DATE'])
        rack_df.columns = rack_df.columns.str.strip().str.upper()
        #rack_df = rack_df[rack_df['DATE'].apply(lambda x: isinstance(x, str) and len(x) > 0)]
        dfs.append(rack_df)
        

    if nymex_df is not None:
        nymex_df.columns = nymex_df.columns.str.strip().str.upper()
        nymex_df = nymex_df.dropna(subset=['DATE'])
        dfs.append(nymex_df)

    if platts_df is not None:
        platts_df = platts_df.dropna(subset=['DATE'])
        platts_df = platts_df[platts_df['DATE'].apply(lambda x: isinstance(x, str) and len(x) > 0)]
        dfs.append(platts_df)

    if dfs:
        for i in range(len(dfs)):
            dfs[i]['DATE'] = pd.to_datetime(dfs[i]['DATE'], errors='coerce')

        from functools import reduce
        merged_df = reduce(lambda left, right: pd.merge(left, right, on="DATE", how="inner"), dfs)

        merged_df["DATE"] = pd.to_datetime(merged_df["DATE"], format='%m-%d-%y', errors='coerce')
        merged_df = merged_df.sort_values(by="DATE")

        for col in merged_df.columns:
            if col != "DATE" and (
                "RACK" in col or "PLATTS" in col or "MARATHON" in col or "branded" in col or "unbranded" in col
            ) and merged_df[col].dtype in ['float64', 'int64']:
                merged_df[col] = merged_df[col] / 100

        merged_df = merged_df.dropna(axis=1, how='all')
        st.session_state.merged_df = merged_df  # Save merged data in session state
    else:
        st.session_state.merged_df = None

# If we have merged data, let the user interact with it
if st.session_state.merged_df is not None:
    
    merged_df = st.session_state.merged_df
    
    st.subheader("📊 Merged Data (Rack + NYMEX + Platts)")
    st.dataframe(merged_df)

    columns = merged_df.columns.tolist()
    if 'DATE' in columns:
        columns.remove('DATE')

    if columns:
        selected_columns = st.multiselect("Select columns to compare", columns, default=[columns[0]])
    else:
        st.warning("No columns available for comparison.")
        st.stop()
    available_dates = merged_df["DATE"].dt.date.unique().tolist()
    if available_dates:
        start_date, end_date = st.select_slider(
            "Select Date Range for Graph",
            options=available_dates,
            value=(available_dates[0], available_dates[-1])
        )
    else:
        st.error("No valid dates available for plotting.")
        st.stop()

    filtered_df = merged_df[
        (merged_df["DATE"].dt.date >= start_date) & (merged_df["DATE"].dt.date <= end_date)
    ]

    line_style = st.selectbox("Select Line Style", ["solid", "dash", "dot"], index=0)
    line_width = st.slider("Select Line Width", 1, 10, 2)

    if selected_columns and not filtered_df.empty:
        fig = px.line(
            filtered_df,
            x="DATE",
            y=selected_columns,
            title="📈 Rack vs NYMEX vs Platts Price Trends",
            line_shape="linear"
        )

        # Update line style and width
        for trace in fig.data:
            trace.line.width = line_width
            trace.line.dash = line_style

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No data to display for the selected range or columns.")