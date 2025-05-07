import pandas as pd
import numpy as np
from scipy import stats
import io
import streamlit as st


def read_file(uploaded_file):

    # Reads CSV, JSON, Excel
    name = uploaded_file.name
    df = None

    # Allow multiple extensions
    try:
        if name.endswith(".csv"):
            # Try standard UTF-8 first, then latin1 as fallback
            try:
                df = pd.read_csv(uploaded_file)
            except UnicodeDecodeError:
                uploaded_file.seek(0)  # Rewind file before trying again
                df = pd.read_csv(uploaded_file, encoding="latin1")
        elif name.endswith(".json"):
            df = pd.read_json(uploaded_file)
        elif name.endswith((".xls", ".xlsx")):
            engine = "openpyxl" if name.endswith(".xlsx") else "xlrd"
            df = pd.read_excel(uploaded_file, engine=engine)
        else:
            st.error(
                f"Unsupported file type: {name}. Please upload CSV, JSON, XLS, or XLSX."
            )
            return None  # Return None for unsupported types
        
    except Exception as e:
        st.error(f"Error reading file '{name}': {e}")
        return None  # Return None on read error



    # remove fully empty rows/columns and check if file not completely empty
    if df is not None:
        df.dropna(axis=0, how="all", inplace=True)
        df.dropna(axis=1, how="all", inplace=True)
        if df.empty:
            st.warning(
                "File loaded successfully, but it appears to be empty after removing blank rows/columns."
            )
            return None
    return df


# EDA (Expolatory Data Analysis) Summary
def generate_eda_report(df):
    report = {"basic_info": {}, "column_analysis": {}}
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()

    report["basic_info"] = {
        "Rows": len(df),
        "Columns": len(df.columns),
        "Duplicate Rows": df.duplicated().sum(),
        "Total Missing Values": df.isnull().sum().sum(),
        "DataFrame Info": info_str,  # Renamed for clarity
    }

    # Analyze each column
    col_summaries = {}
    for col in df.columns:
        data = df[col]
        missing_count = data.isnull().sum()
        missing_pct = (missing_count / len(df) * 100) if len(df) > 0 else 0
        unique_count = data.nunique()

        stats_dict = {
            "Data Type": str(data.dtype),
            "Missing Values": f"{missing_count} ({missing_pct:.1f}%)",
            "Unique Values": unique_count,
        }

        # Add type-specific stats
        if pd.api.types.is_numeric_dtype(data): #numerical data
            stats_dict.update(get_descriptive_stats(data.dropna()))
            #drop na columns for stats

        elif pd.api.types.is_datetime64_any_dtype(data): #date time data
            stats_dict.update({"Minimum Date": data.min(), "Maximum Date": data.max()})
        elif pd.api.types.is_object_dtype(data) or pd.api.types.is_categorical_dtype( #string
            data
        ):
            # Show top 5 most frequent values
            top_5 = data.value_counts().head(5).to_dict()
            stats_dict["Top 5 Values"] = top_5 if top_5 else "N/A"

        col_summaries[col] = stats_dict

    report["column_analysis"] = col_summaries
    return report


# Fills or drops missing values based on strategy per column
def handle_missing_values(df, strategy_dict):

    df_processed = df.copy() #make a copy of dataset
    rows_before = len(df_processed)
    columns_processed = []

    for column, strat in strategy_dict.items():
        if column not in df_processed.columns:
            st.warning(f"Column '{column}' not found for handling missing values.")
            continue 

        method = strat.get("method")
        if method == "drop":
            df_processed.dropna(subset=[column], inplace=True)
            columns_processed.append(column)
        elif method == "fill":
            fill_val = strat.get("value")
            # Basic type check for fill value if needed, though None usually works
            if fill_val is not None:
                df_processed[column].fillna(fill_val, inplace=True)
                columns_processed.append(column)
            else:
                st.warning(
                    f"No fill value provided for column '{column}'. Skipping fill."
                )
        else:
            st.warning(
                f"Unknown missing value strategy '{method}' for column '{column}'."
            )

    rows_dropped = rows_before - len(df_processed)
    if not columns_processed:
        st.info("No missing value operations were performed.")
    return df_processed, rows_dropped


# Removes rows where the Z-score in specified numeric columns exceeds the threshold
def remove_outliers(df, columns, threshold=3.0):
    df_processed = df.copy()
    initial_rows = len(df_processed)
    outlier_indices = set()

    if not columns:
        st.warning("No columns selected for outlier removal.")
        return df_processed, 0 #0 rows removed

    #check for numeric data
    numeric_columns_found = [
        col for col in columns if (col in df_processed.columns and pd.api.types.is_numeric_dtype(df_processed[col]))
    ]

    if not numeric_columns_found:
        st.warning(
            "None of the selected columns are numeric. Cannot remove outliers based on Z-score."
        )
        return df_processed, 0

    for col in numeric_columns_found:
        numeric_data = df_processed[col].dropna() # Drop NA before calculating Z-score to avoid errors/warnings

        # Z-score requires at least 2 data points and non-zero std dev
        if len(numeric_data) > 1 and numeric_data.std() > 0:
            z_scores = np.abs(stats.zscore(numeric_data))
            # Find indices in the original numeric_data where Z-score exceeds threshold
            col_outlier_indices = numeric_data.index[z_scores >= threshold]
            outlier_indices.update(col_outlier_indices)
        elif len(numeric_data) <= 1:
            st.info(
                f"Skipping outlier check for '{col}': Not enough non-NA data points."
            )
        else:  # std == 0
            st.info(
                f"Skipping outlier check for '{col}': All values are the same (zero standard deviation)."
            )

    if outlier_indices:
        df_processed.drop(index=list(outlier_indices), inplace=True)

    rows_removed = initial_rows - len(df_processed)
    return df_processed, rows_removed



# Multiplies specified numeric columns by a factor
def scale_by_factor(df, columns, factor):

    df_processed = df.copy() #safety

    if not columns:
        st.warning("No columns selected for scaling.")
        return df_processed

    # Ensure factor is numeric
    try:
        k = float(factor)
    except (ValueError, TypeError):
        st.error(f"Invalid scaling factor '{factor}'. Please enter a number.")
        return df  # Return original df on error


    scaled_count = 0
    for col in columns:
        if col in df_processed.columns:
            if pd.api.types.is_numeric_dtype(df_processed[col]):
                df_processed[col] = df_processed[col] * k
                scaled_count += 1
            else:
                st.warning(f"Column '{col}' is not numeric. Skipping scaling.")
        else:
            st.warning(f"Column '{col}' not found. Skipping scaling.")

    if scaled_count == 0 and columns:
        st.warning("No numeric columns were found among the selected columns to scale.")

    return df_processed



# Converts a column to the specified target type
def change_column_type(df, column_name, target_type):
    df_processed = df.copy()

    if column_name not in df_processed.columns:
        st.error(f"Column '{column_name}' not found.")
        return df, 0  # Return original df

    original_series = df_processed[column_name]
    initial_na = original_series.isna().sum()
    converted_series = None

    try:
        if "Numeric (Float)" in target_type:
            converted_series = pd.to_numeric(original_series, errors="coerce") #coerce: make it NaN if can't convert
        elif "Numeric (Integer)" in target_type:
            # Convert to float first, then try Int64 if no NaNs introduced and values are whole numbers
            numeric_temp = pd.to_numeric(original_series, errors="coerce")
            if (
                numeric_temp.notna().all()
                and (numeric_temp == numeric_temp.round(0)).all()
            ):
                converted_series = numeric_temp.astype(pd.Int64Dtype())
            else:
                # Keep as float if conversion to int isn't clean (contains NaNs or decimals)
                converted_series = numeric_temp
                if numeric_temp.isna().sum() > initial_na:
                    st.warning(
                        "Some values could not be converted to numeric and became NaN."
                    )
                elif not (numeric_temp == numeric_temp.round(0)).all():
                    st.warning("Column contains non-integer values. Keeping as float.")

        elif "Text (String)" in target_type:
            # Convert all to string; keep Na.
            converted_series = original_series.astype(str)
        elif "Category" in target_type:
            converted_series = original_series.astype("category")
        elif "Date/Time" in target_type:
            converted_series = pd.to_datetime(original_series, errors="coerce")
        else:
            st.error(f"Unsupported target type: {target_type}")
            return df, 0  # Return original df

        # Assign the converted series back
        df_processed[column_name] = converted_series
        # Calculate how many new NaNs were introduced (only relevant for numeric/datetime)
        failed_conversions = (
            max(0, converted_series.isna().sum() - initial_na)
            if pd.api.types.is_numeric_dtype(converted_series.dtype)
            or pd.api.types.is_datetime64_any_dtype(converted_series.dtype)
            else 0
        )

    except Exception as e:
        st.error(f"Error converting column '{column_name}' to {target_type}: {e}")
        return (
            df,
            original_series.isna().sum(),
        )  # Return original df and original NA count on error

    return df_processed, failed_conversions


# Calculates basic descriptive statistics for a numeric series
def get_descriptive_stats(series):

    if not pd.api.types.is_numeric_dtype(series):
        # Return empty dict or specific message if not numeric
        return {"Error": "Column is not numeric"}
    if series.empty:
        return {"Message": "Column is empty or all NaN"}

    # Use describe() for basic stats, then add mode separately
    stats_dict = series.describe().to_dict()

    # Calculate mode, handle multi-modal cases (report first mode)
    mode_val = series.mode()
    stats_dict["Mode"] = mode_val.iloc[0] if not mode_val.empty else "N/A"


    if "50%" in stats_dict:
        stats_dict["Median"] = stats_dict.pop("50%")
    if "count" in stats_dict:
        stats_dict.pop("count")  # Remove count from this specific view
    if "std" in stats_dict:
        stats_dict["Std Dev"] = stats_dict.pop("std")
    if "mean" in stats_dict:
        stats_dict["Mean"] = stats_dict.pop("mean")
    if "min" in stats_dict:
        stats_dict["Min"] = stats_dict.pop("min")
    if "max" in stats_dict:
        stats_dict["Max"] = stats_dict.pop("max")


    # formatting for display
    formatted_stats = {}
    for k, v in stats_dict.items():
        if isinstance(v, (int, float, np.number)) and pd.notna(v):
            formatted_stats[k] = f"{v:.2f}"  # Format numbers to 2 decimal places
        else:
            formatted_stats[k] = v  # Keep others as is (like Mode string)

    return formatted_stats
