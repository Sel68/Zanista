# preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import stats
import io

def preprocess_data(df, handle_missing=True, normalize=True):
    """
    Preprocess the dataframe based on selected options
    """
    # Identify columns
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns

    if handle_missing:
        # Fill numeric with mean, categorical with mode
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
        df[categorical_columns] = df[categorical_columns].fillna(df[categorical_columns].mode().iloc[0])
    
    if normalize:
        scaler = StandardScaler()
        df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    
    return df

def read_file(uploaded_file):
    """Read different file formats and convert to pandas DataFrame"""
    file_type = uploaded_file.name.split('.')[-1].lower()
    
    try:
        if file_type == 'csv':
            df = pd.read_csv(uploaded_file)
        elif file_type == 'json':
            df = pd.read_json(uploaded_file)
        elif file_type in ['xls', 'xlsx']:
            df = pd.read_excel(uploaded_file)
        else:
            raise ValueError("Unsupported file format. Please upload CSV, JSON, XLS, or XLSX.")
        return df
    except Exception as e:
        raise ValueError(f"Error reading file: {e}")

def generate_eda_report(df):
    """Generate a basic EDA report using pandas"""
    report = {}
    
    # Basic Dataset Info
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()

    report['basic_info'] = {
        'Total Rows': len(df),
        'Total Columns': len(df.columns),
        'Duplicate Rows': df.duplicated().sum(),
        'Missing Values (Total)': df.isnull().sum().sum(),
        'Data Types & Non-Null Counts': info_str
    }
    
    # Column-wise Analysis - More detailed
    column_analysis = {}
    for column in df.columns:
        col_data = df[column]
        stats = {
            'Data Type': str(col_data.dtype),
            'Missing Values (%)': f"{col_data.isnull().mean() * 100:.2f}%",
            'Unique Values': col_data.nunique()
        }
        
        if pd.api.types.is_numeric_dtype(col_data):
            desc = col_data.describe()
            stats.update({
                'Mean': f"{desc.get('mean', 'N/A'):.2f}",
                'Std Dev': f"{desc.get('std', 'N/A'):.2f}",
                'Min': f"{desc.get('min', 'N/A'):.2f}",
                '25%': f"{desc.get('25%', 'N/A'):.2f}",
                'Median (50%)': f"{desc.get('50%', 'N/A'):.2f}",
                '75%': f"{desc.get('75%', 'N/A'):.2f}",
                'Max': f"{desc.get('max', 'N/A'):.2f}"
            })
        elif pd.api.types.is_categorical_dtype(col_data) or col_data.dtype == 'object':
             # Show top 5 most frequent values for categorical/object columns
             top_values = col_data.value_counts().head(5).to_dict()
             stats['Top 5 Values'] = top_values

        column_analysis[column] = stats
    
    report['column_analysis'] = column_analysis
    
    return report

def handle_missing_values(df, strategy_dict):
    """
    Handle missing values based on user-defined strategies for specific columns.
    strategy_dict: {column_name: {'method': 'drop'/'fill', 'value': fill_value}}
    Returns the processed DataFrame.
    """
    df_processed = df.copy()
    rows_dropped = 0
    
    for column, strategy in strategy_dict.items():
        if strategy['method'] == 'drop':
            initial_rows = len(df_processed)
            df_processed = df_processed.dropna(subset=[column])
            rows_dropped += initial_rows - len(df_processed)
        elif strategy['method'] == 'fill':
             # Attempt to convert fill_value to the column's dtype if possible
            try:
                fill_val = pd.Series([strategy['value']]).astype(df_processed[column].dtype).iloc[0]
            except (ValueError, TypeError):
                 fill_val = strategy['value'] # Keep original if conversion fails
            df_processed[column] = df_processed[column].fillna(fill_val)
            
    return df_processed, rows_dropped

def remove_outliers(df, columns, threshold=3.0):
    """Remove outliers using z-score method for specified numeric columns"""
    df_processed = df.copy()
    initial_rows = len(df_processed)
    
    for column in columns:
        # Ensure column exists and is numeric before proceeding
        if column in df_processed.columns and pd.api.types.is_numeric_dtype(df_processed[column]):
            # Calculate Z-scores, handling potential NaNs
            z_scores = np.abs(stats.zscore(df_processed[column], nan_policy='omit')) 
            # Keep rows where the Z-score is below the threshold (ignoring NaNs in comparison)
            df_processed = df_processed[ (z_scores < threshold) | (df_processed[column].isnull()) ] 
            
    rows_removed = initial_rows - len(df_processed)
    return df_processed, rows_removed

def scale_columns(df, columns_to_scale):
    """Scale selected numeric columns using StandardScaler"""
    df_processed = df.copy()
    scaler = StandardScaler()
    
    # Ensure columns exist and are numeric
    valid_columns = [col for col in columns_to_scale if col in df_processed.columns and pd.api.types.is_numeric_dtype(df_processed[col])]
    
    if not valid_columns:
        return df_processed # Return original if no valid columns selected

    df_processed[valid_columns] = scaler.fit_transform(df_processed[valid_columns])
    return df_processed

def get_descriptive_stats(series):
    """Calculate descriptive statistics for a pandas Series (column)"""
    if not pd.api.types.is_numeric_dtype(series):
        return {"Error": "Selected column is not numeric."}
        
    stats_dict = {
        'Mean': series.mean(),
        'Median': series.median(),
        'Mode': ', '.join(map(str, series.mode().tolist())), # Handle multiple modes
        'Std Dev': series.std(),
        'Variance': series.var(),
        'IQR': series.quantile(0.75) - series.quantile(0.25),
        'Range': series.max() - series.min(),
        'Min': series.min(),
        'Max': series.max()
    }
    # Format to 2 decimal places for readability
    return {k: (f"{v:.2f}" if isinstance(v, (int, float)) else v) for k, v in stats_dict.items()}

