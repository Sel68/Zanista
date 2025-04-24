# visualization.py
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import pandas as pd
import numpy as np

def create_visualizations(df, viz_type, x_column=None, y_column=None, color_column=None):
    """
    Create different types of visualizations based on the selected type.
    Includes optional color dimension for some plots.
    """
    fig = None
    title = "Visualization" # Default title
    template = "plotly_white" # Consistent theme
    hover_data = list(df.columns) # Show all columns on hover by default

    try:
        if viz_type == "Scatter Plot":
            if x_column and y_column:
                title = f"Scatter Plot: {y_column} vs {x_column}" + (f" by {color_column}" if color_column else "")
                fig = px.scatter(df, x=x_column, y=y_column, color=color_column, 
                                template=template, title=title, hover_data=hover_data)
            else:
                raise ValueError("Please select both X and Y columns for Scatter Plot.")
            
        elif viz_type == "Histogram":
            if x_column:
                title = f"Histogram of {x_column}" + (f" colored by {color_column}" if color_column else "")
                fig = px.histogram(df, x=x_column, color=color_column, 
                                template=template, title=title, marginal="box") # Add marginal box plot
            else:
                 raise ValueError("Please select an X column for Histogram.")
            
        elif viz_type == "Box Plot":
             if y_column:
                 title = f"Box Plot of {y_column}" + (f" grouped by {x_column}" if x_column else "") + (f" colored by {color_column}" if color_column else "")
                 fig = px.box(df, y=y_column, x=x_column, color=color_column,
                             template=template, title=title, points="all") # Show underlying points
             else:
                 raise ValueError("Please select a Y column for Box Plot.")
            
        elif viz_type == "Bar Chart":
            if x_column and y_column:
                title = f"Bar Chart: {y_column} by {x_column}" + (f" colored by {color_column}" if color_column else "")
                # Decide aggregation if y is numeric, otherwise count
                if pd.api.types.is_numeric_dtype(df[y_column]):
                     # Use mean aggregation by default, could be made selectable
                     agg_df = df.groupby(x_column, as_index=False)[y_column].mean()
                     fig = px.bar(agg_df, x=x_column, y=y_column, color=color_column,
                             template=template, title=title, hover_data=hover_data)
                else: # Count occurrences if y is categorical/object
                     count_df = df.groupby(x_column, as_index=False).size()
                     fig = px.bar(count_df, x=x_column, y='size', color=color_column,
                                 template=template, title=f"Count of {x_column}" + (f" colored by {color_column}" if color_column else ""))
            else:
                raise ValueError("Please select both X and Y columns for Bar Chart.")
            
        elif viz_type == "Pie Chart":
            if x_column and y_column:
                 # Typically `names` is categorical and `values` is numeric sum/count
                 title = f"Pie Chart: Distribution of {y_column} by {x_column}"
                 fig = px.pie(df, values=y_column, names=x_column,
                             template=template, title=title, hole=0.3) # Add a donut hole
            else:
                raise ValueError("Please select both 'names' (X) and 'values' (Y) columns for Pie Chart.")
            
        elif viz_type == "Line Chart":
            if x_column and y_column:
                title = f"Line Chart: {y_column} over {x_column}" + (f" grouped by {color_column}" if color_column else "")
                # Sort by x_column for sensible line charts
                df_sorted = df.sort_values(by=x_column) if x_column in df.columns else df
                fig = px.line(df_sorted, x=x_column, y=y_column, color=color_column, markers=True, # Add markers
                             template=template, title=title, hover_data=hover_data)
            else:
                raise ValueError("Please select both X and Y columns for Line Chart.")
            
        elif viz_type == "Correlation Matrix":
            title = "Correlation Matrix of Numeric Features"
            numeric_df = df.select_dtypes(include=np.number) # Select only numeric columns
            if numeric_df.shape[1] >= 2:
                 corr = numeric_df.corr()
                 fig = px.imshow(corr, text_auto=True, # Show correlation values
                                template=template, title=title,
                                color_continuous_scale="RdBu_r", # Red-Blue scale
                                aspect="auto")
            else:
                raise ValueError("Need at least 2 numeric columns for a Correlation Matrix.")

    except Exception as e:
         # Return an empty figure with an error message if plotting fails
         fig = go.Figure()
         fig.add_annotation(text=f"Plotting Error: {e}", showarrow=False, font=dict(size=14, color="red"))
         fig.update_layout(title="Error", xaxis=dict(visible=False), yaxis=dict(visible=False))

    # Apply consistent styling if a figure was created
    if fig:
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font_color='#1f77b4', # Blue text
            title_font_color='#0a3d62', # Darker blue title
            legend_title_font_color='#0a3d62',
            margin=dict(l=40, r=40, t=60, b=40) # Add some margins
        )
        fig.update_traces(marker=dict(line=dict(width=0.5, color='DarkSlateGrey'))) # Outline markers
    
    return fig

def create_whisker_plot(series):
    """Create a whisker plot for a specific pandas Series (column)"""
    if not pd.api.types.is_numeric_dtype(series):
         return go.Figure().add_annotation(text="Whisker plot requires a numeric column.", showarrow=False)
         
    fig = go.Figure()
    fig.add_trace(go.Box(y=series, name=series.name, boxpoints='all', jitter=0.3, pointpos=-1.8)) # Show points
    
    fig.update_layout(
        template="plotly_white",
        title=f"Whisker Plot of {series.name}",
        plot_bgcolor='white',
        paper_bgcolor='white',
        font_color='#1f77b4',
        title_font_color='#0a3d62'
    )
    return fig


# ml_models.py
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score
import pandas as pd

def train_model(df, target_column, model_type):
    """
    Train a machine learning model based on the selected type
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Handle categorical variables
    X = pd.get_dummies(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Choose model
    if model_type == "Linear Regression":
        model = LinearRegression()
        task = 'regression'
    elif model_type == "Random Forest":
        model = RandomForestRegressor()
        task = 'regression'
    else:
        model = RandomForestClassifier()
        task = 'classification'

    model.fit(X_train, y_train)

    # Save feature names for aligning new data
    model.feature_names_in_ = X_train.columns

    # Evaluate
    y_pred = model.predict(X_test)
    score = r2_score(y_test, y_pred) if task == 'regression' else accuracy_score(y_test, y_pred)

    return model, score

def make_predictions(model, X_new):
    """
    Make predictions on new data using the trained model
    """
    X_new = pd.get_dummies(X_new)
    # Align new data columns to training features
    X_new = X_new.reindex(columns=model.feature_names_in_, fill_value=0)
    return model.predict(X_new)