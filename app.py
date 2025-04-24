# app.py
import streamlit as st
import pandas as pd
import numpy as np
from preprocessing import (read_file, generate_eda_report, handle_missing_values,
                         remove_outliers, scale_columns, get_descriptive_stats)
from visualization import create_visualizations, create_whisker_plot
from ml_models import train_model, make_predictions
import io

# --- Page Configuration ---
st.set_page_config(
    page_title="ZANISTA",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "A dashboard for interactive data preprocessing, analysis, and basic ML modeling."
    }
)

st.markdown("""
    <style>
        /* Base text color for the main area */
        .main {
            background-color: #ffffff;
            color: #31333F; /* Default dark grey text for contrast */
        }
        body { /* Ensure body text color is also set */
            color: #31333F;
        }
        /* Ensure all paragraphs and divs inherit base color unless overridden */
        p, div {
             color: #31333F;
        }

        [data-testid="stSidebar"] { background-color: #f0f2f6; }

        /* Buttons - Ensure high contrast */
        .stButton>button {
            background-color: #1f77b4; /* Primary Blue */
            color: white !important; /* White text - Use important to ensure override */
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 0.35rem; /* Slightly more rounded */
            font-weight: 600; /* Slightly bolder */
            transition: background-color 0.3s ease, transform 0.1s ease;
            border: 1px solid #1f77b4;
        }
        .stButton>button:hover {
            background-color: #0a3d62; /* Darker Blue */
            border-color: #0a3d62;
            color: white !important;
            transform: scale(1.02); /* Slight scale effect */
        }
        .stButton>button:active {
            background-color: #0a3d62;
            border-color: #0a3d62;
            transform: scale(0.98); /* Slight press effect */
        }

        /* Titles and Headers - Using a darker blue */
        h1, h2, h3 { color: #0a3d62; font-weight: 600;}

        /* Expander Headers - Ensure text is clearly visible */
        .streamlit-expanderHeader {
             background-color: #e6f3ff; /* Light blue background */
             color: #0a3d62 !important; /* Dark blue text - Added !important */
             font-weight: 600; /* Bolder */
             border-radius: 0.35rem;
             border: 1px solid #cce5ff; /* Subtle border */
             padding: 0.75rem 1rem; /* More padding */
        }
         .streamlit-expanderHeader:hover {
             background-color: #d1eaff;
         }

        /* --- NEW RULE for Expander Content --- */
        [data-testid="stExpander"] div[data-testid="stVerticalBlock"] {
             color: #31333F !important; /* Set content text color */
             background-color: #ffffff !important; /* Ensure content background is white */
        }
        /* --- End NEW RULE --- */


        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
            border-bottom: 2px solid #f0f2f6; /* Underline for tab list */
            padding-bottom: 2px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: transparent; /* Make inactive transparent */
            color: #4a4a4a; /* Dark grey text for inactive */
            border-radius: 4px 4px 0px 0px;
            gap: 1px;
            padding: 10px 15px;
            border: none; /* Remove default border */
            border-bottom: 2px solid transparent; /* Placeholder for active underline */
            transition: background-color 0.3s ease, color 0.3s ease, border-color 0.3s ease;
        }
        .stTabs [data-baseweb="tab"]:hover {
            background-color: #f0f2f6; /* Light grey hover for inactive */
            color: #0a3d62;
            border-bottom: 2px solid #d1eaff; /* Subtle underline on hover */
        }
        .stTabs [aria-selected="true"] {
            background-color: transparent; /* Keep background transparent */
            color: #1f77b4; /* Blue for selected tab text */
            font-weight: bold;
            border: none; /* Remove border */
            border-bottom: 2px solid #1f77b4; /* Blue underline for active */
        }

        /* Input Labels - Clear and bold */
        label, .stWidget label {
            color: #0a3d62 !important;
            font-weight: 600; /* Bold */
            font-size: 0.95rem; /* Slightly smaller label */
            margin-bottom: 0.25rem; /* Space below label */
            display: block; /* Ensure block display */
        }

        /* Input Widgets - General styling for better visibility */
        .stTextInput>div>div>input,
        .stNumberInput>div>div>input,
        .stSelectbox>div>div,
        .stMultiSelect>div>div>div, /* Adjusted selector for multiselect */
        .stTextArea>div>textarea {
            border: 1px solid #cccccc;
            border-radius: 0.35rem;
            background-color: #ffffff;
            color: #31333F; /* Dark text inside inputs */
            padding: 0.4rem 0.6rem; /* Adjust padding */
            font-size: 0.95rem;
        }
        .stTextInput>div>div>input:focus,
        .stNumberInput>div>div>input:focus,
        .stTextArea>div>textarea:focus {
            border-color: #1f77b4; /* Highlight border on focus */
            box-shadow: 0 0 0 1px #1f77b4; /* Add focus ring */
        }

        /* Selectbox/Multiselect dropdown options */
        div[data-baseweb="select"] ul,
        div[data-baseweb="popover"] ul {
             background-color: white !important;
             border-radius: 0.35rem;
             border: 1px solid #cccccc;
             box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        div[data-baseweb="select"] li,
        div[data-baseweb="popover"] li {
             color: #31333F !important; /* Ensure text is visible */
             background-color: #ffffff !important;
             padding: 0.5rem 1rem !important; /* Adjust padding */
             font-size: 0.95rem;
             transition: background-color 0.2s ease;
        }
         div[data-baseweb="select"] li:hover,
         div[data-baseweb="popover"] li:hover {
              background-color: #f0f2f6 !important; /* Light grey hover */
              color: #0a3d62 !important;
         }
         /* Selected item text color in multiselect */
         .stMultiSelect span[data-baseweb="tag"] span {
              color: white !important;
         }

        /* Radio buttons - styling the label part */
        .stRadio label span { /* Target the span containing the text */
            color: #31333F !important; /* Ensure radio option text is visible */
            font-weight: normal !important;
            font-size: 0.95rem;
            padding-left: 0.25rem; /* Space between radio button and text */
        }

         /* Improve spacing */
         .block-container { padding-top: 2rem; padding-bottom: 2rem; padding-left: 2rem; padding-right: 2rem;}
         [data-testid="stExpander"] { margin-bottom: 1rem; }
         .stButton { margin-top: 0.5rem;} /* Add slight top margin to buttons */

    </style>
    """, unsafe_allow_html=True)


# --- Initialize Session State ---
if 'df' not in st.session_state:
    st.session_state['df'] = None
if 'original_df' not in st.session_state: # Keep original for comparison/reset
    st.session_state['original_df'] = None
if 'model' not in st.session_state:
    st.session_state['model'] = None
if 'model_score' not in st.session_state:
    st.session_state['model_score'] = None
if 'model_task' not in st.session_state:
    st.session_state['model_task'] = None


# --- Sidebar ---
with st.sidebar:
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png", use_column_width=True) # Placeholder logo
    st.title("⚙️ Controls")
    st.markdown("---")

    uploaded_file = st.file_uploader("📂 Upload Data File",
                                     type=['csv', 'json', 'xlsx', 'xls'],
                                     help="Upload your dataset (CSV, JSON, Excel)")

    if uploaded_file:
        # Load data only if it's not already loaded or if a new file is uploaded
        # This prevents reloading on every interaction
        if st.session_state['df'] is None or st.session_state.get('uploaded_filename') != uploaded_file.name:
            try:
                df_loaded = read_file(uploaded_file) # Use intermediate variable
                st.session_state['df'] = df_loaded
                st.session_state['original_df'] = df_loaded.copy() # Store the original
                st.session_state['uploaded_filename'] = uploaded_file.name # Track filename
                st.session_state['model'] = None # Reset model if new data is loaded
                st.success("File uploaded successfully!")
            except ValueError as e:
                st.error(f"Error loading file: {e}")
                st.session_state['df'] = None
                st.session_state['original_df'] = None
            except Exception as e:
                 st.error(f"An unexpected error occurred: {e}")
                 st.session_state['df'] = None
                 st.session_state['original_df'] = None

    if st.session_state['df'] is not None:
         st.markdown("---")
         if st.button("Reset Data to Original", key='reset_data'):
              st.session_state['df'] = st.session_state['original_df'].copy()
              st.session_state['model'] = None # Reset model on data reset
              st.success("Dataset reset to its original state.")
              st.rerun() # Rerun to reflect reset


# --- Main Application ---
st.title("📊 ZANISTA")

if st.session_state.get('df') is not None: # Check if df exists in session state
    df = st.session_state['df'] # Use the dataframe from session state

    tab_titles = ["📄 Data Preview", "🔧 Preprocessing", "📈 Analysis & Viz", "🤖 Machine Learning"]
    tabs = st.tabs(tab_titles)

    # == Tab 1: Data Preview ==
    with tabs[0]:
        st.header("📋 Dataset Preview & Info")
        st.markdown(f"**Shape:** {df.shape[0]} rows, {df.shape[1]} columns")
        
        st.subheader("First 5 Rows (Head)")
        st.dataframe(df.head(), use_container_width=True)

        st.subheader("Last 5 Rows (Tail)")
        st.dataframe(df.tail(), use_container_width=True)

        st.subheader("Column Data Types & Non-Null Counts")
        buffer = io.StringIO()
        df.info(buf=buffer)
        st.text(buffer.getvalue())

    # == Tab 2: Preprocessing ==
    with tabs[1]:
        st.header("🔧 Data Preprocessing")
        st.info("Apply preprocessing steps sequentially. Changes are saved automatically.")
        st.markdown("---")

        # 1. EDA Report
        with st.expander("📊 Generate Automated EDA Report", expanded=False):
            if st.button("Generate Report", key='eda_report'):
                with st.spinner("Generating report..."):
                    report = generate_eda_report(df)
                    st.write("#### Dataset Overview")
                    st.json(report['basic_info'], expanded=False)
                    st.write("#### Column-wise Analysis")
                    for column, analysis in report['column_analysis'].items():
                         st.write(f"##### Column: `{column}`")
                         st.json(analysis, expanded=False)

        # 2. Missing Values
        with st.expander("🔍 Handle Missing Values", expanded=False):
            missing_summary = df.isnull().sum()
            missing_cols = missing_summary[missing_summary > 0].index.tolist()
            st.write("Columns with missing values:", missing_cols)

            if not missing_cols:
                st.success("No missing values found!")
            else:
                selected_col_miss = st.selectbox("Select column to handle:", missing_cols, key='miss_col_select')
                if selected_col_miss:
                    col_type = df[selected_col_miss].dtype
                    st.write(f"Selected column: `{selected_col_miss}` (Type: {col_type}, Missing: {df[selected_col_miss].isnull().sum()})")
                    
                    method = st.radio("Choose method:", ('Drop Rows with Missing', 'Fill Missing Values'), key=f'miss_method_{selected_col_miss}')

                    fill_value = None
                    if method == 'Fill Missing Values':
                        if pd.api.types.is_numeric_dtype(col_type):
                            # Use median for potentially skewed data, mean is also fine
                            default_fill = df[selected_col_miss].median() if not df[selected_col_miss].isnull().all() else 0
                            fill_value = st.number_input("Enter fill value:", value=float(default_fill), key=f'miss_fill_{selected_col_miss}')
                        elif pd.api.types.is_datetime64_any_dtype(col_type):
                             default_fill = df[selected_col_miss].mode().iloc[0] if not df[selected_col_miss].mode().empty else pd.Timestamp.now()
                             fill_value = st.text_input("Enter fill value (e.g., date string):", value=str(default_fill), key=f'miss_fill_{selected_col_miss}')
                             # Attempt to convert to datetime later in handle_missing_values if needed
                        else: # Object or Category
                             default_fill = df[selected_col_miss].mode().iloc[0] if not df[selected_col_miss].mode().empty else 'Unknown'
                             fill_value = st.text_input("Enter fill value:", value=str(default_fill), key=f'miss_fill_{selected_col_miss}')

                    if st.button("Apply Missing Value Treatment", key=f'miss_apply_{selected_col_miss}'):
                         strat_dict = {}
                         if method == 'Drop Rows with Missing':
                              strat_dict[selected_col_miss] = {'method': 'drop'}
                         else: # Fill
                              strat_dict[selected_col_miss] = {'method': 'fill', 'value': fill_value}
                         
                         with st.spinner("Applying..."):
                             df_processed, rows_dropped = handle_missing_values(st.session_state['df'], strat_dict)
                             st.session_state['df'] = df_processed # Update session state
                             if method == 'Drop Rows with Missing':
                                  st.success(f"Dropped {rows_dropped} rows with missing values in '{selected_col_miss}'.")
                             else:
                                  st.success(f"Filled missing values in '{selected_col_miss}' with '{fill_value}'.")
                             st.rerun() # Rerun to reflect changes immediately

        # 3. Outlier Removal (Z-score)
        with st.expander("📉 Remove Outliers (Z-Score)", expanded=False):
            numeric_cols_outlier = df.select_dtypes(include=np.number).columns.tolist()
            if not numeric_cols_outlier:
                 st.warning("No numeric columns available for outlier detection.")
            else:
                cols_for_outlier = st.multiselect("Select numeric columns for outlier removal:",
                                                numeric_cols_outlier, key='outlier_cols')
                z_threshold = st.number_input("Z-score threshold:", min_value=1.0, max_value=5.0, value=3.0, step=0.1, key='outlier_thresh')
                
                if cols_for_outlier:
                    if st.button("Remove Outliers", key='outlier_apply'):
                         with st.spinner("Removing outliers..."):
                              df_processed, rows_removed = remove_outliers(st.session_state['df'], cols_for_outlier, threshold=z_threshold)
                              st.session_state['df'] = df_processed
                              st.success(f"Removed {rows_removed} rows identified as outliers (Z-score > {z_threshold}) based on selected columns.")
                              st.rerun()

        # 4. Custom Column Removal
        with st.expander("✂️ Remove Columns", expanded=False):
            all_cols = df.columns.tolist()
            cols_to_remove = st.multiselect("Select columns to remove:", all_cols, key='remove_cols')
            if cols_to_remove:
                if st.button("Remove Selected Columns", key='remove_cols_apply'):
                    with st.spinner("Removing columns..."):
                         df_processed = st.session_state['df'].drop(columns=cols_to_remove)
                         st.session_state['df'] = df_processed
                         st.success(f"Removed columns: {', '.join(cols_to_remove)}")
                         st.rerun()

        # 5. Scaling (StandardScaler)
        with st.expander("📏 Scale Numeric Data", expanded=False):
             numeric_cols_scale = df.select_dtypes(include=np.number).columns.tolist()
             if not numeric_cols_scale:
                  st.warning("No numeric columns available for scaling.")
             else:
                 cols_to_scale = st.multiselect("Select numeric columns to scale:",
                                              numeric_cols_scale, key='scale_cols')
                 if cols_to_scale:
                     if st.button("Apply Scaling (StandardScaler)", key='scale_apply'):
                          with st.spinner("Scaling data..."):
                              df_processed = scale_columns(st.session_state['df'], cols_to_scale)
                              st.session_state['df'] = df_processed
                              st.success(f"Applied StandardScaler to columns: {', '.join(cols_to_scale)}")
                              st.rerun()

    # == Tab 3: Analysis & Visualization ==
    with tabs[2]:
        st.header("📈 Data Analysis & Visualization")
        st.markdown("---")

        # Descriptive Statistics
        with st.expander("📊 Descriptive Statistics", expanded=True):
            numeric_cols_stats = df.select_dtypes(include=np.number).columns.tolist()
            if not numeric_cols_stats:
                st.warning("No numeric columns available for descriptive statistics.")
            else:
                stat_col = st.selectbox("Select numeric column for statistics:", numeric_cols_stats, key='stat_col')
                if stat_col:
                    stats = get_descriptive_stats(df[stat_col])
                    st.json(stats)
                    
                    if st.checkbox("Show Whisker Plot", key='whisker_check'):
                         st.plotly_chart(create_whisker_plot(df[stat_col]), use_container_width=True)

        # Visualizations
        with st.expander("🎨 Create Visualizations", expanded=True):
            plot_types = ["Scatter Plot", "Histogram", "Box Plot", "Bar Chart", "Pie Chart", "Line Chart", "Correlation Matrix"]
            viz_type = st.selectbox("Select Visualization Type:", plot_types, key='viz_type')

            x_col, y_col, color_col = None, None, None
            all_cols_viz = [""] + df.columns.tolist() # Add empty option for optional axes

            if viz_type != "Correlation Matrix":
                 c1, c2, c3 = st.columns(3)
                 with c1:
                     x_col = st.selectbox("Select X-axis column:", all_cols_viz, key='viz_x')
                     x_col = None if x_col == "" else x_col
                 with c2:
                     y_col = st.selectbox("Select Y-axis column:", all_cols_viz, key='viz_y')
                     y_col = None if y_col == "" else y_col
                 with c3:
                      # Allow coloring based on another column
                      color_col = st.selectbox("Select Color column (Optional):", all_cols_viz, key='viz_color')
                      color_col = None if color_col == "" else color_col

            # Generate and display plot
            # Add a button to prevent regenerating plot on every minor change
            if st.button("Generate Plot", key='gen_plot'):
                 with st.spinner("Generating plot..."):
                     try:
                          fig = create_visualizations(st.session_state['df'], viz_type, x_col, y_col, color_col)
                          st.plotly_chart(fig, use_container_width=True)
                     except Exception as e:
                          st.error(f"Could not generate plot: {e}")


    # == Tab 4: Machine Learning ==
    with tabs[3]:
        st.header("🤖 Machine Learning Modeling")
        st.markdown("---")

        # Model Training Section
        with st.expander("🧠 Train a Model", expanded=True):
            if df.empty:
                 st.warning("Cannot train model on empty DataFrame.")
            else:
                ml_target = st.selectbox("Select Target Variable (Y):", df.columns, key='ml_target')

                is_regression_target = False
                model_options = [] # Initialize empty list

                if ml_target: # Ensure a target is selected
                    if pd.api.types.is_numeric_dtype(df[ml_target]):
                         if df[ml_target].nunique() > 15:
                              is_regression_target = True
                    
                    if is_regression_target:
                         model_options = ["Linear Regression", "Decision Tree Regressor"]
                         st.info("Target variable appears numeric. Regression models suggested.")
                    else:
                         model_options = ["Decision Tree Classifier"]
                         st.info("Target variable appears categorical/non-numeric or has few unique numeric values. Classification models suggested.")
                else:
                     st.warning("Please select a target variable.")

                # Only show model selectbox if options are available
                if model_options:
                     ml_model_type = st.selectbox("Select Model Type:", model_options, key='ml_model')

                     if st.button("Train Model", key='ml_train'):
                         with st.spinner(f"Training {ml_model_type}..."):
                              try:
                                  # Pass the current df from session state
                                  model, score, task = train_model(st.session_state['df'], ml_target, ml_model_type)
                                  st.session_state['model'] = model
                                  st.session_state['model_score'] = score
                                  st.session_state['model_task'] = task

                                  score_metric = "R2 Score" if task == 'regression' else "Accuracy"
                                  st.success(f"Model trained successfully! Task: {task.capitalize()}, {score_metric}: {score:.3f}")
                              except ValueError as ve:
                                   st.error(f"Training Error: {ve}")
                              except Exception as e:
                                   st.error(f"An unexpected error occurred during training: {e}")
                else:
                     # If no target selected yet, don't show model selection
                     pass

        # Prediction Section (only shows if a model is trained)
        if st.session_state.get('model') is not None:
             st.markdown("---")
             st.subheader("🔮 Make Predictions")

             model = st.session_state['model']
             task = st.session_state['model_task']
             ml_target = st.session_state.get('ml_target', None) # Get the target used for training

             if hasattr(model, 'feature_names_in_') and ml_target:
                 # Get original features from the original_df state
                 original_features = [col for col in st.session_state['original_df'].columns if col != ml_target]

                 prediction_input = {}
                 st.write("Enter values for prediction:")

                 # Use st.form for better input grouping
                 with st.form(key='prediction_form'):
                     cols_per_row = 3
                     col_idx = 0
                     form_cols = [st.columns(cols_per_row) for _ in range((len(original_features) + cols_per_row - 1) // cols_per_row)]

                     for i, feature in enumerate(original_features):
                         row_idx = i // cols_per_row
                         col_idx_in_row = i % cols_per_row
                         col_to_use = form_cols[row_idx][col_idx_in_row]

                         feature_dtype = st.session_state['original_df'][feature].dtype

                         if pd.api.types.is_numeric_dtype(feature_dtype):
                             default_val = st.session_state['original_df'][feature].median() # Use median as default
                             prediction_input[feature] = col_to_use.number_input(f"{feature}:", key=f'pred_{feature}', value=float(default_val))
                         else:
                             unique_vals = list(st.session_state['original_df'][feature].astype(str).unique()) # Ensure string conversion for options
                             default_idx = 0 # Default to first unique value
                             prediction_input[feature] = col_to_use.selectbox(f"{feature}:", options=unique_vals, index=default_idx, key=f'pred_{feature}')

                     submitted = st.form_submit_button("Predict")
                     if submitted:
                          try:
                              pred_df = pd.DataFrame([prediction_input])

                              # Ensure dtypes match original before processing
                              for col in pred_df.columns:
                                   try:
                                       # Attempt conversion based on original dtype
                                       original_dtype = st.session_state['original_df'][col].dtype
                                       if pd.api.types.is_datetime64_any_dtype(original_dtype):
                                            pred_df[col] = pd.to_datetime(pred_df[col])
                                       elif pd.api.types.is_numeric_dtype(original_dtype):
                                            pred_df[col] = pd.to_numeric(pred_df[col])
                                       else:
                                             pred_df[col] = pred_df[col].astype(original_dtype)
                                   except Exception as e:
                                        st.warning(f"Could not convert input for '{col}' to original dtype ({st.session_state['original_df'][col].dtype}): {e}. Using as string/object.")
                                        pred_df[col] = pred_df[col].astype(object) # Fallback to object

                              with st.spinner("Predicting..."):
                                  prediction_result = make_predictions(model, pred_df, task)
                                  st.success(f"Predicted {ml_target}: **{prediction_result[0]}**")
                          except Exception as e:
                              st.error(f"Prediction Error: {e}")
             else:
                 st.warning("Model information incomplete. Cannot create prediction interface.")


# --- Initial Screen (if no data is loaded) ---
elif not uploaded_file:
    st.info("Welcome! Upload a data file using the sidebar to get started.")
    st.markdown("""
    This dashboard allows you to:
    - **Preview** your dataset.
    - **Preprocess** data (handle missing values, remove outliers, scale, etc.).
    - **Analyze** with statistics and various interactive visualizations.
    - **Train** basic Machine Learning models (Regression/Classification) and make predictions.
    """)

# --- Footer ---
st.markdown("---")
st.caption("Built with Streamlit")