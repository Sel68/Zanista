import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt


#importing features
from preprocessing import (
    read_file,
    generate_eda_report,
    handle_missing_values,
    remove_outliers,
    scale_by_factor,
    change_column_type,
    get_descriptive_stats,
)
from visualization import create_visualizations, create_whisker_plot
from ml_models import train_model, make_predictions  #


# Page setup and styling
st.set_page_config(page_title="ZANISTA", layout="wide")
st.markdown(
    """
    <style>
         .block-container { padding: 2rem;}
         .stButton { margin-top: 0.5rem;}
         /* Removed Plotly specific style */
    </style>
    """,
    unsafe_allow_html=True,
)

# Default state
if "df" not in st.session_state:
    st.session_state["df"] = None
if "original_df" not in st.session_state:
    st.session_state["original_df"] = None
if "model" not in st.session_state:
    st.session_state["model"] = None
if "num_rows_preview" not in st.session_state:
    st.session_state["num_rows_preview"] = 5
if "uploaded_filename" not in st.session_state:
    st.session_state["uploaded_filename"] = None
if (
    "model_trained_features" not in st.session_state
):  # To store features used for training
    st.session_state["model_trained_features"] = None


# Sidebar interface (for controls)
with st.sidebar:
    st.title("Controls")
    st.markdown("---")

    uploaded_file = st.file_uploader(
        "Upload Data File", type=["csv", "json", "xlsx", "xls"]
    )

    # Load data if a new file is uploaded
    if uploaded_file:
        if st.session_state.get("uploaded_filename") != uploaded_file.name:
            st.session_state["uploaded_filename"] = uploaded_file.name
            try:
                df_loaded = read_file(uploaded_file)  #
                st.session_state["df"] = df_loaded
                st.session_state["original_df"] = df_loaded.copy()
                # Reset other states on new upload
                st.session_state["model"] = None
                st.session_state["model_score"] = None
                st.session_state["model_task"] = None
                st.session_state["model_trained_features"] = None
                st.session_state["ml_target_trained"] = None
                st.session_state["num_rows_preview"] = 5
                st.success("File uploaded!")
                st.rerun()  # reload app state
            except Exception as e:
                st.error(f"Error loading file: {e}")
                st.session_state["df"] = None
                st.session_state["original_df"] = None
                st.session_state["uploaded_filename"] = None
                st.session_state["model"] = None
                st.session_state["model_trained_features"] = None

    # Show reset button only if data exists
    if st.session_state.get("df") is not None:
        st.markdown("---")
        if st.button("Reset Data", key="reset_data"):

            # Ensure original_df exists before trying to copy
            if st.session_state.get("original_df") is not None:
                st.session_state["df"] = st.session_state["original_df"].copy()
                st.session_state["model"] = None  # Clear model too
                st.session_state["model_trained_features"] = None
                st.session_state["ml_target_trained"] = None
                st.success("Dataset reset.")
                st.rerun()
            else:
                st.warning("Original data not available for reset.")


# Main app interface
st.title("ZANISTA Dashboard")

if st.session_state.get("df") is not None:
    #df is always the one from session state
    df = st.session_state["df"]

    tabs = st.tabs(["Preview", "Preprocess", "Visualize", "ML Model"])

    # Data Preview Tab
    with tabs[0]:
        st.header("Data Preview")
        if df is not None:
            st.markdown(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")

            num_rows = st.number_input(
                "Rows to show:",
                min_value=1,
                value=st.session_state["num_rows_preview"],
                key="rows_preview",
            )
            if num_rows != st.session_state["num_rows_preview"]:
                st.session_state["num_rows_preview"] = num_rows  # Update state
                st.rerun()

            st.subheader("Head")
            st.dataframe(
                df.head(st.session_state["num_rows_preview"]), use_container_width=True
            )

            st.subheader("Tail")
            st.dataframe(
                df.tail(st.session_state["num_rows_preview"]), use_container_width=True
            )

            st.subheader("Info")
            buffer = io.StringIO()
            df.info(buf=buffer)
            st.text(buffer.getvalue())
        else:
            st.warning("Dataframe is not loaded correctly.")

    # Preprocessing Tab
    with tabs[1]:
        st.header("Preprocessing Tools")
        st.info(
            "Note: Preprocessing steps modify the current dataset used for visualization and modeling."
        )

        # EDA Report Expander
        with st.expander("EDA Report"):
            if st.button("Generate", key="eda_btn"):
                try:
                    report = generate_eda_report(df)  #
                    st.subheader("Basic Info")
                    st.json(report["basic_info"], expanded=False)  #
                    st.subheader("Column Analysis")
                    st.dataframe(
                        pd.DataFrame.from_dict(
                            report["column_analysis"], orient="index"
                        ),
                        use_container_width=True,
                    )  #
                except Exception as e:
                    st.error(f"Could not generate report: {e}")

        # Missing Values Expander
        with st.expander("Handle Missing Values"):
            missing_summary = df.isnull().sum()
            missing_cols = missing_summary[missing_summary > 0].index.tolist()

            if not missing_cols:
                st.info("No missing values found in the current data.")
            else:
                st.dataframe(
                    missing_summary[missing_summary > 0].reset_index(name="Count"),
                    use_container_width=True,
                )
                col_to_handle = st.selectbox(
                    "Column:", missing_cols, key="miss_col", index=0
                )  # Default to first
                if col_to_handle:
                    method = st.radio(
                        "Method:",
                        ("Fill", "Drop Rows"),
                        key=f"miss_method_{col_to_handle}",
                        horizontal=True,
                    )
                    fill_value = None
                    if method == "Fill":
                        col_type = df[col_to_handle].dtype
                        default_fill = ""
                        if pd.api.types.is_numeric_dtype(col_type):
                            default_fill = df[col_to_handle].median()
                            fill_value = st.number_input(
                                "Fill with:",
                                value=default_fill if pd.notna(default_fill) else 0.0,
                                key=f"miss_fill_{col_to_handle}",
                                format="%g",
                            )
                        else:
                            # Use mode, handle cases where mode is empty or non-unique
                            mode_val = df[col_to_handle].mode()
                            default_fill = (
                                mode_val.iloc[0] if not mode_val.empty else ""
                            )
                            fill_value = st.text_input(
                                "Fill with:",
                                value=default_fill,
                                key=f"miss_fill_{col_to_handle}",
                            )

                    if st.button("Apply Treatment", key=f"miss_apply_{col_to_handle}"):
                        strat = {}
                        if method == "Drop Rows":
                            strat[col_to_handle] = {"method": "drop"}
                        else:
                            # Use the captured fill_value which might be number or text
                            strat[col_to_handle] = {
                                "method": "fill",
                                "value": fill_value,
                            }

                        try:
                            df_processed, rows_dropped = handle_missing_values(
                                st.session_state["df"], strat
                            )  #
                            st.session_state["df"] = (
                                df_processed  # Update the main dataframe
                            )
                            st.success(
                                f"Missing values in '{col_to_handle}' handled using '{method}'. Rows dropped: {rows_dropped if method == 'Drop Rows' else 0}."
                            )
                            # Clear model if data changes
                            st.session_state["model"] = None
                            st.session_state["model_trained_features"] = None
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to handle missing values: {e}")

        # Change Type Expander
        with st.expander("Change Data Type"):
            all_cols = df.columns.tolist()
            selected_col = st.selectbox("Column to change:", all_cols, key="dtype_col")
            if selected_col:
                st.write(f"Current type: **{df[selected_col].dtype}**")
                target_types = [
                    "Numeric (Float)",
                    "Numeric (Integer)",
                    "Text (String)",
                    "Category",
                    "Date/Time",
                ]
                new_type = st.selectbox("New type:", target_types, key="dtype_new")
                if st.button(f"Convert", key="dtype_apply"):
                    try:
                        df_processed, failed = change_column_type(
                            st.session_state["df"], selected_col, new_type
                        )  #
                        st.session_state["df"] = df_processed
                        st.success(
                            f"Converted '{selected_col}'. New type: {df_processed[selected_col].dtype}."
                        )
                        if failed > 0:
                            st.warning(
                                f"{failed} values could not be converted to the target numeric/date type and became NaN."
                            )
                        # Clear model if data changes
                        st.session_state["model"] = None
                        st.session_state["model_trained_features"] = None
                        st.rerun()
                    except Exception as e:
                        st.error(f"Type conversion failed: {e}")

        # Outlier Removal Expander
        with st.expander("Remove Outliers (Z-Score)"):
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            if numeric_cols:
                cols_outlier = st.multiselect(
                    "Columns to check:", numeric_cols, key="outlier_cols"
                )
                threshold = st.number_input(
                    "Z-Score Threshold:",
                    min_value=1.0,
                    value=3.0,
                    step=0.1,
                    key="outlier_thresh",
                )
                if cols_outlier and st.button("Remove Rows", key="outlier_apply"):
                    try:
                        df_processed, removed = remove_outliers(
                            st.session_state["df"], cols_outlier, threshold
                        )  #
                        st.session_state["df"] = df_processed
                        st.success(
                            f"Removed {removed} rows containing outliers based on Z-score > {threshold} in selected columns."
                        )
                        # Clear model if data changes
                        st.session_state["model"] = None
                        st.session_state["model_trained_features"] = None
                        st.rerun()
                    except Exception as e:
                        st.error(f"Outlier removal failed: {e}")
            else:
                st.info("No numeric columns available for outlier removal.")

        # Remove Columns Expander
        with st.expander("Remove Columns"):
            cols_remove = st.multiselect(
                "Columns to remove:", df.columns.tolist(), key="remove_cols"
            )
            if cols_remove and st.button("Remove Selected Columns", key="remove_apply"):
                try:
                    df_processed = st.session_state["df"].drop(columns=cols_remove)
                    st.session_state["df"] = df_processed
                    st.success(f"Removed columns: {', '.join(cols_remove)}.")
                    # Clear model if data changes
                    st.session_state["model"] = None
                    st.session_state["model_trained_features"] = None
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to remove columns: {e}")

        # Scale Data Expander
        with st.expander("Scale Data (Multiply)"):
            numeric_cols_s = df.select_dtypes(include=np.number).columns.tolist()
            if numeric_cols_s:
                cols_scale = st.multiselect(
                    "Columns to scale:", numeric_cols_s, key="scale_cols"
                )
                factor = st.number_input(
                    "Scaling Factor (k):", value=1.0, format="%g", key="scale_factor"
                )
                if cols_scale and st.button("Apply Scaling", key="scale_apply"):
                    try:
                        df_processed = scale_by_factor(
                            st.session_state["df"], cols_scale, factor
                        )  #
                        st.session_state["df"] = df_processed
                        st.success(f"Scaled selected columns by factor {factor}.")
                        # Clear model if data changes
                        st.session_state["model"] = None
                        st.session_state["model_trained_features"] = None
                        st.rerun()
                    except Exception as e:
                        st.error(f"Scaling failed: {e}")
            else:
                st.info("No numeric columns available to scale.")

    # Visualization Tab
    with tabs[2]:
        st.header("Visualize Data")

        # Stats Expander
        with st.expander("Descriptive Stats & Box Plot", expanded=True):
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            if numeric_cols:
                stat_col = st.selectbox(
                    "Select Numeric Column:", numeric_cols, key="stat_col"
                )
                if stat_col:
                    stats = get_descriptive_stats(df[stat_col])  #
                    st.dataframe(
                        pd.DataFrame.from_dict(
                            stats, orient="index", columns=["Value"]
                        ),
                        use_container_width=True,
                    )  #

                    if st.checkbox("Show Box Plot", value=True, key="whisker_show"):
                        try:
                            fig = create_whisker_plot(df[stat_col])  #
                            if fig:
                                # Pass the figure object to st.pyplot
                                st.pyplot(fig)
                                plt.clf()  # Clear the figure after displaying
                            else:
                                st.warning(
                                    "Could not generate box plot (column might not be suitable)."
                                )
                        except Exception as e:
                            st.error(f"Box plot generation failed: {e}")
            else:
                st.info("No numeric columns found for descriptive statistics.")

        # Plotting Expander
        with st.expander("Create Plots", expanded=True):
            plot_types = ["Scatter Plot", "Histogram", "Bar Chart", "Pie Chart"]
            viz_type = st.selectbox("Select Plot Type:", plot_types, key="viz_type")

            # Use current df columns
            all_cols_options = [""] + df.columns.tolist()
            numeric_cols_plot = [""] + df.select_dtypes(
                include=np.number
            ).columns.tolist()

            # Define labels and options based on plot type
            x_label, y_label, color_label = "X-axis:", "Y-axis:", "Color by:"
            x_options = all_cols_options
            y_options = all_cols_options
            color_options = all_cols_options
            y_disabled = False

            if viz_type == "Pie Chart":
                x_label = "Categories (Slices):"
                y_label = "Values (Optional Numeric):"
                x_options = [
                    c
                    for c in all_cols_options
                    if c == "" or not pd.api.types.is_numeric_dtype(df.get(c))
                ]  # Usually categorical for pie
                y_options = numeric_cols_plot  # Optional numeric for values
                color_options = [""]  # Color is implicit in pie chart
            elif viz_type == "Histogram":
                x_label = "Column (Numeric):"
                x_options = numeric_cols_plot  # Histograms need numeric data
                y_label = "Y-axis (Count)"  # Clarify y-axis represents count
                y_options = [""]  # Disable Y selection
                y_disabled = True
                color_options = [
                    c
                    for c in all_cols_options
                    if c == "" or not pd.api.types.is_numeric_dtype(df.get(c))
                ]  # Color usually by category
            elif viz_type == "Scatter Plot":
                x_options = numeric_cols_plot  # Scatter needs numeric X
                y_options = numeric_cols_plot  # Scatter needs numeric Y
            elif viz_type == "Bar Chart":
                x_options = [
                    c
                    for c in all_cols_options
                    if c == "" or not pd.api.types.is_numeric_dtype(df.get(c))
                ]  # Bar usually categorical X
                y_label = (
                    "Y-axis (Optional Numeric for Mean):"  # Y can be count or mean
                )

            # Create columns for selectors
            c1, c2, c3 = st.columns(3)
            with c1:
                x_col = st.selectbox(x_label, x_options, key="viz_x")
            with c2:
                y_col = st.selectbox(
                    y_label, y_options, key="viz_y", disabled=y_disabled
                )
            with c3:
                color_col = st.selectbox(
                    color_label,
                    color_options,
                    key="viz_color",
                    disabled=(viz_type == "Pie Chart"),
                )

            # Clean up selections (use None if empty string)
            x_col = None if x_col == "" else x_col
            y_col = None if y_col == "" else y_col
            color_col = None if color_col == "" else color_col

            if st.button("Generate Plot", key="gen_plot"):
                # Basic validation
                plot_possible = True
                if not x_col:
                    st.error("Please select the primary column (X-axis or Categories).")
                    plot_possible = False
                elif viz_type == "Scatter Plot" and not y_col:
                    st.error("Please select a Y-axis column for the Scatter Plot.")
                    plot_possible = False
                elif (
                    viz_type == "Histogram"
                    and x_col
                    and not pd.api.types.is_numeric_dtype(df.get(x_col))
                ):
                    st.error("Histogram requires a numeric column for X-axis.")
                    plot_possible = False
                # Add more specific checks if needed

                if plot_possible:
                    try:
                        # Pass the current df from session state
                        fig = create_visualizations(
                            st.session_state["df"], viz_type, x_col, y_col, color_col
                        )  #
                        if fig:
                            st.pyplot(fig)
                            plt.clf()  # Clear figure after display
                        else:
                            st.warning(
                                "Could not generate plot (check selections and data)."
                            )
                    except Exception as e:
                        st.error(f"Plotting Error: {e}")

    # Machine Learning Tab
    with tabs[3]:
        st.header("Machine Learning")

        # Train Model Expander
        with st.expander("Train Model", expanded=True):
            # Ensure df exists and has columns
            if df is not None and not df.empty:
                potential_targets = [
                    col
                    for col in df.columns
                    if not pd.api.types.is_datetime64_any_dtype(df[col])
                ]  # Exclude datetime targets
                if not potential_targets:
                    st.warning(
                        "No suitable target columns found in the current dataset (datetime columns excluded)."
                    )
                else:
                    ml_target = st.selectbox(
                        "Target Variable (Y):",
                        [""] + potential_targets,
                        key="ml_target_sel",
                        index=0,
                    )  # Default to empty

                    model_options = []
                    task_type = None
                    if ml_target:
                        # Determine task type based on selected target in current df
                        target_series = df[
                            ml_target
                        ].dropna()  # Use dropna to handle potential missings during nunique check
                        if (
                            pd.api.types.is_numeric_dtype(target_series)
                            and target_series.nunique() > 15
                        ):
                            task_type = "regression"
                            # *** MODIFIED HERE ***
                            model_options = [
                                "Linear Regression",
                                "Polynomial Regression",
                                "Decision Tree Regressor",
                            ]  #
                            st.info(
                                "Detected Task: Regression (numeric target with >15 unique values)"
                            )
                        elif (
                            target_series.nunique() > 1
                        ):  # Classification if non-numeric or numeric with <=15 unique values
                            task_type = "classification"
                            model_options = [
                                "Decision Tree Classifier"
                            ]  # Add more classifiers if needed #
                            st.info(
                                "Detected Task: Classification (non-numeric target or <=15 unique numeric values)"
                            )
                        else:
                            st.warning(
                                "Selected target column has only one unique value after dropping NAs. Cannot train a model."
                            )

                    if model_options:  # Only show model selection if task is identified
                        ml_model_type = st.selectbox(
                            "Model Type:", model_options, key="ml_model_sel"
                        )
                        if st.button("Train Model", key="ml_train_btn"):
                            # Train using the current state of df
                            try:
                                model, score, trained_task, training_features = (
                                    train_model(
                                        st.session_state["df"], ml_target, ml_model_type
                                    )
                                )  #
                                st.session_state["model"] = model
                                st.session_state["model_score"] = score
                                st.session_state["model_task"] = trained_task
                                st.session_state["ml_target_trained"] = (
                                    ml_target  # Store trained target name
                                )
                                st.session_state["model_trained_features"] = (
                                    training_features  # Store features used for training input form
                                )

                                score_metric = (
                                    "R2 Score"
                                    if trained_task == "regression"
                                    else "Accuracy"
                                )  #
                                st.success(
                                    f"Model trained successfully! Task: {trained_task}, {score_metric}: {score:.3f}"
                                )  #
                                st.balloons()
                            except Exception as e:
                                st.error(f"Training Failed: {e}")
                                # Clear potentially corrupted model state
                                st.session_state["model"] = None
                                st.session_state["model_trained_features"] = None
                                st.session_state["ml_target_trained"] = None

                    elif (
                        ml_target and not task_type
                    ):  # If target selected but task not identified (e.g., only 1 unique value)
                        pass  # Warning already shown
                    elif not ml_target:
                        st.info("Select a target variable to choose a model.")
            else:
                st.warning("Load data and ensure it's not empty before training.")

        # Prediction Expander
        # Check if a model exists and the features used for training are stored
        if (
            st.session_state.get("model") is not None
            and st.session_state.get("model_trained_features") is not None
        ):
            st.markdown("---")
            st.subheader("Make Predictions")

            model = st.session_state["model"]
            task = st.session_state["model_task"]
            target = st.session_state.get("ml_target_trained")
            # Use the features stored from the training phase to build the form
            # These are the *original* features before transformation
            features_for_input = st.session_state["model_trained_features"]  #
            # Get the current df to extract default values/options for the form
            current_df_for_defaults = st.session_state.get("df")

            if features_for_input and current_df_for_defaults is not None:
                with st.form(key="pred_form"):
                    pred_inputs = {}
                    st.write("Input Features:")
                    cols_per_row = 3
                    # Calculate number of rows needed for the grid
                    num_rows_grid = (
                        len(features_for_input) + cols_per_row - 1
                    ) // cols_per_row
                    form_cols = [st.columns(cols_per_row) for _ in range(num_rows_grid)]

                    # Create input fields based on the features used during training
                    for i, feature in enumerate(features_for_input):
                        # Determine which column in the grid this feature goes into
                        row_idx = i // cols_per_row
                        col_idx = i % cols_per_row
                        col_widget = form_cols[row_idx][col_idx]

                        # Get data for this feature from the *current* df for defaults/options
                        if feature in current_df_for_defaults.columns:
                            feat_series = current_df_for_defaults[feature]
                            dtype = feat_series.dtype

                            if pd.api.types.is_numeric_dtype(dtype):
                                # Use median of the current data as default
                                default_val = feat_series.median()
                                # Ensure default_val is a standard float, handle NaN
                                input_val = (
                                    float(default_val) if pd.notna(default_val) else 0.0
                                )
                                pred_inputs[feature] = col_widget.number_input(
                                    f"{feature}",
                                    value=input_val,
                                    key=f"pred_{feature}",
                                    format="%g",  # Use general format for numbers
                                )
                            elif pd.api.types.is_categorical_dtype(
                                dtype
                            ) or pd.api.types.is_object_dtype(dtype):
                                # Get unique values from the current data for options
                                options = sorted(
                                    [str(v) for v in feat_series.dropna().unique()]
                                )
                                # Use mode of the current data as default
                                default_mode = feat_series.mode()
                                default_val = (
                                    default_mode.iloc[0]
                                    if not default_mode.empty
                                    else (options[0] if options else "")
                                )
                                default_val_str = str(default_val)

                                # Find index, handle case where default might not be in options (e.g., after filtering)
                                try:
                                    default_index = (
                                        options.index(default_val_str)
                                        if default_val_str in options
                                        else 0
                                    )
                                except ValueError:
                                    default_index = 0  # Fallback to the first option

                                pred_inputs[feature] = col_widget.selectbox(
                                    f"{feature}",
                                    options=options,
                                    index=default_index,
                                    key=f"pred_{feature}",
                                )
                            else:  # Handle other types like boolean maybe? Default to text input
                                pred_inputs[feature] = col_widget.text_input(
                                    f"{feature}", key=f"pred_{feature}"
                                )
                        else:
                            # If a feature used in training is somehow missing now, add a placeholder. This shouldn't normally happen with the current logic.
                            col_widget.text_input(
                                f"{feature} (Missing?)",
                                value="",
                                key=f"pred_{feature}",
                                disabled=True,
                            )
                            pred_inputs[feature] = (
                                None  # Mark as missing for later handling if needed
                            )

                    submitted = st.form_submit_button("Predict")
                    if submitted:
                        # Create DataFrame from inputs
                        pred_df = pd.DataFrame([pred_inputs])

                        # Important: Ensure the prediction DataFrame columns match the order of original features
                        # The make_predictions function now expects the columns in the original order
                        pred_df = pred_df[features_for_input]  #

                        # Basic type conversion based on the *current* dataframe's dtypes is less critical now
                        # as imputation/encoding happens inside make_predictions based on stored info.
                        # We'll skip the explicit conversion loop here.

                        # Make prediction using the model and the prepared DataFrame
                        try:
                            prediction_result = make_predictions(
                                model, pred_df, task
                            )  #
                            # Display the first prediction (assuming single input row)
                            st.success(
                                f"Predicted {target}: **{prediction_result[0]}**"
                            )  #
                        except Exception as e:
                            st.error(f"Prediction Failed: {e}")
                            st.error(
                                "Ensure input values are appropriate for the model. Check logs if errors persist."
                            )

            else:
                st.warning(
                    "Cannot create prediction interface. Model features or current data state missing."
                )

elif not st.session_state.get("uploaded_filename"):
    st.info("Upload a data file using the sidebar to start.")
