from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np


def train_model(df, target_column, model_type):
    # Trains model on provided dataframe state
    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in the provided dataframe."
        )
    if df[target_column].isnull().all():
        raise ValueError(
            f"Target column '{target_column}' contains only missing values."
        )

    # Use a copy to avoid modifying the original df
    df_processed = df.copy()

    # Drop rows where the target variable itself is missing
    df_processed.dropna(subset=[target_column], inplace=True)
    if df_processed.empty:
        raise ValueError(
            "Dataset is empty after removing rows with missing target values."
        )

    y = df_processed[target_column]
    X = df_processed.drop(columns=[target_column])

    # Store original feature names (before encoding) for later use in prediction form
    original_features = X.columns.tolist()

    # Identify categorical and numerical features in the current feature set X
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()

    # Simple Imputation (before encoding)
    imputation_values = {}  # Store imputation values for prediction
    for col in numeric_cols:
        if X[col].isnull().any():
            median_val = X[col].median()
            X[col].fillna(median_val, inplace=True)
            imputation_values[col] = median_val
    for col in categorical_cols:
        if X[col].isnull().any():
            mode_val = X[col].mode()
            fill_val = mode_val.iloc[0] if not mode_val.empty else "Missing"
            X[col].fillna(fill_val, inplace=True)
            imputation_values[col] = fill_val

    # Determine task type (regression/classification) based on target variable y
    task = None
    le = None  # LabelEncoder for classification target
    if pd.api.types.is_numeric_dtype(y) and y.nunique() > 15:
        task = "regression"
    elif y.nunique() > 1:  # Also covers non-numeric types
        task = "classification"
        le = LabelEncoder()
        y = le.fit_transform(y)  # Encode target variable
    else:
        raise ValueError("Target variable has only one unique value. Cannot train.")

    # One-Hot Encode categorical features
    X_encoded = pd.get_dummies(
        X, columns=categorical_cols, drop_first=True, dummy_na=False
    )
    # *** STORE feature names AFTER encoding but BEFORE potential polynomial transformation ***
    encoded_feature_names_before_poly = X_encoded.columns.tolist()

    if X_encoded.empty:
        raise ValueError("No features left after preprocessing and encoding.")

    # --- Polynomial Features Transformation ---
    poly_transformer = None
    X_final_features = X_encoded  # Default to encoded features
    final_feature_names = encoded_feature_names_before_poly  # Default names

    if task == "regression" and model_type == "Polynomial Regression":
        poly_transformer = PolynomialFeatures(degree=3, include_bias=False)
        # Fit on the encoded data
        X_poly = poly_transformer.fit_transform(X_encoded)
        # Get potentially very long list of feature names after polynomial transformation
        try:
            # Use the stored intermediate names as input to get_feature_names_out
            final_feature_names = poly_transformer.get_feature_names_out(
                encoded_feature_names_before_poly
            )
        except AttributeError:
            final_feature_names = [f"poly_{i}" for i in range(X_poly.shape[1])]

        # Convert X_poly back to DataFrame for splitting (optional but safer)
        X_final_features = pd.DataFrame(
            X_poly, index=X_encoded.index, columns=final_feature_names
        )

    # ----------------------------------------

    # Split data into training and testing sets using the final feature set
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_final_features,
            y,
            test_size=0.2,
            random_state=42,
            stratify=(y if task == "classification" else None),
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X_final_features, y, test_size=0.2, random_state=42
        )

    # Select and train the appropriate model
    model = None
    if task == "regression":
        if model_type in ["Linear Regression", "Polynomial Regression"]:
            model = LinearRegression()
        elif model_type == "Decision Tree Regressor":
            model = DecisionTreeRegressor(random_state=42)
        else:
            raise ValueError(f"Unsupported regression model type: {model_type}")
    elif task == "classification":
        if model_type == "Decision Tree Classifier":
            model = DecisionTreeClassifier(random_state=42, class_weight="balanced")
        else:
            raise ValueError(f"Unsupported classification model type: {model_type}")

    # Train on the (potentially transformed) training data
    model.fit(X_train, y_train)

    # Store information needed for prediction within the model object
    model.feature_names_in_ = (
        final_feature_names  # Features names model expects (AFTER poly if used)
    )
    model.original_features_ = (
        original_features  # Feature names *before* any encoding/transformation
    )
    model.numeric_features_ = numeric_cols  # List of original numeric features used
    model.categorical_features_ = (
        categorical_cols  # List of original categorical features used
    )
    model.poly_transformer_ = (
        poly_transformer  # Store the fitted PolynomialFeatures object (or None)
    )
    model.imputation_values_ = imputation_values  # Store imputation values
    # *** Store the intermediate feature names ***
    model.encoded_feature_names_before_poly_ = encoded_feature_names_before_poly

    if task == "classification" and le:
        model.target_classes_ = le.classes_  # Store original class labels

    # Evaluate the model on the test set
    y_pred = model.predict(X_test)  # X_test is already transformed if needed
    score = (
        r2_score(y_test, y_pred)
        if task == "regression"
        else accuracy_score(y_test, y_pred)
    )

    return model, score, task, original_features


def make_predictions(model, X_new, task):
    # Predicts using the trained model on new data X_new

    required_attrs = [
        "feature_names_in_",
        "numeric_features_",
        "categorical_features_",
        "poly_transformer_",
        "imputation_values_",
        "original_features_",
        "encoded_feature_names_before_poly_",
    ]  # Added check
    if not all(hasattr(model, attr) for attr in required_attrs):
        raise ValueError(
            "Model is missing required attributes. Was it trained correctly with the updated train_model function?"
        )

    if not isinstance(X_new, pd.DataFrame):
        raise ValueError("Input data (X_new) must be a pandas DataFrame.")

    try:
        X_pred = X_new[model.original_features_].copy()
    except KeyError as e:
        raise ValueError(
            f"Input data is missing expected original feature columns: {e}"
        )

    numeric_cols = model.numeric_features_
    categorical_cols = model.categorical_features_
    poly_transformer = model.poly_transformer_
    imputation_values = model.imputation_values_
    # Use the appropriately stored feature names
    encoded_features_before_poly = model.encoded_feature_names_before_poly_
    final_expected_features = model.feature_names_in_

    # Impute missing values using stored training imputation values
    for col in numeric_cols:
        if col in X_pred.columns and X_pred[col].isnull().any():
            fill_val = imputation_values.get(col)
            if fill_val is not None:
                X_pred[col].fillna(fill_val, inplace=True)
            else:
                X_pred[col].fillna(X_pred[col].median(), inplace=True)  # Fallback

    for col in categorical_cols:
        if col in X_pred.columns and X_pred[col].isnull().any():
            fill_val = imputation_values.get(col)
            if fill_val is not None:
                X_pred[col].fillna(fill_val, inplace=True)
            else:  # Fallback
                mode_val = X_pred[col].mode()
                fill_val_fallback = (
                    mode_val.iloc[0] if not mode_val.empty else "Missing"
                )
                X_pred[col].fillna(fill_val_fallback, inplace=True)

    # One-Hot Encode categorical features
    cols_to_encode = [col for col in categorical_cols if col in X_pred.columns]
    if cols_to_encode:
        X_pred_encoded = pd.get_dummies(
            X_pred, columns=cols_to_encode, drop_first=True, dummy_na=False
        )
    else:
        X_pred_encoded = X_pred  # No categorical columns to encode

    # --- Align columns and Apply Polynomial Features if necessary ---
    X_pred_final = None

    if poly_transformer:
        # 1. Align columns to match the input the poly transformer expects
        #    (These are the columns AFTER one-hot encoding during training)
        try:
            # Use the stored list: encoded_features_before_poly
            X_pred_aligned_pre_poly = X_pred_encoded.reindex(
                columns=encoded_features_before_poly, fill_value=0
            )
        except Exception as e:
            raise ValueError(
                f"Error aligning columns before polynomial transformation during prediction. Columns expected: {encoded_features_before_poly}. Columns found after encoding: {X_pred_encoded.columns}. Error: {e}"
            )

        # Check if alignment resulted in an empty DataFrame (can happen if columns mismatch severely)
        if X_pred_aligned_pre_poly.empty and not X_pred_encoded.empty:
            raise ValueError(
                f"Column alignment before polynomial transformation resulted in an empty DataFrame. Check feature consistency between training and prediction."
            )
        elif X_pred_aligned_pre_poly.empty:
            raise ValueError(
                "Prediction data became empty after encoding and alignment."
            )

        # 2. Apply the polynomial transformation
        try:
            X_poly_pred = poly_transformer.transform(X_pred_aligned_pre_poly)
        except ValueError as e:
            raise ValueError(
                f"Error applying polynomial transformation during prediction. Input columns might not match expected format. Details: {e}"
            )

        # 3. Convert back to DataFrame with correct FINAL feature names for the linear model
        #    (These are the columns AFTER polynomial transformation during training)
        X_pred_final = pd.DataFrame(
            X_poly_pred, index=X_pred.index, columns=final_expected_features
        )

    else:
        # No polynomial transformation
        # Align directly to the final features expected by the model (Linear or Tree)
        try:
            X_pred_aligned = X_pred_encoded.reindex(
                columns=final_expected_features, fill_value=0
            )
        except Exception as e:
            raise ValueError(
                f"Error aligning columns for non-polynomial model during prediction. Columns expected: {final_expected_features}. Columns found after encoding: {X_pred_encoded.columns}. Error: {e}"
            )

        # Check if alignment resulted in an empty DataFrame
        if X_pred_aligned.empty and not X_pred_encoded.empty:
            raise ValueError(
                f"Column alignment for non-polynomial model resulted in an empty DataFrame. Check feature consistency."
            )
        elif X_pred_aligned.empty:
            raise ValueError(
                "Prediction data became empty after encoding and alignment."
            )

        # Ensure the column order matches exactly
        X_pred_final = X_pred_aligned[final_expected_features]

    # --- Make predictions ---
    if X_pred_final is None:
        raise RuntimeError(
            "Prediction feature DataFrame (X_pred_final) was not generated."
        )
    if X_pred_final.isnull().any().any():
        # Add a warning or handle potential NaNs introduced during reindex/transform if necessary
        print(
            "Warning: NaNs detected in final prediction features. This might indicate issues."
        )
        # Optionally, fill NaNs here if appropriate, e.g., X_pred_final.fillna(0, inplace=True)

    predictions_numeric = model.predict(X_pred_final)

    # Decode predictions if it was a classification task
    if task == "classification" and hasattr(model, "target_classes_"):
        predictions_numeric = predictions_numeric.astype(int)
        predictions_numeric = np.clip(
            predictions_numeric, 0, len(model.target_classes_) - 1
        )
        return model.target_classes_[predictions_numeric]
    else:
        return predictions_numeric  # Return raw numeric predictions for regression
