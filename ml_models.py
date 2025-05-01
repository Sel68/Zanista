from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np


def train_model(df, target_column, model_type):
    # Trains model on provided dataframe state
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in the provided dataframe.")
    if df[target_column].isnull().all():
         raise ValueError(f"Target column '{target_column}' contains only missing values.")

    # Use a copy to avoid modifying the original df passed from Streamlit state
    df_processed = df.copy()

    # Drop rows where the target variable itself is missing
    df_processed.dropna(subset=[target_column], inplace=True)
    if df_processed.empty:
        raise ValueError("Dataset is empty after removing rows with missing target values.")

    y = df_processed[target_column]
    X = df_processed.drop(columns=[target_column])

    # Store original feature names (before encoding) for later use in prediction form
    original_features = X.columns.tolist()

    # Identify categorical and numerical features in the current feature set X
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()

    # Simple Imputation (before encoding)
    # Use median for numeric, mode for categorical
    for col in numeric_cols:
        if X[col].isnull().any():
            median_val = X[col].median()
            X[col].fillna(median_val, inplace=True)
    for col in categorical_cols:
        if X[col].isnull().any():
            mode_val = X[col].mode()
            # Handle cases where mode might be empty
            fill_val = mode_val.iloc[0] if not mode_val.empty else 'Missing'
            X[col].fillna(fill_val, inplace=True)

    # Determine task type (regression/classification) based on target variable y
    task = None
    le = None # LabelEncoder for classification target
    if pd.api.types.is_numeric_dtype(y) and y.nunique() > 15:
        task = 'regression'
    elif y.nunique() > 1: # Also covers non-numeric types
        task = 'classification'
        le = LabelEncoder()
        y = le.fit_transform(y) # Encode target variable
    else:
        raise ValueError("Target variable has only one unique value. Cannot train.")


    # One-Hot Encode categorical features
    # drop_first=True helps reduce multicollinearity, dummy_na=False avoids creating NA columns explicitly
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True, dummy_na=False)
    encoded_feature_names = X_encoded.columns.tolist() # Get feature names *after* encoding

    if X_encoded.empty:
        raise ValueError("No features left after preprocessing and encoding.")

    # Split data into training and testing sets
    try:
        # Stratify for classification tasks to maintain class proportions if possible
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=0.2, random_state=42,
            stratify=(y if task == 'classification' else None)
        )
    except ValueError:
        # Fallback if stratification fails (e.g., too few samples in a class)
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=0.2, random_state=42
        )

    # Select and train the appropriate model
    model = None
    if task == 'regression':
        if model_type == "Linear Regression":
            model = LinearRegression()
        elif model_type == "Decision Tree Regressor":
            model = DecisionTreeRegressor(random_state=42)
        else: raise ValueError(f"Unsupported regression model type: {model_type}")
    elif task == 'classification':
        if model_type == "Decision Tree Classifier":
            # Using class_weight='balanced' can help with imbalanced datasets
            model = DecisionTreeClassifier(random_state=42, class_weight='balanced')
        else: raise ValueError(f"Unsupported classification model type: {model_type}")

    model.fit(X_train, y_train)

    # Store information needed for prediction within the model object
    model.feature_names_in_ = encoded_feature_names # Features names *after* encoding
    model.original_features_ = original_features   # Feature names *before* encoding
    model.numeric_features_ = numeric_cols         # List of numeric features used
    model.categorical_features_ = categorical_cols # List of categorical features used
    if task == 'classification' and le:
        model.target_classes_ = le.classes_ # Store original class labels

    # Evaluate the model on the test set
    y_pred = model.predict(X_test)
    score = r2_score(y_test, y_pred) if task == 'regression' else accuracy_score(y_test, y_pred)

    # Return the trained model, its score, the task type, and original features list
    return model, score, task, original_features


def make_predictions(model, X_new, task):
    # Predicts using the trained model on new data X_new

    # Check if the model has the necessary attributes stored during training
    if not hasattr(model, 'feature_names_in_') or \
       not hasattr(model, 'numeric_features_') or \
       not hasattr(model, 'categorical_features_'):
         raise ValueError("Model is missing required attributes. Was it trained correctly with the updated train_model function?")

    # Ensure X_new is a DataFrame
    if not isinstance(X_new, pd.DataFrame):
        raise ValueError("Input data (X_new) must be a pandas DataFrame.")

    X_pred = X_new.copy()

    # Use the stored lists of numeric and categorical features
    numeric_cols = model.numeric_features_
    categorical_cols = model.categorical_features_
    encoded_training_features = model.feature_names_in_

    # Impute missing values in the new data - use the same strategy as training (median/mode)
    for col in numeric_cols:
        if col in X_pred.columns and X_pred[col].isnull().any():
            median_val = X_pred[col].median() # Ideally use training median
            X_pred[col].fillna(median_val, inplace=True)
    for col in categorical_cols:
         if col in X_pred.columns and X_pred[col].isnull().any():
            mode_val = X_pred[col].mode() # Ideally use training mode
            fill_val = mode_val.iloc[0] if not mode_val.empty else 'Missing'
            X_pred[col].fillna(fill_val, inplace=True)

    # One-Hot Encode categorical features consistent with training
    # Only encode columns that were categorical during training
    cols_to_encode = [col for col in categorical_cols if col in X_pred.columns]
    if cols_to_encode:
        X_pred_encoded = pd.get_dummies(X_pred, columns=cols_to_encode, drop_first=True, dummy_na=False)
    else:
        X_pred_encoded = X_pred # No categorical columns to encode

    # Align columns with the training data
    # Add missing columns (that were present during training) with value 0
    # Remove extra columns (not present during training)
    X_pred_aligned = X_pred_encoded.reindex(columns=encoded_training_features, fill_value=0)

    # Ensure the column order matches exactly
    X_pred_aligned = X_pred_aligned[encoded_training_features]

    # Make predictions
    predictions_numeric = model.predict(X_pred_aligned)

    # Decode predictions if it was a classification task
    if task == 'classification' and hasattr(model, 'target_classes_'):
        # Ensure predicted indices are within the bounds of the learned classes
        # Convert predictions to integers for indexing
        predictions_numeric = predictions_numeric.astype(int)
        # Clip values to be safe indices
        predictions_numeric = np.clip(predictions_numeric, 0, len(model.target_classes_) - 1)
        # Map numeric predictions back to original labels
        return model.target_classes_[predictions_numeric]
    else:
        # Return raw numeric predictions for regression
        return predictions_numeric