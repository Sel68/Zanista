# ml_models.py
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

def train_model(df, target_column, model_type):
    """
    Train a machine learning model (Linear Regression or Decision Tree).
    Determines task (regression/classification) based on target column dtype.
    Handles categorical features using one-hot encoding.
    Returns trained model, score, feature names, and task type.
    """
    df_processed = df.copy()
    
    if target_column not in df_processed.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame.")
        
    y = df_processed[target_column]
    X = df_processed.drop(columns=[target_column])
    
    # Determine task type
    if pd.api.types.is_numeric_dtype(y):
        task = 'regression'
        # If model selected is Classification, raise error or switch? Let's raise error.
        if model_type == "Decision Tree Classifier":
            raise ValueError("Target column is numeric. Cannot use Decision Tree Classifier. Choose Regression.")
    else:
        task = 'classification'
        # Convert target to numerical labels for classification models
        le = LabelEncoder()
        y = le.fit_transform(y) 
        # Store encoder mapping if needed later: model.target_classes_ = le.classes_
        if model_type == "Linear Regression" or model_type == "Decision Tree Regressor":
             raise ValueError("Target column is categorical/object. Cannot use Regression models. Choose Classifier.")

    # Handle categorical features in X using one-hot encoding
    X = pd.get_dummies(X, drop_first=True) # drop_first avoids multicollinearity

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Choose and train model
    if task == 'regression':
        if model_type == "Linear Regression":
            model = LinearRegression()
        elif model_type == "Decision Tree Regressor":
            # Basic Decision Tree - parameters can be tuned
            model = DecisionTreeRegressor(random_state=42) 
        else:
             raise ValueError(f"Unsupported regression model type: {model_type}")
            
    elif task == 'classification':
         if model_type == "Decision Tree Classifier":
              # Basic Decision Tree Classifier
              model = DecisionTreeClassifier(random_state=42)
         else:
              raise ValueError(f"Unsupported classification model type: {model_type}")
    else:
         raise ValueError("Could not determine task type (regression/classification).")


    model.fit(X_train, y_train)

    # Store feature names used during training for consistent prediction
    model.feature_names_in_ = list(X_train.columns)
    if task == 'classification':
         model.target_classes_ = le.classes_ # Store original class names

    # Evaluate
    y_pred = model.predict(X_test)
    
    if task == 'regression':
        score = r2_score(y_test, y_pred)
        # Optionally add RMSE: rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    else: # classification
        score = accuracy_score(y_test, y_pred)

    return model, score, task # Return task type as well

def make_predictions(model, X_new, task):
    """
    Make predictions on new data using the trained model.
    Aligns columns with those used during training.
    Handles categorical features using one-hot encoding.
    For classification, returns original class labels.
    """
    if not hasattr(model, 'feature_names_in_'):
         raise ValueError("Model has not been trained properly or feature names are missing.")
         
    X_new_processed = X_new.copy()
    
    # Handle categorical features in the new data
    X_new_processed = pd.get_dummies(X_new_processed, drop_first=True)
    
    # Align columns - add missing columns with 0, remove extra columns
    training_features = model.feature_names_in_
    X_new_aligned = X_new_processed.reindex(columns=training_features, fill_value=0)
    
    # Ensure order is exactly the same
    X_new_aligned = X_new_aligned[training_features]

    # Make prediction
    predictions_numeric = model.predict(X_new_aligned)

    # If classification, map predictions back to original labels
    if task == 'classification' and hasattr(model, 'target_classes_'):
        return model.target_classes_[predictions_numeric]
    else:
        return predictions_numeric # Return numeric predictions for regression
