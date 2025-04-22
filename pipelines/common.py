"""Common utilities for ML pipelines."""
import logging
import os
import pandas as pd
import numpy as np
import mlflow
from metaflow import current, Parameter

# Python version for conda environment
PYTHON = "3.10.0"

def configure_logging():
    """Configure logging for the pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(__name__)

#################################################
# Package Management
#################################################
# Dictionary of packages with specific versions
PACKAGES = {
    "scikit-learn": "1.3.0",
    "pandas": "2.0.3",
    "numpy": "1.24.4",
    "matplotlib": "3.7.2",
    "mlflow": "2.8.0",
    "tensorflow": "2.12.0",
    "keras": "2.12.0",
}

def packages(*names):
    """Return a dictionary of packages needed for the pipeline.
    
    This function handles both packages with specific versions (from PACKAGES)
    and packages where we want the latest version.
    
    Args:
        *names: Package names to include
        
    Returns:
        Dictionary of packages with their versions
    """
    # Start with an empty dict
    package_dict = {}
    
    # If no names provided, return all packages
    if not names:
        return PACKAGES.copy()
    
    # Add requested packages
    for name in names:
        if name in PACKAGES:
            package_dict[name] = PACKAGES[name]
        else:
            package_dict[name] = None
            
    return package_dict


#################################################
# Dataset Management
#################################################
class DatasetMixin:
    """Mixin class for loading and preprocessing the Auto MPG dataset."""
    
    def load_dataset(self, shuffle=True, seed=42):
        """Load the Auto MPG dataset.
        
        Args:
            shuffle: Whether to shuffle the dataset
            seed: Random seed for reproducibility
            
        Returns:
            Pandas DataFrame with the dataset
        """
        logging.info("Loading Auto MPG dataset (production mode: %s)", current.is_production)
        
        # Define column names for the dataset
        column_names = ['mpg', 'cylinders', 'displacement', 'horsepower', 
                        'weight', 'acceleration', 'model_year', 'origin', 'car_name']
        
        # Load the dataset with custom column names and proper delimiter
        data = pd.read_csv('data/auto-mpg.data', 
                         delim_whitespace=True, 
                         names=column_names, 
                         na_values='?')
        
        # Clean the dataset
        data = self._clean_dataset(data)
        
        # Shuffle if requested
        if shuffle:
            data = data.sample(frac=1, random_state=seed).reset_index(drop=True)
        
        logging.info("Loaded %d samples with %d features", len(data), len(data.columns))
        
        return data
    
    def _clean_dataset(self, data):
        """Clean the dataset.
        
        Args:
            data: Raw dataset
            
        Returns:
            Cleaned dataset
        """
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Convert horsepower to numeric (it may have '?' values)
        df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
        
        # Convert origin to categorical values
        df['origin'] = df['origin'].astype(int).map({1: 'USA', 2: 'Europe', 3: 'Japan'})
        
        # Handle missing values
        for col in df.columns:
            if df[col].isna().sum() > 0:
                if df[col].dtype in ['int64', 'float64']:
                    # Fill numeric columns with median
                    df[col] = df[col].fillna(df[col].median())
                else:
                    # Fill categorical columns with mode
                    df[col] = df[col].fillna(df[col].mode()[0])
        
        return df


#################################################
# MLflow Integration
#################################################
class MLflowMixin:
    """Mixin class for MLflow integration with Metaflow."""
    
    # Define MLflow tracking URI parameter
    mlflow_tracking_uri = Parameter(
        "mlflow_tracking_uri",
        default=os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"),
        help="URI for MLflow tracking server"
    )
    
    mlflow_experiment_name = Parameter(
        "mlflow_experiment_name",
        default="Auto MPG Prediction",
        help="MLflow experiment name"
    )
    
    def setup_mlflow(self):
        """Set up MLflow tracking.
        
        This method:
        1. Sets the tracking URI
        2. Sets the experiment
        3. Starts a run with the Metaflow run ID as the name
        
        Returns:
            The MLflow run ID
        """
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        
        # Set experiment
        mlflow.set_experiment(self.mlflow_experiment_name)
        
        # Start MLflow run using Metaflow run ID as name
        run = mlflow.start_run(run_name=current.run_id)
        
        logging.info("MLflow tracking set up:")
        logging.info("  Tracking URI: %s", self.mlflow_tracking_uri)
        logging.info("  Experiment: %s", self.mlflow_experiment_name)
        logging.info("  Run ID: %s", run.info.run_id)
        logging.info("  Run Name: %s", current.run_id)
        
        # Return the run ID so we can store it
        return run.info.run_id
    
    def log_metrics(self, metrics, step=None):
        """Log metrics to MLflow."""
        # Ensure we have an active run
        with mlflow.start_run(run_id=self.mlflow_run_id):
            mlflow.log_metrics(metrics, step=step)
    
    def log_params(self, params):
        """Log parameters to MLflow."""
        # Ensure we have an active run
        with mlflow.start_run(run_id=self.mlflow_run_id):
            mlflow.log_params(params)
    
    def log_model(self, model, artifact_path, registered_model_name=None):
        """Log a model to MLflow."""
        # Ensure we have an active run
        with mlflow.start_run(run_id=self.mlflow_run_id):
            return mlflow.sklearn.log_model(
                model, 
                artifact_path, 
                registered_model_name=registered_model_name
            )


#################################################
# Data Transformation
#################################################
def build_features_transformer(X, categorical_strategy="most_frequent", numerical_strategy="median"):
    """Build a transformer for feature columns."""
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.impute import SimpleImputer
    
    # Identify numerical and categorical columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    
    # Create transformer for numerical features
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=numerical_strategy)),
        ('scaler', StandardScaler())
    ])
    
    # Create transformer for categorical features
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=categorical_strategy)),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine transformers
    features_transformer = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    return features_transformer

def build_target_transformer(strategy="passthrough"):
    """Build a transformer for the target column."""
    from sklearn.preprocessing import FunctionTransformer, StandardScaler
    import numpy as np
    
    if strategy == "passthrough":
        return None
    elif strategy == "log":
        # Log transformation (useful for skewed targets)
        return FunctionTransformer(np.log1p, np.expm1)
    elif strategy == "standardize":
        # Standardize target
        return StandardScaler()
    else:
        # Default to no transformation
        return None


#################################################
# Model Building
#################################################
def build_model(input_dim, hidden_units=[64, 32], activation='relu', learning_rate=0.001):
    """Build a neural network model for regression."""
    import os
    
    # Set the Keras backend if not already set
    if 'KERAS_BACKEND' not in os.environ:
        os.environ['KERAS_BACKEND'] = 'tensorflow'
    
    import keras
    from keras import layers
    
    # Create model
    model = keras.Sequential()
    
    # Add input layer
    model.add(layers.Dense(hidden_units[0], activation=activation, input_dim=input_dim))
    
    # Add hidden layers
    for units in hidden_units[1:]:
        model.add(layers.Dense(units, activation=activation))
        model.add(layers.Dropout(0.2))  # Add dropout for regularization
    
    # Add output layer (regression has 1 output)
    model.add(layers.Dense(1))
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mean_squared_error',
        metrics=['mean_absolute_error', 'mean_squared_error']
    )
    
    return model