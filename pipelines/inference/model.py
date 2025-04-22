import mlflow.pyfunc
import tensorflow as tf
import joblib
import pandas as pd
import numpy as np

class AutoMPGModel(mlflow.pyfunc.PythonModel):
    """Python model for Auto MPG prediction."""
    
    def load_context(self, context):
        """Load model artifacts when the model is loaded."""
        # Load the transformers and model
        self.features_transformer = joblib.load(context.artifacts["features_transformer"])
        self.model = tf.keras.models.load_model(context.artifacts["model"])
    
    def predict(self, context, model_input):
        """Make predictions on new data.
        
        Args:
            context: MLflow model context
            model_input: Pandas DataFrame with input features
        
        Returns:
            Pandas DataFrame with predictions
        """
        # Transform input features
        X_transformed = self.features_transformer.transform(model_input)
        
        # Make predictions
        predictions = self.model.predict(X_transformed).flatten()
        
        # Return as DataFrame
        return pd.DataFrame({
            'mpg_prediction': predictions
        })