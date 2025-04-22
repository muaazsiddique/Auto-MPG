import logging
import os
from pathlib import Path

from pipelines.common import (
    PYTHON,
    DatasetMixin,
    MLflowMixin,
    build_features_transformer,
    build_model,
    build_target_transformer,
    configure_logging,
    packages,
)
from metaflow import (
    FlowSpec,
    Parameter,
    card,
    conda_base,
    current,
    environment,
    project,
    resources,
    step,
)

# Set up logging configuration for the pipeline
configure_logging()


@project(name="auto_mpg")  # Define the Metaflow project name
@conda_base(
    python=PYTHON,  # Use the Python version defined in common module
    packages=packages(  # Define required Python packages for the conda environment
        "scikit-learn",  # For data preprocessing and metrics
        "pandas",        # For data manipulation
        "numpy",         # For numerical operations
        "keras",         # High-level neural networks API
        "tensorflow",    # ML framework (backend for Keras)
        "mlflow",        # For experiment tracking and model registry
    ),
)
class Training(FlowSpec, DatasetMixin, MLflowMixin):
    """Training pipeline for Auto MPG prediction.

    This pipeline trains, evaluates, and registers a model to predict the miles per gallon
    (MPG) of automobiles based on their features.
    """

    # Pipeline parameters - configurable via command line or environment variables
    mlflow_tracking_uri = Parameter(
        "mlflow-tracking-uri",
        help="Location of the MLflow tracking server.",
        default=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),  # Default to local server if not specified
    )

    training_epochs = Parameter(
        "training-epochs",
        help="Number of epochs that will be used to train the model.",
        default=50,  # Train for 50 epochs by default
    )

    training_batch_size = Parameter(
        "training-batch-size",
        help="Batch size that will be used to train the model.",
        default=32,  # Process 32 samples at a time during training
    )

    accuracy_threshold = Parameter(
        "accuracy-threshold",
        help="Minimum R² score required to register the model.",
        default=0.7,  # Model must achieve at least 0.7 R² score to be registered
    )

    @card  # Generate a Metaflow card with visualizations for this step
    @step  # Define this method as a step in the Metaflow pipeline
    def start(self):
        """Start and prepare the Training pipeline."""
        logging.info("Starting the Auto MPG training pipeline...")
        
        # Set up MLflow tracking - creates or gets a run ID and configures the connection
        self.mlflow_run_id = self.setup_mlflow()
        logging.info("MLflow tracking server: %s", self.mlflow_tracking_uri)
        
        # Log basic pipeline parameters to MLflow for reproducibility
        self.log_params({
            "production_mode": current.is_production,  # Whether this is a production run
            "metaflow_flow_name": current.flow_name,   # Name of the Metaflow pipeline
            "metaflow_run_id": current.run_id,         # Metaflow run ID (different from MLflow run ID)
        })
        
        # Load the Auto MPG dataset from the DatasetMixin
        self.data = self.load_dataset(shuffle=True)  # Shuffle data for better training
        
        # Split features (X) and target variable (y)
        self.X = self.data.drop(['mpg', 'car_name'], axis=1)  # Features: all columns except mpg and car_name
        self.y = self.data['mpg']  # Target: mpg column (what we want to predict)
        
        logging.info("Dataset loaded with %d samples and %d features", 
                    len(self.X), self.X.shape[1])
        
        # Log dataset information to MLflow
        self.log_params({
            "dataset_samples": len(self.X),       # Number of data points
            "dataset_features": self.X.shape[1]   # Number of feature columns
        })
        
        # Branch the pipeline into two parallel paths:
        # 1. cross_validation: To evaluate model performance with cross-validation
        # 2. transform: To prepare data for training the final model
        self.next(self.cross_validation, self.transform)

    @card  # Generate a Metaflow card for this step
    @step  # Define this method as a step in the pipeline
    def cross_validation(self):
        """Generate the indices to split the data for the cross-validation process."""
        from sklearn.model_selection import KFold
        
        logging.info("Creating cross-validation folds...")
        
        # Number of folds for cross-validation - split data into 5 parts
        n_folds = 5
        
        # Create KFold object with shuffling to ensure random distribution across folds
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)  # Fixed random state for reproducibility
        
        # Generate fold indices as list of (fold_idx, (train_idx, val_idx)) tuples
        # This creates indices for 5 different train/test splits
        self.folds = []
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(self.X)):
            self.folds.append((fold_idx, (train_idx, val_idx)))
        
        # Log cross-validation parameters to MLflow
        self.log_params({
            "cv_strategy": "k-fold",   # Type of cross-validation used
            "n_folds": n_folds,        # Number of folds
            "random_state": 42         # Seed for reproducibility
        })
        
        logging.info("Created %d cross-validation folds", n_folds)
        
        # Process each fold independently in parallel
        # The foreach="folds" will run the next step once for each item in self.folds
        self.next(self.transform_fold, foreach="folds")

    @step  # Define this method as a step in the pipeline
    def transform_fold(self):
        """Transform the dataset for the current fold."""
        # Unpack the fold information from the input
        # self.input contains the current fold data from the foreach loop
        self.fold, (train_idx, test_idx) = self.input
        logging.info("Transforming fold %d...", self.fold)
        
        # Get train/validation split for this fold using the indices
        X_train = self.X.iloc[train_idx]  # Training features for this fold
        y_train = self.y.iloc[train_idx]  # Training targets for this fold
        X_test = self.X.iloc[test_idx]    # Testing features for this fold
        y_test = self.y.iloc[test_idx]    # Testing targets for this fold
        
        # Build features transformer (from common module)
        # This likely includes scaling, normalization, one-hot encoding, etc.
        features_transformer = build_features_transformer(X_train)
        
        # Fit and transform training data - learn parameters from training data and apply them
        self.x_train = features_transformer.fit_transform(X_train)
        
        # Transform test data (without fitting) - apply same transformation without learning new parameters
        self.x_test = features_transformer.transform(X_test)
        
        # Store the target data unchanged (assuming no target transformation needed)
        self.y_train = y_train
        self.y_test = y_test
        
        # Store transformer for future use (model registration will need it)
        self.features_transformer = features_transformer
        
        # Log preprocessing info to MLflow
        self.log_params({
            f"fold_{self.fold}_train_samples": len(X_train),  # Number of training samples in this fold
            f"fold_{self.fold}_test_samples": len(X_test)     # Number of test samples in this fold
        })
        
        logging.info("Transformed data for fold %d - X_train: %s", 
                    self.fold, str(self.x_train.shape))
        
        # Continue to the training step for this fold
        self.next(self.train_fold)

    @card  # Generate a Metaflow card for this step
    @environment(vars={"KERAS_BACKEND": "tensorflow"})  # Set environment variable for Keras
    @resources(memory=4096)  # Allocate 4GB of memory for this step
    @step  # Define this method as a step in the pipeline
    def train_fold(self):
        """Train a model as part of the cross-validation process."""
        import mlflow
        import time
        
        logging.info("Training model on fold %d...", self.fold)
        
        # Set up nested MLflow run for this fold
        # This creates a child run under the main pipeline run
        with (
            mlflow.start_run(run_id=self.mlflow_run_id),  # Parent run (main pipeline)
            mlflow.start_run(
                run_name=f"cross-validation-fold-{self.fold}", 
                nested=True  # Create a child run for this fold
            ) as run,
        ):
            # Store fold-specific run ID for later use in evaluation
            self.mlflow_fold_run_id = run.info.run_id
            
            # Turn off MLflow auto-logging to control what gets logged
            mlflow.autolog(log_models=False)
            
            # Get input dimension (number of features after transformation)
            input_dim = self.x_train.shape[1]
            
            # Build the neural network model using common module function
            model = build_model(
                input_dim=input_dim,          # Number of input features
                hidden_units=[64, 32],        # Two hidden layers with 64 and 32 neurons
                activation='relu',            # ReLU activation function
                learning_rate=0.001           # Learning rate for optimizer
            )
            
            # Log model parameters to MLflow
            mlflow.log_params({
                f"fold_{self.fold}_model_type": "neural_network",
                f"fold_{self.fold}_hidden_units": str([64, 32]),
                f"fold_{self.fold}_activation": "relu",
                f"fold_{self.fold}_learning_rate": 0.001,
                f"fold_{self.fold}_input_dim": input_dim,
                f"fold_{self.fold}_epochs": self.training_epochs,
                f"fold_{self.fold}_batch_size": self.training_batch_size
            })
            
            # Track training time for performance monitoring
            start_time = time.time()
            
            # Train the model using Keras fit method
            history = model.fit(
                self.x_train, self.y_train,             # Training data
                epochs=self.training_epochs,            # Number of epochs to train
                batch_size=self.training_batch_size,    # Batch size
                verbose=0                               # No output during training
            )
            
            # Calculate total training time
            training_time = time.time() - start_time
            
            # Store the model and history for evaluation
            self.model = model                 # Trained model
            self.history = history.history     # Training metrics history
            
            # Log training time to MLflow
            mlflow.log_metric(f"fold_{self.fold}_training_time", training_time)
            
            # Log final training metrics
            final_loss = history.history['loss'][-1]                        # Final loss value
            final_mae = history.history['mean_absolute_error'][-1]         # Final MAE value
            
            logging.info(
                "Fold %d - train_loss: %f - train_mae: %f",
                self.fold, final_loss, final_mae
            )
        
        # Continue to the evaluation step for this fold
        self.next(self.evaluate_fold)

    @card  # Generate a Metaflow card for this step
    @environment(vars={"KERAS_BACKEND": "tensorflow"})  # Set environment variable for Keras
    @step  # Define this method as a step in the pipeline
    def evaluate_fold(self):
        """Evaluate the model trained on the current fold."""
        import mlflow
        from sklearn.metrics import mean_squared_error, r2_score
        import numpy as np
        
        logging.info("Evaluating model on fold %d...", self.fold)
        
        # Set up nested MLflow run for this fold's evaluation
        # Reuse the same fold run created in train_fold
        with (
            mlflow.start_run(run_id=self.mlflow_run_id),             # Parent run
            mlflow.start_run(
                run_id=self.mlflow_fold_run_id,                       # Same run as in train_fold
                nested=True
            ),
        ):
            # Evaluate model using Keras evaluate() method
            # Returns [loss, mae, mse] as configured in the model compilation
            self.test_loss, self.test_mae, self.test_mse = self.model.evaluate(
                self.x_test,
                self.y_test,
                verbose=0  # No output during evaluation
            )
            
            # Make predictions for additional metrics calculation
            y_pred = self.model.predict(self.x_test, verbose=0).flatten()
            
            # Calculate additional metrics
            self.test_rmse = np.sqrt(self.test_mse)              # Root Mean Squared Error
            self.test_r2 = r2_score(self.y_test, y_pred)         # R² coefficient of determination
            
            # Log all evaluation metrics to MLflow
            mlflow.log_metrics({
                f"fold_{self.fold}_test_loss": self.test_loss,   # Test loss
                f"fold_{self.fold}_test_mse": self.test_mse,     # Mean Squared Error
                f"fold_{self.fold}_test_rmse": self.test_rmse,   # Root Mean Squared Error
                f"fold_{self.fold}_test_mae": self.test_mae,     # Mean Absolute Error
                f"fold_{self.fold}_test_r2": self.test_r2        # R² score
            })
        
        logging.info(
            "Fold %d - test_rmse: %f - test_r2: %f",
            self.fold, self.test_rmse, self.test_r2
        )
        
        # Continue to the step that averages scores from all folds
        # This is a join point where all fold evaluations converge
        self.next(self.average_scores)

    @card  # Generate a Metaflow card for this step
    @step  # Define this method as a step in the pipeline
    def average_scores(self, inputs):
        """Average the scores from all folds of the cross-validation process."""
        import mlflow
        import numpy as np
        
        logging.info("Averaging scores from all folds...")
        
        # Propagate the MLflow run ID from inputs (all folds have the same main run ID)
        self.merge_artifacts(inputs, include=["mlflow_run_id"])
        
        # Collect metrics from all folds into a list of dictionaries
        metrics = []
        for inp in inputs:
            metrics.append({
                'rmse': inp.test_rmse,     # Root Mean Squared Error
                'r2': inp.test_r2,         # R² score
                'mae': inp.test_mae,       # Mean Absolute Error
                'mse': inp.test_mse,       # Mean Squared Error
                'loss': inp.test_loss      # Loss value
            })
        
        # Calculate average metrics across all folds
        self.test_rmse = np.mean([m['rmse'] for m in metrics])           # Average RMSE
        self.test_rmse_std = np.std([m['rmse'] for m in metrics])        # Standard deviation of RMSE
        self.test_r2 = np.mean([m['r2'] for m in metrics])               # Average R²
        self.test_r2_std = np.std([m['r2'] for m in metrics])            # Standard deviation of R²
        self.test_mae = np.mean([m['mae'] for m in metrics])             # Average MAE
        self.test_mse = np.mean([m['mse'] for m in metrics])             # Average MSE
        self.test_loss = np.mean([m['loss'] for m in metrics])           # Average loss
        
        # Store all metrics in a dictionary for later use
        self.cv_metrics = {
            'avg_mse': self.test_mse,
            'avg_rmse': self.test_rmse,
            'avg_mae': self.test_mae,
            'avg_r2': self.test_r2,
            'std_rmse': self.test_rmse_std,
            'std_r2': self.test_r2_std
        }
        
        # Log average metrics to MLflow
        with mlflow.start_run(run_id=self.mlflow_run_id):
            mlflow.log_metrics(self.cv_metrics)
        
        logging.info("Cross-validation results:")
        logging.info("  Average RMSE: %f ± %f", self.test_rmse, self.test_rmse_std)
        logging.info("  Average R²: %f ± %f", self.test_r2, self.test_r2_std)
        
        # Continue to the register step
        # This joins with the transform-train branch
        self.next(self.register)

    @card  # Generate a Metaflow card for this step
    @step  # Define this method as a step in the pipeline
    def transform(self):
        """Transform the entire dataset for final model training."""
        import time
        
        logging.info("Transforming the entire dataset for final model training...")
        
        # Start timing the transformation process
        start_time = time.time()
        
        # Build the feature transformer on the entire dataset
        # This runs in parallel with the cross-validation branch
        self.features_transformer = build_features_transformer(self.X)
        
        # Fit and transform the entire features dataset
        self.x_transformed = self.features_transformer.fit_transform(self.X)
        self.y_transformed = self.y  # No transformation for the target
        
        # Calculate transformation time
        transform_time = time.time() - start_time
        
        # Log transformation details to MLflow
        self.log_params({
            "transformed_features_shape": str(self.x_transformed.shape),  # Shape after transformation
            "transform_time_seconds": round(transform_time, 2)            # Time taken to transform
        })
        
        logging.info("Dataset transformation completed in %.2f seconds", transform_time)
        logging.info("Transformed features shape: %s", str(self.x_transformed.shape))
        
        # Continue to the training step for the final model
        self.next(self.train)

    @card  # Generate a Metaflow card for this step
    @environment(vars={"KERAS_BACKEND": "tensorflow"})  # Set environment variable for Keras
    @resources(memory=4096)  # Allocate 4GB of memory for this step
    @step  # Define this method as a step in the pipeline
    def train(self):
        """Train the final model on the entire dataset."""
        import mlflow
        import time
        import numpy as np
        
        logging.info("Training final model on the entire dataset...")
        
        # Get input dimension for the model
        input_dim = self.x_transformed.shape[1]
        
        # Set up nested MLflow run for final model training
        with (
            mlflow.start_run(run_id=self.mlflow_run_id),          # Parent run
            mlflow.start_run(
                run_name="final_model_training",                   # Name this run for clarity
                nested=True                                        # Create as a child run
            ) as run,
        ):
            # Store run ID for later reference
            self.final_run_id = run.info.run_id
            
            # Disable automatic model logging
            mlflow.autolog(log_models=False)
            
            # Log training parameters to MLflow
            mlflow.log_params({
                "model_type": "neural_network",
                "hidden_layers": "[64, 32]",
                "activation": "relu",
                "learning_rate": 0.001,
                "batch_size": self.training_batch_size,
                "epochs": self.training_epochs,
                "input_dim": input_dim
            })
            
            # Build the neural network model with same architecture as in CV
            self.model = build_model(
                input_dim=input_dim,          # Number of input features
                hidden_units=[64, 32],        # Two hidden layers
                activation='relu',            # ReLU activation
                learning_rate=0.001           # Learning rate for optimizer
            )
            
            # Track training time
            start_time = time.time()
            
            # Train the model on the entire dataset
            history = self.model.fit(
                self.x_transformed, 
                self.y_transformed,
                epochs=self.training_epochs,
                batch_size=self.training_batch_size,
                verbose=1  # Show progress during training
            )
            
            # Calculate training time
            training_time = time.time() - start_time
            
            # Log training time
            mlflow.log_metric("training_time_seconds", training_time)
            
            # Log final training metrics
            final_loss = history.history['loss'][-1]                         # Final loss
            final_mae = history.history['mean_absolute_error'][-1]          # Final MAE
            final_mse = history.history['mean_squared_error'][-1]           # Final MSE
            final_rmse = np.sqrt(final_mse)                                 # Calculate RMSE
            
            mlflow.log_metrics({
                "final_train_loss": final_loss,
                "final_train_mae": final_mae,
                "final_train_mse": final_mse,
                "final_train_rmse": final_rmse
            })
        
        logging.info("Training completed in %.2f seconds", training_time)
        logging.info("Final training loss (MSE): %.4f", final_loss)
        
        # Continue to the register step - joining with cross-validation branch
        self.next(self.register)

    @environment(vars={"KERAS_BACKEND": "tensorflow"})  # Set environment variable for Keras
    @resources(memory=4096)  # Allocate 4GB of memory for this step
    @step  # Define this method as a step in the pipeline
    def register(self, inputs):
        """Register the model in the MLflow model registry if it meets performance criteria."""
        import os
        import mlflow
        import mlflow.pyfunc
        import tempfile
        import joblib
        from pathlib import Path
        
        logging.info("Preparing to register model in MLflow registry...")
        
        # Merge artifacts from both branches
        cv_branch = inputs[0]  # Cross-validation branch with metrics
        model_branch = inputs[1]  # Model training branch with the model
        
        # Merge artifacts from both branches
        self.merge_artifacts(
            inputs,
            include=["mlflow_run_id", "cv_metrics", "model", "features_transformer"]
        )
        
        # Get CV performance metrics
        avg_r2 = self.cv_metrics.get('avg_r2', 0)           # Average R² from CV
        avg_rmse = self.cv_metrics.get('avg_rmse', float('inf'))  # Average RMSE from CV
        
        logging.info("Model performance from cross-validation:")
        logging.info("  Average R² score: %.4f", avg_r2)
        logging.info("  Average RMSE: %.4f", avg_rmse)
        logging.info("  Required R² threshold: %.4f", self.accuracy_threshold)
        
        # Check if model meets performance threshold
        if avg_r2 >= self.accuracy_threshold:
            self.registered = True
            logging.info("Model meets performance threshold. Proceeding with registration...")
            
            # Create a temporary directory for model artifacts
            with (
                mlflow.start_run(run_id=self.mlflow_run_id),    # Parent run
                tempfile.TemporaryDirectory() as tmp_dir         # Temp directory for artifacts
            ):
                # Prepare model artifacts (trained model and transformers)
                self.artifacts = self._get_model_artifacts(tmp_dir)
                
                # Get pip requirements for the model
                self.pip_requirements = self._get_model_pip_requirements()
                
                # Get path to the model implementation file
                root = Path(__file__).parent
                self.code_paths = [(root / "inference" / "backend.py").as_posix()]
                
                # Register the model in MLflow model registry
                mlflow.pyfunc.log_model(
                    python_model=Path(__file__).parent / "inference" / "model.py",  # Custom PyFunc model class
                    registered_model_name="auto_mpg_predictor",                     # Name in the registry
                    artifact_path="model",                                          # Path within the MLflow run
                    code_paths=self.code_paths,                                     # Additional code files
                    artifacts=self.artifacts,                                       # Model artifacts
                    pip_requirements=self.pip_requirements,                         # Package dependencies
                    example_no_conversion=True                                      # Don't convert example input
                )
                
                # Store the model URI for reference
                self.model_uri = f"models:/auto_mpg_predictor/latest"
                logging.info("Model successfully registered as 'auto_mpg_predictor'")
                logging.info("Model URI: %s", self.model_uri)
        else:
            # Model doesn't meet threshold criteria
            self.registered = False
            logging.info(
                "Model doesn't meet the performance threshold (R² score %.4f < %.4f)",
                avg_r2, self.accuracy_threshold
            )
            logging.info("Model will not be registered.")
            
            # Log the decision not to register
            with mlflow.start_run(run_id=self.mlflow_run_id):
                mlflow.log_params({
                    "model_meets_threshold": False,
                    "cv_avg_r2": avg_r2,
                    "cv_avg_rmse": avg_rmse,
                    "accuracy_threshold": self.accuracy_threshold
                })
            
            # Still store model URI as None
            self.model_uri = None
        
        # Continue to end step
        self.next(self.end)

    @step  # Define this method as a step in the pipeline
    def end(self):
        """End the pipeline."""
        logging.info("Pipeline completed successfully!")
        
        # Print final summary
        if hasattr(self, 'registered') and self.registered:
            logging.info("Registered model URI: %s", self.model_uri)
            logging.info("The model can be loaded using: mlflow.pyfunc.load_model('%s')", self.model_uri)
        else:
            logging.info("No model was registered (performance below threshold or error occurred)")
        
        logging.info("MLflow Run ID: %s", self.mlflow_run_id)
        logging.info("MLflow Experiment URL: %s/#/experiments/1/runs/%s", 
                    self.mlflow_tracking_uri, self.mlflow_run_id)

    def _get_model_artifacts(self, directory: str):
        """Return the list of artifacts that will be included with model.
        
        Args:
            directory: Temporary directory to store artifacts
            
        Returns:
            Dictionary mapping artifact names to their file paths
        """
        import joblib
        from pathlib import Path
        
        # Save the Keras model
        model_path = (Path(directory) / "model.keras").as_posix()
        self.model.save(model_path)
        
        # Save the feature transformer
        features_transformer_path = (Path(directory) / "features_transformer.joblib").as_posix()
        joblib.dump(self.features_transformer, features_transformer_path)
        
        # Return a dictionary mapping artifact names to their file paths
        return {
            "model": model_path,                              # TensorFlow model
            "features_transformer": features_transformer_path, # Feature preprocessing pipeline
        }

    def _get_model_pip_requirements(self):
        """Return the list of required packages to run the model in production.
        
        Returns:
            List of package requirements in pip format
        """
        # Convert from generic package names to specific package==version format
        return [
            f"{package}=={version}" if version else package
            for package, version in packages(
                "scikit-learn",  # For preprocessing
                "pandas",        # For data handling
                "numpy",         # For numerical operations
                "keras",         # For neural network operations
                "tensorflow",    # For backend
            ).items()
        ]


if __name__ == "__main__":
    # Execute the pipeline if this file is run directly
    Training()