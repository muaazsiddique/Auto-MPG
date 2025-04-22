"""Backend utilities for inference."""

import logging
import os
from typing import Any, Dict

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def load_data_from_json(data: Dict[str, Any]) -> pd.DataFrame:
    """Load data from a JSON dictionary into a pandas DataFrame.
    
    Args:
        data: Dictionary with feature values
        
    Returns:
        DataFrame ready for prediction
    """
    # Check that the data is a dictionary
    if not isinstance(data, dict):
        raise ValueError("Input data must be a dictionary")
    
    # Convert to DataFrame
    try:
        df = pd.DataFrame([data])
        logger.info("Successfully created DataFrame with shape %s", df.shape)
        return df
    except Exception as e:
        logger.error("Failed to create DataFrame: %s", e)
        raise