"""
Federated Learning Coordinator
"""
import numpy as np
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class FederatedLearningCoordinator:
    """
    Coordinates federated learning between households.
    Uses Federated Averaging (FedAvg) algorithm to aggregate local models.
    """
    
    def __init__(self):
        self.global_model = self._initialize_model()
        self.local_updates = []
        self.num_households = 0
        
    def _initialize_model(self) -> Dict:
        """
        Initialize a simple linear regression model for consumption prediction.
        The model uses 24 features (one per hour) to predict consumption.
        """
        return {
            "weights": np.random.randn(24) * 0.1,  # One weight per hour
            "bias": 0.5
        }
    
    def get_global_model(self) -> Dict:
        """Get the current global model"""
        return {
            "weights": self.global_model["weights"].copy(),
            "bias": self.global_model["bias"]
        }
    
    def add_local_update(self, local_weights: Dict):
        """Add a local model update from a household"""
        self.local_updates.append(local_weights)
        logger.debug(f"Added local update. Total updates: {len(self.local_updates)}/{self.num_households}")
        
    def ready_to_aggregate(self) -> bool:
        """Check if ready to aggregate (all households reported)"""
        is_ready = len(self.local_updates) >= self.num_households and self.num_households > 0
        return is_ready
    
    def aggregate_models(self) -> Dict:
        """
        Aggregate local models using Federated Averaging (FedAvg).
        Takes the average of all local model weights.
        """
        if not self.local_updates:
            logger.warning("No local updates to aggregate")
            return self.global_model
        
        # Average all weights (FedAvg algorithm)
        avg_weights = np.mean([u["weights"] for u in self.local_updates], axis=0)
        avg_bias = np.mean([u["bias"] for u in self.local_updates])
        
        self.global_model = {
            "weights": avg_weights,
            "bias": avg_bias
        }
        
        logger.info(f"Aggregated {len(self.local_updates)} local models using FedAvg")
        
        # Clear local updates for next round
        self.local_updates = []
        
        return self.get_global_model()
    
    def predict(self, hour: int) -> float:
        """
        Predict consumption for a given hour using the global model.
        Simple linear model: prediction = weights[hour] + bias
        """
        if 0 <= hour < 24:
            prediction = self.global_model["weights"][hour] + self.global_model["bias"]
            return max(0.0, prediction)
        return 0.0
    
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Evaluate model predictions using MSE, MAE, and RMSE metrics.
        
        Args:
            y_true: Actual consumption values
            y_pred: Predicted consumption values
            
        Returns:
            Dict with MSE, MAE, and RMSE values
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        
        # Calculate metrics
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        
        logger.info(f"Model Evaluation - MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
        
        return {
            "MSE": float(mse),
            "MAE": float(mae),
            "RMSE": float(rmse)
        }
