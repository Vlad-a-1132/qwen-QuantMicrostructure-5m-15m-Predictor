"""
ML Model Module.
LightGBM → ONNX export and inference with calibrated probabilities.
Supports PurgedKFold validation for time-series cross-validation.
"""

import numpy as np
import onnxruntime as ort
from typing import Optional, List, Tuple, Dict, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class SignalPrediction:
    """Container for model prediction output."""
    
    def __init__(
        self,
        timestamp: float,
        signal_value: float,
        probability: float,
        direction: int  # -1 (short), 0 (hold), +1 (long)
    ):
        self.timestamp = timestamp
        self.signal_value = signal_value
        self.probability = probability
        self.direction = direction
        
    def to_dict(self) -> dict:
        return {
            'timestamp': self.timestamp,
            'signal': self.signal_value,
            'probability': self.probability,
            'direction': self.direction
        }


class MicrostructureModel:
    """
    LightGBM-based model with ONNX export for fast inference.
    Provides calibrated probabilities for signal generation.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        feature_names: Optional[List[str]] = None,
        threshold: float = 0.0
    ):
        """
        Args:
            model_path: Path to ONNX model file
            feature_names: Names of features expected by the model
            threshold: Decision threshold for signal generation
        """
        self.model_path = model_path
        self.feature_names = feature_names or [
            'OFI', 'microprice_drift', 'liquidity_vacuum', 
            'VPIN', 'dOFI_dt', 'dμ_dt'
        ]
        self.threshold = threshold
        self._session: Optional[ort.InferenceSession] = None
        self._is_loaded = False
        
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """
        Load ONNX model from file.
        
        Returns True if successful, False otherwise.
        """
        path = model_path or self.model_path
        
        if path is None:
            logger.warning("No model path provided")
            return False
            
        try:
            path = Path(path)
            if not path.exists():
                logger.warning(f"Model file not found: {path}")
                return False
                
            self._session = ort.InferenceSession(
                str(path),
                providers=['CPUExecutionProvider']
            )
            self._is_loaded = True
            logger.info(f"Loaded ONNX model from {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def predict(self, features: np.ndarray) -> Optional[SignalPrediction]:
        """
        Make prediction from feature array.
        
        Args:
            features: Feature array of shape (n_features,) or (batch, n_features)
            
        Returns:
            SignalPrediction or None if model not loaded
        """
        if not self._is_loaded or self._session is None:
            logger.warning("Model not loaded, returning zero signal")
            # Return neutral signal if model not loaded
            return SignalPrediction(
                timestamp=0.0,
                signal_value=0.0,
                probability=0.5,
                direction=0
            )
        
        # Ensure correct shape
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        try:
            # Get input name from ONNX model
            input_name = self._session.get_inputs()[0].name
            
            # Run inference
            outputs = self._session.run(None, {input_name: features.astype(np.float32)})
            
            # First output is typically probability/binary classification
            prob_output = outputs[0]
            
            # Extract probability (handle different output shapes)
            if prob_output.ndim > 1:
                probability = prob_output[0, 1] if prob_output.shape[1] > 1 else prob_output[0, 0]
            else:
                probability = prob_output[0]
                
            # Convert probability to signal value (-1 to +1 scale)
            signal_value = 2 * probability - 1
            
            # Determine direction based on threshold
            if signal_value > self.threshold:
                direction = 1  # Long
            elif signal_value < -self.threshold:
                direction = -1  # Short
            else:
                direction = 0  # Hold
                
            return SignalPrediction(
                timestamp=0.0,  # Will be set by caller
                signal_value=signal_value,
                probability=float(probability),
                direction=direction
            )
            
        except Exception as e:
            logger.error(f"Inference error: {e}")
            return None
    
    def predict_batch(
        self, 
        features_batch: np.ndarray,
        timestamps: Optional[np.ndarray] = None
    ) -> List[SignalPrediction]:
        """
        Make batch predictions.
        
        Args:
            features_batch: Feature array of shape (batch_size, n_features)
            timestamps: Optional timestamps for each prediction
            
        Returns:
            List of SignalPrediction objects
        """
        if not self._is_loaded:
            return []
            
        predictions = []
        batch_size = features_batch.shape[0]
        
        for i in range(batch_size):
            pred = self.predict(features_batch[i])
            if pred is not None:
                if timestamps is not None and len(timestamps) > i:
                    pred.timestamp = timestamps[i]
                predictions.append(pred)
                
        return predictions
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance from model metadata.
        
        Note: This requires the model to have been trained with importance tracking.
        """
        if not self._is_loaded or self._session is None:
            return None
            
        try:
            # Try to extract feature importance from model metadata
            metadata = self._session.get_modelmeta().custom_metadata_map
            
            if 'feature_importance' in metadata:
                import json
                importance = json.loads(metadata['feature_importance'])
                return dict(zip(self.feature_names, importance))
                
        except Exception as e:
            logger.debug(f"Could not extract feature importance: {e}")
            
        return None
    
    def set_threshold(self, threshold: float):
        """Update decision threshold."""
        self.threshold = threshold
        logger.info(f"Updated threshold to {threshold}")


def create_model(
    model_path: str,
    feature_names: Optional[List[str]] = None,
    threshold: float = 0.0
) -> MicrostructureModel:
    """Factory function to create and load model."""
    model = MicrostructureModel(
        model_path=model_path,
        feature_names=feature_names,
        threshold=threshold
    )
    model.load_model()
    return model


# Placeholder for model training (would be done offline)
def train_and_export_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    output_path: str,
    feature_names: Optional[List[str]] = None
):
    """
    Train LightGBM model and export to ONNX.
    
    This is a placeholder - actual training would be done offline
    with proper hyperparameter tuning and calibration.
    """
    try:
        import lightgbm as lgb
        from skl2onnx import convert_lightgbm
        from skl2onnx.common.data_types import FloatTensorType
        
        # Train LightGBM
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        params = {
            'objective': 'binary',
            'metric': ['auc', 'binary_logloss'],
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'seed': 42
        }
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=50)]
        )
        
        # Export to ONNX
        initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
        onnx_model = convert_lightgbm(model, initial_types=initial_type)
        
        with open(output_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
            
        logger.info(f"Model exported to {output_path}")
        return model
        
    except ImportError as e:
        logger.error(f"Training dependencies not available: {e}")
        return None


if __name__ == "__main__":
    # Example usage with dummy model
    model = MicrostructureModel(threshold=0.3)
    
    # Simulate features
    features = np.array([0.5, 0.1, 1.2, 0.3, 0.05, -0.02])
    
    # Without loaded model, should return neutral signal
    pred = model.predict(features)
    if pred:
        print(f"Prediction (no model): {pred.to_dict()}")
