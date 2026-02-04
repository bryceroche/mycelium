"""Signal mapper to predict Qwen attention signals from MiniLM embeddings."""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


class SignalMapper(nn.Module):
    """MLP to map MiniLM embeddings (384-dim) to Qwen signals (3-dim)."""

    def __init__(self, input_dim=384, hidden_dim=128, output_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Sigmoid()  # Signals are normalized to [0, 1]
        )

    def forward(self, x):
        return self.net(x)


class QwenSignalPredictor:
    """Predicts Qwen attention signals (entropy, received, connection) from MiniLM embeddings."""

    _instance: Optional['QwenSignalPredictor'] = None

    def __new__(cls, model_path: Optional[str] = None):
        """Singleton pattern to avoid loading model multiple times."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_path: Optional[str] = None):
        if self._initialized:
            return

        if model_path is None:
            # Find model relative to this file or in common locations
            possible_paths = [
                Path(__file__).parent.parent.parent / "models" / "minilm_to_qwen_mapping.pt",
                Path("models/minilm_to_qwen_mapping.pt"),
                Path.home() / "Desktop" / "mycelium" / "models" / "minilm_to_qwen_mapping.pt"
            ]
            for p in possible_paths:
                if p.exists():
                    model_path = str(p)
                    break
            else:
                raise FileNotFoundError(
                    f"Could not find signal mapper model. Tried: {possible_paths}"
                )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SignalMapper().to(self.device)

        # Load trained weights (weights_only=False because stats dict may have numpy types)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Store stats for reference
        self.stats = checkpoint.get('stats', {})
        self._initialized = True

    def predict(self, embedding: np.ndarray) -> Tuple[float, float, float]:
        """
        Predict Qwen attention signals from a MiniLM embedding.

        Args:
            embedding: 384-dimensional MiniLM embedding

        Returns:
            Tuple of (entropy, received, connection) signals
        """
        with torch.no_grad():
            if embedding.ndim == 1:
                x = torch.tensor(embedding, dtype=torch.float32, device=self.device).unsqueeze(0)
            else:
                x = torch.tensor(embedding, dtype=torch.float32, device=self.device)

            signals = self.model(x)

            if signals.shape[0] == 1:
                signals = signals.squeeze(0)

            return float(signals[0]), float(signals[1]), float(signals[2])

    def predict_batch(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Predict Qwen attention signals for a batch of embeddings.

        Args:
            embeddings: (N, 384) array of MiniLM embeddings

        Returns:
            (N, 3) array of signals [entropy, received, connection]
        """
        with torch.no_grad():
            x = torch.tensor(embeddings, dtype=torch.float32, device=self.device)
            signals = self.model(x)
            return signals.cpu().numpy()


# Global predictor instance
_predictor: Optional[QwenSignalPredictor] = None


def predict_qwen_signals(embedding: np.ndarray) -> Tuple[float, float, float]:
    """
    Convenience function to predict Qwen signals from MiniLM embedding.

    Args:
        embedding: 384-dimensional MiniLM embedding

    Returns:
        Tuple of (entropy, received, connection) signals
    """
    global _predictor
    if _predictor is None:
        _predictor = QwenSignalPredictor()
    return _predictor.predict(embedding)
