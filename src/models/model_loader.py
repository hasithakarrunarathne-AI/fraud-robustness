import sys
from pathlib import Path
from typing import Dict, Tuple

import joblib
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.models.train_mlp_torch_v2 import FraudMLP


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MLP_THRESHOLD = 0.9
ENSEMBLE_THRESHOLD = 0.9


def get_saved_model_dir() -> Path:
    return PROJECT_ROOT / "results" / "saved_models"


def get_processed_data_dir() -> Path:
    return PROJECT_ROOT / "data" / "processed"


def load_processed_test_data() -> Tuple[pd.DataFrame, pd.Series]:
    processed_dir = get_processed_data_dir()
    X_test = pd.read_csv(processed_dir / "X_test.csv")
    y_test = pd.read_csv(processed_dir / "y_test.csv").squeeze("columns")
    return X_test, y_test


def load_mlp_model() -> FraudMLP:
    model_dir = get_saved_model_dir()

    meta = joblib.load(model_dir / "mlp_torch_meta.pkl")
    input_dim = int(meta["input_dim"])

    model = FraudMLP(input_dim).to(DEVICE)
    model.load_state_dict(torch.load(model_dir / "mlp_torch.pth", map_location=DEVICE))
    model.eval()

    return model


def load_baseline_models(include_lr: bool = True) -> Dict[str, object]:
    model_dir = get_saved_model_dir()

    models = {
        "SVM": joblib.load(model_dir / "SVM.pkl"),
        "DecisionTree": joblib.load(model_dir / "DecisionTree.pkl"),
        "RandomForest": joblib.load(model_dir / "RandomForest.pkl"),
    }

    if include_lr:
        models["LogisticRegression"] = joblib.load(model_dir / "LogisticRegression.pkl")

    return models