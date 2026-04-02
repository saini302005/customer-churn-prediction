import joblib
import os

def save_object(obj, filepath):
    """
    Save an object to a file using joblib.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(obj, filepath)
    print(f"Object saved to {filepath}")

def load_object(filepath):
    """
    Load an object from a file using joblib.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    return joblib.load(filepath)
