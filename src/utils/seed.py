import random
import os
import numpy as np

def set_seed(seed: int = 42):
    """
    Set seed for reproducibility across python, numpy, and potentially deep learning frameworks.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
    # Try importing PyTorch to set seeds if available
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except ImportError:
        pass

    # Try importing TensorFlow to set seeds if available
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass

    print(f"Seed set to: {seed}")
