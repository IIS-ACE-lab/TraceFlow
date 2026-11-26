#services/side_channel_core/utils/save_utils.py
'''
This module provides utility functions for saving PyTorch models.

It includes a function to save the model's state_dict to a specified path,
and handles the creation of necessary directories if they do not exist.
A failed save raises (TODO §20): a model that was not written must never look
saved.


Author: Shraddha Mishra

'''
import torch

from pathlib import Path
from utils.logger import get_logger

logger = get_logger("Save")

def save_model(model, path, verbose):
    """
    Saves a PyTorch model's state_dict to the given path.

    Args:
        model (torch.nn.Module): The model to save.
        path (str or Path): The file path where the model should be saved.
        verbose (bool): If True, logs a confirmation.
    """
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), path)
    except Exception as e:
        logger.error("Failed to save model to %s: %s", path, e)
        raise
    if verbose:
        logger.info("Model successfully saved to: %s", path)
