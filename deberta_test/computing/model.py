from sentence_transformers import SentenceTransformer, models
import torch
from transformers import AutoTokenizer, AutoModel


"""
Loads a pre-trained model from the specified module path.

Parameters:
    module_path (str): The path to the pre-trained model module.
    force_cuda (bool): (optional) Whether to force the model to use CUDA if available. Defaults to False.
Returns:
    model: The loaded pre-trained model.

Raises:    
    If an error occurs during model loading, the function will print the error message and return -1.
"""
def load_model(module_path: str, force_cuda=False):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if force_cuda and device == 'cpu':
            print(f"Using CPU for {module_path} - Exiting...")
            return False
        model = SentenceTransformer(module_path, device=device)
    except Exception as e:
        print(e)
        return False

    return model


"""
Load a base model for sentence transformation from huggingface.

Parameters:
    module_name (str): The name of the module to load the base model from. Default is 'sentence-transformers/all-mpnet-base-v2'.
    force_cuda (bool): Whether to force the usage of CUDA. Default is False.

Returns:
    model: The loaded base model for sentence transformation.

Raises:
    Exception: If an error occurs during the loading of the base model.
"""
def load_base_model(module_name='microsoft/deberta-v2-xxlarge', force_cuda=False):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if force_cuda and device == 'cpu':
            print(f"Using CPU for {module_name} - Exiting...")
            return False
        model = AutoModel.from_pretrained(module_name).to(device)
    except Exception as e:
        print(e)
        return False

    return model


def load_tokenizer(module_name='microsoft/deberta-base', force_cuda=False):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if force_cuda and device == 'cpu':
            print(f"Using CPU for {module_name} - Exiting...")
            return False
        tokenizer = AutoTokenizer.from_pretrained(module_name)
    except Exception as e:
        print(e)
        return False

    return tokenizer

    

"""
Check the availability of cuda and return the appropriate device.

Parameters:
    force_cuda (bool): If set to True, force the use of CUDA device and return False if not available. Default is False.

Returns:
    device (str or int): The device to be used for computation. If force_cuda is True and CUDA device is not available, False is returned. Otherwise, the device name ('cuda' or 'cpu') is returned.
"""
def check_device(force_cuda=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if force_cuda and device == 'cpu':
        return False
    return device