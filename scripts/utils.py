import numpy as np
import torch 
import os 
import json 
import random 

from typing import Dict, List, Tuple, Any 
from ray.data import DatasetContext
from ray.train.torch import get_device
from scripts.config import mlflow

DatasetContext.get_current().execution_options.preserve_order = True

# due to diff sized batches, we need to pad the arrays
def pad_array(arr, dtype=np.int32):
    max_len = max(len(row) for row in arr)
    padded_arr = np.zeros((arr.shape[0], max_len), dtype=dtype)
    
    for i, row in enumerate(arr):
        padded_arr[i][:len(row)] = row
    return padded_arr

def collate_fn(batch: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
    batch["ids"] = pad_array(batch["ids"])
    batch["masks"] = pad_array(batch["masks"])
    dtypes = {"ids": torch.int32, "masks": torch.int32, "targets": torch.int64}
    tensor_batch = {}

    for key, array in batch.items():
        tensor_batch[key] = torch.as_tensor(array, dtype=dtypes[key], device=get_device())
    return tensor_batch

def load_dict(path: str) -> Dict:
    with open(path, "r") as fp:
        return json.load(fp)

def set_seeds(seed=42):
    """Set seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def save_dict(d: Dict, path: str, cls: Any = None, sort_keys: bool = False) -> None:
    dir_path = os.path.dirname(path)

    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    with open(path, "w") as fp:
        json.dump(d, fp, cls=cls, indent=2, sort_keys=sort_keys)
        fp.write("\n")

def get_run_id(exp_name: str, trial_id: str) -> str:
    trial_name = f"TorchTrainer_{trial_id}"
    run = mlflow.search_runs(experiment_names=[exp_name], filter_string=f"tags.trial_name = '{trial_name}'").iloc[0]
    return run.run_id

def dict_to_list(data: Dict, keys: List[str]) -> List[Dict[str, Any]]:
    list_of_dicts = []

    for i in range(len(data[keys[0]])):
        list_of_dicts.append({key: data[key][i] for key in keys})
    
    return list_of_dicts
