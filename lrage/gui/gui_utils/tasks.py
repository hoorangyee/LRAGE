import torch
from lrage.tasks import TaskManager

def get_all_tasks():
    task_manager = TaskManager()
    return task_manager.all_tasks

def get_all_devices():
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices += [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        devices += ["cuda"]
    return devices