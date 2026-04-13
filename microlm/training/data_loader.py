import torch
import numpy as np
import numpy.typing as npt

def get_batch(
        dataset: npt.NDArray,
        batch_size:int,
        context_length: int,
        device: str
) ->tuple[torch.Tensor, torch.Tensor]:
    
    dataset_len = len(dataset)
    max_id = dataset_len - context_length -1
    ix = torch.randint(0, max_id+1, (batch_size,))
    x_stack = [dataset[i:i+context_length] for i in ix]
    y_stack = [dataset[i+1:i+context_length+1] for i in ix]
    x = torch.from_numpy(np.array(x_stack)).to(device).long()
    y = torch.from_numpy(np.array(y_stack)).to(device).long()
    return x, y  