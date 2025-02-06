from typing import List, Dict
from torch import Tensor


def collate_tensor_with_padding(batch: List[Tensor]) -> Tensor:
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas


def collate_pairs_and_text(lst_elements: List, ) -> Dict:
    if 'features' in lst_elements[0]: # test set
        batch = {"motion_feats": collate_tensor_with_padding([el["features"] for el in lst_elements]),
                "length": [x["length"] for x in lst_elements],
                "text": [x["text"] for x in lst_elements],
                "is_transition": collate_tensor_with_padding([el["is_transition"] for el in lst_elements])
                }
    else:
        raise ValueError("Should always have features in lst_elements")
    return batch

