import torch
from colorama import Fore
import json


def coco_to_pascal_voc_bbox(bbox):
    xmin = bbox[0]
    ymin = bbox[1]
    xmax = bbox[0] + bbox[2]
    ymax = bbox[1] + bbox[3]
    return [xmin, ymin, xmax, ymax]


def run_model_on():
    """
    select device (whether GPU or CPU)
    :return: wi
    """
    # select device (whether GPU or CPU)
    if torch.cuda.is_available():
        print(Fore.GREEN + 'CUDA is available calculation will be preformed on GPU')
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def save_dict_to_json(file_path: str, dict_obj: dict):
    with open(file_path + '.json', 'w') as json_file:
        json.dump(dict_obj, json_file)


def unique(list1: list):
    """
    return a list with only the unique values
    :param list1:
    :return:
    """
    # insert the list to the set
    list_set = set(list1)
    # convert the set to the list
    unique_list = (list(list_set))
    return unique_list
