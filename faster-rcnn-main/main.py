import os
import numpy as np
import torch
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
import torchvision
from PIL import Image
from pycocotools.coco import COCO
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN
from colorama import Fore
from tqdm import tqdm, trange
import cv2 as cv
import json
import albumentations as A
from albumentations.pytorch import ToTensorV2
import myutils
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# DONE: DetectionDataset
# DONE: ClassificationDataset
# TODO: upgrade to pytorch 11.
# TODO: SegmentationDataset
# TODO: Loader
# TODO: Training


class DATASET(torch.utils.data.Dataset):
    """
    general class for setting up a data set
    """

    def __init__(self, folder_dir: str, coco_dir: str, transforms=None):
        super().__init__()
        self.folder_dir = folder_dir
        self.transforms = transforms
        self.coco = COCO(coco_dir)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.mapping_label = self.set_mapping()  # dict that connect id of label and the name of the label

    def set_mapping(self):
        """

        :return: dict that connect id of label and the name of the label
        """
        coco = self.coco
        catego = coco.cats
        mapping = {}
        for i in list(catego.keys()):
            mapping[catego[i]["id"]] = catego[i]["name"]
        return mapping


class DetectionDataset(DATASET):
    """
    this calls will manage all loading and access of the data
    """

    def __init__(self, folder_dir: str, coco_dir: str, transforms=None):
        super().__init__(folder_dir, coco_dir, transforms)

    def __getitem__(self, index):
        # Own coco file
        coco = self.coco
        # Image ID
        img_id = self.ids[index]
        # List: get annotation id from coco
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # Dictionary: target coco_annotation file for an image
        coco_annotation = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']
        # open the input image
        img: np.ndarray = cv.imread(os.path.join(self.folder_dir, path)) # BGR
        # Convert the image from BGR to RGB format
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # RGB
        # number of objects in the image
        num_objs = len(coco_annotation)

        # Get category of annotation also known as label
        labels = []
        for i in range(num_objs):
            category_id = torch.as_tensor(coco_annotation[i]['category_id'], dtype=torch.int64)
            labels.append(category_id)
        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        for i in range(num_objs):
            xmin = coco_annotation[i]['bbox'][0]
            ymin = coco_annotation[i]['bbox'][1]
            xmax = xmin + coco_annotation[i]['bbox'][2]
            ymax = ymin + coco_annotation[i]['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])
        # Labels (In my case, only one class: target class or background)
        # labels = torch.ones((num_objs,), dtype=torch.int64)
        # Tensorise img_id
        img_id = torch.tensor([img_id])
        # Size of bbox (Rectangular)
        areas = []
        for i in range(num_objs):
            areas.append(coco_annotation[i]['area'])
        areas = torch.as_tensor(areas, dtype=torch.float32)
        # Iscrowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # make the wanted pipe of transformation to image and bbox
        augmented = self.transforms(image=img,
                                    bboxes=boxes,
                                    category_ids=labels)
        # convert to torch
        boxes: torch.tensor = torch.as_tensor(augmented['bboxes'], dtype=torch.float32)
        labels = torch.as_tensor(augmented['category_ids'], dtype=torch.int64)
        img: torch.tensor = augmented['image']
        # Show image
        # image_array = img.numpy().transpose(1, 2, 0)
        # img_normal = (image_array * 0.5 + 0.5) * 255.0
        # # Display the image using imshow
        # plt.imshow(img_normal)
        # plt.show()

        # Annotation is in dictionary format
        my_annotation = {"boxes": boxes,
                         "labels": labels,
                         "image_id": img_id,
                         "area": areas,
                         "iscrowd": iscrowd}

        return img, my_annotation

    def __len__(self):
        return len(self.ids)

    def set_mapping(self):
        """

        :return: dict that connect id of label and the name of the label
        """
        coco = self.coco
        catego = coco.cats
        mapping = {}
        for i in list(catego.keys()):
            mapping[catego[i]["id"]] = catego[i]["name"]
        return mapping


class ClassificationDataset(DATASET):
    """
    this calls will manage all loading and access of the data
    """

    def __init__(self, folder_dir: str, coco_dir: str, transforms=None):
        super().__init__(folder_dir, coco_dir, transforms)

    def __getitem__(self, index):
        # Own coco file
        coco = self.coco
        # Image ID
        img_id = self.ids[index]
        # List: get annotation id from coco
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # Dictionary: target coco_annotation file for an image
        coco_annotation = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']
        # open the input image
        img: np.ndarray = cv.imread(os.path.join(self.folder_dir, path))

        # Get classification of image
        label = torch.as_tensor(coco_annotation[0]['category_id'], dtype=torch.int64)
        # Tensorise img_id
        img_id = torch.tensor([img_id])
        # make the wanted pipe of transformation to image and bbox
        augmented = self.transforms(image=img)
        # convert to torch
        img = augmented['image']
        # Annotation is in dictionary format
        # my_annotation = {}
        # my_annotation["labels"] = label
        # my_annotation["image_id"] = img_id

        label = label - 1  # the labels start from 0
        return img, label

    def __len__(self):
        return len(self.ids)

    def set_mapping(self):
        """

        :return: dict that connect id of label and the name of the label
        """
        coco = self.coco
        catego = coco.cats
        mapping = {}
        for i in list(catego.keys()):
            mapping[catego[i]["id"]] = catego[i]["name"]
        return mapping


class Loader:
    def __init__(self, data_name: str, data_dir: str, coco: str, _type: str = "detection",
                 trans_settings: dict = {}):

        self.data_name: str = data_name
        self.data_dir = data_dir
        self.task: str = _type
        self.trans_settings: dict = trans_settings
        # self.trans_parm: dict = self.transform_settings(scale_f)

        if _type == "classification":
            dataset = ClassificationDataset
            trans = self.classification_transformed()

        else:
            dataset = DetectionDataset
            trans = self.detection_transformed()

        ## TRAIN SET ##

        # create own Dataset
        self.dataset = dataset(
            folder_dir=data_dir,
            coco_dir=coco,
            transforms=trans
        )

        print(Fore.GREEN + f'{data_name} dataset Was loaded..')

        ## for Documentation ##
        self.docu_data = {
            "data_name": data_name,
            "data length": len(self.dataset),
            "data_dir": data_dir,
            "coco_dir": coco,
            "mapping_label": self.dataset.mapping_label
        }

        # # train DataLoader
        # self.loader = torch.utils.data.DataLoader(
        #     self.dataset,
        #     batch_size=batch_size,
        #     shuffle=True,
        #     num_workers=num_workers,
        #     collate_fn=collate_fn
        # )
        #
        # print(Fore.GREEN + f'{name} Data loader is set..')

    def __call__(self):
        return self.dataset

    def __getitem__(self, index):
        return self.dataset.__getitem__(index)

    def __len__(self):
        return len(self.dataset)

    def Resize(self):
        """
        add a field to the trans_settings dict that will hold the new size of the images
        """
        scale_f: int = self.trans_settings["ScaleFactor"]
        ## get image size
        size = Image.open(
            os.path.join(self.data_dir,
                         os.listdir(self.data_dir)[0])).size
        # setting new size to image
        new_size = lambda w, h: (round(size[0] / w), round(size[1] / h))
        new_size = new_size(scale_f, scale_f)  # (width, height)
        print(f'New size of images {new_size}')
        # this will hold the parameters that is needed to create a trans pipe for all images
        self.trans_settings["Resize"] = new_size

    def detection_transformed(self):
        """
        :return:
        """
        custom_transforms: list = []
        if "ScaleFactor" in self.trans_settings:
            self.Resize()
            custom_transforms.append(
                A.Resize(self.trans_settings["Resize"][1], self.trans_settings["Resize"][0],
                         interpolation=cv.INTER_LANCZOS4))

        custom_transforms += [
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ToTensorV2(),
        ]
        return A.Compose(transforms=custom_transforms,
                         bbox_params=A.BboxParams(format="pascal_voc",
                                                  label_fields=["category_ids"]
                                                  ),
                         )

    def classification_transformed(self):
        """
        :return:
        """
        custom_transforms: list = []
        if "ScaleFactor" in self.trans_settings:
            self.Resize()
            custom_transforms.append(
                A.Resize(self.trans_settings["Resize"][1], self.trans_settings["Resize"][0],
                         interpolation=cv.INTER_LANCZOS4))

        custom_transforms += [A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                              ToTensorV2()]
        return A.Compose(transforms=custom_transforms)


class Model:
    """
    Create a model
    """

    def __init__(self, model_name: str, pretrained: bool = True, classes_num: int = 2, **kwargs):
        models: dict = {
            "fasterRCNN": self.fasterRCNN,
            "ResNet50": self.ResNet50

        }
        self.model_name: str = model_name
        self.pretrained: bool = pretrained
        self.classes_num: int = classes_num
        self.model_kwargs = kwargs
        self.model, self.params = models[self.model_name]()
        # move model to the right device
        self.device = myutils.run_model_on()  # select device (whether GPU or CPU)
        self.model.to(self.device)

        ## for Documentation ##
        self.Docu_Model = {
            "model_name": model_name,
            "pretrained": pretrained,
            "classes_num": classes_num,
            "model_kwargs": self.model_kwargs,
            "device": str(self.device)
        }

    def fasterRCNN(self):
        # load an instance segmentation model pre-trained on COCO
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=self.pretrained)
        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.classes_num)
        # parameters
        params: list = [p for p in model.parameters() if p.requires_grad]
        return model, params

    def ResNet50(self):
        # Define the ResNet50 model
        model = torchvision.models.resnet50(pretrained=self.pretrained)
        # Replace the last fully-connected layer with a new one that has the correct number of classes
        model.fc = nn.Linear(model.fc.in_features, self.classes_num)
        # parameters
        params: list = [p for p in model.fc.parameters() if p.requires_grad]
        return model, params

    def get_model(self):
        return self.model

    def __call__(self, images, target=None):
        if target:
            output = self.model(images, target)
            return output
        output = self.model(images)
        return output


class OptimizerFactory:
    """
    Create an optimizer
    """
    def __init__(self, optimizer_name: str, model_parameters: list, **kwargs):
        self.optimizer_name = optimizer_name
        self.model_parameters = model_parameters
        self.optimizer_kwargs = kwargs
        self.optimizer = self.get_optimizer()

        ## for Documentation ##
        self.Docu_Optimizer = {
            "optimizer_name": optimizer_name,
            "optimizer_kwargs": self.optimizer_kwargs,
        }

    def get_optimizer(self):
        if self.optimizer_name == "SGD":
            optimizer = optim.SGD(self.model_parameters, **self.optimizer_kwargs)
        elif self.optimizer_name == "adam":
            optimizer = optim.Adam(self.model_parameters, **self.optimizer_kwargs)
        elif self.optimizer_name == "adagrad":
            optimizer = optim.Adagrad(self.model_parameters, **self.optimizer_kwargs)
        elif self.optimizer_name == "adadelta":
            optimizer = optim.Adadelta(self.model_parameters, **self.optimizer_kwargs)
        elif self.optimizer_name == "rmsprop":
            optimizer = optim.RMSprop(self.model_parameters, **self.optimizer_kwargs)
        else:
            raise ValueError(f"Unknown optimizer name: {self.optimizer_name}")
        return optimizer

    def __call__(self):
        return self.optimizer


class LossFunctionFactory:
    """
    Define a loss function
    """
    def __init__(self, loss_name, **kwargs):
        self.loss_name = loss_name
        self.loss_kwargs = kwargs
        self.criterion = self.get_loss_function()

        self.device = myutils.run_model_on()  # select device (whether GPU or CPU)
        self.criterion.to(self.device)

        ## for Documentation ##
        self.Docu_LossFunction = {
            "loss_name": loss_name,
            "loss_kwargs": self.loss_kwargs,
        }

    def get_loss_function(self):
        if self.loss_name == "MSE":  # for detection
            loss_function = nn.MSELoss(**self.loss_kwargs)
        elif self.loss_name == "crossentropy":  # for classification
            loss_function = nn.CrossEntropyLoss(**self.loss_kwargs)
        elif self.loss_name == "binarycrossentropy":
            loss_function = nn.BCELoss(**self.loss_kwargs)
        elif self.loss_name == "kldivergence":
            loss_function = nn.KLDivLoss(**self.loss_kwargs)
        elif self.loss_name == "l1":
            loss_function = nn.L1Loss(**self.loss_kwargs)
        elif self.loss_name == "hinge":
            loss_function = nn.HingeEmbeddingLoss(**self.loss_kwargs)
        elif self.loss_name == "triplet":
            loss_function = nn.TripletMarginLoss(**self.loss_kwargs)
        else:
            raise ValueError(f"Unknown loss function name: {self.loss_name}")

        return loss_function

    def __call__(self, inputs, target, mission, mymodel, device):
        if mission == "detection":
            inputs = list(img.to(device) for img in inputs)
            target: list = [{k: v.to(device) for k, v in t.items()} for t in target]
            loss_dict: dict = mymodel(inputs, target)
            losses = sum(loss for loss in loss_dict.values())
            # compute the loss
            return self.criterion(losses, torch.zeros(1).to(device))
        else:

            # Define the loss function and optimizer
            inputs, target = inputs.to(device), target.to(device)
            outputs = mymodel(inputs)
            loss = self.criterion(outputs, target)
            return loss


class Train:

    def __init__(self, dataset: dict, model: Model, optimizer: OptimizerFactory, loss_func: LossFunctionFactory,
                 name: str,
                 batch_size: int = 1,
                 num_workers: int = os.cpu_count(),
                 epoch: int = 10):

        self.training_name: str = name
        self.dataset = dataset
        self.model = model
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.device = model.device
        self.epoch: int = epoch

        # check if we want data back in tuple or all together
        if dataset["train"].task == "classification":
            fn = None
        else:
            fn = collate_fn
        # DataLoader
        self.dataloaders = {
            "train": torch.utils.data.DataLoader(
                dataset["train"],
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                # TODO - maybe this can speed computation
                #pin_memory=True
                collate_fn=fn
            ),
            "val": torch.utils.data.DataLoader(
                dataset["val"],
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                #pin_memory=True
                collate_fn=fn
            )
        }

        print(Fore.GREEN + f'{name} Data loader is set..')
        # this will hold all the losses from each epoch (every epoch will have one loss which is the average of all the
        # loss in the current epoch
        self.loss: pd.DataFrame = pd.DataFrame({
            'train': [],
            'val': []
        })
        ## for Documentation ##
        self.docu_Train = {
            "Train_name": self.training_name,
            "time of training": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            "batch_size": batch_size,
            "num_workers": num_workers,
        }

    def train(self):
        # this will hold all the losses from each epoch (every epoch will have one loss which is the average of all the
        # loss in the current epoch
        loss = self.loss
        device = self.device
        # to run GUI event loop to present the loss graph

        with plt.ion():
            # here we are creating sub plots
            loss_figure, ax = plt.subplots(figsize=(10, 8))
            train_p, val_p = None, None
            with trange(self.epoch) as t:
                for epoch in t:
                    train_loss: torch.tensor = 0.0
                    val_loss: torch.tensor = 0.0
                    self.model.get_model().train()
                    i = 0
                    for imgs, target in self.dataloaders['train']:
                        i += 1

                        losses = self.loss_func(inputs=imgs, target=target,
                                                mission=self.dataset["train"].task,
                                                mymodel=self.model, device=device)

                        # imgs = list(img.to(device) for img in imgs)
                        # target = [{k: v.to(device) for k, v in t.items()} for t in target]
                        # # For Faster R cnn
                        # loss_dict: dict = self.model(imgs, target)
                        # losses = sum(loss for loss in loss_dict.values())
                        # # compute the loss
                        # losses = self.loss_func(losses=losses, target=torch.zeros(1).to(device))
                        self.optimizer().zero_grad()
                        losses.backward()
                        self.optimizer().step()
                        train_loss += losses  # the loss
                    train_loss = train_loss / len(self.dataloaders['train'])  # average
                    # model.eval()  # Optional when not using Model Specific layer
                    with torch.no_grad():
                        for imgs, target in self.dataloaders['val']:
                            losses = self.loss_func(inputs=imgs, target=target,
                                                    mission=self.dataset["train"].task,
                                                    mymodel=self.model, device=device)
                            # imgs: list = list(img.to(device) for img in imgs)
                            # target: list = [{k: v.to(device) for k, v in t.items()} for t in target]
                            # loss_dict = self.model(imgs, target)
                            # losses = sum(loss for loss in loss_dict.values())
                            # losses = self.loss_func(losses=losses, target=torch.zeros(1).to(device))
                            val_loss += losses
                    val_loss = val_loss / len(self.dataloaders['val'])  # average
                    # print(Fore.GREEN + f'the loss on Validation set: {val_loss}')
                    # adding new losses
                    new_loss: pd.DataFrame = pd.DataFrame({
                        'train': [float(train_loss)],
                        'val': [float(val_loss)]
                    })
                    loss = loss.append(new_loss, ignore_index=True)
                    loss.index.name = "epoch"
                    if epoch > 0:
                        loss_figure, ax, train_p, val_p = plot_loss(loss, loss_figure, ax, train_p, val_p)
                    # Description will be displayed on the left
                    t.set_description(f'Epoch: {epoch + 1}')
                    # Postfix will be displayed on the right,
                    # formatted automatically based on argument's datatype
                    t.set_postfix(TrainingLoss=float(train_loss), ValidationLoss=float(val_loss))
            self.loss = loss

    def saving_prot(self):
        """
        the protocol of saving all the needed data for testing and running the trained net
        :param model_name: saving model, test images file and all in needed for future use
        :return:
        """
        # checking if the directory demo_folder
        # exist or not.
        new_dir: bool = False
        while os.path.exists(self.training_name):
            self.training_name += '_new'
        os.makedirs(self.training_name)

        # saving the model, contains the weight and bias matrices for each layer in the model
        torch.save(self.model.get_model().state_dict(), self.training_name + "//" + 'model.pth')

        # saving the dict that connect id to the name of the labels
        with open(f'{self.training_name}//mapping.json', 'w') as fp:
            json.dump(self.dataset["train"].dataset.mapping_label, fp)

        # saving the dict with the transformation parameters that is needed to preform trans pipe for an incoming image
        # that enter the model
        with open(f'{self.training_name}//trans_parm.json', 'w') as fp:
            json.dump(self.dataset["train"].trans_settings, fp, indent=4)

        # general information about the model
        documentation = {"train dataset": self.dataset["train"].docu_data,
                         "validation dataset": self.dataset["val"].docu_data,
                         "model": self.model.Docu_Model,
                         "optimizer": self.optimizer.Docu_Optimizer,
                         "loss_func": self.loss_func.Docu_LossFunction,
                         "Training": self.docu_Train,
                         "transformation parameters": self.dataset["train"].trans_settings,
                         "train loss": float(self.loss["train"].tolist()[-1]),
                         "validation loss": float(self.loss["val"].tolist()[-1])}
        with open(f'{self.training_name}//general_info.json', 'w') as fp:
            json.dump(documentation, fp, indent=4)

        print(Fore.GREEN + f"The model and all is saved in a folder named  - '{self.training_name}'")

        # save the loss plot
        plt.savefig(f'{self.training_name}//Loss Plot.png')


class DetectionDataloader(Loader):

    def __init__(self, data_name: str, data_dir: str, coco: str,
                 scale_f: int = 1, batch_size: int = 1,
                 num_workers: int = os.cpu_count(),
                 num_classes: int = 2, num_epochs: int = 5):
        # inheritance
        super().__init__(data_name=data_name, data_dir=data_dir, coco=coco, _type='detection')
        # add info about net
        self.general_info.update(
            {
                "num classes": num_classes,
                "num epochs": num_epochs,
                "train loss": None,
                "validation loss": None  # 2 classes; Only target class or background
             }
        )

        self.device = myutils.run_model_on()  # select device (whether GPU or CPU)
        # self.train, self.valid, self.test = self.split_data()  # torch type

        self.model = None
        self.optimizer = None


# class TrainDetectionData:
#
#     def __init__(self, loaders: dict, num_classes: int = 2, num_epochs: int = 5):
#         """"""
#         self.loaders: dict = loaders
#         # will hold all the general info on the net
#         self.general_info: dict = {
#             # "split_ratio": s_ratio,
#             "num classes": num_classes,
#             "num epochs": num_epochs,
#             "train loss": None,
#             "validation loss": None  # 2 classes; Only target class or background
#         }
#         self.load_info()
#
#         self.device = myutils.run_model_on()  # select device (whether GPU or CPU)
#         # self.train, self.valid, self.test = self.split_data()  # torch type
#
#         self.model = None
#         self.optimizer = None
#
#     def load_info(self):
#         """
#         load to general info info about all the loaders
#         :return: nothing
#         """
#         for load in self.loaders:
#             self.general_info.update(self.loaders[load].general_info)
#
#     def set_fasterRCNN(self, lr: float = 0.01, momentum: float = 0.9, weight_decay: float = 0.0005):
#         self.model = get_model_instance_segmentation(self.general_info["num classes"])
#
#         # move model to the right device
#         self.model.to(self.device)
#         # parameters
#         params = [p for p in self.model.parameters() if p.requires_grad]
#         self.optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
#
#     def set_ResNet50(self):
#         # Define the ResNet50 model
#         self.model = torchvision.models.resnet50(pretrained=True)
#
#         # move model to the right device
#         self.model.to(self.device)
#         # parameters
#         params = [p for p in self.model.parameters() if p.requires_grad]
#         self.optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
#
#     def saving_prot(self, model_name: str, losses: pd.DataFrame):
#         """
#         the protocol of saving all the needed data for testing and running the trained net
#         :param model_name: saving model, test images file and all in needed for future use
#         :return:
#         """
#         # checking if the directory demo_folder
#         # exist or not.
#         new_dir: bool = False
#         while os.path.exists(model_name):
#             model_name += '_new'
#         os.makedirs(model_name)
#
#         # saving the model, contains the weight and bias matrices for each layer in the model
#         torch.save(self.model.state_dict(), model_name + "//" + 'model.pth')
#
#         # saving the dict that connect id to the name of the labels
#         with open(f'{model_name}//mapping.json', 'w') as fp:
#             json.dump(self.loaders['test'].dataset.mapping_label, fp)
#
#         # saving the dict with the transformation parameters that is needed to preform trans pipe for an incoming image
#         # that enter the model
#         with open(f'{model_name}//trans_parm.json', 'w') as fp:
#             json.dump(self.loaders['test'].trans_parm, fp, indent=4)
#
#         # general information about the model
#         self.general_info["train loss"] = float(losses["train"].tolist()[-1])
#         self.general_info["validation loss"] = float(losses["val"].tolist()[-1])
#         with open(f'{model_name}//general_info.json', 'w') as fp:
#             json.dump(self.general_info, fp, indent=4)
#
#         print(Fore.GREEN + f"The model and all is saved in a folder named  - '{model_name}'")
#
#         # save the loss plot
#         plt.savefig(f'{model_name}//Loss Plot.png')

# Not in use
    def split_data(self):
        if sum(self.general_info["split_ratio"]) != 1:
            raise Exception("Split ratio sum should be one")
        #  splitting will hold the length of the train, valid, test sets
        splitting: list = [round(len(self.train_dataset) * self.general_info["split_ratio"][0]),
                           round(len(self.train_dataset) * self.general_info["split_ratio"][1])]
        splitting.append(len(self.train_dataset) - splitting[0] - splitting[1])
        train, valid, test = torch.utils.data.random_split(self.train_dataset, splitting)
        # creating text file that saves the test images
        all_files_names = [im['file_name'] for im in test.dataset.coco.imgs.values()]
        index = test.indices
        self.general_info["test_images"]: list = []
        for i in index:
            im_id = self.train_dataset.ids[i]
            # get all ID's of the annotations of image
            anno_ids: list = self.train_dataset.coco.getAnnIds(imgIds=[im_id])
            self.general_info["test_images"].append(
                {"name": all_files_names[i],
                 "annotations": self.train_dataset.coco.loadAnns(anno_ids)}
            )
        return train, valid, test


# class TrainingModel:
#
#     def __init__(self, train_data_dir: str, train_coco: str,
#                  val_data_dir: str, val_coco: str,  # s_ratio: tuple,
#                  batch_size: int = 1, num_workers: int = os.cpu_count(),
#                  num_classes: int = 2, num_epochs: int = 5, scale_f: int = None):
#         """"""
#         # will hold all the general info on the net
#         self.general_info: dict = {
#             'train_data_dir': train_data_dir,
#             "train_coco": train_coco,
#             'val_data_dir': val_data_dir,
#             'val_coco': val_coco,
#             # "split_ratio": s_ratio,
#             "batch_size": batch_size,  # number of images per batch
#             "num_classes": num_classes,
#             "num_epochs": num_epochs,
#             "train loss": None,
#             "validation loss": None  # 2 classes; Only target class or background
#         }
#
#         self.trans_parm: dict = self.transform_settings(scale_f)
#
#         ## TRAIN SET ##
#
#         # create own Dataset
#         self.train_dataset = DetectionDataset(
#             root=self.general_info['train_data_dir'],
#             annotation=self.general_info["train_coco"],
#             transforms=transformed(self.trans_parm)
#         )
#
#         print(Fore.GREEN + 'Train dataset Was loaded..')
#
#         # train DataLoader
#         self.train_loader = torch.utils.data.DataLoader(
#             self.train_dataset,
#             batch_size=self.general_info["batch_size"],
#             shuffle=True,
#             num_workers=num_workers,
#             collate_fn=collate_fn
#         )
#
#         print(Fore.GREEN + 'Train Data loader is set..')
#
#         ## VALIDATION SET ##
#
#         # create own Dataset
#         self.val_dataset = DetectionDataset(
#             root=self.general_info['val_data_dir'],
#             annotation=self.general_info["val_coco"],
#             transforms=transformed(self.trans_parm)
#         )
#
#         print(Fore.GREEN + 'Validation dataset Was loaded..')
#
#         # train DataLoader
#         self.valid_loader = torch.utils.data.DataLoader(
#             self.val_dataset,
#             batch_size=self.general_info["batch_size"],
#             shuffle=True,
#             num_workers=num_workers,
#             collate_fn=collate_fn
#         )
#
#         print(Fore.GREEN + 'Validation Data loader is set..')
#
#         self.num_workers = num_workers
#         self.device = myutils.run_model_on()  # select device (whether GPU or CPU)
#         # self.train, self.valid, self.test = self.split_data()  # torch type
#
#         self.model = None
#         # Define loss function and optimizer
#         self.criterion = torch.nn.CrossEntropyLoss()
#         self.optimizer = None
#
#     def transform_settings(self, scale_f: int):
#         """
#
#         :param scale_f: factor of downsampling
#         :return: dict
#         """
#         ## get image size
#         size = Image.open(
#             os.path.join(self.general_info['train_data_dir'],
#                          os.listdir(self.general_info['train_data_dir'])[0])).size
#         # setting new size to image
#         new_size = lambda w, h: (round(size[0] / w), round(size[1] / h))
#         new_size = new_size(scale_f, scale_f)  # (width, height)
#         print(f'New size of images {new_size}')
#         # this will hold the parameters that is needed to create a trans pipe for all images
#         return {'ScaleFactor': scale_f,
#                 'Resize': new_size}
#
#     # Not in use
#     def split_data(self):
#         if sum(self.general_info["split_ratio"]) != 1:
#             raise Exception("Split ratio sum should be one")
#         #  splitting will hold the length of the train, valid, test sets
#         splitting: list = [round(len(self.train_dataset) * self.general_info["split_ratio"][0]),
#                            round(len(self.train_dataset) * self.general_info["split_ratio"][1])]
#         splitting.append(len(self.train_dataset) - splitting[0] - splitting[1])
#         train, valid, test = torch.utils.data.random_split(self.train_dataset, splitting)
#         # creating text file that saves the test images
#         all_files_names = [im['file_name'] for im in test.dataset.coco.imgs.values()]
#         index = test.indices
#         self.general_info["test_images"]: list = []
#         for i in index:
#             im_id = self.train_dataset.ids[i]
#             # get all ID's of the annotations of image
#             anno_ids: list = self.train_dataset.coco.getAnnIds(imgIds=[im_id])
#             self.general_info["test_images"].append(
#                 {"name": all_files_names[i],
#                  "annotations": self.train_dataset.coco.loadAnns(anno_ids)}
#             )
#         return train, valid, test
#
#     def set_fasterRCNN(self, lr: float = 0.01, momentum: float = 0.9, weight_decay: float = 0.0005):
#         self.model = get_model_instance_segmentation(self.general_info["num_classes"])
#
#         # move model to the right device
#         self.model.to(self.device)
#         #  train parameters that are meant to be trained
#         params = [p for p in self.model.parameters() if p.requires_grad]
#         self.optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
#
#     def set_ResNet50(self, lr: float = 0.01, momentum: float = 0.9):
#         # Load the ResNet50 model pre-trained on ImageNet
#         self.model = torchvision.models.resnet50(pretrained=True)
#         # move model to the right device
#         self.model.to(self.device)
#         #  train parameters that are meant to be trained
#         params = [p for p in self.model.parameters() if p.requires_grad]
#         num_ftrs = self.model.fc.in_features
#         self.model.fc = torch.nn.Linear(num_ftrs, self.general_info["num_classes"])
#         # Define loss function and optimizer
#         self.optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum)
#
#     def saving_prot(self, model_name: str, losses: pd.DataFrame):
#         """
#         the protocol of saving all the needed data for testing and running the trained net
#         :param model_name: saving model, test images file and all in needed for future use
#         :return:
#         """
#         # checking if the directory demo_folder
#         # exist or not.
#         if not os.path.exists(model_name):
#             # if the demo_folder directory is not present
#             # then create it.
#             os.makedirs(model_name)
#         # saving the model, contains the weight and bias matrices for each layer in the model
#         torch.save(self.model.state_dict(), model_name + "//" + 'model.pth')
#
#         # saving the dict that connect id to the name of the labels
#         with open(f'{model_name}//mapping.json', 'w') as fp:
#             json.dump(self.train_dataset.mapping_label, fp)
#
#         # saving the dict with the transformation parameters that is needed to preform trans pipe for an incoming image
#         # that enter the model
#         with open(f'{model_name}//trans_parm.json', 'w') as fp:
#             json.dump(self.trans_parm, fp, indent=4)
#
#         # general information about the model
#         self.general_info["train loss"] = float(losses["train"].tolist()[-1])
#         self.general_info["validation loss"] = float(losses["val"].tolist()[-1])
#         with open(f'{model_name}//general_info.json', 'w') as fp:
#             json.dump(self.general_info, fp, indent=4)
#
#         print(Fore.GREEN + f"The model and all is saved in a folder named  - '{model_name}'")
#
#         # save the loss plot
#         plt.savefig(f'{model_name}//Loss Plot.png')


# collate_fn needs for batch

def collate_fn(batch):
    return tuple(zip(*batch))


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def plot_loss(data: pd.DataFrame, figure, ax, train_p, val_p):
    if not train_p:
        train_p, = ax.plot(np.array(data.index) + 1, np.array(data["train"]), label="Training loss")
        val_p, = ax.plot(np.array(data.index) + 1, np.array(data["val"]), label="Validation loss")
        # setting x-axis label and y-axis label
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title('Loss vs. Epochs')
        plt.legend()
    else:
        # updating data values
        # train plot
        train_p.set_xdata(np.array(data.index) + 1)
        train_p.set_ydata(np.array(data["train"]))
        # validation plot
        val_p.set_xdata(np.array(data.index) + 1)
        val_p.set_ydata(np.array(data["val"]))
        # changing the limits of the axis so we can see the new loss
        plt.xlim(1, len(data))
        # find the minimum and max number in the y scale
        plt.ylim(min(list(data["train"]) + list(data["train"])), max(list(data["train"]) + list(data["train"])))
    # convert y-axis to Logarithmic scale
    plt.yscale("logit")

    figure.canvas.draw()
    # This will run the GUI event
    # loop until all UI events
    # currently waiting have been processed
    figure.canvas.flush_events()
    return figure, ax, train_p, val_p


# def train(training: TrainDetectionData):
#     # this will hold all the losses from each epoch (every epoch will have one loss which is the average of all the
#     # loss in the current epoch
#     loss: pd.DataFrame = pd.DataFrame({
#         'train': [],
#         'val': []
#     })
#
#     # to run GUI event loop to present the loss graph
#     with plt.ion():
#         # here we are creating sub plots
#         loss_figure, ax = plt.subplots(figsize=(10, 8))
#         train_p, val_p = None, None
#         with trange(training.general_info["num epochs"]) as t:
#             for epoch in t:
#                 train_loss: torch.tensor = 0.0
#                 val_loss: torch.tensor = 0.0
#                 training.model.train()
#                 i = 0
#                 for imgs, target in training.loaders['train']:
#                     i += 1
#                     imgs = list(img.to(training.device) for img in imgs)
#                     target = [{k: v.to(training.device) for k, v in t.items()} for t in target]
#                     loss_dict: dict = training.model(imgs, target)
#                     # For Faster R cnn
#                     losses = sum(loss for loss in loss_dict.values())
#                     training.optimizer.zero_grad()
#                     losses.backward()
#                     training.optimizer.step()
#                     train_loss += losses  # the loss
#                 train_loss = train_loss / len(training.loaders['train'])  # average
#
#                 # model.eval()  # Optional when not using Model Specific layer
#                 with torch.no_grad():
#                     for imgs, target in training.loaders['val']:
#                         imgs: list = list(img.to(training.device) for img in imgs)
#                         target: list = [{k: v.to(training.device) for k, v in t.items()} for t in target]
#                         loss_dict = training.model(imgs, target)
#                         losses = sum(loss for loss in loss_dict.values())
#
#                         val_loss += losses
#                 val_loss = val_loss / len(training.loaders['val'])  # average
#                 # print(Fore.GREEN + f'the loss on Validation set: {val_loss}')
#                 # adding new losses
#                 new_loss: pd.DataFrame = pd.DataFrame({
#                     'train': [float(train_loss)],
#                     'val': [float(val_loss)]
#                 })
#                 loss = loss.append(new_loss, ignore_index=True)
#                 loss.index.name = "epoch"
#                 if epoch > 0:
#                     loss_figure, ax, train_p, val_p = plot_loss(loss, loss_figure, ax, train_p, val_p)
#                 # Description will be displayed on the left
#                 t.set_description(f'Epoch: {epoch + 1}')
#                 # Postfix will be displayed on the right,
#                 # formatted automatically based on argument's datatype
#                 t.set_postfix(TrainingLoss=float(train_loss), ValidationLoss=float(val_loss))
#         return loss


def trainResNet50(training):
    # this will hold all the losses from each epoch (every epoch will have one loss which is the average of all the
    # loss in the current epoch
    loss: pd.DataFrame = pd.DataFrame({
        'train': [],
        'val': []
    })

    # to run GUI event loop to present the loss graph
    with plt.ion():
        # here we are creating sub plots
        loss_figure, ax = plt.subplots(figsize=(10, 8))
        train_p, val_p = None, None
        with trange(training.general_info["num epochs"]) as t:
            for epoch in t:
                train_loss: torch.tensor = 0.0
                val_loss: torch.tensor = 0.0
                training.model.train()
                i = 0
                for imgs, target in training.loaders['train']:
                    i += 1
                    imgs = list(img.to(training.device) for img in imgs)
                    target = [{k: v.to(training.device) for k, v in t.items()} for t in target]
                    loss_dict: dict = training.model(imgs, target)
                    # For Faster R cnn
                    losses = sum(loss for loss in loss_dict.values())
                    training.optimizer.zero_grad()
                    losses.backward()
                    training.optimizer.step()
                    train_loss += losses  # the loss
                train_loss = train_loss / len(training.loaders['train'])  # average

                # model.eval()  # Optional when not using Model Specific layer
                with torch.no_grad():
                    for imgs, target in training.loaders['val']:
                        imgs: list = list(img.to(training.device) for img in imgs)
                        target: list = [{k: v.to(training.device) for k, v in t.items()} for t in target]
                        loss_dict = training.model(imgs, target)
                        losses = sum(loss for loss in loss_dict.values())

                        val_loss += losses
                val_loss = val_loss / len(training.loaders['val'])  # average
                # print(Fore.GREEN + f'the loss on Validation set: {val_loss}')
                # adding new losses
                new_loss: pd.DataFrame = pd.DataFrame({
                    'train': [float(train_loss)],
                    'val': [float(val_loss)]
                })
                loss = loss.append(new_loss, ignore_index=True)
                loss.index.name = "epoch"
                if epoch > 0:
                    loss_figure, ax, train_p, val_p = plot_loss(loss, loss_figure, ax, train_p, val_p)
                # Description will be displayed on the left
                t.set_description(f'Epoch: {epoch + 1}')
                # Postfix will be displayed on the right,
                # formatted automatically based on argument's datatype
                t.set_postfix(TrainingLoss=float(train_loss), ValidationLoss=float(val_loss))
        return loss


if __name__ == '__main__':

    #Multi SIM Full Data
    M_data_dir_t = r'C:\Users\VisionTeam\Pictures\Data For Deep testing\splitted - SIM_Multi_models\Train - Original + Augmented'
    M_coco_t = r'C:\Users\VisionTeam\Pictures\Data For Deep testing\splitted - SIM_Multi_models\Train_SIM_Multi_modelsAugmented.json'
    M_data_dir_v = r'C:\Users\VisionTeam\Pictures\Data For Deep testing\splitted - SIM_Multi_models\Validation - Original + Augmented'
    M_coco_v = r'C:\Users\VisionTeam\Pictures\Data For Deep testing\splitted - SIM_Multi_models\Validation_SIM_Multi_modelsAugmented.json'
    M_type = "detection"


    # WW ST3 Screw Val Data
    c_data_dir_t = r'C:\Users\VisionTeam\Pictures\Data For Deep testing\WW ST3 Screw Val Data\images'
    c_coco_t = r'C:\Users\VisionTeam\Pictures\Data For Deep testing\WW ST3 Screw Val Data\Train_WW ST3 Screw Val Data.json'
    c_data_dir_v = r'C:\Users\VisionTeam\Pictures\Data For Deep testing\WW ST3 Screw Val Data\images'
    c_coco_v = r'C:\Users\VisionTeam\Pictures\Data For Deep testing\WW ST3 Screw Val Data\Validation_WW ST3 Screw Val Data.json'
    C_type = "classification"
    # Load data
    train_loader = Loader(data_name='train',
                          data_dir=M_data_dir_t,
                          coco=M_coco_t,
                          _type=M_type, trans_settings={"ScaleFactor": 1})

    val_loader = Loader(data_name='val',
                        data_dir=M_data_dir_v,
                        coco=M_coco_v,
                        _type=M_type, trans_settings={"ScaleFactor": 1})

    # load model
    model = Model(model_name="fasterRCNN", pretrained=True, classes_num=4)

    # load optimizer
    optimizer = OptimizerFactory(optimizer_name="SGD", model_parameters=model.params,
                                 lr=0.001, momentum=0.5, weight_decay=0.005)

    # Loss function
    loss_funct = LossFunctionFactory(loss_name='MSE')

    # Training
    datasets = {"train": train_loader,
                "val": val_loader}
    train = Train(dataset=datasets, model=model, optimizer=optimizer, loss_func=loss_funct,
                  name="Multi SIM F 04_02", batch_size=6, epoch=40, num_workers=6)
    train.train()

    train.saving_prot()












    # Loading the data
    # loader = {
    #     'train': Loader(data_name='Train', data_dir=r'C:\Users\VisionTeam\Pictures\Data For Deep testing\splitted - SIM_Multi_models\Train - Original + Augmented',
    #                     coco=r'C:\Users\VisionTeam\Pictures\Data For Deep testing\splitted - SIM_Multi_models\Train_SIM_Multi_modelsAugmented.json',
    #                     scale_f=10, batch_size=6),
    #     'val': Loader(data_name='Val',
    #                   data_dir=r'C:\Users\VisionTeam\Pictures\Data For Deep testing\splitted - SIM_Multi_models\Validation - Original + Augmented',
    #                   coco=r'C:\Users\VisionTeam\Pictures\Data For Deep testing\splitted - SIM_Multi_models\Validation_SIM_Multi_modelsAugmented.json',
    #                   scale_f=10, batch_size=6)
    # }
    #
    # trainer = TrainDetectionData(loaders=loader, num_classes=4, num_epochs=3)

    # training = TrainingModel(
    #     train_data_dir=r'C:\Users\VisionTeam\Pictures\Data For Deep testing\splitted - SIM_Multi_models\Train - Original + Augmented',
    #     train_coco=r'C:\Users\VisionTeam\Pictures\Data For Deep testing\splitted - SIM_Multi_models\Train_SIM_Multi_modelsAugmented.json',
    #     val_data_dir=r'C:\Users\VisionTeam\Pictures\Data For Deep testing\splitted - SIM_Multi_models\Validation - Original + Augmented',
    #     val_coco=r'C:\Users\VisionTeam\Pictures\Data For Deep testing\splitted - SIM_Multi_models\Validation_SIM_Multi_modelsAugmented.json',
    #     batch_size=4,
    #     num_classes=4,
    #     num_epochs=10,
    #     scale_f=10
    #     )
    # #training.set_fasterRCNN()
    # trainer.set_fasterRCNN()
    # loss_pd: pd.DataFrame = trainResNet50(trainer)
    #
    # trainer.saving_prot(
    #     model_name='faster_rcnn_3_15_SIM_Multi_models_factor10',
    #     losses=loss_pd)
