import os
from glob import glob
from termcolor import colored

from pygments.lexer import default
from sklearn.model_selection import train_test_split
import torch
from tqdm import tqdm

import numpy as np
from utilities import copy_files_to_folder, read_json, check_ratio, test_train_data_path, move_files_to_folder
from Models.ODs.Yolo5_API import YOLO_MODELS, YOLO_MODELS_FILES
from AnnotationAssistant.AutoAnnotator import CocoData
import cv2 as cv

YOLO5_MODEL_PATH: str = r'../Models/yolov5s.pt'
YOLO5_PATH: str = ""
YOLO5_MODELS_PATH: str = r'C:\Models\Yolo5\yolov5_62\yolov5'



MODELS = {'yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x', 'ResNet50', "MaskRCNN"}
YOLO5_MODELS = {'yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x'}
ONJ_CLR = 'magenta'
ERR_CLR = "red"
INFO_CLR = "blue"
WRN_CLR = "yellow"
UPD_CLR = "cyan"
OK_CLR = "green"

TESTING = False
Yolo5_V6 =  True



def coco2yolo5(coco:CocoData, outDir: str = "", outFolderName: str = "labels"):
    """ gets CCOCO JSON dict 'data' and convert it into Yolo5 labels files in 'outDir' folder

    :param outFolderName:
    :param data: coco dict obj
    :type data: dict
    :param outDir:  where to create the Yolo5 annotations labels files, for example: c:\\MyLabels
    :type outDir: str
    :return: None
    """
    #data = self.data

    # Creates annotation file for each image:
    i_anno = 0
    n_anno = len(coco.annotations)
    for img_data in coco.images:
        lines = []
        # Collect all annotations regarding the image and convert to Yolo5 annotation format:
        while (i_anno < n_anno) and coco.annotations[i_anno].image_id == img_data.id:
            anno_id = coco.annotations[i_anno].category_id - 1
            x_, y_ = coco.annotations[i_anno].bbox[0], coco.annotations[i_anno].bbox[1]
            w_, h_ = coco.annotations[i_anno].bbox[2], coco.annotations[i_anno].bbox[3]
            x = (x_ + 0.5 * w_) / img_data.width
            y = (y_ + 0.5 * h_) / img_data.height
            w = w_ / img_data.width
            h = h_ / img_data.height
            lines.append(f"{anno_id} {round(x, 4)} {round(y, 4)} {round(w, 4)} {round(h, 4)}")
            i_anno += 1

        # Creates output folder:
        #outDir = os.path.join(self.train_data_dir, "labels")
        out_dir: str = os.path.join(outDir, outFolderName)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        # Write the file:

        with open(os.path.join(out_dir, img_data.file_name.split('.')[0] + ".txt"), 'w') as f:
            content = "\n".join(lines)
            f.write(content)
            f.close()


def crop_annotations(data: dict, images_path: str, outDir: str):

    cropped_path: str = os.path.join(outDir, 'patches')
    if not os.path.exists(cropped_path):
        os.mkdir(cropped_path)

    categories: dict = {c["id"]: c["name"] for c in data['categories']}
    coco = CocoData(data)

    ids = [0 for i in range(len(coco.categories))]
    out_dirs = [os.path.join(outDir, "patches", cat.name, cat.status) for cat in coco.categories]

    for _dir in out_dirs:
        if not os.path.exists(_dir):
            os.makedirs(_dir)
    # n_images = len(self.data['images'])
    n_images = len(coco.images)


    for i in tqdm(range(len(data['images'])), ncols=100):
        indx = i
        d = data['images'][i]
    # for indx, d in enumerate(data['images']):
        img_id = d['id']
        file_name = os.path.join(images_path, d['file_name'])
        suffix = file_name.split(".")[-1]
        suffix = 'png'
        img = cv.imread(file_name)
        # print(f"[{indx + 1}/{n_images}] Cropping items from {d['file_name']}...", end=" ")
        for anno in data['annotations']:
            if img_id != anno['image_id']:  # we want only the annotations in our image
                continue
            if anno['image_id'] > img_id:  # annotations ara sorted by image id
                break

            i = anno['category_id'] - 1

            # category_id: str = categories[anno['category_id']]
            # crop_img_file = os.path.join(out_dirs[i], f"{category_id}_{str(ids[i] + 1).rjust(3, '0')}.{suffix}")

            cat_class: str = coco.categories[anno['category_id']-1].supercategory
            cat_obj: str = coco.categories[anno['category_id'] - 1].name
            crop_img_file = os.path.join(out_dirs[i], f"{cat_class}_{cat_obj}_{str(ids[i] + 1).rjust(3, '0')}.{suffix}")

            x0, y0 = anno['bbox'][0], anno['bbox'][1]
            x1, y1 = anno['bbox'][0] + anno['bbox'][2], anno['bbox'][1] + anno['bbox'][3]

            x0 = x0 if type(x0) == int else int(x0)
            x1 = x1 if type(x1) == int else int(x1)
            y0 = y0 if type(y0) == int else int(y0)
            y1 = y1 if type(y1) == int else int(y1)

            cv.imwrite(crop_img_file, img[y0:y1, x0:x1])
            ids[i] += 1
            # print(f"| {categories[anno['category_id']]} ", end=" ")
        # print(f"...Done.")


def test_coco_anno(file_path:str, data: dict):

    categories: dict = {c["id"]: c["name"] for c in data['categories']}
    for image_anno in data['images']:
        annos = [anno for anno in data['annotations'] if anno['image_id'] == image_anno['id']]

        image = cv.imread(os.path.join(file_path,'images',image_anno['file_name']))

        for anno in annos:
            box = anno['bbox']
            pt1 = (box[0],  box[1])
            pt2 = (box[0] + box[2],  box[1] + box[3])
            anno_cat = categories[anno['category_id']]
            image = cv.rectangle(image, pt1, pt2, (255, 0, 0), 5)
            image = cv.putText(image, f"[{anno['category_id']}]{anno_cat}", (pt1[0] - 10, pt1[1] - 10),
                                  cv.FONT_HERSHEY_SIMPLEX, 1,
                                  (0, 255, 255), 2, cv.LINE_AA)

        image = cv.putText(image, f"[{image_anno['file_name']}]", (0, 100), cv.FONT_HERSHEY_SIMPLEX, 1,(0, 255, 255), 2, cv.LINE_AA)
        cv.namedWindow("Test COCO ANNO", cv.WINDOW_NORMAL)
        cv.imshow("Test COCO ANNO", image)
        cv.waitKey(350)


def test_yolo5_annotations(fileDir: str):
    labels_files = glob(os.path.join(fileDir, "labels", "*.txt"))
    images_files = glob(os.path.join(fileDir, "images", "*"))
    data = read_json(glob(os.path.join(fileDir, "*full.json"))[0])
    categories: dict = {c["id"]: c["name"] for c in data['categories']}
    for img_file, lbl_file in zip(images_files, labels_files):
        imgc = cv.imread(img_file)
        img = cv.cvtColor(imgc, cv.COLOR_BGR2GRAY)
        img_anno = imgc.copy()
        h, w = img.shape
        with open(lbl_file) as f:
            data = f.readlines()
            for line in data:
                anno_line = line.split(" ")
                anno_class = categories[int(anno_line[0]) + 1]
                anno_line[-1] = anno_line[-1].replace("\n", "")
                anno_id = int(anno_line.pop(0))  # Store annotation's id
                anno_line = [float(e) for e in anno_line]

                x_c = anno_line[0] * w
                y_c = anno_line[1] * h
                w_ = anno_line[2] * w
                h_ = anno_line[3] * h
                pt1 = (round(x_c - 0.5 * w_), round(y_c - 0.5 * h_))
                pt2 = (round(x_c + 0.5 * w_), round(y_c + 0.5 * h_))
                img_anno = cv.rectangle(img_anno, pt1, pt2, (255, 0, 0), 5)
                img_anno = cv.putText(img_anno, f"[{anno_id}]{anno_class}", (pt1[0] - 10, pt1[1] - 10),
                                      cv.FONT_HERSHEY_SIMPLEX, 1,
                                      (0, 255, 255), 2, cv.LINE_AA)
        cv.namedWindow("Yolo52Coco Converter Test", cv.WINDOW_NORMAL)
        cv.imshow("Yolo52Coco Converter Test", img_anno)
        cv.waitKey(1)
    cv.destroyAllWindows()


def anno2masks(data: dict, outDir: str = ""):
    """ Reads CCOCO JSON file from 'fileDir' and convert it into Yolo5 labels files in 'outDir' folder
    :param data: coco dict obj
    :type data: dict
    :param outDir:  where to create the Yolo5 annotations labels files, for example: c:\\MyLabels
    :type outDir: str
    :return: None
    """
    # Creates output folder:
    out_dir: str = os.path.join(outDir, "masks")
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # Creates image mask(annotations) file for each image:
    i_anno = 0
    n_anno = len(data['annotations'])
    n_cat: int = len(data['categories'])
    # colors = [(255//n_cat)*i for i in range(n_cat+1)]
    colors = [i for i in range(n_cat + 1)]
    cv.namedWindow("Masks", cv.WINDOW_NORMAL)
    for img_data in data['images']:
        lines = []
        mask = np.zeros((img_data['height'], img_data['width'], 3), dtype='uint8')

        # Collect all annotations regarding the image and convert to Yolo5 annotation format:
        while (i_anno < n_anno) and data['annotations'][i_anno]['image_id'] == img_data['id']:
            pts = np.array(data['annotations'][i_anno]['segmentation'][0]).reshape(1, -1, 2)
            c = (colors[data['annotations'][i_anno]['category_id']],
                 colors[data['annotations'][i_anno]['category_id']],
                 colors[data['annotations'][i_anno]['category_id']])
            mask = cv.fillPoly(img=mask, pts=[pts], color=c)

            i_anno += 1

        cv.imshow("Masks", mask*40)
        cv.waitKey(1)

        # Write the file:
        mask_file = os.path.join(out_dir, img_data['file_name'].split('.')[0] + ".bmp")
        cv.imwrite(mask_file, mask)

    cv.destroyWindow("Masks")


class DataPreparatorBase:
    """
    Splits data (images and labels folders) into train | val | test folders by given ratio:
    """
    _model_name: str = ""
    _folder_path: str = ""
    _train_ratio: float = 0.0
    _val_ratio: float = 0.0
    _test_ratio: float = 0.0

    def __init__(self, folderPath: str = "", rTrain: float = 0.8, rVal: float = 0.1, rTest: float = 0.1) -> None:
        """
        :param folderPath: [str] ....\\TrainData
        :param rTrain: [float]
        :param rVal: [float
        :param rTest: [float
        """
        self.train_ratio = rTrain
        self.val_ratio = rVal
        self.test_ratio = rTest
        if folderPath != "":
            self.prepare(folderPath)
            self.split(rTrain, rVal, rTest)
            self.write_files()

    def create_data_folders(self) -> None:
        # images folder:
        if not os.path.exists(os.path.join(self.od_path, "images")):
            os.mkdir(os.path.join(self.od_path, "images"))
            os.mkdir(os.path.join(self.od_path, "images", "train"))
            os.mkdir(os.path.join(self.od_path, "images", "val"))
            os.mkdir(os.path.join(self.od_path, "images", "test"))

        # labels folder:
        if not os.path.exists(os.path.join(self.od_path, "labels")):
            os.mkdir(os.path.join(self.od_path, "labels"))
            os.mkdir(os.path.join(self.od_path, "labels", "train"))
            os.mkdir(os.path.join(self.od_path, "labels", "val"))
            os.mkdir(os.path.join(self.od_path, "labels", "test"))

    def prepare(self, folderPath: str):
        """
        gets the location of the data folder [...\\TrainDada].
        Create Models folder contains all the data, files, images, and default models.
        in short. all the information needed from training model.

        :param folderPath:
        :return: NOne
        """

        self.folder_path = folderPath

        coco_file: str = glob(os.path.join(folderPath, "*_full.json"))[0]
        self.model_name = coco_file[:coco_file.index("_full")].split("\\")[-1]  # folderPath.split("\\")[-1]
        self.od_path = os.path.join(folderPath, "../Models", "OD", "Yolo5")

        self.create_data_folders()
        self.images_files = glob(os.path.join(folderPath, "images", "*"))
        self.labels_files = glob(os.path.join(folderPath, "labels", "*"))
        self.coco_data = read_json(glob(os.path.join(folderPath, "*.json"))[0])

        self.split(rTrain=0.8, rVal=0.1, rTest=0.1)
        self.write_files()

    def split(self, rTrain: float, rVal: float, rTest: float):
        """ Splits the images and labels folders into train,test and vol folders,
        :param rTrain:
        :type rTrain: float
        :param rVal:
        :type rVal: float
        :param rTest:
        :type rTest: float
        :return: None
        """
        assert (rTrain + rVal + rTest) == 1.0, ValueError("Wrong Ratio values. have to be SUM() == 1")
        assert 0 < rTrain < 1 and 0 < rVal < 1 and 0 < rTest < 1, ValueError("Ratio values- have to be >0")
        assert isinstance(rTrain, float) and isinstance(rVal, float) and isinstance(rTest, float)

        # Split the dataset into train-valid-test splits
        train_images, val_images, train_annotations, val_annotations = train_test_split(self.images_files,
                                                                                        self.labels_files,
                                                                                        train_size=rTrain,
                                                                                        test_size=rTest + rVal,
                                                                                        random_state=1)

        val_images, test_images, val_annotations, test_annotations = train_test_split(val_images,
                                                                                      val_annotations,
                                                                                      test_size=0.5,
                                                                                      random_state=1)

        n_test, n_val, n_train = len(test_images), len(val_images), len(train_images)

        print(f"Test images & Annotations ({n_test}):\n", test_images, "\n", test_annotations)
        print(f"Val images & Annotations ({n_val}): \n", val_images, "\n", val_annotations)
        print(f"Train images & Annotations {n_train}): \n", train_images, "\n", train_annotations)

        # Copy images:
        img_dst_dir = os.path.join(os.path.join(self.od_path, "images"))
        copy_files_to_folder(test_images, os.path.join(img_dst_dir, "test"))
        copy_files_to_folder(val_images, os.path.join(img_dst_dir, "val"))
        copy_files_to_folder(train_images, os.path.join(img_dst_dir, "train"))

        # Copy Labels:
        lbl_dst_dir = os.path.join(os.path.join(self.od_path, "labels"))
        copy_files_to_folder(test_annotations, os.path.join(lbl_dst_dir, "test"))
        copy_files_to_folder(val_annotations, os.path.join(lbl_dst_dir, "val"))
        copy_files_to_folder(train_annotations, os.path.join(lbl_dst_dir, "train"))

        print(f"Data as been split into Train-{n_train}, Val-{n_val}, Test-{n_test}")

    def write_files(self):
        pass

    # Model Name
    @property
    def model_name(self):
        return self._model_name

    @model_name.setter
    def model_name(self, name: str):
        assert isinstance(name, str), TypeError("name input must be type string")
        assert len(name) > 0, ValueError(f"project name is empty. got: {name}")
        print(f"updating project name {name}")
        self._model_name = name

    # Folder Path
    @property
    def folder_path(self):
        return self._folder_path

    @folder_path.setter
    def folder_path(self, folderPath: str):
        assert isinstance(folderPath, str), TypeError("folderPath input must be type string")
        assert os.path.exists(folderPath), ValueError(f"{folderPath} doesn't exist")
        print(f"updating folder path to {folderPath}")
        self._folder_path = folderPath

    # train ratio
    @property
    def train_ratio(self):
        return self._train_ratio

    @train_ratio.setter
    def train_ratio(self, value: float):
        if not check_ratio(value):
            return
        self._train_ratio = value

    # val ratio
    @property
    def val_ratio(self):
        return self._val_ratio

    @val_ratio.setter
    def val_ratio(self, value: float):
        if not check_ratio(value):
            return
        self._val_ratio = value

    # test ratio
    @property
    def test_ratio(self):
        return self._test_ratio

    @test_ratio.setter
    def test_ratio(self, value: float):
        if not check_ratio(value):
            return
        self._test_ratio = value


class Yolo5Preparator:
    """
    Splits data (images and labels folders) into train | val | test folders by given ratio:

    Creates Folders:
    responsible for annotation conversion:
    [{MainFolder}]
            - > [TrainData]
                    -> [Models]
                            -> [Yolo5]
                                    -> Training Files and folders
                                    ->[raw]



    """

    hyperPrams = {
        "lr0": 0.01,  # initial learning rate (SGD=1E-2, Adam=1E-3)
        "lrf": 0.2,  # final OneCycleLR learning rate (lr0 * lrf)
        "momentum": 0.937,  # SGD momentum/Adam beta1
        "weight_decay": 0.0005,  # optimizer weight decay 5e-4
        "warmup_epochs": 3.0,  # warmup epochs (fractions ok)
        "warmup_momentum": 0.8,  # warmup initial momentum
        "warmup_bias_lr": 0.1,  # warmup initial bias lr
        "box": 0.05,  # box loss gain
        "cls": 0.5,  # cls loss gain
        "cls_pw": 1.0,  # cls BCELoss positive_weight
        "obj": 1.0,  # obj loss gain (scale with pixels)
        "obj_pw": 1.0,  # obj BCELoss positive_weight
        "iou_t": 0.20,  # IoU training threshold
        "anchor_t": 4.0,  # anchor-multiple threshold

        # anchors: 3  # anchors per output layer (0 to ignore)
        "fl_gamma": 0.0,  # focal loss gamma (efficientDet default gamma=1.5)
        "hsv_h": 0.0,  # 0.015,  # image HSV-Hue augmentation (fraction)
        "hsv_s": 0.0,  # 0.7,  # image HSV-Saturation augmentation (fraction)
        "hsv_v": 0.0,  # 0.4,  # image HSV-Value augmentation (fraction)
        "degrees": 0.0,  # image rotation (+/- deg)
        "translate": 0.2,  # image translation (+/- fraction)
        "scale": 0.0,  # image scale (+/- gain)
        "shear": 0.0,  # image shear (+/- deg)
        "perspective": 0.0,  # image perspective (+/- fraction), range 0-0.001
        "flipud": 0.0,  # image flip up-down (probability)
        "fliplr": 0.0,  # 0.5,  # image flip left-right (probability)
        "mosaic": 0.0,  # image mosaic (probability)
        "mixup": 0.0,  # image mixup (probability)
        "copy_paste": 0.0  # segment copy-paste (probability)
    }

    hyp_header = "# Hyperparameters for COCO training from scratch\n" \
                 "# python train.py --batch 40 --cfg yolov5m.yaml --weights '' --data coco.yaml --img 640 --epochs 300\n" \
                 "# See tutorials for hyperparameter evolution https://github.com/ultralytics/yolov5#tutorials\n\n\n"

    _model_name: str = ""
    _model_type: str = "yolov5s"
    _folder_path: str = ""
    _coco: CocoData
    _model_path: str = ""  # Object Detector Object Folder Path
    _images_files: list[str] = default
    _labels_files: list[str] = default

    def __init__(self, coco: CocoData, modelType: str, folderPath: str = "", modelName: str = "MyObjDet",
                 rTrain: float = 0.8,
                 rVal: float = 0.1,
                 rTest: float = 0.1
                 ) -> None:
        """
        :param folderPath: [str] ....\\TrainData
        :param rTrain: [float]
        :param rVal: [float
        :param rTest: [float
        """
        self.coco = coco
        self.model_name = modelName
        self.model_type = modelType
        if folderPath != "":
            self.prepare(folderPath)
            self.split(rTrain, rVal, rTest)
            self.write_files()

    def create_data_folders(self) -> None:
        # images folder:
        print(colored("Creating folders...", INFO_CLR), end=" ")

        if not os.path.isdir(self.model_path):
            os.mkdir(self.model_path)

        # NEW : Creates raw fodler for storing annotation conversion results before splitting:

        images_path: str = os.path.join(self.model_path, "images")
        if not os.path.exists(images_path):
            os.mkdir(images_path)
            os.mkdir(os.path.join(images_path, "train"))
            os.mkdir(os.path.join(images_path, "val"))
            os.mkdir(os.path.join(images_path, "test"))

        # labels folder:
        labels_path: str = os.path.join(self.model_path, "labels")
        if not os.path.exists(labels_path):
            os.mkdir(labels_path)
            os.mkdir(os.path.join(labels_path, "train"))
            os.mkdir(os.path.join(labels_path, "val"))
            os.mkdir(os.path.join(labels_path, "test"))
            os.mkdir(os.path.join(labels_path, "raw"))

        print(colored("Done", 'green'))


    def prepare(self, folderPath: str, modelName: str = ""):
        """
        gets the location of the data folder [...\\TrainDada].
        Create Models folder contains all the data, files, images, and default models.
        in short. all the information needed from training model.

        :param folderPath: str  [...\\TrainDada].
        :param modelName: str
        :return: None
        """

        self.folder_path = folderPath
        self.model_name = modelName if modelName != "" else self.model_name
        self.model_path = os.path.join(folderPath, "Models", "Yolo5")


        self.create_data_folders()

        # Annotation Conversion:

        # coco_file: str = glob(os.path.join(folderPath, "*.json"))[0]
        # self.coco_data = read_json(coco_file)

        print(colored("Converting COCO 2 Yolo5 lables...", INFO_CLR), end=" ")
        coco2yolo5(self.coco, folderPath)
        print(colored("Done", OK_CLR))

        if TESTING:
            print(colored("Testing COCO 2 Yolo5 lables...", INFO_CLR), end=" ")
            test_yolo5_annotations(folderPath)
            print(colored("Done", OK_CLR))

        self.images_files = glob(os.path.join(folderPath, "images", "*"))
        self.labels_files = glob(os.path.join(folderPath, 'labels', "*"))



    def split(self, rTrain: float, rVal: float, rTest: float):
        """ Splits the images and labels folders into train,test and vol folders,
        :param rTrain:
        :type rTrain: float
        :param rVal:
        :type rVal: float
        :param rTest:
        :type rTest: float
        :return: None
        """
        assert (rTrain + rVal + rTest) == 1.0, ValueError("Wrong Ratio values. have to be SUM() == 1")
        assert 0 < rTrain < 1 and 0 < rVal < 1 and 0 < rTest < 1, ValueError("Ratio values- have to be >0")
        assert isinstance(rTrain, float) and isinstance(rVal, float) and isinstance(rTest, float)

        print(colored("Splitting the data...",INFO_CLR), end=" ")
        # Split the dataset into train-valid-test splits
        train_images, val_images, train_annotations, val_annotations = train_test_split(self.images_files,
                                                                                        self.labels_files,
                                                                                        train_size=rTrain,
                                                                                        test_size=rTest + rVal,
                                                                                        random_state=1)

        val_images, test_images, val_annotations, test_annotations = train_test_split(val_images,
                                                                                      val_annotations,
                                                                                      test_size=0.5,
                                                                                      random_state=1)

        n_test, n_val, n_train = len(test_images), len(val_images), len(train_images)

        print(colored("Done. Data as been split into", OK_CLR), end=" ")
        print(colored(f"Train-{n_train}", "magenta", attrs=['underline', 'bold']),
              colored(f"Val-{n_val}", "green", attrs=['underline', 'bold']),
              colored(f"Test-{n_test}", "yellow", attrs=['underline', 'bold']))


        print(colored(f"\ttrain images & Annotations ({n_train}):", 'magenta'),
              colored(f"\n\t\t{str(train_images[:3])}"),
              colored(f"\n\t\t{str(train_annotations[:3])}"))

        print(colored(f"\tVal images & Annotations ({n_val}):", 'green'),
              colored(f"\n\t\t{str(val_images[:3])}"),
              colored(f"\n\t\t{str(val_annotations[:3])}"))

        print(colored(f"\tTest images & Annotations ({n_test}):", 'yellow'),
              colored(f"\n\t\t{str(test_images[:3])}"),
              colored(f"\n\t\t{str(test_annotations[:3])}"))

        # Copy images:
        img_dst_dir = os.path.join(os.path.join(self.model_path, "images"))
        copy_files_to_folder(test_images, os.path.join(img_dst_dir, "test"))
        copy_files_to_folder(val_images, os.path.join(img_dst_dir, "val"))
        copy_files_to_folder(train_images, os.path.join(img_dst_dir, "train"))

        # Copy Labels:
        lbl_dst_dir = os.path.join(os.path.join(self.model_path, "labels"))
        copy_files_to_folder(test_annotations, os.path.join(lbl_dst_dir, "test"))
        copy_files_to_folder(val_annotations, os.path.join(lbl_dst_dir, "val"))
        copy_files_to_folder(train_annotations, os.path.join(lbl_dst_dir, "train"))

        # Move Labels:
        #lbl_dst_dir = os.path.join(os.path.join(self.model_path, "labels"))
        #move_files_to_folder(test_annotations, os.path.join(lbl_dst_dir, "test"))
        #move_files_to_folder(val_annotations, os.path.join(lbl_dst_dir, "val"))
        #move_files_to_folder(train_annotations, os.path.join(lbl_dst_dir, "train"))



    def write_files(self):

        # YOLO5 Data Config File [yaml]:
        print(colored("Writing training files... "))
        print(colored(f"\tData File: {self.model_name}_data.yaml"))

        if Yolo5_V6:
            with open(os.path.join(self.model_path, f"{self.model_name}_data.yaml"), "w") as data_yaml_file:
                data_yaml_file.write("path: " + os.path.join(self.model_path, "images") + "\n")
                data_yaml_file.write("train: " + "train" + "\n")
                data_yaml_file.write("val: " + "val" + "\n")
                data_yaml_file.write("test: " + "test" + "\n")
                data_yaml_file.write("names:\n")
                for cat in self.coco.categories:
                    data_yaml_file.write(f"\t{cat.id}: {cat.name}\n")

        else:
            with open(os.path.join(self.model_path, f"{self.model_name}_data.yaml"), "w") as data_yaml_file:
                data_yaml_file.write("train: " + os.path.join(self.model_path, "images", "train") + "\n")
                data_yaml_file.write("val: " + os.path.join(self.model_path, "images", "val") + "\n")
                data_yaml_file.write("test: " + os.path.join(self.model_path, "images", "test") + "\n")
                data_yaml_file.write(f"\n# number of classes\nnc: {len(self.coco.categories)}\n")
                data_yaml_file.write(f"\n# class names\nnames: {[d.name for d in self.coco.categories]}\n")

        # YOLO5 Hyperparameter Config File [yaml]:
        print(colored(f"\tHyp File: {self.model_name}_hyp.yaml"))
        with open(os.path.join(self.model_path, f"{self.model_name}_hyp.yaml"), "w") as hyp_yaml_file:
            hyp_yaml_file.write(self.hyp_header)  # write header
            for prm, value in self.hyperPrams.items():  # write all prams and values
                hyp_yaml_file.write(f"{prm}: {value}\n")

        # Download Pretraind model:
        print(colored(f"\tDownloading model File: {self.model_type}.pt"))
        model = torch.hub.load('ultralytics/yolov5', self.model_type, pretrained=True)
        torch.load
        torch.save(model, os.path.join(self.model_path, self.model_type + ".pt"))

        # Copy model Config File
        config_file: str = ""
        for file in YOLO_MODELS_FILES:
            if self.model_type not in file:
                continue
            config_file = file
            break
        assert config_file != "", FileNotFoundError(f"{self.__class__.__name__}-{self.model_type} file not found in {YOLO_MODELS_FILES}")

        copy_files_to_folder([config_file], os.path.join(self.model_path, config_file.split('\\')[-1]))

        print(colored("Done", "green"))

    # Model Type
    @property
    def model_type(self):
        return self._model_type

    @model_type.setter
    def model_type(self, name: str):
        assert isinstance(name, str), TypeError(f"{self.__class__.__name__}-model_type.setter: must be type string")
        assert len(name) > 0, ValueError(f"{self.__class__.__name__}-model_type.setter: empty. got: {name}")
        self._model_type = name

    # Model Name
    @property
    def model_name(self):
        return self._model_name

    @model_name.setter
    def model_name(self, name: str):
        assert isinstance(name, str), TypeError(f"{self.__class__.__name__}-model_name.setter: must be type string")
        assert len(name) > 0, ValueError(f"{self.__class__.__name__}-model_name.setter: empty. got: {name}")
        # print(colored(f"updating project name {name}"))
        self._model_name = name

    # Folder Path
    @property
    def folder_path(self):
        return self._folder_path

    @folder_path.setter
    def folder_path(self, folderPath: str):
        assert isinstance(folderPath, str), TypeError("folderPath input must be type string")
        assert os.path.exists(folderPath), ValueError(f"{folderPath} doesn't exist")
        # print(colored(f"updating folder path to {folderPath}"))
        self._folder_path = folderPath

    # Object Detector Object Folder Path
    @property
    def model_path(self):
        return self._model_path

    @model_path.setter
    def model_path(self, folderPath: str):
        assert isinstance(folderPath, str), TypeError("folderPath input must be type string")
        if not os.path.exists(folderPath):
            # print(colored(f"Creating new folders: {folderPath}"))
            os.makedirs(folderPath)
        # print(colored(f"updating data path to {folderPath}"))
        self._model_path = folderPath

    # COCO Data:
    @property
    def coco(self):
        return self._coco

    @coco.setter
    def coco(self, data: CocoData):
        assert isinstance(data, CocoData), TypeError("data input must be type dict")
        self._coco = data

    # Images Files
    @property
    def images_files(self):
        return self._images_files

    @images_files.setter
    def images_files(self, files: list[str]):
        assert isinstance(files, list), TypeError("images_files input must be type list")
        assert all(map(lambda f: isinstance(f, str), files)), TypeError("images_files must be list of string")
        self._images_files = files

    # Labels Files
    @property
    def labels_files(self):
        return self._labels_files

    @labels_files.setter
    def labels_files(self, files: list[str]):
        assert isinstance(files, list), TypeError("labels_files input must be type list")
        assert all(map(lambda f: isinstance(f, str), files)), TypeError("labels_files must be list of string")
        self._labels_files = files


class DataPreparator:
    """
    DataPreparator class responsible for creating the folders for model training.

    [{MainFolder}]
            - > [TrainData]
                    -> [Models]
                            -> [{ModelType}]
                                    -> Training Files and folders

    Inputs:
        base_folder = {MainFolder}\\TrainData
        coco_file = {MainFolder}\\TrainData\\COCO_FULL.json

    models_path = {MainFolder}\\TrainData\\Models
    model_type_path = {MainFolder}\\TrainData\\Models\\{ModelType}

    Process (Given ModelType):
        (1) Creates 'Models' folder under base_folder.   -> {MainFolder}\\TrainData\\Models
        (2) Creates '{ModelType}' Folder under 'Models' folder -> -> {MainFolder}\\TrainData\\Models\\{ModelType}
             For example Yolo5, ResNet50, MaskRCNN etc...
        (3) Perform annotation conversion according to given ModelType
        (4) Prepare all files for model training in


    Outputs:
        Folders with files ready to train according to given "ModelType"
    """

    def __init__(self, base_folder: str = ""):
        print(colored(f"Initiating {self.__class__.__name__} object...", 'magenta'))
        self.folder: str = ""  # ...\\TrainData
        self.coco_file: str = ""
        self.coco_data: dict = {}
        self.coco = None  # : CocoData(data) TODO: Add empty initiate for CocoData Class

        if base_folder != "":
            self.init(base_folder)

    def init(self, base_folder: str = "") -> None:
        """
        (1) Storing base_folder path.
        (2) Reads and parse CCOCO JSON file into CocoData obj.
        :param base_folder: ...\\TrainData
        :return:None
        """
        test_train_data_path(base_folder)

        self.folder: str = base_folder  # ...\\TrainData

        # extract all labeled data forms:
        self.coco_file: str = glob(os.path.join(base_folder, "*.json"))[0]
        self.coco_data: dict = read_json(self.coco_file)
        self.coco: CocoData = CocoData(self.coco_data)

    def prepare(self, modelType: str, modelName: str = "MyModel", ratios: tuple = (0.8, 0.1, 0.1)):

        # Validate inputs
        if modelType not in MODELS:
            print(colored(f"Invalid model Type [{modelType}]. choose form {str(MODELS)}", ERR_CLR))
            return False

        # Creating main Models Folder:
        models_folder: str = os.path.join(self.folder, "Models")
        if not os.path.exists(models_folder):
            os.mkdir(models_folder)



        print(colored(f"preparing {modelType} model named {modelName}...", 'magenta'))

        # For ResNet50:
        if modelType == "ResNet50":
            self.prepare_resnet50()

        # For Yolo5 Family:
        elif modelType.lower().startswith("yolov5"):
            yolo5_dp = Yolo5Preparator(self.coco, modelType)
            yolo5_dp.prepare(self.folder, modelName=modelName)
            yolo5_dp.split(rTrain=ratios[0], rVal=ratios[1], rTest=ratios[2])
            yolo5_dp.write_files()

        # For MaskRCNN:
        elif modelType == "MaskRCNN":
            pass  # TODO: Prepare MaskRCNN


    def prepare_resnet50(self):
        # (1) Creates Folders:
        # make main model folder
        print(colored("Creating folders...", 'yellow'), end="")
        resnet50_folder: str = os.path.join(self.folder, "Models", "ResNet50")
        if not os.path.exists(resnet50_folder):
            os.mkdir(resnet50_folder)
        print(colored("Done", "green"))


        #print("Annotation Conversion. COCO2Crop")
        images_path: str = os.path.join(self.folder, 'images')
        crop_annotations(self.coco_data, images_path, self.folder)

        print(colored("Copping files..."), end ="")
        patches_folder: str = os.path.join(self.folder, 'patches')
        objs: list[str] = os.listdir(patches_folder)

        # copy obj folder from patches folder to MaskRcnn50\\Obj folder
        for obj in objs:

            dst_obj_folder: str = os.path.join(resnet50_folder, obj)
            if not os.path.exists(dst_obj_folder):
                os.mkdir(dst_obj_folder)


            src_obj_folder: str = os.path.join(patches_folder, obj)
            obj_classes: list[str] = os.listdir(src_obj_folder)
            for cls in obj_classes:
                src_cls_folder: str = os.path.join(src_obj_folder, cls)
                dst_cls_folder: str = os.path.join(dst_obj_folder, cls)
                if not os.path.exists(dst_cls_folder):
                    os.mkdir(dst_cls_folder)

                # Copy all files from src object class to dst object class
                cls_files: list[str] = [os.path.join(src_cls_folder, file) for file in os.listdir(src_cls_folder)]
                copy_files_to_folder(cls_files, dst_cls_folder)



            #obj_folder: str = os.path.join(resnet50_folder, obj)
            #src: str = os.path.join(patches_folder, obj)
            #shutil.copytree(src, obj_folder)
        print(colored("Done", "green"))

    def prepare_maskrcnn(self, modelName: str = "MyObjSeg", train_ratio: float = 0.8, val_ratio: float = 0.2):
        # creates folders:
        # (1) Creates Folders:
        # make main model folder
        model_folder: str = os.path.join(self.folder, "Models", "MaskRCNN", modelName)
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)



if __name__ == "__main__":
    # base_folder = r'C:\BM\Data\WW'
    #my_folder = r'C:\BM\Data\AutoAnnotation\TrainData'
    my_folder = r'C:\BM\Data\AugTest\TrainData'
    # base_folder = r'C:\BM\Data\ITRenew\STN2\TrainData'

    dp = DataPreparator(my_folder)
    dp.prepare("ResNet50")
    dp.prepare("yolov5s")
    # dp.prepare("MaskRCNN")
