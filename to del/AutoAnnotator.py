import json
import os
from glob import glob
import math
from typing import Union
from tqdm import trange
import time
import cv2 as cv
import numpy as np
from termcolor import colored

import matplotlib.pyplot as plt

import utilities
from utilities import read_json, test_base_folder
from DataSet import DataSet
from AnnotationAssistant.CocoAPI import CocoData

plt.style.use('dark_background')

GOOD_MATCH_PERCENT = 0.95
MAX_FEATURES = 1000

# Microsoft's Common Objects in Context (COCO) dataset Json Format: https://cocodataset.org/#format-data
#
# TODO: Merging multi annotated data files\folders.
# TODO: 




# In Progress....
class ImageRegistrator:
    kp1: tuple
    des1: np.ndarray
    kp2: tuple
    des2: np.ndarray

    bkup_kp1: tuple
    bkup_des1: np.ndarray
    bkup_kp2: tuple
    bkup_des2: np.ndarray

    def __init__(self):
        print(colored(f"Initiating {self.__class__.__name__} object...", 'cyan'))
        # Create KeyPoints detector:
        self.detector = cv.ORB_create(MAX_FEATURES)  # cv.SIFT_create(MAX_FEATURES)
        self.matcher = cv.DescriptorMatcher_create(cv.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)

        # Create backup KeyPoints detector (in cases where main detector fails):
        self.backup_detector = cv.SIFT_create(MAX_FEATURES)
        self.backup_matcher = cv.DescriptorMatcher_create(cv.DESCRIPTOR_MATCHER_BRUTEFORCE_L1)


    def init(self, img: np.ndarray):
        # Calculates template image's keypoints and descriptors:
        self.kp1, self.des1 = self.detector.detectAndCompute(img, None)
        self.bkup_kp1, self.bkup_des1 = self.backup_detector.detectAndCompute(img, None)


    def process(self, img: np.ndarray):
        # Find Transform matrix and parse parameters:
        self.kp2, self.des2 = self.detector.detectAndCompute(img, None)
        h = find_transform(self.kp1, self.des1, self.kp2, self.des2, self.matcher)  # find H
        tx, ty, s, a = affine2prm(h)  # extract H's params
        print(f"T=[tx={round(tx, 3)}, ty={round(ty, 3)}, s={round(s, 3)}, a={round(a, 3)}]", end="...")


def affine2prm(H: np.ndarray) -> list[float, float, float, float]:
    """ Extract physical parameters from the Affine matrix. such as:
    tx - translation in X axis
    ty - translation in Y axis
    s - scale
    alpha - rotation

    :param H: Affine transformation matrix
    :type H: np.ndarray
    :return: list of params [tx, ty, s, alpha]
    :rtype: list[float,float,float,float]
    """
    tx = H[0, 2]
    ty = H[1, 2]
    s = math.sqrt(H[0, 0] ** 2 + H[0, 1] ** 2)
    alpha = math.atan2(H[0, 1], H[0, 0])
    return [tx, ty, s, alpha]


def find_transform(kp1: tuple[cv.KeyPoint], ds1: np.ndarray, kp2: tuple[cv.KeyPoint], ds2: np.ndarray,
                   matcher: cv.DescriptorMatcher) -> np.ndarray:
    global GOOD_MATCH_PERCENT
    # Match features.

    matches = list(matcher.match(ds1, ds2, None))

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    num_good_matches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:num_good_matches]

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = kp1[match.queryIdx].pt
        points2[i, :] = kp2[match.trainIdx].pt

    # Find homography
    # h, _ = cv.estimateAffinePartial2D(points1, points2, cv.RANSAC)
    h, _ = cv.estimateAffine2D(points1, points2, cv.RANSAC)
    # h, mask = cv.findHomography(points1, points2, cv.RANSAC)

    return h


def valid_transform(tx: float, ty: float, s: float, a: float) -> bool:
    result: bool = True
    if not 0.9 < s < 1.1 or not abs(a) < 0.1:
        result = False
    return result




'''
Functions Using Detectron - for MaskRCNN base od Detectron 2
def create_maskrcnn_coco(data: dict,  img_folder: str, outDir: str, fileName: str = "Model"):
    """
    Creating COCO.json file for data annotation. based on Detectron2 Mask RCNN
    :param data: COCO json dict obj
    :param img_folder: location of all images
    :param outDir:  location of result MaskRCNN COCO.json file
    :param fileName:  name of the result file (for example, if "MyModel" given, then MyModel_maskrcnn.json
    :return: None
    """
    dataset_dicts: list[dict] = []  # Result coco object. list of images dicts.
    coco_data = CocoData(data)
    coco: dict = data  # read_json(json_file)  # Read Full annotated COCO Json file (COCO Format)
    coco_images: list[dict] = coco['images']  # Extract images section
    coco_annos: list[dict] = coco['annotations']  # Extract annotations section
    n_annos: int = len(coco_annos)
    indx_anno = 0  # For efficiency, loop all annotations ones. assuming annotations and images are sorted in file.

    # For every image, (1) create 'Image' based Detectron2 Mask RCNN structure. (2) collect all annotations:
    # 'Image' Detectron2 Mask RCNN structure is as follows:
    #      id -> rename to image_id, same value.
    #      file_name-> full path
    #      height - > same
    #      width - > same
    #      annotations -> list of all annotations related to that image
    for c_img in coco_images:
        d: dict = {}
        filename: str = os.path.join(img_folder, c_img['file_name'])
        height, width = c_img['height'], c_img['width']
        idx = c_img['id']
        d["file_name"] = filename
        d["image_id"] = idx
        d["height"] = height
        d["width"] = width
        d["annotations"]: list
        annos = []
        for i_nno in range(indx_anno, n_annos):
            anno: dict = coco_annos[i_nno]

            if anno['image_id'] < idx:
                continue
            elif anno['image_id'] > idx:
                break

            # Assume  anno['image_id'] == idx:
            anno_t = anno.copy()
            anno_t.pop("image_id")
            anno_t["bbox_mode"] = BoxMode.XYXY_ABS
            # anno_t.pop("iscrowd")
            annos.append(anno_t)

            # Option #2: copy all info. TODO: Rty to Train model with that.
            # anno_t: dict = {}
            # anno_t['bbox_mode'] = BoxMode.XYXY_ABS
            # anno_t['bbox'] = anno['bbox'].copy()
            # anno_t['segmentation'] = anno['segmentation'].copy()
            # anno_t['category_id'] = anno['category_id']
            # anno_t['area'] = anno['area']
            # anno_t['id'] = anno['id']
            # annos.append(anno_t)

        d["annotations"] = annos.copy()

        dataset_dicts.append(d)

        new_json_file = os.path.join(outDir, fileName + "_maskRcnn.json")
        with open(new_json_file, 'w') as f:
            json.dump(dataset_dicts, f)

    return dataset_dicts


def coco2maskrcnn(data: dict,  img_folder: str, outDir: str, fileName: str = "Model"):
    """
    Creating COCO.json file for data annotation. based on Detectron2 Mask RCNN
    :param data: COCO json dict obj
    :param img_folder: location of all images
    :param outDir:  location of result MaskRCNN COCO.json file
    :param fileName:  name of the result file (for example, if "MyModel" given, then MyModel_maskrcnn.json
    :return: None
    """
    dataset_dicts: list[dict] = []  # Result coco object. list of images dicts.
    coco: CocoData = CocoData(data)  # Read Full annotated COCO Json file (COCO Format)

    n_annos: int = len(coco.annotations)
    indx_anno = 0  # For efficiency, loop all annotations ones. assuming annotations and images are sorted in file.

    # For every image, (1) create 'Image' based Detectron2 Mask RCNN structure. (2) collect all annotations:
    # 'Image' Detectron2 Mask RCNN structure is as follows:
    #      id -> rename to image_id, same value.
    #      file_name-> full path
    #      height - > same
    #      width - > same
    #      annotations -> list of all annotations related to that image
    for c_img in coco.images:
        d: dict = {}
        filename: str = os.path.join(img_folder, c_img.file_name)
        height, width = c_img.height, c_img.width
        idx = c_img.id
        d["file_name"] = filename
        d["image_id"] = idx
        d["height"] = height
        d["width"] = width
        d["annotations"]: list
        annos = []
        for i_nno in range(indx_anno, n_annos):
            anno: CocoAnnotation = coco.annotations[i_nno]

            if anno.image_id < idx:
                continue
            elif anno.image_id > idx:
                break

            # Assume  anno['image_id'] == idx:
            anno_t = anno.to_dict()
            anno_t.pop("image_id")
            anno_t["bbox_mode"] = BoxMode.XYXY_ABS
            # anno_t.pop("iscrowd")
            annos.append(anno_t)

            # Option #2: copy all info. TODO: Rty to Train model with that.
            # anno_t: dict = {}
            # anno_t['bbox_mode'] = BoxMode.XYXY_ABS
            # anno_t['bbox'] = anno['bbox'].copy()
            # anno_t['segmentation'] = anno['segmentation'].copy()
            # anno_t['category_id'] = anno['category_id']
            # anno_t['area'] = anno['area']
            # anno_t['id'] = anno['id']
            # annos.append(anno_t)

        d["annotations"] = annos.copy()

        dataset_dicts.append(d)

    new_json_file = os.path.join(os.path.join(outDir, "masks"), fileName + "_maskRcnn.json")
    with open(new_json_file, 'w') as f:
        json.dump(dataset_dicts, f)

    return dataset_dicts
'''


class AutoAnnotator:
    folder_dir: str = ""  # 'BaseFolder' ...\Folder that contains "images" folder and "{PrjName}_coco.json" file
    train_data_dir: str = ""  # output folder. TrainData -> ..\{fileDir}\TrainData
    coco_file_dir: str = ""
    data: dict
    show_images: bool = True
    data_size: int = 0
    coco: Union[None, CocoData] = None
    categories: dict
    data_dir: str = ""
    datasets: list[DataSet]

    def __init__(self, folderDir: str = ""):

        print(colored(f"Initiating {self.__class__.__name__} object...", 'cyan'))

        self.coco = CocoData()

        if folderDir != "":
            self.init(folderDir)



    def init(self, folderDir: str, base_name: str):

        if not os.path.isdir(folderDir):
            raise FileNotFoundError(f"{self.__class__.__name__}.folder_dir setter: Cannot find {folderDir}")

        self.data_size = 0

        print(f"Testing Base Folder", colored(folderDir, "yellow"), "...", end="")
        self.folder_dir: str = folderDir  # ...\Folder that contains "images" folder and "{PrjName}_coco.json" file
        test_base_folder(self.folder_dir)
        print(colored("Done", 'green'))

        # Creating output folders:
        print(f"Creating output folders.", end=" ")
        self.train_data_dir: str = os.path.join(folderDir, base_name + "TrainData")  # ..\{fileDir}\TrainData
        if not os.path.exists(self.train_data_dir):  # ..\{fileDir}\TrainData
            os.makedirs(self.train_data_dir)

        new_dir: str = os.path.join(self.train_data_dir, 'images')  # ..\{fileDir}\TrainData\images
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        print(colored("Done", 'green'))
        # ???
        #new_dir: str = os.path.join(self.train_data_dir, 'labels')  # ..\{fileDir}\TrainData\labels
        #if not os.path.exists(new_dir):
        #    os.makedirs(new_dir)



        # init CocoData- Read COCO-JSON File then init CocoData Obj:

        self.read_coco_file(folderDir)

        self.coco.init(self.data)


        print(f"Annotations Found:")
        for cat in self.coco.categories:
            n_anno: int = len([anno for anno in self.coco.annotations if anno.category_id == cat.id])
            print(f"\t[{n_anno} {cat.name}-{cat.status}]")

        self.categories = {c["id"]: c["name"] for c in self.data['categories']}

        # Init DataSet info:
        self.data_dir = os.path.join(folderDir, "images")

        # Parse Folders(DataSets) Information:
        f: str
        folders: list[str] = glob(os.path.join(self.data_dir, "*"))  # load folders dirs
        folders_names: list[str] = [f.split('\\')[-1] for f in folders if os.path.isdir(f)]  # extract names
        self.datasets = [DataSet(n, f) for n, f in zip(folders_names, folders)]
        print(f"{len(folders_names)} folder/s was found in {self.data_dir}: {folders_names}")

        for i, ds in enumerate(self.datasets):
            self.data_size += ds.n_images
            print(ds, f", Annotated Image: {self.coco.images[i].file_name}")
        print(colored("Total images", 'yellow'), f" = {self.data_size}.")


    def read_coco_file(self, folderDir: str) -> None:
        """Gets the folder path, search, read and parse the {Proejct_Name}_coco.json file
        :param folderDir: location of the coco.json file
        :type folderDir: str
        :return: None
        """
        # Read COCO-JSON File:
        json_file: str = glob(os.path.join(folderDir, "*.json"))[0]

        print("Loading COCO file ", colored(f"{json_file}", 'yellow'), "...", end="")
        self.data: dict = read_json(json_file)

        for i, image_anno in enumerate(self.data['images']):
            self.data['images'][i]['file_name'] = self.data['images'][i]['file_name'].split("\\")[-1]

        self.coco_file_dir: str = json_file

        print(colored("Done", 'green'))

    def annotate(self) -> dict:

        # Create KeyPoints detector:
        detector = cv.ORB_create(MAX_FEATURES)  # cv.SIFT_create(MAX_FEATURES)
        matcher = cv.DescriptorMatcher_create(cv.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)

        # Create backup KeyPoints detector (in cases where main detector fails):
        backup_detector = cv.SIFT_create(MAX_FEATURES)
        backup_matcher = cv.DescriptorMatcher_create(cv.DESCRIPTOR_MATCHER_BRUTEFORCE_L1)

        # every image in origin COCO file refer to dataset folder:
        i_all = 0
        i_anno = len(self.data['annotations']) + 1  # Count currently annotations in COCO file. to add
        i_img = len(self.datasets) + 1

        # creating a show object for presenting annotation on images
        show = utilities.ShowAnno("annotation", (1200, 800))
        # cv.namedWindow("Annotations", cv.WINDOW_NORMAL)
        # cv.resizeWindow("Annotations", 1200, 800)

        for coco_img, ds in zip(self.coco.images, self.datasets):
            # coco_img: CocoImage
            #
            images_files: list[str] = ds.files.copy()  # copy all images files
            img_ref_file = os.path.join(ds.data_path, coco_img.file_name)  # get template image file

            images_files.remove(img_ref_file)  # remove template image file from the list. we want to all others.

            images_names = [file.split("\\")[-1] for file in images_files]  # extract all files names (without dirs)

            utilities.copy_file(file_dir=img_ref_file, destination_folder=os.path.join(self.train_data_dir, "images"))

            img_ref = cv.imread(img_ref_file)  # Read template image as color image
            img_ref = cv.cvtColor(img_ref, cv.COLOR_BGR2GRAY)  # convert template image to gray

            # Calculates template image's keypoints and descriptors:
            kp1, des1 = detector.detectAndCompute(img_ref, None)
            bkup_kp1, bkup_des1 = backup_detector.detectAndCompute(img_ref, None)

            # Count number of images to add into COCO file
            n_images = self.data_size - len(self.datasets)  # len(images_files)
            new_annotations = []
            new_images = []

            # Perform annotation alignment between template image and all images, and add to COCO:
            with trange(len(images_files)) as t:
                for i in t:

                    img_file = images_files[i]

                    # Description will be displayed on the left
                    t.set_description(f"Dataset: {ds.name}, Processing {img_file}...",)

                    # read image from folder(list):
                    imgc = cv.imread(img_file)
                    img = cv.cvtColor(imgc, cv.COLOR_BGR2GRAY)
                    rows, cols = img.shape  # extract image's width and height

                    # Find Transform matrix and parse parameters:
                    kp2, des2 = detector.detectAndCompute(img, None)
                    h = find_transform(kp1, des1, kp2, des2, matcher)  # find H
                    tx, ty, s, a = affine2prm(h)  # extract H's params

                    # Postfix will be displayed on the right,
                    # formatted automatically based on argument's datatype

                    # Check if transformation is valid:
                    if not valid_transform(tx, ty, s, a):
                        # Tray again, with backup keypoint detector:
                        #print(f"Fail. using {type(backup_detector)}.", end="...")
                        kp2, des2 = backup_detector.detectAndCompute(img, None)
                        h = find_transform(bkup_kp1, bkup_des1, kp2, des2, backup_matcher)
                        tx, ty, s, a = affine2prm(h)

                        if not valid_transform(tx, ty, s, a):
                            # Both Transformations failed. skipping that image.
                            i_all += 1
                            t.set_postfix(tx=round(tx, 3), ty=round(ty, 3), s=round(s, 3), a=round(a, 3),
                                          status="Fail. skipping that image.")
                            continue

                    t.set_postfix(tx=round(tx, 3), ty=round(ty, 3), s=round(s, 3), a=round(a, 3), status="pass")
                    t.set_postfix(T=f"[tx={round(tx, 3)}, ty={round(ty, 3)}, s={round(s, 3)}, a={round(a, 3)}]", status="pass")
                    # date and time the image was created
                    created_time = os.path.getctime(img_ref_file)
                    date_captured = time.ctime(created_time)
                    date_obj = time.strptime(date_captured)
                    date_captured = time.strftime("%d/%m/%Y , %H:%M", date_obj)
                    # create img dict object:
                    img_dict = {"id": i_img,
                                "width": cols,
                                "height": rows,
                                "file_name": images_names[i],
                                "license": 0,
                                "date_captured": date_captured}
                    new_images.append(img_dict)

                    # Perform transformation for all pre-define annotations to that image:

                    img_anno = imgc.copy()
                    # loading image to show object
                    show.load_im(img_anno)
                    for anno in self.coco.annotations:  # self.data['annotations']:

                        if anno.image_id != coco_img.id:  # take only the annotations if that image
                            continue

                        # PreProcessing- reshaping annotation obj for matrix multiplication:
                        #                                                            |x1,x2,...xN|
                        # from A=[[x1,y1,x2,y2,...,xN,yN]], shape=(1,2N)   to    A = |y1,y2,...yN|, shape(3,N)
                        #                                                            |1,1,.....,1|
                        # then perform  A' = H*A
                        # where:
                        # N=number of points in annotation,
                        # A [3xN] - annotation points matrix
                        # H [3x3] - transformation matrix
                        # A'[3xN] - transformed annotation points matrix
                        segmentation = anno.segmentation  # [[x,y,x,y,...,x,y]], shape=(1,2N), N=number of points
                        pts_np = np.ones((3, int(len(segmentation[0]) / 2)), dtype=int)  # to shape=(3,N)
                        pts_np[0:2, :] = np.asarray(segmentation).reshape((-1, 2)).transpose()
                        seg_np = np.matmul(h, pts_np).astype(int)  # Apply the transformation on annotation

                        # Store new annotation's info:
                        segmentation = seg_np.transpose().reshape((1, -1)).tolist()
                        x0, y0 = int(min(seg_np[0, :])), int(min(seg_np[1, :]))  # Convert Xs to int -> [pixel domain]
                        x1, y1 = int(max(seg_np[0, :])), int(max(seg_np[1, :]))  # Convert Ys to int -> [pixel domain]

                        # Calculate transformed annotation's new information (area, bbox):
                        area = (x1 - x0) * (y1 - y0)
                        bbox = [x0, y0, x1 - x0, y1 - y0]  # anno['bbox']
                        anno_dict = {
                            "segmentation": segmentation,
                            "area": area,
                            "bbox": bbox,
                            "iscrowd": 0,
                            "id": i_anno,
                            "image_id": i_img,
                            "category_id": anno.category_id}
                        new_annotations.append(anno_dict)
                        i_anno += 1

                        # Draw transformed polygon on the image:
                        segmentation = np.array(segmentation, dtype=np.int32).reshape((-1, 1, 2))
                        # adding annotations to image
                        show.anno_maker(
                            poly_points=segmentation,
                            rec_points=((x0 - 10, y0 - 10), (x1 + 10, y1 + 10)),
                            texts=[self.coco.categories[anno.category_id - 1].name,
                                   self.coco.categories[anno.category_id - 1].status],
                            text_loc=((bbox[0], bbox[1]), (50, 50))
                        )
                        """
                        img_anno = cv.polylines(img_anno, [segmentation], isClosed=True, color=(255, 255, 0), thickness=3)
                        img_anno = cv.rectangle(img_anno, (x0 - 10, y0 - 10), (x1 + 10, y1 + 10), (255, 0, 255),
                                                thickness=5)
                        txt_cat: str = self.coco.categories[anno.category_id - 1].name
                        txt_staus: str = self.coco.categories[anno.category_id - 1].status
                        img_anno = cv.putText(img_anno,
                                              f"{txt_cat}-{txt_staus}",
                                              (bbox[0], bbox[1]),
                                              cv.FONT_HERSHEY_SIMPLEX, 1,
                                              (0, 255, 255), 2, cv.LINE_AA)
                    # Show result image:
                    img_anno = cv.putText(img_anno, f"[Aligned] {images_names[i]}", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv.imshow("Annotations", img_anno)
                    cv.waitKey(1)
                    # Cancel operation if run is terminated (the exit x sign was pressed)
                    if not cv.getWindowProperty("Annotations", cv.WND_PROP_VISIBLE):
                        print("Operation Cancelled")
                        break
                    """

                    # Show image with annotations on cv platform
                    # if window was terminated stop running on new images
                    if not show.show_image():
                        break
                    utilities.copy_file(file_dir=img_file, destination_folder=os.path.join(self.train_data_dir, "images"))
                    i_all += 1
                    i_img += 1


            self.data['images'] += new_images
            self.data['annotations'] += new_annotations


        cv.destroyAllWindows()
        new_json_file = os.path.join(self.train_data_dir, self.coco_file_dir.split('\\')[-1].split('.')[0] + "Full.json")
        with open(new_json_file, 'w') as f:
            json.dump(self.data, f, indent=4)

        print(colored("Auto Annotation Summery:", 'blue'), f"{len(self.data['images'])} images, "
                                                           f"{len(self.data['annotations'])} annotations,")

        return self.data


if __name__ == "__main__":

    # location of coco.json file and 'images' folder
    # folder = r'C:\BM\Data\AutoAnnotation'
    # folder = r'C:\BM\Data\ITRenew\STN2'
    folder = r'C:\BM\Data\AugTest'  # location of coco.json file and 'images' folder

    aa = AutoAnnotator()
    aa.init(folder)
    aa.annotate()


