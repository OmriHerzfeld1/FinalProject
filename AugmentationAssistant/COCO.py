import json
import os
import PySimpleGUI as sg
import datetime
from termcolor import colored


class Anno:
    def __init__(self, segmentation: list, area: int, bbox: list, iscrowd: str, id: int,
                 image_id: int, category_id: int):
        """
        annotation object, holds all info relevant for the annotation of every image
        """
        self.segmentation = segmentation
        self.area = area
        self.bbox = bbox
        self.iscrowd = iscrowd
        self.id = id
        self.image_id = image_id
        self.category_id = category_id


class Image:
    def __init__(self, id: int,  path: str, name: str, width: int, height: int, license: int, date_captured: str):
        self.id = id
        self.width = width
        self.height = height
        self.path = path
        self.name = name  # hold's the name of the file with no dir name
        self.license = license
        self.date_captured = date_captured
        self.anno = []  # list of anno objects

    def add_anno(self, segmentation: list, area: int, bbox: list, iscrowd: str, id: int,
                 image_id: int, category_id : int):
        """
        adding annotation objects to a list in Image object
        :param segmentation:
        :param area:
        :param bbox:
        :param iscrowd:
        :param id:
        :param image_id:
        :param category_id:
        :return:
        """
        self.anno.append(Anno(segmentation, area, bbox, iscrowd, id, image_id, category_id))


class Coco:
    def __init__(self, coco_dir: str):
        self.original_coco_dir: str = coco_dir
        self.coco_info: dict = {}  # all general info of coco file
        self.images: list = []  # list of image objects
        self.licenses: list = []
        self.categories: list = []
        self.output_images: list = []  # will hold image objects after augmentation (including original)

    def read_coco(self, im_folder_name: str = None):
        # TODO - path is image file name
        """
        this will load a coco file into a list of Image object
        :param im_folder_name: hold the name of the folder that the images are stored in, needed only if name is not
        "images"
        :return: if the file was loaded properly than return True, else False.
        """
        file = open(self.original_coco_dir, 'r')  # reading file
        coco_dict: dict = json.load(file)  # converting file to dict
        self.coco_info = coco_dict["info"]  # all general info of coco file
        folder = im_folder_name if im_folder_name else "images"
        for im in coco_dict["images"]:
            try:  # this will check if the format is ok
                self.images.append(Image(id=im['id'],
                                         width=im['width'],
                                         height=im['height'],
                                         path="{}/{}/{}".format(os.path.dirname(self.original_coco_dir),
                                                                folder, im['file_name']),
                                         name=os.path.basename(im['file_name']),
                                         license=im['license'],
                                         date_captured=im['date_captured']
                                         ))
            except():
                sg.popup_error("COCO format is not correct, Pleas check COCO file")
                return False
        for anno in coco_dict["annotations"]:
            try:  # this will check if the format is ok
                corresponding_im = anno["image_id"]
                self.images[corresponding_im - 1].add_anno(segmentation=anno["segmentation"],
                                                           area=anno["area"],
                                                           bbox=anno["bbox"],
                                                           iscrowd=anno["iscrowd"],
                                                           id=anno["id"],
                                                           image_id=anno["image_id"],
                                                           category_id=anno["category_id"])
            except():
                sg.popup_error("COCO format is not correct, Pleas check COCO file")
                return False
        try:  # this will check if the format is ok
            self.licenses: list = coco_dict["licenses"]
            self.categories: list = coco_dict["categories"]
        except():
            sg.popup_error("COCO format is not correct, Pleas check COCO file")
            return False
        return True  # if all is good

    def export_coco(self, final: str):
        """
        write JSON coco file
        :type final: add string to the end of the file name
        :return:
        """
        time = datetime.datetime.now()
        self.coco_info["date_created"] = "%s/%s/%s , %s:%s" % (time.day, time.month,
                                                               time.year, time.hour, time.minute)
        images_list = []  # will hold the relevant info of all images for coco file, list of dicts
        anno_list = []  # will hold the relevant info of all annotations for coco file, list of dicts
        for im in self.output_images:  # running on all images
            cur_image = {'id': im.id,
                         'width': im.width,
                         'height': im.height,
                         'file_name': im.name,
                         'license': im.license,
                         'date_captured': im.date_captured
                         }  # creating mini dict for every images
            images_list.append(cur_image)  # adding mini dict to list of images info
            for anno in im.anno:  # running on all images
                cur_anno = {'segmentation': anno.segmentation,
                            "area": anno.area,
                            'bbox': anno.bbox,
                            'iscrowd': anno.iscrowd,
                            'id': anno.id,
                            'image_id': anno.image_id,
                            'category_id': anno.category_id,
                            }  # mini dict for current annotation
                anno_list.append(cur_anno)
        coco_dict = {
            'info': self.coco_info,
            'images': images_list,
            'annotations': anno_list,
            'licenses': self.licenses,
            'categories': self.categories
        }
        coco_name = os.path.basename(self.original_coco_dir).split(".")[0] + 'final'
        out_file = open(r'{}\{}.json'.format(os.path.dirname(self.original_coco_dir), coco_name), 'w')
        json.dump(coco_dict, out_file, indent=4)
        out_file.close()
        print(colored("{} was saved in the following dir - ".format(coco_name), "green"),
              colored(os.path.dirname(self.original_coco_dir), "yellow"))
