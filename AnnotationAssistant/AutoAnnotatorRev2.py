import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
import time
import COCO
from AnnotationAssistant import BM_Annotation_Tool
import AugmentationAssistant.utiles.Convert as convert
import utilities
from colorama import Fore
from tqdm import tqdm
import math


class AutoAnnotator:
    """
    Will handle all the process of the Auto Annotation
    This will happen in the following way ..
    by reading a coco file containing the prime all ready annotated image
    assuming the wanted images for the AA is in the same folder.
    this class will write in the end a coco file with all the new images and the respected annotations

    """
    cur_area: int

    def __init__(self, coco_dir):
        self.MAX_FEATURES: int = 5000
        self.GOOD_MATCH_PERCENT: float = 0.2
        self.status: bool = True  # Indicates if the process running fine
        self.coco = COCO.Coco(coco_dir)
        self.og_im: np.array = np.array([])  # will hold original image
        self.og_im_obj = ''  # original image object
        self.cur_h: np.array = np.array([])  # transformation matrix
        self.cur_seg: list = []
        self.cur_area: int = 0
        self.cur_bbox: list = []

    def process(self, image_folder: str):
        """
        will handale the process of the AA
        :param image_folder: hold the name of the folder that the images are stored in
        :return:
        """
        self.coco.read_coco(image_folder)  # reading temp coco to ge needed information
        os.remove(self.coco.original_coco_dir)  # removing temp file, not needed anymore
        self.og_im_obj = self.coco.images[0]
        self.og_im = cv.imread(self.og_im_obj.path)
        status, full_status, im_name_l = self.create_im_list()
        if not status:
            print(Fore.RED + full_status)
            # cprint(full_status, color='red')
            return full_status
        print(Fore.GREEN + full_status)
        # cprint(full_status, color='green')
        im_name_l.remove(os.path.basename(self.og_im_obj.path))  # remove original image
        show_anno = utilities.ShowAnno(name="Auto Annotation")  # creating object for showing images with annotations
        anno_id: int = 1  # index start in 1
        # tqdm creates progress line on the run
        for i in tqdm(range(0, len(im_name_l)), desc='Annotation Progress'):  # run on all images in folder
            name = im_name_l[i]
            im_path = os.path.join(os.path.dirname(self.og_im_obj.path), name)
            cur_im: np.array = cv.imread(im_path)
            homag_ok = self.align_images(cur_im)  # find homography
            # date and time the image was created
            created_time = os.path.getctime(im_path)
            date_captured = time.ctime(created_time)
            date_obj = time.strptime(date_captured)
            date_captured = time.strftime("%d/%m/%Y , %H:%M", date_obj)
            # add image object to list of images
            self.coco.output_images.append(COCO.Image(
                id=i + 1,  # index start in 1
                width=cur_im.shape[1],
                height=cur_im.shape[0],
                path=im_path,
                name=im_path,  # name is path name so we could know location when importing image to gui
                # name=''.join([im_path.split('\\')[-2], '//', im_path.split('\\')[-1]]),  # folder and name of file
                license=0,
                date_captured=date_captured
            ))
            ## Showing annotations
            show_anno.load_im(cur_im)
            if homag_ok:  # homography make sens
                # run on all annotation of the image
                for j, anno in enumerate(self.og_im_obj.anno):
                    og_segmentation = convert.seg_to_key(anno.segmentation[0])  # format - [(x0,y0), (x1,y1), (x2,y2), ....]
                    og_bbox = anno.bbox
                    og_area = anno.area
                    self.cal_annotation(og_segmentation, og_area, og_bbox)  # calculate the new annotations
                    # add new annotation
                    self.coco.output_images[-1].add_anno(segmentation=[convert.key_to_seg(self.cur_seg)],  # format [x0,y0,x1..]
                                                         area=self.cur_area,
                                                         bbox=self.cur_bbox,
                                                         iscrowd=0,
                                                         id=anno_id,
                                                         image_id=i + 1,  # index start in 1
                                                         category_id=anno.category_id)
                    anno_id += 1
                    # show image on cv screen with its annotations
                    show_anno.anno_maker(poly_points=np.array(self.cur_seg),
                                         rec_points=((int(self.cur_bbox[0]), int(self.cur_bbox[1])),
                                                     (int(self.cur_bbox[0] + self.cur_bbox[2]), int(self.cur_bbox[1] +
                                                                                                    self.cur_bbox[3]))),
                                         texts=["[{}/{}  {} %]".format(i, len(im_name_l),
                                                                       round(i / len(im_name_l) * 100)),
                                                self.coco.categories[anno.category_id-1]["name"]],
                                         text_loc=np.array([[50, 50], [self.cur_bbox[0], self.cur_bbox[1]]])
                                         )
            else:
                print(Fore.RED + f"Failed to Auto Annotate image :'{self.coco.output_images[-1].name}'")
                show_anno.anno_maker(texts=["[Failed to Auto Annotate image {}/{}  {} %]".format(i, len(im_name_l),
                                                                   round(i / len(im_name_l) * 100))],
                                     text_loc=np.array([[50, 50]])
                                     )
            # if window was terminated stop running on new images
            if not show_anno.show_image():
                break
        self.coco.export_coco('')
        cv.destroyAllWindows()
        return 'All good'

    def cal_annotation(self, seg, area, bbox):
        a00, a01, a10, a11 = self.cur_h[0][0], self.cur_h[0][1], self.cur_h[1][0], self.cur_h[1][1]  # shear
        b0, b1 = self.cur_h[0][2], self.cur_h[1][2]  # shift
        circle = False
        if math.sqrt(abs(((seg[0][0] - seg[1][0]) ** 2 + (seg[0][1] - seg[1][1]) ** 2) - (
                (seg[2][0] - seg[3][0]) ** 2 + (seg[2][1] - seg[3][1]) ** 2))) < 6 and len(seg) > 4:  # check if Circle
            circle = True
        det = a00 * a11 - a01 * a10  # determination
        self.cur_area = int(area * det)  # cal new area with determinant
        self.cur_seg, x_points, y_points = self.cal_transform(a00, a01, a10, a11, b0, b1, seg, bbox, circle)
        left_top = (min(x_points), min(y_points))
        right_bot = (max(x_points), max(y_points))
        self.cur_bbox = [round(left_top[0]), round(left_top[1]),
                         round(right_bot[0] - left_top[0]), round(right_bot[1] - left_top[1])]

    def cal_transform(self, a00: float, a01: float, a10: float, a11: float, b0: float, b1: float, point_list: list,
                      bbox: list, c: bool):
        point_trans: list = []
        x_points: list = []
        y_points: list = []
        if c:  # check if circle
            x0_orig = int(bbox[0] + bbox[2] / 2)
            y0_orig = int(bbox[1] + bbox[3] / 2)
            radius = math.sqrt(self.cur_area / 4)  # area is area of the bbox
            # point of middle of circle
            point: tuple = round(x0_orig * a00 + y0_orig * a01 + b0), round(x0_orig * a10 + y0_orig * a11 + b1)
            for theta in np.linspace(0, 2 * np.pi, 40):
                point_trans.append((point[0] + int(radius * np.cos(theta)), point[1] + int(radius * np.sin(theta))))
                x_points.append(point[0] + int(radius * np.cos(theta)))
                y_points.append(point[1] + int(radius * np.sin(theta)))
        else:
            for point in point_list:
                point_trans.append((round(point[0] * a00 + point[1] * a01 + b0),
                                    round(point[0] * a10 + point[1] * a11 + b1)))
                x_points.append(point[0] * a00 + point[1] * a01 + b0)
                y_points.append(point[0] * a10 + point[1] * a11 + b1)
        return point_trans, x_points, y_points

    def align_images(self, im2: np.array) -> bool:
        # Detect ORB features and compute descriptors.
        orb = cv.ORB_create(self.MAX_FEATURES)
        keypoints1, descriptors1 = orb.detectAndCompute(self.og_im, None)
        keypoints2, descriptors2 = orb.detectAndCompute(im2, None)

        # Match features.
        matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        matches = list(matcher.match(descriptors1, descriptors2, None))

        # Sort matches by score
        matches.sort(key=lambda x: x.distance, reverse=False)

        # Remove not so good matches
        numGoodMatches = int(len(matches) * self.GOOD_MATCH_PERCENT)
        matches = matches[:numGoodMatches]

        # Extract location of good matches
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)

        for i, match in enumerate(matches):
            points1[i, :] = keypoints1[match.queryIdx].pt
            points2[i, :] = keypoints2[match.trainIdx].pt

        # Find homography using RANSAC to cut outliers
        ransac_reproj_threshold = 1
        self.cur_h, mask = cv.estimateAffine2D(points1, points2, cv.RANSAC, ransacReprojThreshold=ransac_reproj_threshold)
        a00, a01, a10, a11 = self.cur_h[0][0], self.cur_h[0][1], self.cur_h[1][0], self.cur_h[1][1]  # shear
        b0, b1 = self.cur_h[0][2], self.cur_h[1][2]  # shift
        # check if matrix make sense
        det = a00 * a11 - a01 * a10  # determination
        # a very big change in area of annotation which means problem with algo
        if not 0.9 < det < 1.1 or b0 > self.og_im.shape[0] * 0.1 or b0 > self.og_im.shape[1] * 0.1:
            return False
        return True

    def create_im_list(self):
        ok_file_types = utilities.okfiletypes()
        folder_dir = os.path.dirname(self.coco.images[0].path)  # folder of wanted images for annotations
        print(folder_dir)
        im_name_l = os.listdir(folder_dir)
        for dir in im_name_l:
            try:
                if os.path.basename(dir).split('.')[1] not in ok_file_types:
                    return False, 'Some files not in the right format exist in the folder', []
            except:
                return False, 'Only files are allowed in this folder', []
        return True, 'Image list was loaded ok', im_name_l


if __name__ == '__main__':
    pass
    # coco_dir = 'C://Users//omri.herzfeld//OneDrive - Bright Machines//Pictures//images for annotaion test//TestTrainData\\temp_coco.json'
    # AA = AutoAnnotator(coco_dir)
    # folder_name = 'images'
    # AA.process(folder_name)