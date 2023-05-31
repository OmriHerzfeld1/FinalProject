from AugmentationAssistant.utiles.COCO import *
from utilities import ShowAnno
import cv2 as cv
from termcolor import colored



if __name__ == '__main__':
    coco_dir = "C:\\Users\\omri.herzfeld\\OneDrive - Bright Machines\\Pictures\\images for annotaion test\\TestTrainData\\TestFullAugmented.json"
    coco = Coco(coco_dir)
    if coco.read_coco(im_folder_name="Augmented"):
        print(colored("Coco file was loaded", color="green"))
    show = ShowAnno("Anno")
    for im in coco.images:
        im = cv.imread(im.path)
        show.load_im(im)


    print("done")
