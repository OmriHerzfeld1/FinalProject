import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as F
from main import *
from torchvision.io import read_image
import myutils
from time import sleep


class Model:
    """
    this class with help a user to run a trained model
    """
    def __init__(self, model_path: str, images_path: str):
        self.images_path: str = images_path
        self.MODEL_PATH = model_path
        self.mapping: dict = {}  # connects id to name of the label
        self.trans_parm: dict = {}  # connects id to name of the label
        self.general_info: dict = {}  # dict with all the general info of the net
        self.loading_parms()
        self.device = None  # the device the net is running on
        self.model = self.model_create()
        self.transform = self.transformed()  # the transformation pipe for incoming im

    def __call__(self, im: np.array):
        """
        this will pass the incoming image throw the net
        :param im: incoming image
        :return: a list with one dict, the dict fields are: "image", "boxes", "labels"
        """
        im_t = self.transform(image=im)
        im_t = im_t['image']  # image in tensor
        im_t_copy = im_t.to(self.device)
        # getting model outputs faster RCNN
        #outputs = self.model([im_t_copy])

        # for rcnn
        t = im_t_copy.unsqueeze(0).shape
        outputs = self.model(im_t_copy.unsqueeze(0))

        return outputs

    def transformed(self):
        """
        :return:
        """
        custom_transforms: list = []
        resize: list = self.trans_parm["Resize"]
        custom_transforms.append(A.Resize(resize[1], resize[0],
                         interpolation=cv.INTER_LANCZOS4))
        custom_transforms += [
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ToTensorV2(),
        ]
        return A.Compose(transforms=custom_transforms)

    def present_ouputs(self, outputs: dict, im_path: str, im_name: str):
        """
        here we can visualize the annotations that the identified on the incoming image
        :param outputs: a dict that holds all you outcomes of the net
        :param im_path: path to the incoming image
        :param im_name: name of the incoming image
        :return: nothing
        """
        bbox = outputs[0]["boxes"]
        labels_id = outputs[0]["labels"]
        labels_names = []  # holdes the label names in list of strings
        for label in labels_id:
            labels_names.append(self.mapping[str(label.item())])
        anno_im = torchvision.utils.draw_bounding_boxes(
            image=read_image(im_path).to(self.device, dtype=torch.uint8),  # convert to a torch tensor of type torch.uint8
            boxes=bbox,
            width=8,
            labels=labels_names,
            font_size=10,
            fill=True,
            colors=[(0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255),
                    (255, 0, 0), (0, 0, 255), (0, 255, 255)]
            )
        anno_im = anno_im.detach()
        anno_im = F.to_pil_image(anno_im)  # convert to PIL
        # anno_im = anno_im.numpy().reshape([2064, -1, 3])
        np_im = np.array(anno_im)[:, :, ::-1]  # converting PIL to np and RGB to BRG
        # shape = np.array(np_im.shape[:-1]) // 4  # downscale by factor of 4
        # show_im = cv.resize(np_im, shape)
        scale_percent = 40  # percent of original size
        width = int(np_im.shape[1] * scale_percent / 100)
        height = int(np_im.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        resized = cv.resize(np_im, dim, interpolation=cv.INTER_AREA)
        # Destroying All the windows
        cv.destroyAllWindows()
        cv.imshow(f'image name: {im_name}', resized)
        cv.waitKey(1000)
        return labels_names, bbox

    def loading_parms(self):
        """
        this function is called i the init stage off the class and will load needed data for the init stage of the class
        :return: nothing
        """
        mapping_path = os.path.join(self.MODEL_PATH, "mapping.json")
        trans_path = os.path.join(self.MODEL_PATH, "trans_parm.json")
        general_info = os.path.join(self.MODEL_PATH, "general_info.json")
        # Open the input file in read mode and use json.load() to load the data into a dictionary
        with open(mapping_path, "r") as input_file:
            self.mapping = json.load(input_file)  # connects id to name of the label
        # Importing the dict with the transformation parameters that is needed to preform trans pipe for an incoming image
        with open(trans_path, "r") as input_file:
            self.trans_parm = json.load(input_file)  # connects id to name of the label
        # Importing general info of the net
        with open(general_info, "r") as input_file:
            self.general_info = json.load(input_file)  # connects id to name of the label
        return

    def model_create(self):
        """
        here we are creating the model of the input model location,
        if GPU is available model will run on it
        :return: the model (CNN net)
        """
        model_path = os.path.join(self.MODEL_PATH, 'model.pth')
        num_classes = self.general_info["model"]["classes_num"]
        # for faster RCNN
        model = get_model_instance_segmentation(num_classes)
        # model.load_state_dict(torch.load(model_path))
        # model = torchvision.models.resnet50(pretrained=True)
        # model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        # select device (whether GPU or CPU)
        if torch.cuda.is_available():
            print(Fore.GREEN + 'CUDA is available calculation will be preformed on GPU')
            self.device = torch.device('cuda')
        else:
            print(Fore.RED + 'CUDA is  not available calculation will be preformed on CPU')
            self.device = torch.device('cpu')
        # moving model to device
        model.to(self.device)
        return model


class Accuracy:

    def __init__(self):
        # unique_labels: list = myutils.unique(labels)
        # create a dict this dict will have unique labels as keys and a list of iou as values
        self.IOU = {}
        # this will have as key image name, and in value tuple - (how many labels in image, how many was right)
        self.labels_accuracy = {
            'total': [0, 0]
        }
        self.iou_mean = {}
        self.iou_std = {}
        self.iou_tot_mean: float = 0.0


def iou(boxA, boxB):
    """
    This will evaluate the score of the prediction of the bbox relative to the ground truth
    :param boxA: ground truth
    :param boxB: predict of the model
    :return:  the score, higher is better, one is perfect (cant be done), above 0.5 is good
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


class TestingModel(Model):

    def __init__(self, model_path: str, images_path: str, test_coco: str):
        super().__init__(model_path, images_path, )
        self.coco = COCO(test_coco)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def test(self):
        # object that will hold the info about the accuracy of the model
        accuracy = Accuracy()
        with trange(len(self.ids)) as t:
            for i in t:
                image_id = i + 1  # image id's start from 1
                # dict that hold's all the info of the ground truth
                test_im_info: dict = self.coco.imgs[image_id]
                im_name: str = test_im_info["file_name"]
                im_path: str = os.path.join(self.images_path, im_name)
                im = cv.imread(im_path)  # image in np.array format
                outputs = self(im)
                # multiply by the factor of down sampling
                outputs[0]["boxes"] = outputs[0]["boxes"] * self.trans_parm["ScaleFactor"]
                # presenting the output on the input image
                label, bbox = self.present_ouputs(outputs, im_path, im_name)
                cur_im_iou: list = []  # needed to show iou in the tqdm print
                ground_labels: list = []  # needed to check labeling accuracy
                # calculation the accuracy of the model on the input image compare to the ground truth
                annotations: dict = self.coco.imgToAnns[image_id]  # all annotations of cur img
                for ground_anno in annotations:
                    # holds all the info of the annotation
                    ground_label: int = ground_anno["category_id"]  # the category of the ground truth annotation
                    ground_labels.append(ground_label)
                    ground_bbox: list = ground_anno["bbox"]
                    # let's check if the model outputs have this output
                    try:
                        # location of the category in the output tensor
                        location: int = torch.where(outputs[0]["labels"] == ground_label)[0].tolist()[0]
                        output_bbox: list = outputs[0]["boxes"][location].tolist()
                        # here we compute the accuracy
                        cur_iou: float = iou(myutils.coco_to_pascal_voc_bbox(ground_bbox), output_bbox)
                    except IndexError:
                        # in the case of not recognizing the category the iou will be the lowest as possible - 0
                        cur_iou: float = 0.0
                    finally:
                        # check if the label all ready exist in the iou dict
                        if self.mapping[str(ground_label)] in accuracy.IOU:
                            accuracy.IOU[self.mapping[str(ground_label)]].append(cur_iou)
                        else:
                            accuracy.IOU[self.mapping[str(ground_label)]] = [cur_iou]

                        cur_im_iou.append(cur_iou)

                output_labels: list = outputs[0]["labels"].tolist()
                common_labels: int = len(set(output_labels).intersection(ground_labels))
                accuracy.labels_accuracy[im_name]: tuple = (len(ground_labels), common_labels)
                accuracy.labels_accuracy['total'] = [accuracy.labels_accuracy['total'][0] + len(ground_labels),
                                                     accuracy.labels_accuracy['total'][1] + common_labels]

                # Description will be displayed on the left
                t.set_description(f'Image {image_id}')
                # Postfix will be displayed on the right,
                # formatted automatically based on argument's datatype
                t.set_postfix(iou=cur_im_iou, label=label)
                #sleep(0.1)
        # convert to np for future use
        for label in accuracy.IOU:
            accuracy.IOU[label] = np.array(accuracy.IOU[label], dtype=float)
            accuracy.iou_mean[label] = np.mean(accuracy.IOU[label])
            print(f"for category: {label} the mean iou was - {accuracy.iou_mean[label]}")
            accuracy.iou_std[label] = np.std(accuracy.IOU[label])
            print(f"for category: {label} the std iou was - {accuracy.iou_std[label]}")
            accuracy.iou_tot_mean += accuracy.iou_mean[label]
        accuracy.iou_tot_mean = accuracy.iou_tot_mean / len(accuracy.IOU)  # divide by the number of unique labels
        print(f"Overall correct labeling - {accuracy.labels_accuracy['total'][0]} / {accuracy.labels_accuracy['total'][1]}")
        return accuracy




if __name__ == '__main__':
    # SIM multi model
    # model_path: str = r'C:\Users\VisionTeam\Documents\Python Projects\faster-rcnn-master\SIM FULL 29.03'
    # images_path = r'C:\Users\VisionTeam\Pictures\Data For Deep testing\splitted - SIM_Multi_models\images'
    # test_coco = r'C:\Users\VisionTeam\Pictures\Data For Deep testing\splitted - SIM_Multi_models\Test_SIM_Multi_models.json'

    # classification WW
    model_path: str = r'C:\Users\omri.herzfeld\OneDrive - Bright Machines\Desktop\Final Project\faster-rcnn-main\SIM No Augmentation'
    images_path = r'C:\Users\omri.herzfeld\OneDrive - Bright Machines\Pictures\images for annotaion test\SIM\images'
    test_coco = r'C:\Users\omri.herzfeld\OneDrive - Bright Machines\Pictures\images for annotaion test\SIM\SIM_Test.json'

    model = TestingModel(model_path=model_path,
                         images_path=images_path,
                         test_coco=test_coco


    )
    # test
    # dir = r'C:\\Users\\VisionTeam\\Pictures\\Data For Deep testing\\WW ST3 Screw Val Data\\images\\ImageLB0015.bmp'
    # im = cv.imread(dir)
    # print(model(im))
    #
    accuracy_obj: Accuracy = model.test()
    myutils.save_dict_to_json(
        file_path=os.path.join(model_path, 'model_performance'),
        dict_obj={
            'Mean Error': accuracy_obj.iou_mean,
            'Standard Deviation': accuracy_obj.iou_std,
            'Corrected labeling': accuracy_obj.labels_accuracy
        }
    )


def test_tq():
    with trange(10) as t:
        for i in t:
            # Description will be displayed on the left
            t.set_description('Image %i' % i)
            # Postfix will be displayed on the right,
            # formatted automatically based on argument's datatype
            t.set_postfix(loss=i/2, gen=25, str='h',
                          lst=[1, 2])
            sleep(1)
