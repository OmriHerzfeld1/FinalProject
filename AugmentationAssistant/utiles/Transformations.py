import random
import cv2
from matplotlib import pyplot as plt
import skimage
import albumentations as A
from AugmentationAssistant.COCO import *
from AugmentationAssistant.utiles.Convert import *
import time


def visualize(image_og, image_tr):
    fig = plt.figure(figsize=(10, 7))
    fig.add_subplot(2, 2, 1)
    plt.imshow(image_og)
    plt.axis('off')
    plt.title("First")
    fig.add_subplot(2, 2, 2)
    plt.imshow(image_tr)
    plt.axis('off')
    plt.title("Second")
    plt.show()


class Transformation:

    def __init__(self, name: str, trans: dict):
        self.name: str = name
        self.present_name: str = trans['name']
        self.parameters: list = []  # list that will hold all parameter elements
        self.build_parameters(trans["parameters"])
        # float number that will hold the wanted probability for the transformation to be applied every iteration
        self.probability: float = 0.0

    def build_parameters(self, parameters: list):
        """
        adding parameters elements to parameters list
        :param parameters: list of dictionary that holds parameters settings
        :return: nothing
        """
        for para in parameters:
            self.parameters.append(Parameter(para))


class Parameter:
    """
    creating parameter elements to be stored in parameter list as a part of "Transformation" element
    """

    def __init__(self, parameter: dict):
        self.type = parameter["type"]
        self.param_name = parameter["param_name"]
        self.present_name = parameter["present_name"]
        self.defaults = None
        self.limits_list = None
        self.options = None
        self.selected = None  # the selected value for the parameter
        if parameter["type"] == "slider":
            self.slider(parameter)
        elif parameter["type"] == "double_slider":
            self.double_slider(parameter)
        elif parameter["type"] == "combo":
            self.combo(parameter)
        elif parameter["type"] == "no_parameters":
            self.no_parameters(parameter)

    def slider(self, parameter):
        self.defaults: float = parameter["defaults"]
        self.limits_list: list = parameter["limits_list"]

    def double_slider(self, parameter):
        self.defaults: list = parameter["defaults"]
        self.limits_list: list = parameter["limits_list"]

    def combo(self, parameter):
        self.options: dict = parameter["options"]

    def no_parameters(self, parameter):
        pass


class Transform:

    def __init__(self):
        file = open('AugmentationAssistant\\utiles\\ConfigAugmentation.json', 'r')
        # dictionary of the transformations
        # This dic holds all the transformations available to make
        # In every transformation we have a list that holds the name of the transformation function and the settings
        self.transform: dict = json.load(file)
        self.trans_names: dict = {}  # connects presented name of transformation to the name itself
        self.names()  # makes the dict from line up
        # this will hold the wanted transformations to display in Transformation elements
        # holds all selected transformation for presentation Transformation element
        self.selected_transformations: list = []
        self.pipe_type: str = 'order'  # hold pipe order type
        self.transform_pipe = None  # will hold the "A.Compose" for transformation pipeline
        self.augmented_images: int = 0  # number of images that where augmented
        self.augmented_anno: int = 1  # number of current total annotations that had been made
        self.final_augmented_amount: int = 0  # number of images that is expected in the end

        self.total_t = 0

    def reset(self):
        """
        reset all parameters to only the init setting so a new augmentation processes can happen
        :return:
        """
        self.selected_transformations = []
        self.transform_pipe = None
        self.augmented_images = 0
        self.augmented_anno = 1
        self.final_augmented_amount = 0

    def names(self):
        """
        connects presented name of transformation to the name itself
        :return: nothing
        """
        for trans in list(self.transform.keys()):
            self.trans_names[self.transform[trans]["name"]] = trans

    def load_selected_parameters(self, values):
        """
        this will put selected parameters settings for every setting parameter and probability
        :param values: values dict form settings gui
        :return:
        """
        for trans in self.selected_transformations:
            trans.probability = float(values[trans.name + ' prob'])
            for para in trans.parameters:
                if para.type == 'double_slider':
                    para.selected = (values[para.present_name + ' min'], values[para.present_name + ' max'])
                elif para.selected == 'no_parameters':
                    pass
                elif para.type == 'slider':
                    para.selected = values[para.present_name]
                elif para.type == 'combo':
                    para.selected = para.options[values[para.present_name]]
        print(colored('Loading parameters from UI was done', color='green'))

    def trans_list(self):
        """
        builds a list in albumentation formate that will hold all wanted transformations and parameters
        :return: list
        """
        transform: list = []  # this will hold all the info for a wanted transform_pipe
        for trans in self.selected_transformations:
            param_values: dict = {"p": trans.probability}  # will hold parameters as dict
            for para in trans.parameters:
                if para.selected:  # check if the parameter has needed settings
                    param_values[para.param_name] = para.selected
            print(param_values)
            transform.append(getattr(A, trans.name)(**param_values))  # getattr - convert string to object
        # transform_pipe for all future transformations, order is set but parameters can change by settings entered
        # to transform list
        self.transform_pipe = A.Compose(
            transforms=transform,
            bbox_params=A.BboxParams(format="coco",
                                     label_fields=["bbox_classes"],
                                     ),
            keypoint_params=A.KeypointParams(
                format('xy')
            )
        )  # transform_pipe for all future transformations
        return transform

    def albu_trans(self, image, transform: list, new_folder_path: str,
                   iterations: int, coco, i: int):
        """
        will create and save augmented images
        :param coco: COCO Object
        :param i: number of coco file
        :param image: image object to transform
        :param transform: needed for random or single order pipeline
        :param iterations: how many synthetic images do we want
        :param new_folder_path: new folder dir
        :return:  True if the program was stopped
        """
        # raed image from the folder
        # print(image.path)
        image_to_transform = cv2.imread(image.path)
        print(image.path)
        # saving original image in the new folder #
        # we don't want to lose data if it's a jpg format, losing data when saving in jpg format
        ending = 'png' if image.name.split(".")[-1].lower() == 'jpg' else image.path[-3:].lower()
        new_file_path = '%s/%s.%s' % (new_folder_path, image.name.split(".")[0], ending)  # losing the jpg ending and adding png
        # write original image
        cv2.imwrite(new_file_path, image_to_transform)
        og_image = Image(id=self.augmented_images + 1,  # append original image
                         width=image.width,
                         height=image.height,
                         path=new_file_path,
                         name=os.path.basename(new_file_path),
                         license=image.license,
                         date_captured=image.date_captured)
        self.augmented_images += 1  # adding image to counter
        coco.output_images.append(og_image)
        # adding original annotations for matching original image
        for anno in image.anno:
            og_image.add_anno(
                segmentation=anno.segmentation,
                area=anno.area,
                bbox=anno.bbox,
                iscrowd=anno.iscrowd,
                id=self.augmented_anno,
                image_id=self.augmented_images,
                category_id=anno.category_id
            )
            self.augmented_anno += 1
        # this is the format in Albumentation
        image_to_transform = cv2.cvtColor(image_to_transform, cv2.COLOR_BGR2RGB)
        im_name = image.name.split(".")[0]  # image name with no ending.. ".png"
        for counter in range(1, iterations + 1):
            if self.pipe_type == 'random':  # for every iteration make a new random order pipeline
                self.transform_pipe = A.Compose(
                    transforms=random.sample(transform, len(transform)),
                    bbox_params=A.BboxParams(
                        format="coco",
                        label_fields=["bbox_classes"]),
                    keypoint_params=A.KeypointParams(
                        format('xy'))
                )
            if self.pipe_type == 'single':
                self.transform_pipe = A.Compose(
                    transforms=[A.OneOf(random.sample(transform, len(transform)))],
                    bbox_params=A.BboxParams(
                        format="coco",
                        label_fields=["bbox_classes"]),
                    keypoint_params=A.KeypointParams(
                        format('xy'))
                )
            begin_time = time.time()
            bbox = [anno.bbox for anno in image.anno]  # create one list that contains all bbox lists
            bbox_classes = [''] * len(bbox)  # Name had no meaning
            keypoints: list = []  # format - [(x0,y0), (x1,y1), (x2,y2), ....]
            for anno in image.anno:  # combining all segmentations into one list of points keypoints format
                keypoints = keypoints + seg_to_key(anno.segmentation[0])  # covert to keypoints format
            transformed = self.transform_pipe(
                image=image_to_transform,
                bboxes=bbox,
                bbox_classes=bbox_classes,
                keypoints=keypoints
            )
            new_file_path = '%s/%s_%s.%s' % (new_folder_path, im_name, counter, ending)
            # convert to regular format
            transformed_im = cv2.cvtColor(transformed["image"], cv2.COLOR_RGB2BGR)
            # write image to the disk
            cv2.imwrite(new_file_path, transformed_im)
            end_time = time.time()
            delta_t = end_time - begin_time
            self.total_t += delta_t
            now_time = datetime.datetime.now()
            cur_im = Image(id=self.augmented_images + 1,  # creating im object for augmented images
                           width=image.width,
                           height=image.height,
                           path=new_file_path,
                           name=os.path.basename(new_file_path),
                           license=image.license,
                           date_captured="%s/%s/%s , %s:%s" % (now_time.day, now_time.month,
                                                               now_time.year, now_time.hour, now_time.minute)
                           )
            index = 0  # indicate which bbox to put for each annotation
            seg_index = 0  # indicates where the last segmentation points ended
            for anno in image.anno:  # adding annotations to the Image object
                bbox = [round(i) for i in transformed["bboxes"][index]]  # need to round number
                relevant_keypoints = transformed["keypoints"][seg_index: seg_index + len(anno.segmentation[0]) // 2]
                seg_index += len(anno.segmentation[0]) // 2
                segmentation = [key_to_seg(relevant_keypoints)]  # convert keypoints format to segmentation
                cur_im.add_anno(segmentation=segmentation,
                                area=anno.area,
                                bbox=bbox,
                                iscrowd=anno.iscrowd,
                                id=self.augmented_anno,
                                image_id=self.augmented_images + 1,
                                category_id=anno.category_id)
                index += 1
                self.augmented_anno += 1
            coco.output_images.append(cur_im)  # adding Image object to list
            self.augmented_images += 1  # number of images that where augmented
            if not sg.one_line_progress_meter('Progress', self.augmented_images,
                                              self.final_augmented_amount,
                                              f'Coco file number {i} \nNumber of files done from total', key='-cancel-',
                                              orientation='horizontal',
                                              grab_anywhere=True,
                                              bar_color=('#F47264', '#FFFFFF')):  # Show progress bar to the user
                return True

    def combo_list(self, pipe_counter: int):
        """
        this will return a sg_combo element inside a [[]] and the element is setting a dropdown for all the different
        transformations
        this is use for dynamic amount of wanted transformations
        :param pipe_counter: the serial number of the element that is being prepared here
        :return: [[sg_combo element]]
        """
        transforms: list = [trans["name"] for trans in list(self.transform.values())]  # transformation list
        sg_combo = sg.Combo(transforms, size=(30, 7), key='transformations_' + str(pipe_counter),
                            visible=True, background_color='#181c21', default_value=transforms[0],
                            readonly=True)
        return [[sg_combo]]

    def build_selected_trans(self, transformations_names: list):
        """
        this will make Transformation element that will hold all wanted info to upload the wanted settings for
        transformations
        :param transformations_names: list of selected transformations in their presented names
        :return: nothing
        """
        print(colored("selected transformations are", color='green'),
              colored(str(transformations_names), color='yellow'))
        for trans in transformations_names:
            self.selected_transformations.append(Transformation(self.trans_names[trans],
                                                                self.transform[self.trans_names[trans]]))

    def transform_settings_list(self, settings_width):
        """
        build's the sttings window GUI
        :param settings_width: give an equal width value to every setting frame
        :return: the sg element, wanted height for the col int settings window
        """
        build: dict = {  # this will connect type of settings to the function
            'slider': self.build_slider,
            'double_slider': self.build_doubleslider,
            'combo': self.build_combo,
            'no_parameters': self.build_no_para
        }
        sg_list: list = []
        height: int = 0  # will hold the wanted height for the col int settings window
        for trans in self.selected_transformations:
            # running on all different settings, 0 is the name of the function in albumentations
            parameters: list = []
            for para in trans.parameters:
                # adding settings for list that will hold of parameters
                parameters = build[para.type](para, parameters)
            left_col = sg.Col(parameters)
            right_col = sg.Col([[sg.Push(), sg.Text('Probability', tooltip='Enter a float number 0.0 - 1.0'),
                                 sg.InputText(default_text='1.0', size=(3, 2), tooltip='Enter a float number 0.0 - 1.0',
                                              key=trans.name + ' prob', enable_events=False)],
                                [sg.VPush()]])
            sg_list.append([sg.Frame(trans.present_name, [[left_col, sg.Push(), right_col]],
                                     size=(settings_width, len(parameters) * 48))])
            height += len(parameters) * 48
        return sg_list, height + 120

    def build_slider(self, para, parameters):
        """
        build generic slider for the UI
        :param parameters: this hold's the parameters, and will get bigger till all parameters for UI is set
        :param para: the element of the wanted transformation to be added
        :return: list that hold the UI to define the parameter
        """
        name = para.present_name
        defaults = para.defaults
        limits = para.limits_list
        slider = [
            sg.Text(name, visible=True, background_color='#181c21'),
            sg.Slider(range=(limits[0], limits[1]), default_value=defaults, resolution=1,
                      orientation='horizontal',
                      font=('Helvetica', 12), size=(100, 7), key=name,
                      visible=True, background_color='#181c21', trough_color='white')
        ]
        return parameters + [slider]

    def build_doubleslider(self, para, parameters):
        """
        build generic double slider for the UI
        :param parameters: this hold's the parameters, and will get bigger till all parameters for UI is set
        :param para: the element of the wanted transformation to be added
        :return: list that hold the UI to define the parameter
        """
        name = para.present_name
        defaults = para.defaults
        limits = para.limits_list
        slider1 = [
            sg.Text(name + ' - minimum', visible=True, background_color='#181c21'),
            sg.Slider(range=(limits[0], limits[1]), default_value=defaults[0], resolution=1, orientation='horizontal',
                      font=('Helvetica', 12), size=(100, 7), key=name + ' min',
                      visible=True, background_color='#181c21', trough_color='white')
        ]
        slider2 = [
            sg.Text(name + ' - maximum', visible=True, background_color='#181c21'),
            sg.Slider(range=(limits[0], limits[1]), default_value=defaults[1], resolution=1, orientation='horizontal',
                      font=('Helvetica', 12), size=(100, 7), key=name + ' max',
                      visible=True, background_color='#181c21', trough_color='white')
        ]
        return parameters + [slider1, slider2]

    def build_combo(self, para, parameters):
        """
        build generic combo for the UI
        :param parameters: this hold's the parameters, and will get bigger till all parameters for UI is set
        :param para: the element of the wanted transformation to be added
        :return: list that hold the UI to define the parameter
        """
        name = para.present_name
        options = list(para.options.keys())
        combo = [
            sg.Text(name, visible=True, background_color='#181c21'),
            sg.Combo(options, size=(30, 7), key=name, visible=True,
                     background_color='#181c21', default_value=options[0],
                     button_arrow_color='#FFFFFF', text_color='#FFFFFF', button_background_color='#181c21')
        ]
        return parameters + [combo]

    def build_no_para(self, para, parameters):
        """
        build generic no parameters for the UI
        :param parameters: this hold's the parameters, and will get bigger till all parameters for UI is set
        :param para: the element of the wanted transformation to be added
        :return: list that hold the UI to define the parameter
        """
        name = para.present_name
        nothing = [
            sg.Text(name + ' - No parameters is needed', visible=True, background_color='#181c21')
        ]
        return parameters + [nothing]


class Transformations:

    @staticmethod
    def random_cropping(indata: list):
        """
        :param indata: [image, boundary_box: list [x, y, w, h],
        patch size: list [x, y]]
        :return: transformed image
        """
        # Top left pixel position
        #  boundary_box: list[int] = []  # [x,y,w,h]
        x = indata[1][0]
        y = indata[1][1]
        # size of boundary box
        w = indata[1][2]
        h = indata[1][3]
        size = indata[2]  # Size of crop
        crop_x = random.randint(x, x + w - size[0])
        crop_y = random.randint(y, y + h - size[1])
        cropped = indata[0][crop_y:crop_y + size[1], crop_x:crop_x + size[0]]  # x is y and y is x in this mathematics
        return cropped

