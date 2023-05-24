import random
import os.path
from termcolor import colored
import PySimpleGUI as sg
from AugmentationAssistant.utiles.Transformations import *
from COCO import *
import PIL.Image
import io
import numpy as np
import cv2
import os
import PIL.ImageGrab


class Augment:

    def __init__(self, folder_path: str, new_path: str,
                 multi_transform: bool, trans_amount: int, wanted_transformations: dict):
        self.folder_path = folder_path
        self.new_path = new_path
        self.transform = Transform()
        # TODO - delete multi
        self.multi_transform = multi_transform  # Bool that holds if to make random transforms on transforms
        self.trans_amount = trans_amount  # Will hold the number of images to augment for every image if 'multi_transform' == Ture
        # else number of images to augment for every transformation for every image
        # Transformation inputs:
        # "wanted_transformations" will hold all the transformation we want to make on the images and
        # the settings for them, the format is a list [transformations function name, first settings input,
        # second settings input,..]
        self.wanted_transformations = wanted_transformations
        self.im_amount = 0  # Number of input images


class AugmentationGui(Augment):
    mon_size: tuple

    def __init__(self):
        super().__init__('',
                         '', False,
                         1, {})
        # this is setting a custom them for Gui that will be used later in the window element of PySimpleGui
        sg.LOOK_AND_FEEL_TABLE['MyCreatedTheme'] = {'BACKGROUND': '#181c21',
                                                    'TEXT': '#FFFFFF',
                                                    'INPUT': '#181c21',
                                                    'TEXT_INPUT': '#FFFFFF',
                                                    'SCROLL': '# 99CC99',
                                                    'BUTTON': ('#FFFFFF', '#181c21'),
                                                    'PROGRESS': ('# 99CC99', '#FFFFFF'),
                                                    'BORDER': 1, 'SLIDER_DEPTH': 0,
                                                    'PROGRESS_DEPTH': 0, }
        self.window_gui = 'None'
        self.window_settings = 'None'
        # Transformation inputs:
        # "wanted_transformations" will hold all the transformation we want to make on the images and
        # the settings for them, the format is a list [transformations function name, first settings input,
        # second settings input,..]
        self.coco = [None, None]  # will hold Coco objects
        self.second_coco = False  # will hold if there is a second coco or not
        self.stop = False  # If Ture this indicates if not to run anymore transformations
        self.selected: list = []  # will hold all selected transformations

    def selected_trans(self, values, order: int):
        """
        this will send to transform a list of all the selected transformation and over there a list of info
        for all wanted transformation will be build
        :param values: all the values of gui
        :param order: number of total add transformation, including the deleted once's, they are only not being display
        :return:
        """
        for i in range(0, order):
            if values["transformations_" + str(i)] != '*Exception occurred*':
                self.selected.append(values["transformations_" + str(i)])
        # this will build a selected transformation list of all transformation in pipe
        self.transform.build_selected_trans(self.selected)

    def gui(self):  # This function is making th GUI for the project
        self.mon_size = PIL.ImageGrab.grab().size
        sg.theme('MyCreatedTheme')  # this is implementing the custom them that was build earlier
        # this will set the first dropdown element of selecting a transformation pipeline
        trans_l_col = self.transform.combo_list(0)
        # trans_r_col holds the add and remove transformation buttons
        trans_r_col = [[sg.Button("Add transformation", size=(15, 2), button_color='#181c21', key='-add trans-')],
                       [sg.Button("remove last transformation", size=(15, 2), button_color='#181c21', key='-del trans-'),
                        sg.VPush(),
                        ]]

        coco_f = [[sg.Text("load COCO"), sg.Input(key='-coco dir-'),
                   sg.FileBrowse('Browse', file_types=(("ALL Files", "json"),),  # read only 'json' files
                                 button_color='#181c21'),
                   sg.Button(' Import ', font=2, key='-IMPORT-', button_color='#181c21')],
                  [sg.Checkbox('Second coco', key='Second coco')],
                  # visible only when 'Second coco' is pressed
                  [sg.Text("load COCO", visible=False, key="-load COCO 2-"),
                   sg.Input(key='-coco dir 2-', visible=False),
                   sg.FileBrowse('Browse', file_types=(("ALL Files", "json"),),  # read only 'json' files
                                 button_color='#181c21', key="-Browse 2-", visible=False),
                   sg.Button(' Import ', font=2, key='-IMPORT 2-', button_color='#181c21', visible=False)]
                  ]

        trans_f = [[
            sg.Col(layout=trans_l_col, background_color='#181c21', key='-trans col-'),  # this is holding the trans pipe
            sg.Col(layout=trans_r_col, background_color='#181c21', key='-add col-')
        ]]
        layout = [[sg.Frame('Select COCO', layout=coco_f, background_color='#181c21',
                            key='-trans frame-')],
                  [sg.Frame('Select transformation pipeline', layout=trans_f, background_color='#181c21',
                            key='-trans frame-')
                   ],
                  [
                      sg.Radio('Make transformations in the selected order', "RADIO1", key='order',
                               enable_events=True, default=True),
                      sg.Radio('Make transformations in a random order', "RADIO1", key='random',
                               enable_events=True),
                  ],
                  [
                      sg.Radio('Make single transformation (all the transformations but one at a time)',
                               "RADIO1", key='single', enable_events=True)
                  ],
                  [sg.Text("Number of images to augment for every image", visible=True,
                           background_color='#181c21'),
                   sg.In(size=(5, 1), enable_events=False, key='-num of images-', default_text='1',
                         visible=True, background_color='#181c21', text_color='white')],  # Number of images input
                  [sg.Push(background_color='#181c21'), sg.Button("Next", size=(10, 5), button_color='#181c21')]]
        self.window_gui = sg.Window("Bright Machines Augmentation tool", layout, background_color='#181c21',
                                    icon='bm_logo.ico', resizable=False)
        print(colored('Showing GUI', color='blue'))
        cancel: bool = False  # To know if to open settings window (if false run function if true the app shut down)
        pipe_counter: int = 1
        deleted_trans: int = 0  # number of deleted transformations, need to hold because they are not really deleted
        coco_load: list = [False] * 2  # indicates if a coco file was loaded

        while True:
            event, values = self.window_gui.read(timeout=100)
            if event == sg.WIN_CLOSED:
                break
            # else:
            #     # This function is in charge of showing the right information to the user by updating the gui screen
            #     self.visible_checkbox(values)
            elif event == '-add trans-':
                #if pipe_counter in deleted_trans:
                #    self.window_gui.extend_layout(self.window_gui['-trans col-'],
                #                                  self.transform.combo_list(pipe_counter))
                self.window_gui.extend_layout(self.window_gui['-trans col-'], self.transform.combo_list(pipe_counter))
                # this is made to keep the add and delete buttons at the same place
                if pipe_counter - deleted_trans > 2:
                    self.window_gui.extend_layout(self.window_gui['-add col-'], [[sg.Text('', size=(2, 1))]])
                self.window_gui.visibility_changed()
                pipe_counter += 1
                print(colored('transformation nuber {} was added'.format(pipe_counter - deleted_trans), color='green'))
            elif event == '-del trans-':
                if pipe_counter > 1:
                    # lets make the last visible transform not visible
                    # self.window_gui['transformations_' + str(pipe_counter - 1)].update(visible=False)
                    # self.window_gui['transformations_' + str(pipe_counter - 1)].widget.destroy()
                    # deleting the last trans from GUI but the memory is still taken
                    self.window_gui['-trans col-'].Widget.winfo_children()[-1].destroy()
                    deleted_trans += 1  # adding deleted trans counters
                    if pipe_counter - deleted_trans > 3:
                        self.window_gui['-add col-'].Widget.winfo_children()[-1].destroy()  # deleting the last ''
                    # self.window_gui['-add col-'].Widget.update()
                    # self.window_gui['-trans frame-'].Widget.update()
                    # self.window_gui.Refresh()
                    # self.window_gui['-trans frame-'].contents_changed()
                    # pipe_counter -= 1
                    print(colored('transformation nuber {} was deleted'.format(pipe_counter - deleted_trans + 1),
                                  color='red'))
            elif event == '-IMPORT-':  # importing coco file to make the augmentations on
                coco_dir = values['-coco dir-']
                if coco_dir != '':
                    # creating list of Image objects that hold all info from coco in a convenient way
                    self.coco[0] = Coco(coco_dir)
                    if self.coco[0].read_coco():  # the file was read in a success
                        coco_load[0] = True
                        sg.popup('Loading is done')
                else:
                    sg.PopupError('Please enter a coco file before pressing the Import button')

            elif event == '-IMPORT 2-':  # importing coco file to make the augmentations on
                coco_dir = values['-coco dir 2-']
                if coco_dir != '':
                    # creating list of Image objects that hold all info from coco in a convenient way
                    self.coco[1] = Coco(coco_dir)
                    if self.coco[1].read_coco():  # the file was read in a success
                        coco_load[1] = True
                        sg.popup('Loading is done')
                else:
                    sg.PopupError('Please enter a coco file before pressing the Import button')

            # change the visibility of the second coco import
            elif self.second_coco != values['Second coco']:
                self.window_gui["-load COCO 2-"].update(visible=values['Second coco'])
                self.window_gui['-coco dir 2-'].update(visible=values['Second coco'])
                self.window_gui['-Browse 2-'].update(visible=values['Second coco'])
                self.window_gui['-IMPORT 2-'].update(visible=values['Second coco'])
                self.second_coco = values['Second coco']
            elif event == 'Next':
                if values['order']:
                    self.transform.pipe_type = 'order'
                elif values['random']:
                    self.transform.pipe_type = 'random'
                elif values['single']:
                    self.transform.pipe_type = 'single'
                try:  # check if integer
                    self.trans_amount = int(values['-num of images-'])
                except():
                    sg.popup_error('Error message',
                                   'Pleas enter a number in "Number of images..')  # Shows red error button
                else:
                    if self.trans_amount > 0:  # must be bigger than 0
                        # check inputs
                        if not coco_load[0] or (self.second_coco and not coco_load[1]):
                            sg.popup_error('Please load a coco file to continue')
                        else:
                            self.selected_trans(values, pipe_counter)
                            self.window_gui.Hide()  # Hide the gui window but the object is still alive
                            # save input values
                            self.im_amount = values['-num of images-']
                            cancel = self.settings_window()  # All the settings related stuff
                            if cancel:  # the exit button was pressed
                                break
                            self.window_gui.UnHide()  # Show gui window
                            print(colored("Back to select transformation GUI", color='blue'))
                            # reset some values
                            self.transform.reset()
        return True

    def run_augmentation(self, values):
        """
        after this augmentation is over
        :param values: all the values from UI
        :return:
        """
        for i, coco in enumerate(self.coco):
            # if there is nor second coco then let's end run
            if i == 1 and not self.second_coco:
                break
            self.transform.load_selected_parameters(values)
            # loop on all images
            im_amount: int = len(coco.images)
            # Number of output images including original images
            self.transform.final_augmented_amount = im_amount * (self.trans_amount + 1)
            transform: list = self.transform.trans_list()
            new_folder_path = os.path.dirname(coco.original_coco_dir) + '/' + \
                              os.path.basename(coco.original_coco_dir).split('_')[0] + \
                              " - Original + Augmented"
            # creating new path if it doesn't exist
            if not os.path.exists(new_folder_path):
                print(colored("new dir doesn't exist | writing new folder", color='red'))
                os.makedirs(new_folder_path)
            for im_num in range(0, im_amount):
                if self.stop:
                    break
                print(colored("image number: {}".format(im_num), color='yellow'))
                image_obj = coco.images[im_num]  # current images object
                # This function is making the transform to an incoming image and will save it
                if self.transform.albu_trans(image=image_obj,
                                             transform=transform,
                                             new_folder_path=new_folder_path,
                                             iterations=self.trans_amount,
                                             coco=coco,
                                             i=i + 1):
                    break
            coco.export_coco('Augmented')
            # need to restart counting
            self.transform.augmented_images = 0
            self.transform.augmented_anno = 1
        # State the augmentation is done
        sg.popup('Augmentation is done')

    #  This functions in creating the settings window for the wanted transformations,
    #  activation_list contains True for wanted transformations and False other ways
    def settings_window(self):
        print(colored('moving to settings window', color='blue'))
        sg.theme('MyCreatedTheme')  # this is implementing the custom them that was build earlier
        settings_width: int = 1100  # width of settings in settings window
        # TODO: Check if cam do nicer
        # Dir to one image' need for display of patch
        # im_dis_patch = [os.path.join(self.folder_path, f) for f in os.listdir(self.folder_path) if
        #                  os.path.isfile(os.path.join(self.folder_path, f))][0]
        # For patch settings # TODO - make size relative to screen size
        canvas_size = (585, 390)  # The size of the canvas object as well as im shower

        right_col, height = self.transform.transform_settings_list(settings_width)
        right_col.append([sg.Push(),
                          sg.Button("Back", size=(10, 5)),
                          sg.Button("Run", size=(10, 5))])
        # enabling scrollable when to much augmentation setting
        col_height = min(round(self.mon_size[1] / 1.5), height)
        scroll = False if col_height == height else True
        layout = [[sg.Col(right_col, scrollable=scroll, size=((settings_width + 30,
                                                               col_height)))]]
        self.window_settings = sg.Window('Settings for transformations', layout, finalize=True, resizable=True,
                                         background_color='#181c21', icon='bm_logo.ico', grab_anywhere=True,
                                         scaling=True)

        while True:
            event, values = self.window_settings.read()
            if event == sg.WIN_CLOSED:
                return True
            elif event == "Back":
                self.window_settings.close()
                return False
            elif event == "Run":
                self.run_augmentation(values)
                # save augmentation parameters to json
                file_dir = open(r'{}\{}.json'.format(os.path.dirname(self.coco[0].original_coco_dir), "Augmentation parameters"), 'w')
                parameters = {
                    "Number of Augmentations per image": self.trans_amount,
                    "Transformation pipeline": self.transform.pipe_type,
                    "Selected transformations": self.selected,
                    "Selected Parameters": values
                }
                json.dump(parameters, file_dir, indent=4)
                file_dir.close()
                self.window_settings.close()
                return False

    def rec_draw(self):
        pass

    def convert_to_bytes(self, file_dir: str, resize: tuple, crop: tuple = None):
        """
        Will convert into bytes and optionally resize an image that is a file or a base64 bytes object.
        Turns into  PNG format in the process so that can be displayed by tkinter
        :param crop:
        :param file_dir: either a string filename or a bytes base64 image object
        :type file_dir:  (Union[str, bytes])
        :param resize:  optional new size
        :type resize: (Tuple[int, int] or None)
        :return: (bytes) a byte-string object
        :rtype: (bytes)
        """
        img = PIL.Image.open(file_dir)
        if crop is not None:  # If cropping is needed
            img = img.crop(crop)
        cur_width, cur_height = img.size  # Incoming im size
        new_width, new_height = resize  # Size of graph
        scale: float = min(new_height / cur_height, new_width / cur_width)  # Scale needed for perfect fit
        img = img.resize((int(cur_width * scale), int(cur_height * scale)), PIL.Image.LANCZOS)  # Scaling im
        bio = io.BytesIO()
        img.save(bio, format="PNG")
        del img
        return bio.getvalue(), scale


def run_aug():
    augmentation = AugmentationGui()
    augmentation.gui()  # This function is making th GUI for the project


def test():
    pass


if __name__ == "__main__":
    Transformations()
    run_aug()
    # test()
