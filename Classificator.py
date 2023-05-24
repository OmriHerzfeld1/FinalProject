import PySimpleGUI as sg
import os
from PIL import Image as I
import datetime
import time
from COCO import Coco, Image, Anno
import utilities


def worm(dir_path: str, category: str, coco):
    """
    recursive flow that adds image objects to the coco object
    :param dir_path: the dir of the folder with all the data
    :param category: the name of the category until this run
    :param coco: the coco object
    :return: coco object
    """
    # get all folders in the dir_path
    folders = [f for f in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, f))]

    if folders:
        for f in folders:
            coco = worm(dir_path=os.path.join(dir_path, f), category=f'{category}_{f}' if category else f, coco=coco)
    # get all files in the dir_path
    files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
    if not files:  # if there are no images...
        return coco
    im_exist = False  # lets check if there is an im in folder
    for i in files:
        ext = i.split('.')[-1].lower()  # ending of file
        valid_types = ["bmp", "jpg", 'jpeg', 'png']
        if ext in valid_types:
            im_exist = True
            break
    if not im_exist:
        return coco
    # create a new category
    elif not category:
        sg.popup_error('There is an image file which is not in a folder!', background_color='#181c21')
        return False  # abort process
    coco.categories.append(
        {
            'supercategory': 'None',
            'id': len(coco.categories) + 1,
            'name': category,
            'status': 'Pass'
                   }
    )
    for f in files:
        ext = f.split('.')[-1].lower()  # ending of file
        valid_types = ["bmp", "jpg", 'jpeg', 'png']
        if ext not in valid_types:
            # raise TypeError("Invalid type of file")
            pass
        else:
            full_path = os.path.join(dir_path, f)
            im = I.open(full_path)
            width, height = im.size
            # date and time the image was created
            created_time = os.path.getctime(full_path)
            date_captured = time.ctime(created_time)
            date_obj = time.strptime(date_captured)
            date_captured = time.strftime("%d/%m/%Y , %H:%M", date_obj)
            # create Image object
            im_o = Image(
                id=len(coco.output_images) + 1,
                path=full_path,  # full path
                name=f,
                width=width,
                height=height,
                license=0,
                date_captured=date_captured
            )
            # load annotations, in this case we dont care about seg, area, bbox, iscrowd
            im_o.add_anno(
                segmentation=[[width / 2 - 10, height / 2 - 10,
                               width / 2 - 10, height / 2 + 10,
                               width / 2 + 10, height / 2 + 10,
                               width / 2 + 10, height / 2 - 10]],
                area=100,
                bbox=[width / 2 - 10, height / 2 - 10, width / 2 + 10, height / 2 + 10],
                iscrowd=0,
                id=len(coco.output_images) + 1,
                image_id=len(coco.output_images) + 1,
                category_id=len(coco.categories),
                          )
            coco.output_images.append(im_o)
    return coco


def GUI():
    """
    build the window object
    :return: Window object
    """
    # setting up theme for gui window
    sg.LOOK_AND_FEEL_TABLE['MyCreatedTheme'] = {'BACKGROUND': '#181c21',
                                                'TEXT': '#FFFFFF',
                                                'INPUT': '#181c21',
                                                'TEXT_INPUT': '#FFFFFF',
                                                'SCROLL': '# 99CC99',
                                                'BUTTON': ('#FFFFFF', '#181c21'),
                                                'PROGRESS': ('# 99CC99', '#FFFFFF'),
                                                'BORDER': 1, 'SLIDER_DEPTH': 0,
                                                'PROGRESS_DEPTH': 0, }
    sg.theme('MyCreatedTheme')  # this is implementing the custom them that was build earlier
    # UI setup
    font: tuple = ("Arial", 20)
    frame_width: int = 1000

    frame = [[sg.Text('Main folder', font=font, background_color='#181c21'),
              sg.Input(key='-IMPORT DIR-', font=font, size=(32, 1)),
              sg.FolderBrowse('Browse',
              font=font,  # read only 'json' files
              button_color='#181c21')],
             [sg.Text('New folder name', font=font, background_color='#181c21'),
              sg.Input(key='-COCO NAME-', font=font, size=(35, 1))],
             [sg.Text('Folder to export to', font=font, background_color='#181c21'),
              sg.Input(key='-EXPORT DIR-', font=font, size=(29, 1)),
              sg.FolderBrowse('Browse',
                              font=font,  # read only 'json' files
                              button_color='#181c21')
              ],
             [sg.Push(), sg.Button('Export to COCO', font=font, key='-run-', button_color='#181c21')]
             ]

    layout = [
        [sg.Frame('Select folder with classified folders',
                  frame,
                  font=font,
                  background_color='#181c21',
                  size=(frame_width, 270))
         ]
    ]

    # GUI object
    window = sg.Window("Bright Machines Classificator tool",
                       layout=layout,
                       icon='bm_logo.ico',
                       finalize=True,
                       return_keyboard_events=True,
                       grab_anywhere=True,
                       background_color='#181c21')

    return window


def run_gui(window: sg.Window):
    """
    will iterate until a run command is made
    :param window: sg object
    :return:
    """
    while True:
        event, values = window.read(timeout=100)
        if event == sg.WIN_CLOSED:
            break
        elif event == '-run-':
            if values['-IMPORT DIR-'] == '' or values['-EXPORT DIR-'] == '':
                sg.popup_error('Please enter a folder direction', background_color='#181c21')
            elif values['-COCO NAME-'] == '':
                sg.popup_error('Please enter wanted coco file name', background_color='#181c21')
            else:
                dir_name, base_name = values['-EXPORT DIR-'], values['-COCO NAME-']
                full_dir: str = os.path.join(dir_name, base_name)
                run_coco_build(folder=values['-IMPORT DIR-'], dir_for_coco=full_dir)


def copy_images(ims: list, destination: str):
    ims_d: list = [im.path for im in ims]  # list of full dirs of all images
    ok: bool = utilities.move_files_to_folder_sg(list_of_files=ims_d, destination_folder=destination)
    if ok:
        sg.popup('Images where copied to a single folder', background_color='#181c21',
                 auto_close=True, auto_close_duration=3)
    return


def run_coco_build(folder: str, dir_for_coco: str):
    """
    this will be the backbone for building and exporting the coco file
    :param folder: the dir of the folder with all the data
    :param dir_for_coco: the full dir of the location and name for the new coco json file
    :return:
    """
    # create coco object
    new_coco = Coco(coco_dir=dir_for_coco)
    time_now = datetime.datetime.now()
    new_coco.coco_info = {
        "year": time_now.year,
        "version": "1.0",
        "description": "BMA",
        "contributor": "",
        "url": "",
        "date_created": "%s/%s/%s , %s:%s" % (
            time_now.day, time_now.month, time_now.year, time_now.hour, time_now.minute)
    }
    new_coco.licenses = [{
        "id": 0,
        "name": "Unknown License",
        "url": ""
    }]
    coco = worm(dir_path=folder, category='', coco=new_coco)
    if not isinstance(coco, bool):  # the process went OK
        # create a folder
        # checking if the directory demo_folder
        # exist or not.
        while os.path.exists(dir_for_coco):
            dir_for_coco += '_new'
        os.makedirs(dir_for_coco)
        # save coco
        coco.export_coco('/' + os.path.basename(dir_for_coco))
        sg.popup('Export is done, the file was created', background_color='#181c21',
                 auto_close=True, auto_close_duration=3)
        # copy all images to the same folder
        os.makedirs(os.path.join(dir_for_coco, 'images'))
        copy_images(ims=coco.output_images, destination=os.path.join(dir_for_coco, 'images'))


def main():
    window = GUI()
    run_gui(window)


if __name__ == "__main__":
    main()
