from enum import Enum, auto
import shutil as sh
import os
import json
from glob import glob
import numpy as np
from termcolor import colored
from tqdm import trange
import cv2 as cv
import PySimpleGUI as sg


def okfiletypes():
    return ["bmp", "jpg", "jpeg", "png"]


class ReadFileError(BaseException):
    pass


class ImagesTypes(str, Enum):
    PNG: str = "png"
    JPG: str = "jpg"
    BMP: str = "bmp"
    Unknown: str = "Unknown"


class ImageSource(Enum):
    Camera = auto()
    DataSet = auto()
    NotSelected = auto()


def move_files_to_folder(list_of_files: list[str], destination_folder: str) -> None:
    with trange(len(list_of_files)) as t:
        for i in t:
            # Description will be displayed on the left
            f = list_of_files[i]
            t.set_description(f"Moving... to {destination_folder}:")
            t.set_postfix(File=f)
            sh.move(f, destination_folder)


def move_files_to_folder_sg(list_of_files: list[str], destination_folder: str) -> bool:
    l: int = len(list_of_files)
    for i, _dir in enumerate(list_of_files):
        run: bool = sg.one_line_progress_meter(title='My Meter', current_value=i + 1,
                                               bar_color=('red', '#181c21'),
                                               max_value=l, key='key', orientation='h',
                                               )
        # Update the progress bar every iteration
        # event, values = window.read(timeout=10)
        if run:
            f = _dir
            sh.copy(f, destination_folder)
        else:
            return False
    return True


def copy_files_to_folder(list_of_files: list[str], destination_folder: str) -> None:
    with trange(len(list_of_files)) as t:
        for i in t:
            f = list_of_files[i]
            t.set_description(f"Copying... to {destination_folder}:")
            t.set_postfix(File=f)
            sh.copy(f, destination_folder)


def del_file(path: str) -> None:
    """
    delete file or folder and all the files inside with recursion
    :param path: dir for file or folder
    :return:
    """
    if os.path.exists(path):  # check if exists
        if os.path.isdir(path):  # check if fir and not file
            for f in os.listdir(path):  # recursion of deleting file
                del_file(os.path.join(path, f))
            os.rmdir(path)
        else:
            os.remove(path)


def copy_file(file_dir: str, destination_folder: str) -> None:
    sh.copy(file_dir, destination_folder)


def move_file(file_dir: str, destination_folder: str) -> None:
    sh.move(file_dir, destination_folder)


def check_file(file_path: str, suffix: str) -> None:
    """ Help function for reading files. Check inputs types, correctness, and file existence.
    :param file_path: file name with directory. example: c:\\MyFolder\\MyFile.txt
    :type file_path: str
    :param suffix: type of the file, such as txt,json,csv etc.
    :type suffix: str
    :return: None
    """
    if type(file_path) != str:
        raise ReadFileError(f"{file_path} not sting. got {type(file_path)}")
    elif file_path.__len__() == 0:
        raise ReadFileError("File name is empty")
    elif not os.path.exists(file_path) or not os.path.isfile(file_path):
        raise ReadFileError(f"{file_path} is not exists.")
    elif not file_path.endswith("." + suffix):
        raise ReadFileError(f"file is not a {suffix} file.")



def read_json(file_path: str) -> dict:
    """ Function for reading json file. gets file directory. Check inputs and that file exits. return parsed data.
    :param file_path: file name with directory. example: c:\\MyFolder\\MyFile.json
    :type file_path: str
    :return: data -> parsed data from the json file.
    :rtype: dict
    """
    check_file(file_path, "json")
    with open(file_path) as f:
        data = json.load(f)
    return data


def write_json(path: str, file_name: str, data: dict) -> None:
    """ Writing json file. gets what and where to write, and how to call it.
    example:path = c:\\MyFolder file_name=MyFile data= {"Data1":"MyData1, "Data2":"MyData2, "Data3":"MyData3}
    the function will write the file MyFile.json into c:\\MyFolder contains data.
    :param path:
    :type path: str
    :param file_name:
    :type file_name:  str
    :param data:
    :type data: dict
    :return: None
    """
    assert os.path.isdir(path), ValueError("path is not valid.")
    assert isinstance(file_name, str), TypeError("file_name have to be string")
    assert isinstance(data, dict), TypeError("data have to be dictionary")

    json_file_name = os.path.join(path, file_name) + ".json"

    with open(json_file_name, 'w') as f:
        json.dump(data, f, indent=4)


# This class objective is to prepare and present image with annotation
class ShowAnno:
    def __init__(self, name: str, size: tuple = (1200, 800)):
        cv.namedWindow(name, cv.WINDOW_NORMAL)
        cv.resizeWindow(name, size[0], size[1])
        self.name = name
        self.im = np.array([])

    def load_im(self, im: np.array):
        self.im = im

    def anno_maker(self, poly_points: np.array = None, rec_points: tuple = None,
                   texts: list[str] = None, text_loc: np.array = None):
        """
        Putting annotations on top of the selected image
        :param poly_points: points that make a polygon
        :param rec_points: ((x1, y1), (x2, y2))
        :param texts: list of strings ("text 1", "text 2", ...)
        :param text_loc: tuple of tuples ((x1, y1), (x2, y2))
        :return:
        """
        if poly_points is not None:
            self.im = cv.polylines(self.im, [poly_points], isClosed=True, color=(255, 255, 0), thickness=3)
        if isinstance(rec_points, tuple):
            self.im = cv.rectangle(self.im, rec_points[0], rec_points[1], (255, 0, 255), thickness=5)
        if texts:
            for text, loc in zip(texts, text_loc):
                self.im = cv.putText(img=self.im, text=text,
                                     org=loc,  # location of text
                                     fontFace=cv.FONT_HERSHEY_SIMPLEX,  # font
                                     fontScale=2,  # font scale
                                     color=(0, 255, 255),  # color
                                     thickness=2,
                                     lineType=cv.LINE_AA)

    def show_image(self):
        """
        Showing image in cv platform
        :return: True if all is good, and False if the run was terminated
        """
        cv.imshow(self.name, self.im)
        cv.waitKey(5)
        # Cancel operation if run is terminated (the exit x sign was pressed)
        if not cv.getWindowProperty(self.name, cv.WND_PROP_VISIBLE):
            print("Operation Cancelled")
            return False
        return True


def load_files(path: str, suffix: str = ""):
    assert type(path) == str, TypeError(f"path have to be string. got {type(path)}.")
    assert os.path.exists(path), ValueError(f"{path} does not exists.")
    file_type = f"*.{suffix}" if suffix != "" else "*"
    files = glob(os.path.join(path, file_type))
    return files


def rename_dataset(fileDir: str, template: str = "Image", file_type: str = "") -> None:
    """

    :param fileDir:  folder's path. where all the images files are located
    :param template:  template: name template
    :param file_type: specified the file's type to rename. if not given -> rename all files types in folder.
    :return:
    """
    old_dirs = glob(os.path.join(fileDir, f"*.{file_type}" if file_type != "" else "*"))
    n_digits: int = len(str(len(old_dirs))) # get number of digits
    i: int
    name: str
    for i, name in enumerate(old_dirs):
        suffix: str = name.split(".")[-1]  # extract suffix
        file_name: str = f"{template}_{str(i).rjust(n_digits, '0')}.{suffix}"  # build new file name (indexed)
        new_file_dir: str = os.path.join(fileDir, file_name)  # set full file directory
        os.rename(name, new_file_dir)  # Rename file from name to new_file_dir
        print(colored(f"Rename {name} to {new_file_dir}."))


def check_ratio(value: float) -> bool:
    status: bool = False
    try:
        assert isinstance(value, float), TypeError("folderPath input must be type string")
        assert value >= 0.0, ValueError("Train ratio have to be positive non zero value. got ", value)
        status = True

    except TypeError as e:
        print(colored(e.__class__.__name__, 'red'), colored(e.args[0], "red"))
    except ValueError as e:
        print(colored(e.__class__.__name__, 'red'), colored(e.args[0], "red"))
    except AssertionError as e:
        print(colored(e.__class__.__name__, 'red'), colored(e.args[0], "red"))

    return status


def make_dir(new_dir: str) -> bool:
    try:
        if not os.path.exists(new_dir):
            os.mkdir(new_dir)  # making dir
        return True
    except FileExistsError as e:
        print(colored(f"Error {e}"))
        return False


def is_unsigned_normal_float(v: float) -> tuple[bool, str]:
    """
    UnsignedNormalFloat
    """
    status: bool = False
    msg: str = "All Good"

    if not isinstance(v, float):
        msg = f"WRONG TYPE: have to be float. got {v.__class__.__name__}"
        print(msg)

    elif not (0 <= v <= 1):
        msg = f"WRONG VALUE: have to be 0 < value < 1, got {v}"
        print(msg)

    else:
        status = True

    return status, msg


def test_base_folder(base_path: str):
    """
    Function checks the follows:
     1) Path exist
     2) Contain images Folder
     3) Contain COCO JSON file

    :param base_path:
    :return:
    """
    # Valid Base Folder Existence:
    if not os.path.isdir(base_path):
        raise FileExistsError(f'{base_path} not found nor exist.')

    # Valid images Folder Existence:
    images_path: str = os.path.join(base_path, 'images')
    if not os.path.isdir(images_path):
        raise FileExistsError(f'{images_path} not found.')

    # Valid coco File Existence:
    json_files: list[str] = glob(os.path.join(base_path, "*.json"))
    if len(json_files) == 0:
        raise FileExistsError(f'COCO JSON file not found in {base_path}.')

