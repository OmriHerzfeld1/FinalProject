from MainGUI import *
from pycocotools.coco import COCO
import numpy as np
import PIL
from tqdm import trange


def is_valid_coco(file_path: str):
    """
    check if the file is a valid coco format
    :param file_path: full dir of json
    :return: if valid or not
    """
    try:
        with open(file_path, 'r') as f:
            coco_dict = json.load(f)
        coco = COCO(file_path)
        return True
    except Exception as e:
        return False


def test_ratio(ratio: tuple):
    """
    check that ratio inputs is good
    :param ration: tuple of floats
    :return: message if the inputs is bad and True if valid
    """
    sum: float = 0.0
    try:
        for r in ratio:
            r = float(r)
            if isinstance(r, float) or r == 0 or r == 1:
                if 0.0 <= r <= 1.0:
                    pass
                else:
                    return 'inputs must be in the range of 0.0-1.0', ratio
            else:
                return 'inputs must be in the range of 0.0-1.0', ratio

            sum += r
    except ValueError:
        return 'inputs must be in the range of 0.0-1.0', ratio
    if sum != 1.0:
        return 'sum of inputs must be 1.0', ratio

    return True, (float(ratio[0]), float(ratio[0]) + float(ratio[1]), float(ratio[2]))


def splitting(coco_dir: str, ratio: tuple):
    """
    will split and save a coco for each set - Train, Validation and Test
    will save TrainCoco.json and so on files in the coco_dir folder
    :param coco_dir: full dir
    :param ration: tuple with numbers between [0,1] (Train, Validation, Test)
    :return:
    """
    valid, ratio = test_ratio(ratio)  # return validation text and ratio tuple of floats

    if valid != True:
        return valid

    # Load the COCO annotations file
    coco = COCO(coco_dir)

    # Get all image IDs
    imgIds = coco.getImgIds()

    # Randomly shuffle image IDs
    np.random.shuffle(imgIds)

    # Split image IDs into train, val, and test sets
    trainIds = imgIds[:int(ratio[0] * len(imgIds))]
    valIds = imgIds[int(ratio[0] * len(imgIds)):int(ratio[1] * len(imgIds))]
    testIds = imgIds[int(ratio[1] * len(imgIds)):]

    # Create new COCO objects for train, val, and test sets
    trainCoco = coco.loadImgs(trainIds)
    valCoco = coco.loadImgs(valIds)
    testCoco = coco.loadImgs(testIds)

    # Create new annotations dictionaries for train, val, and test sets
    trainAnn: dict = {'info': coco.dataset['info'], 'licenses': coco.dataset['licenses'], 'images': [], 'annotations': [], 'categories': coco.dataset['categories']}
    valAnn: dict = {'info': coco.dataset['info'], 'licenses': coco.dataset['licenses'], 'images': [], 'annotations': [], 'categories': coco.dataset['categories']}
    testAnn: dict = {'info': coco.dataset['info'], 'licenses': coco.dataset['licenses'], 'images': [], 'annotations': [], 'categories': coco.dataset['categories']}

    # dict that will help iterate on all sets
    sets: dict = {
        'train': [trainCoco, trainAnn, '_Train'],
        'val': [valCoco, valAnn, '_Validation'],
        'test': [testCoco, testAnn, '_Test']
    }

    folder_dir: str = os.path.dirname(coco_dir)
    # Iterate over set in sets
    for set in sets:
        # Iterate over images in set
        with trange(len(sets[set][0])) as t:
            for i in t:
                img: dict = sets[set][0][i]
                # Add image to new annotations dictionary
                sets[set][1]['images'].append(img)
                # Get annotations for image from original dataset
                annIds: list = coco.getAnnIds(imgIds=img['id'])
                anns: list = coco.loadAnns(annIds)
                # Add annotations to new annotations dictionary
                sets[set][1]['annotations'].extend(anns)

                # staring the images ids form 0
                sets[set][1]['images'][-1]['id']: int = i + 1
                for j in range(len(anns)):
                    sets[set][1]['annotations'][-1 * (j + 1)]["image_id"] = i + 1

            t.set_description(f'set: {set}')
            # Postfix will be displayed on the right,
            # formatted automatically based on argument's datatype
            # t.set_postfix(TrainingLoss=float(train_loss), ValidationLoss=float(val_loss))

        # save annotations to JSON file
        annotations = sets[set][1]
        file_name: str = os.path.basename(coco_dir).split('.')[0] + sets[set][2] + '.json'
        full_dir: str = os.path.join(folder_dir, file_name)
        with open(full_dir, 'w') as f:
            json.dump(annotations, f, indent=4)
    return True


def run_gui(window):
    valid: bool = False
    while True:
        event, values = window.read(timeout=100)
        if event == sg.WIN_CLOSED:
            break
        elif event == '-IMPORT-':
            if values['-IMPORT DIR-'] == '':
                sg.popup_error('Please enter a Coco (json format) file direction', background_color='#181c21')
            else:
                coco: str = values['-IMPORT DIR-']
                # check if coco is valid
                valid = is_valid_coco(coco)
                if not valid:
                    sg.popup_error('Please enter a valid coco file', background_color='#181c21')
                else:
                    sg.popup('Loading is done')

        elif event == '-split-':
            if not valid:
                sg.popup_error('Please load coco file before', background_color='#181c21')
            else:
                ratio: tuple = (values['-Train ratio-'],
                                values['-Val ratio-'],
                                values['-Test ratio-'])
                output = splitting(coco, ratio)
                if output == True:
                    print(Fore.GREEN + 'Splitting is done')
                    sg.popup('The data has been split')
                else:
                    sg.popup_error(output, background_color='#181c21')
    window.Close()


def build_gui():
    # UI setup
    font: tuple = ("Arial", 20)
    frame_width: int = 1000

    impo_coco = [[sg.Text('Coco file', font=font, background_color='#181c21'),
                  sg.Input(key='-IMPORT DIR-', font=font, size=(40, 1)),
                    sg.FileBrowse('Browse',
                                  font=font,
                                  file_types=(("ALL Files", "json"),),  # read only 'json' files
                                  button_color='#181c21'),
                    sg.Button(' Import ', font=font, key='-IMPORT-', button_color='#181c21')]
                   ]

    impo_coco_frame = [
        [sg.Frame('Import coco',
                  impo_coco,
                  font=font,
                  background_color='#181c21',
                  size=(frame_width, 100))
         ]
    ]

    splitting_rows = [
        [sg.Text('Train       ', font=font, background_color='#181c21'),
         sg.Input(key='-Train ratio-', font=font, size=(4, 1))],
        [sg.Text('Validation', font=font, background_color='#181c21'),
         sg.Input(key='-Val ratio-', font=font, size=(4, 1))],
        [sg.Text('Test        ', font=font, background_color='#181c21'),
         sg.Input(key='-Test ratio-', font=font, size=(4, 1))]
    ]

    splitting_frame = [
        [sg.Frame('Splitting Ratio',
                  splitting_rows,
                  element_justification='left',
                  font=font,
                  background_color='#181c21',
                  size=(frame_width, 200))]
                ]

    layout = [[impo_coco_frame],
              [splitting_frame],
              [sg.Push(background_color='#181c21'), sg.Button('Split', font=font, key='-split-', button_color='#181c21')]
              ]
    # GUI object
    window = sg.Window("Bright Machines Splitting Data tool",
                       layout=layout,
                       icon='bm_logo.ico',
                       finalize=True,
                       return_keyboard_events=True,
                       grab_anywhere=True,
                       background_color='#181c21')

    return window


def main():
    window = build_gui()
    run_gui(window)


if __name__ == '__main__':
    main()
