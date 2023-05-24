import PySimpleGUI as sg
from utilities import *
from AnnotationAssistant.BM_Annotation_Tool import gui
from AugmentationAssistant.BM_Augmentation_Tool import run_aug
from colorama import Fore
import DataSplit
import Classificator


def main_gui():
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
    button_size: tuple = (20, 10)
    font_size: int = 20
    layout = [
        [sg.Button("Annotation Maker", key="-anno-", font=font_size, size=button_size),
         sg.Button("Classification Maker", key="-Classificator-", font=font_size, size=button_size)
         ],
        [sg.Button("Split Data", key="-split-", font=font_size, size=button_size),
         sg.Button("Augmentation Maker", key="-augment-", font=font_size, size=button_size)]
              ]
    window = sg.Window("Bright Data Preparation Tool",
                       layout=layout,
                       icon='bm_logo.ico',
                       finalize=True,
                       grab_anywhere=True)

    while True:
        event, values = window.read(timeout=100)
        if event == sg.WIN_CLOSED:
            break
        elif event == "-anno-":  # open annotation gui and hiding main gui
            print(colored("hiding main GUI", color='blue'))
            window.Hide()
            gui()
            window.UnHide()

        elif event == "-Classificator-":
            print(colored("hiding main GUI", color='blue'))
            window.Hide()
            Classificator.main()
            window.UnHide()

        elif event == "-augment-":  # open augmentation gui and hiding main gui
            print(colored("hiding main GUI", color='blue'))
            window.Hide()
            run_aug()
            window.UnHide()
        elif event == "-split-":  # open augmentation gui and hiding main gui
            print(colored("hiding main GUI", color='blue'))
            window.Hide()
            DataSplit.main()
            window.UnHide()


if __name__ == "__main__":
    main_gui()
