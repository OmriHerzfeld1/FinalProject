import math
import os
import PIL.Image
import PIL.ImageGrab
import io
import PySimpleGUI as sg
import json
import datetime
import time
import numpy as np
import utilities
# from termcolor import cprint
from colorama import Fore
from AnnotationAssistant.AutoAnnotatorRev2 import AutoAnnotator


class Anno:
    def __init__(self, superclass: str, part: str, status: str, shape: str, anno_obj, bbox: tuple):
        """
        annotation object, holds all info relevant for the annotation of every image
        :param superclass:
        :param part:
        :param status:
        :param shape: 'Rectangle' or 'Circle' or 'Polynom'
        :param anno_obj: the link to the object itself, when wanting to delete or move..
        :param bbox: the points that make the form of the shape
        """
        self.superclass = superclass
        self.part = part
        self.status = status
        self.shape = shape
        self.anno_obj = anno_obj
        self.bbox = bbox
        # will hold info to move object by x, y
        self.delta_x = None
        self.delta_y = None


class Image:
    def __init__(self, path: str, name: str, scale: float = 0.0, width=-1, height=-1, cur_anno=None):
        self.path = path
        self.name = name
        self.scale = scale
        self.width = width
        self.height = height
        self.anno = []
        self.cur_anno = cur_anno  # the current selected annotation
        self.cur_anno_obj = None  # the link to the object itself when it is selected and colord in yellow
        self.red_points: list = []  # list of red points objects for stretching annotation
        self.moving_anno: bool = False  # indicate if an object is being moved

    def add_anno(self, superclass: str, part: str, status: str, shape: str, anno_obj, bbox: tuple):
        self.anno.append(Anno(superclass, part, status, shape, anno_obj, bbox))

    def show_im(self, resize: tuple = (500, 500)):
        """
        Will convert into bytes and optionally resize an image that is a file or a base64 bytes object.
        Turns into  PNG format in the process so that can be displayed by tkinter
        :param resize:  optional new size
        :type resize: (Tuple[int, int] or None)
        :return: (bytes) a byte-string object
        :rtype: (bytes)
        """
        img = PIL.Image.open(self.path)
        # self.width, self.height = img.size  # Incoming im size
        new_width, new_height = resize  # Size of graph
        scale: float = min(new_height / self.height, new_width / self.width)  # Scale needed for perfect fit
        img = img.resize((int(self.width * scale), int(self.height * scale)), PIL.Image.LANCZOS)  # Scaling im
        bio = io.BytesIO()
        img.save(bio, format="PNG")
        del img
        return bio.getvalue(), scale

    def drawpoly(self, graph, points, width=5, color='black'):
        """
        Gets points and draws polygon on graph form lines
        :param graph:
        :param points: list of point [[x1,y1],[x2,y2]]
        :param width:
        :param color:
        :return: return a list of all the line elements that creates the polygon
        """
        lines: list = []
        x, y = points[0]
        for x1, y1 in points[1:] + [points[0]]:
            line = graph.DrawLine((x, y), (x1, y1), width=width, color=color)
            x, y = x1, y1
            lines.append(line)
            # print(line)
        return lines

    def load_im(self, graph, canvas_size, color: str = 'black'):
        """
        Will load all ready exist image with all the current annotation
        :param color: color of drawings
        :param graph:
        :param canvas_size:
        """
        graph.erase()  # Delete all old annotations from graph
        # Scale = scale factor from original to new
        # Converts the incoming im to PIL file and resize to fit canvas size
        im_show, scale = self.show_im(resize=canvas_size)
        image_el = graph.draw_image(data=im_show, location=(0, 0))  # Put's the images on the graph
        # Load annotations to graph
        for anno in self.anno:
            if anno.shape == 'Rectangle':
                anno.anno_obj = graph.draw_rectangle(anno.bbox[0], anno.bbox[1], line_color=color, line_width=5)
            elif anno.shape == 'Circle':
                anno.anno_obj = graph.draw_circle(anno.bbox[0], anno.bbox[1], line_color=color, line_width=5)
            elif anno.shape == 'Polygon':
                anno.anno_obj = self.drawpoly(graph, anno.bbox, color=color)
        return image_el

    def colord_anno(self, graph, color_m: str, selected_anno=None):
        """
        Will color the selected annotation in the input color, the colord annotation object (graph object) will be
        saved in 'self.cur_anno_obj'
        :param color_m: color of drawings
        :param selected_anno: the annotation we want to color in a different color
        :param graph:
        """
        if self.red_points:  # let get rid of old red points
            for i in range(0, len(self.red_points)):
                graph.delete_figure(self.red_points[0])
                self.red_points.remove(self.red_points[0])
        # Load annotations to graph
        for anno in self.anno:
            graph.delete_figure(anno.anno_obj)
            if anno == selected_anno:  # If it's the selected annotation make color as entered
                color = color_m
                graph.delete_figure(self.cur_anno_obj)  # delete the old yellow form if exist
                if anno.shape == 'Rectangle':
                    self.cur_anno_obj = graph.draw_rectangle(anno.bbox[0], anno.bbox[1], line_color=color, line_width=5)
                    self.red_points.append(graph.draw_circle(anno.bbox[0], 6, line_color='red', fill_color='red'))
                    self.red_points.append(graph.draw_circle((anno.bbox[1][0], anno.bbox[0][1]), 6,
                                                             line_color='red', fill_color='red'))
                    self.red_points.append(graph.draw_circle(anno.bbox[1], 6, line_color='red', fill_color='red'))
                    self.red_points.append(graph.draw_circle((anno.bbox[0][0], anno.bbox[1][1]), 6,
                                                             line_color='red', fill_color='red'))
                elif anno.shape == 'Circle':
                    self.cur_anno_obj = graph.draw_circle(anno.bbox[0], anno.bbox[1], line_color=color, line_width=5)
                    self.red_points.append(graph.draw_circle((anno.bbox[0][0], anno.bbox[0][1] - anno.bbox[1]), 6,
                                                             line_color='red', fill_color='red'))
                    self.red_points.append(graph.draw_circle((anno.bbox[0][0] + anno.bbox[1], anno.bbox[0][1]), 6,
                                                             line_color='red', fill_color='red'))
                    self.red_points.append(graph.draw_circle((anno.bbox[0][0], anno.bbox[0][1] + anno.bbox[1]), 6,
                                                             line_color='red', fill_color='red'))
                    self.red_points.append(graph.draw_circle((anno.bbox[0][0] - anno.bbox[1], anno.bbox[0][1]), 6,
                                                             line_color='red', fill_color='red'))
                elif anno.shape == 'Polygon':
                    for line in anno.anno_obj:
                        graph.delete_figure(line)  # deleting all old line objects
                    self.cur_anno_obj = self.drawpoly(graph, anno.bbox, color=color, width=5)
                anno.anno_obj = self.cur_anno_obj
            else:
                color = 'black'
                if anno.shape == 'Rectangle':
                    anno.anno_obj = graph.draw_rectangle(anno.bbox[0], anno.bbox[1], line_color=color, line_width=5)
                elif anno.shape == 'Circle':
                    anno.anno_obj = graph.draw_circle(anno.bbox[0], anno.bbox[1], line_color=color, line_width=5)
                elif anno.shape == 'Polygon':
                    for line in anno.anno_obj:  # deleting all old line objects
                        graph.delete_figure(line)
                    anno.anno_obj = self.drawpoly(graph, anno.bbox, color=color)

    def stretching_anno(self, graph, location: list, stretching: bool, start_p: tuple, event,
                        point: int, new_anno: tuple):
        """
        This function will stretch annotation if a red point is pressed and dragged (working only for 'Rectangle' and
        'Circle'
        :param graph:
        :param location: the x, y location of mouse on the graph object
        :param stretching: indicts if the object is being stretched (start point exist's)
        :param start_p: the first point that was pressed
        :param event:
        :param point: gives the position of the red point:
        when 'Rectangle' - 0 - right top, 1 - left top, 2 - left bottom, 3 - right bottom
        when 'Circle' - 0 - top, 1 - right, 2 - bottom, 3 - left
        :param new_anno: points that define the new annotation
        :return: things that is needed for future run's for this function
        """
        if stretching and event.endswith('+UP'):
            print("end")
            self.cur_anno.bbox = new_anno  # updating boundary box
            self.colord_anno(graph, 'yellow', self.cur_anno)  # need this to put back red points
            stretching, self.moving_anno = False, False
            self.cur_anno.delta_x, self.cur_anno.delta_y = None, None
        if event == "-GRAPH-":
            x, y = location
            if not stretching:  # the annotation didn't start to stretch
                figure = graph.get_figures_at_location(location)
                for i in range(0, len(self.red_points)):
                    if self.red_points[i] in figure:
                        print('red point number {} was selected'.format(i))
                        point = i  # the selected point
                        start_p = (x, y)  # Beginning point
                        stretching = True  # Next points are not beginning anymore
                        if self.red_points:  # let get rid of old red points
                            for j in range(0, len(self.red_points)):
                                graph.delete_figure(self.red_points[0])
                                self.red_points.remove(self.red_points[0])
                        break
            else:
                end_p = [x, y]  # In the end will hold the end point
                print(end_p)
                if self.cur_anno_obj:
                    self.cur_anno.delta_x, self.cur_anno.delta_y = end_p[0] - start_p[0], end_p[1] - start_p[1]
                    graph.delete_figure(self.cur_anno_obj)
                    if self.cur_anno.shape == 'Rectangle':
                        if point == 0:
                            new_anno = ((self.cur_anno.bbox[0][0] + self.cur_anno.delta_x,
                                         self.cur_anno.bbox[0][1] + self.cur_anno.delta_y),
                                        (self.cur_anno.bbox[1][0],
                                         self.cur_anno.bbox[1][1]
                                         ))
                        elif point == 1:
                            new_anno = ((self.cur_anno.bbox[0][0],
                                         self.cur_anno.bbox[0][1] + self.cur_anno.delta_y),
                                        (self.cur_anno.bbox[1][0] + self.cur_anno.delta_x,
                                         self.cur_anno.bbox[1][1]
                                         ))
                        elif point == 2:
                            new_anno = ((self.cur_anno.bbox[0][0],
                                         self.cur_anno.bbox[0][1]),
                                        (self.cur_anno.bbox[1][0] + self.cur_anno.delta_x,
                                         self.cur_anno.bbox[1][1] + self.cur_anno.delta_y
                                         ))
                        elif point == 3:
                            new_anno = ((self.cur_anno.bbox[0][0] + self.cur_anno.delta_x,
                                         self.cur_anno.bbox[0][1]),
                                        (self.cur_anno.bbox[1][0],
                                         self.cur_anno.bbox[1][1] + self.cur_anno.delta_y
                                         ))
                        graph.delete_figure(self.cur_anno_obj)
                        self.cur_anno_obj = graph.draw_rectangle(new_anno[0], new_anno[1],
                                                                 line_color='yellow', line_width=5)
                    elif self.cur_anno.shape == 'Circle':
                        if point == 0:
                            new_anno = (self.cur_anno.bbox[0],
                                        self.cur_anno.bbox[1] - self.cur_anno.delta_y)
                        elif point == 1:
                            new_anno = (self.cur_anno.bbox[0],
                                        self.cur_anno.bbox[1] + self.cur_anno.delta_x)
                        elif point == 2:
                            new_anno = (self.cur_anno.bbox[0],
                                        self.cur_anno.bbox[1] + self.cur_anno.delta_y)
                        elif point == 3:
                            new_anno = (self.cur_anno.bbox[0],
                                        self.cur_anno.bbox[1] - self.cur_anno.delta_x)
                        graph.delete_figure(self.cur_anno_obj)
                        self.cur_anno_obj = graph.draw_circle(new_anno[0], new_anno[1],
                                                              line_color='yellow', line_width=5)
                    print('bbox is now {}'.format(new_anno))
        return stretching, start_p, point, new_anno

    def find_anno(self, location):
        """
        Given a location on graph this will return annotation in the input location
        :param location: [x, y]
        :return: annotation in the input location or None if there is no annotation in location
        """
        for an in self.anno:
            if an.shape == 'Rectangle':
                if an.bbox[0][0] - 5 < location[0] < an.bbox[1][0] + 5 and \
                        an.bbox[0][1] - 5 < location[1] < an.bbox[1][1] + 5:
                    self.cur_anno = an
                    return
            elif an.shape == 'Circle':
                if an.bbox[1] >= ((an.bbox[0][0] - location[0]) ** 2 + (an.bbox[0][1] - location[1]) ** 2) ** 0.5:
                    self.cur_anno = an
                    return
            elif an.shape == 'Polygon':
                if ray_tracing(location[0], location[1], an.bbox):
                    self.cur_anno = an
                    return
        self.cur_anno = None

    def del_anno(self, canvas_size, graph, location):
        """
        delete all annotations falling on the input location
        :param canvas_size:
        :param graph:
        :param location: [x, y]
        """
        self.find_anno(location)
        if self.cur_anno:
            self.anno.remove(self.cur_anno)
            self.load_im(graph, canvas_size)

    def anno_info(self, window, graph, location, category=None):
        """
        This will show the current info of a selected annotation and color the annotation
        :param graph:
        :param window:
        :param location:
        :param category: dict of all category's available
        :return: return the annotations which falls in the location entered
        """
        self.find_anno(location)
        cur_anno = self.cur_anno
        if cur_anno:
            print('the selected annotation is a {} ["{}", "{}", "{}"]'.format(cur_anno.shape,
                                                                              cur_anno.superclass,
                                                                              cur_anno.part,
                                                                              cur_anno.status))
            window['-CURR ATT-'].Update(visible=True)
            window['cur ' + list(category.keys())[0]].update(value=cur_anno.superclass)
            window['cur ' + list(category.keys())[1]].update(value=cur_anno.part)
            window['cur ' + list(category.keys())[2]].update(value=cur_anno.status)
            self.colord_anno(graph, 'yellow', cur_anno)  # color the annotation
            self.moving_anno = True
        else:
            window['-CURR ATT-'].Update(visible=False)

    def move_anno(self, window, graph, event, values, dragging, start_p):
        """
        will move selected annotation on graph element
        :param window:
        :param graph:
        :param event:
        :param values:
        :param dragging: True if the annotation is being dragged, else False
        :param start_p:
        :return:
        """
        if self.red_points:  # let get rid of old red points
            for i in range(0, len(self.red_points)):
                graph.delete_figure(self.red_points[0])
                self.red_points.remove(self.red_points[0])
        if event.endswith('+UP') and dragging:  # when moving is done
            # reset all parameters
            dragging = False
            start_p, end_p = None, None
            self.moving_anno = False  # the object is done being moved and is now set in wanted place
            self.colord_anno(graph, 'yellow', self.cur_anno)
            print('{} moved to points {}'.format(self.cur_anno.shape, self.cur_anno.bbox))
        else:
            x, y = values["-GRAPH-"]
            if self.cur_anno.shape == 'Polygon':  # TODO need to do nicer
                if not dragging:  # The mouse was pressed just now
                    start_p = [x, y]  # Beginning point
                    dragging = True  # Next points are not beginning anymore
                else:
                    end_p = [x, y]  # In the end will hold the end point
                    # setting delta's
                    self.cur_anno.delta_x, self.cur_anno.delta_y = end_p[0] - start_p[0], end_p[1] - start_p[1]
                    # moving the object by delta
                    for line in self.cur_anno_obj:
                        # print('moving', line)
                        graph.move_figure(line, self.cur_anno.delta_x, self.cur_anno.delta_y)
                    start_p = [x, y]
                    for point in self.cur_anno.bbox:
                        point[0] = point[0] + self.cur_anno.delta_x
                        point[1] = point[1] + self.cur_anno.delta_y
            else:
                if not dragging:  # The mouse was pressed just now
                    start_p = [x, y]  # Beginning point
                    dragging = True  # Next points are not beginning anymore
                else:
                    end_p = [x, y]  # In the end will hold the end point
                    # setting delta's
                    self.cur_anno.delta_x, self.cur_anno.delta_y = end_p[0] - start_p[0], end_p[1] - start_p[1]
                    # moving the object by delta
                    graph.move_figure(self.cur_anno_obj, self.cur_anno.delta_x, self.cur_anno.delta_y)
                    start_p = [x, y]
                    if self.cur_anno.shape == 'Rectangle':
                        # updating the object location on real time
                        self.cur_anno.bbox = ((self.cur_anno.bbox[0][0] + self.cur_anno.delta_x,
                                               self.cur_anno.bbox[0][1] + self.cur_anno.delta_y),
                                              (self.cur_anno.bbox[1][0] + self.cur_anno.delta_x,
                                               self.cur_anno.bbox[1][1] + self.cur_anno.delta_y))
                    elif self.cur_anno.shape == 'Circle':
                        # updating the object location on real time
                        self.cur_anno.bbox = ((self.cur_anno.bbox[0][0] + self.cur_anno.delta_x,
                                               self.cur_anno.bbox[0][1] + self.cur_anno.delta_y),
                                              self.cur_anno.bbox[1])
            # Show's location of mouse real time when left click is pressed
            window["-INFO-"].update(
                value='mouse x:{} , y:{}'.format(int(x / self.scale), int(y / self.scale)))
        return dragging, start_p


# @jit(nopython=True)
def ray_tracing(x, y, poly):
    """
    Detects if a point is inside a polygon
    :param x: x position
    :param y: y position
    :param poly: list of tuples that make a polygon
    :return: True if x,y in polygon other ways False
    """
    n = len(poly)
    inside = False
    p2x = 0.0
    p2y = 0.0
    xints = 0.0
    p1x, p1y = poly[0]
    for i in range(n + 1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def anno_location(annotation: list, shape: str, scale: int):
    """
    this function is getting some information about an annotation and return all the needed info for a given annotation
    in Coco format
    :param scale: scale -  canvas size / image
    :param annotation: list that has the information to build a shape
    :param shape: the type of shape - Circle ext
    :return:segmentation, area, bbox
    """
    # bbox format - [top left x , top left y, width , height] of surrounding
    if shape == 'Circle':
        r = int(annotation[1] / scale)
        x0, y0 = int(annotation[0][0] / scale), int(annotation[0][1] / scale)
        bbox = [
            x0 - r,
            y0 - r,
            r * 2,
            r * 2,
        ]
        segmentation = []
        for theta in np.linspace(0, 2 * np.pi, 40):
            segmentation.append(x0 + int(r * np.cos(theta)))
            segmentation.append(y0 + int(r * np.sin(theta)))
    elif shape == 'Rectangle':
        bbox = [
            int(annotation[0][0] / scale),  # X0
            int(annotation[0][1] / scale),  # Y0
            int(abs((annotation[1][0] - annotation[0][0])) / scale),  # width
            int(abs((annotation[1][1] - annotation[0][1])) / scale),  # Height
        ]
        segmentation = [
            bbox[0], bbox[1],  # Top-left coordinate
            bbox[0] + bbox[2], bbox[1],  # Top-right coordinate
            bbox[0] + bbox[2], bbox[1] + bbox[3],  # Bottom-right coordinate
            bbox[0], bbox[1] + bbox[3],  # Bottom-left coordinate
        ]
    elif shape == 'Polygon':
        max_x, max_y = 0, 0
        min_x, min_y = float('inf'), float('inf')
        segmentation = []

        for point in annotation:
            x = point[0] / scale
            y = point[1] / scale
            segmentation.append(x)
            segmentation.append(y)
            if x < min_x:
                min_x = x
            if y < min_y:
                min_y = y
            if x > max_x:
                max_x = x
            if y > max_y:
                max_y = y
        bbox = [int(min_x), int(min_y), int(max_x - min_x), int(max_y - min_y)]
    else:
        bbox = segmentation = 'null'
    area = bbox[2] * bbox[3]  # Width * Height, this is the number of pixels of the boundary box
    return [segmentation], area, bbox


def export_coco(images: list, folder_dir: str, folder_name: str, ims_dir=None):
    """
    this will export a coco format file from the images and annotations that was made in the gui
    :param ims_dir: tuple of all the dir's of all the images if None than export is for Auto Annotation
    :param folder_name: name for the folder file
    :param folder_dir: location of the destination folder to create folder in
    :param images: list of Image objects
    :return:
    """
    time_now = datetime.datetime.now()
    info_list = {
        "year": time_now.year,
        "version": "1.0",
        "description": "BMA",
        "contributor": "",
        "url": "",
        "date_created": "%s/%s/%s , %s:%s" % (time_now.day, time_now.month, time_now.year, time_now.hour, time_now.minute)
    }

    images_list = []  # will hold the relevant info of all images for coco file, list of dicts
    anno_list = []  # will hold the relevant info of all annotations for coco file, list of dicts
    anno_id = 1  # holds the current id number for the annotation
    categories = {}  # will hold all the deference categories, 'an.superclass, an.part, an.status' is key
    category_id = 1  # holds the current id number for the new category

    # Delete old file is exist:
    # old_cocos_files: list[str] = glob(os.path.join(coco_dir, "*.json"))
    # if len(old_cocos_files) > 0:
    #     [os.remove(file) for file in old_cocos_files]

    for i in range(0, len(images)):  # running on all images
        # date and time the image was created
        created_time = os.path.getctime(images[i].path)
        date_captured = time.ctime(created_time)
        date_obj = time.strptime(date_captured)
        date_captured = time.strftime("%d/%m/%Y , %H:%M", date_obj)
        cur_image = {'id': i + 1,
                     'width': images[i].width,
                     'height': images[i].height,
                     # 'file_name': images[i].path,  - Full path
                     'file_name': os.path.basename(images[i].path),  # Only base name
                     'license': 0,
                     'date_captured': date_captured
                     }  # creating mini dict for every images
        images_list.append(cur_image)  # adding mini dict to list of images info
        for an in images[i].anno:  # running on every annotation for every image
            # checking if category exists
            if '{},{},{}'.format(an.superclass, an.part, an.status) in categories.keys():
                # if exist
                cur_cat_id = categories['{},{},{}'.format(an.superclass, an.part, an.status)]  # id for current anno
            else:  # doesn't exist
                categories['{},{},{}'.format(an.superclass, an.part, an.status)] = category_id
                cur_cat_id = category_id  # id for current anno
                category_id += 1  # next new category
            seg, area, bbox = anno_location(an.bbox, an.shape, images[i].scale)
            seg = [[round(x) for x in seg[0]]]
            cur_anno = {'segmentation': seg,
                        "area": area,
                        'bbox': bbox,
                        'iscrowd': 0,
                        'id': anno_id,
                        'image_id': i + 1,
                        'category_id': cur_cat_id
                        }  # mini dict for current annotation
            anno_list.append(cur_anno)
            anno_id += 1
    categories_list = []  # will hold the wanted info about all different categories
    for category in categories.keys():  # running on all keys
        chunks = category.split(',')  # splitting and getting a list [an.superclass, an.part, an.status]
        mini_dict = {'supercategory': chunks[0],
                     'id': categories[category],
                     'name': chunks[1],
                     'status': chunks[2]
                     }  # dict that hold's new category info
        categories_list.append(mini_dict)  # hold's all uniq categories

    licenses = [{
        "id": 0,
        "name": "Unknown License",
        "url": ""
    }]
    coco_dict = {
        'info': info_list,
        'images': images_list,
        'annotations': anno_list,
        'licenses': licenses,
        'categories': categories_list
    }
    if ims_dir is not None:  # the export is not for Auto Annotation
        folder_dir: str = os.path.join(folder_dir, folder_name)
        if not utilities.make_dir(folder_dir):  # creating the dir for folder
            sg.popup_error('Folder dir is not in the right format')
            return
    utilities.write_json(path=folder_dir, file_name=folder_name, data=coco_dict)  # writing coco
    if ims_dir is not None:  # the export is not for Auto Annotation
        utilities.del_file(os.path.join(folder_dir, 'images'))  # delete folder if exists
        utilities.make_dir(os.path.join(folder_dir, 'images'))  # creating the dir for "images" folder
        utilities.copy_files_to_folder(ims_dir, os.path.join(folder_dir, 'images'))  # copy all images to images folder
        sg.popup('Export is done, a folder named - "{}" has all the wanted data'.format(folder_name),
                 background_color='#181c21')
    print('exporting is done')
    return


def import_coco(file_dir: str, resize: tuple):
    """
    Imports images and annotations form Coco file format
    :param file_dir: file location
    :param resize: size we want the image to be in the end of process
    :return: list of Image objects, list of all images names, list of all images locations, dictionary that connect the
    two lists, and lists of all the unique names of the classifications
    """
    dir_name = os.path.dirname(file_dir)  # dir of folder
    file = open(file_dir, 'r')  # reading file
    coco_dict: dict = json.load(file)  # converting file to dict
    images: dict = {}  # values are Image objects
    im_dir: list = []  # list of all the directions of all the images
    im_name: list = []  # list of all the names of all the images
    im_id: int = 1  # image id
    for im in coco_dict["images"]:
        new_width, new_height = resize  # Size of graph
        # Scale needed for perfect fit
        scale: float = min(new_height / im['height'], new_width / im['width'])
        cur_dir = im['file_name'] if im['file_name'] != os.path.basename(im['file_name']) \
            else os.path.join(dir_name, "images", im['file_name'])
        # check if this is Augmented json
        if 'Augmented' in os.path.basename(file_dir):
            cur_dir = os.path.join(dir_name, "Original + Augmented", im['file_name'])
        im_dir.append(cur_dir)
        im_name.append(os.path.basename(im['file_name']))
        # creating all the Image objects
        images[im_id] = Image(path=im_dir[-1], name=im_name[-1],
                              scale=scale, width=im['width'], height=im['height'])
        im_id += 1
    im_dic: dict = dict(zip(im_name, im_dir))
    categories: dict = {}  # dictionary that connect file name to dir for all the images
    # this will hold all the unique category names
    unique_names: dict = {'Supercategory': [], 'name': [], 'status': []}
    cat_id: int = 1  # category id
    for cat in coco_dict['categories']:  # creating dict (of all categories) of dicts
        # checking if name exist in category
        if cat['supercategory'] not in unique_names['Supercategory']:
            unique_names['Supercategory'].append(cat['supercategory'])
        if cat['name'] not in unique_names['name']:
            unique_names['name'].append(cat['name'])
        if cat['status'] not in unique_names['status']:
            unique_names['status'].append(cat['status'])
        #
        categories[cat_id] = cat
        cat_id += 1
    for an in coco_dict['annotations']:
        im_id = an['image_id']  # image id
        cat_id = an['category_id']  # category id
        scale: float = images[im_id].scale
        seg = an['segmentation'][0]  # it's a list of lists

        if [seg[4], seg[5]] == [an['bbox'][0] + an['bbox'][2], an['bbox'][1] + an['bbox'][3]]:
            shape = 'Rectangle'
            bbox = ((int(seg[0] * scale), int(seg[1] * scale)),
                    (int(seg[4] * scale), int(seg[5] * scale)))
        elif math.sqrt(abs(((seg[0] - seg[2]) ** 2 + (seg[1] - seg[3]) ** 2) - (
                (seg[2] - seg[4]) ** 2 + (seg[3] - seg[5]) ** 2))) < 10:  # check if Circle
            shape = 'Circle'
            x_location = int(an['bbox'][0] + an['bbox'][2] / 2) * scale
            y_location = int(an['bbox'][1] + an['bbox'][3] / 2) * scale
            radius = int(an['bbox'][2] / 2) * scale
            bbox = ([x_location, y_location], radius)
        else:
            shape = 'Polygon'
            bbox = []
            for i in range(0, len(seg)):
                if i % 2 == 0:  # the x part of a point
                    point = [seg[i] * scale]
                else:  # # the y part of a point
                    point.append(seg[i] * scale)  # this is a new point
                    bbox.append(point)
        images[im_id].add_anno(superclass=categories[cat_id]['supercategory'], part=categories[cat_id]['name'],
                               status=categories[cat_id]['status'], shape=shape,
                               anno_obj=None, bbox=bbox)
    print('importing is done')
    return list(images.values()), im_name, im_dir, im_dic, \
           unique_names['Supercategory'], unique_names['name'], unique_names['status']


def gui():
    mon_y, mon_x = PIL.ImageGrab.grab().size
    print("Monitor size is {}".format((mon_x, mon_y)))
    # TODO - canvas size in relation with screen size
    canvas_size = (int(mon_x * 0.9), int(mon_y * 0.3))  # The size of the canvas object as well as im shower (x, y)
    anno_min_size = (200, 15)  # (min size for area of rectangle annotation, same for circle but min radius)
    supers, parts, status = [], [], ['Pass', 'Fail']  # Lists of classification
    category: dict = {'Supercategory': supers, 'Name': parts, 'Status': status}  # connect chosen category to correct list
    # TODO: name_index
    # name_index: int = 0  # Hold's the index for Annotation name presentation
    attributes = [[sg.Text('Choose', background_color='#181c21'),  # UI for attributes
                   # readonly - can't type in
                   sg.Combo(list(category.keys()), default_value='Supercategory', readonly=True,
                            size=(15, 1), enable_events=True, key='-CATEGORY-', background_color='#181c21',
                            button_arrow_color='#FFFFFF', text_color='#FFFFFF', button_background_color='#181c21'),
                   sg.In(size=(20, 1), enable_events=True, key='-ADDING NAME-', tooltip='Enter new classification'),
                   sg.Button(' + ', key='-PLUS-', size=(2, 1), button_color='#181c21'),
                   sg.Button(' - ', key='-MINOS-', size=(2, 1), button_color='#181c21')],
                  [sg.Text(list(category.keys())[0], background_color='#181c21'),
                   sg.Combo(supers, size=(15, 1), readonly=True, key=list(category.keys())[0],
                            background_color='#181c21', button_arrow_color='#FFFFFF', text_color='#FFFFFF',
                            button_background_color='#181c21'),
                   sg.Text(list(category.keys())[1], background_color='#181c21'),
                   sg.Combo(parts, size=(15, 1), readonly=True, key=list(category.keys())[1],
                            background_color='#181c21', button_arrow_color='#FFFFFF',
                            text_color='#FFFFFF', button_background_color='#181c21'),
                   sg.Text(list(category.keys())[2], background_color='#181c21'),
                   sg.Combo(status, default_value=status[0], size=(15, 1), readonly=True, key=list(category.keys())[2],
                            background_color='#181c21', button_arrow_color='#FFFFFF',
                            text_color='#FFFFFF', button_background_color='#181c21')]]
    impo_coco_f = [[sg.Text('Coco file', background_color='#181c21'), sg.Input(key='-IMPORT DIR-', size=(40, 1)),
                    sg.FileBrowse('Browse', file_types=(("ALL Files", "json"),),  # read only 'json' files
                                  button_color='#181c21'),
                    sg.Button(' Import ', font=2, key='-IMPORT-', button_color='#181c21')]
                   ]
    expo_coco_f = [[sg.Text('Name', background_color='#181c21'), sg.Input(key='-COCO NAME-', size=(40, 1)),
                    sg.Button('  Export  ', font=8, key='-SAVE-', button_color='#181c21')],
                   [sg.Text('Path', background_color='#181c21'), sg.In(size=(40, 1), key='-COCO DIR-'),
                    sg.FolderBrowse('    Browse    ', button_color='#181c21')]
                   ]

    # auto_anno_f = [[sg.Button('  Auto annotate current image ', font=8, key='-AUTO ANNO-', button_color='#181c21')
    #                ]]

    # General Ui for the current attributes
    left_col = [[sg.Frame('Import COCO', impo_coco_f, size=(570, 60), background_color='#181c21')],
                [sg.Text('Folder', background_color='#181c21'),
                 sg.In(size=(30, 1), enable_events=True, key='-FOLDER-', disabled=True),
                 sg.FilesBrowse('Add files', file_types=(("ALL Files", utilities.okfiletypes()),),
                                button_color='#181c21'),
                 sg.Button('Delete file', key='-DEL FILE-', button_color='#181c21'),
                 sg.Button('Delete all files', key='-DEL ALL FILE-')],
                [sg.Listbox(values=[], enable_events=True, size=(40, 20), key='-IMAGE LIST-',
                            background_color='#181c21', sbar_background_color='#181c21', text_color='#FFFFFF')],
                [sg.Button('  Auto annotate current image ', font=8, key='-AUTO ANNO-', button_color='#181c21')],
                [sg.Text('Select Shape', background_color='#181c21'),
                 sg.Button(image_filename='AnnotationAssistant/images/RedRectangle.PNG', image_subsample=11, tooltip='Rectangle',
                           key='-RECTANGLE-'),
                 sg.Button(image_filename='AnnotationAssistant/images/Circle.PNG', image_subsample=11, tooltip='Circle',
                           key='-CIRCLE-'),
                 sg.Button(image_filename='AnnotationAssistant/images/Polygon.PNG', image_subsample=11, tooltip='Polygon',
                           key='-POLYGON-')],
                [sg.Frame('Adding attributes', attributes, background_color='#181c21')],  # attributes frame
                [sg.Frame('Export COCO', expo_coco_f, size=(570, 100), background_color='#181c21')],
                # [sg.Frame('Auto Annotation', auto_anno_f, size=(570, 100), background_color='#181c21')]
                ]
    curr_att = [[sg.Text(list(category.keys())[0], background_color='#181c21'),
                 sg.Combo(supers, size=(15, 1), readonly=True, key='cur ' + list(category.keys())[0],
                          background_color='#181c21', button_arrow_color='#FFFFFF',
                          text_color='#FFFFFF', button_background_color='#181c21'),
                 sg.Text(list(category.keys())[1], background_color='#181c21'),
                 sg.Combo(parts, size=(15, 1), readonly=True, key='cur ' + list(category.keys())[1],
                          background_color='#181c21', button_arrow_color='#FFFFFF',
                          text_color='#FFFFFF', button_background_color='#181c21'),
                 sg.Text(list(category.keys())[2], background_color='#181c21'),
                 sg.Combo(status, size=(15, 1), readonly=True, key='cur ' + list(category.keys())[2],
                          background_color='#181c21', button_arrow_color='#FFFFFF',
                          text_color='#FFFFFF', button_background_color='#181c21'),
                 sg.Button('Update', key='-UPDATE CUR ATT-', button_color='#181c21')]]
    right_col = [[sg.Push(background_color='#181c21'),
                  sg.Text('Draw annotation on the selected image', font=[6],
                          background_color='#181c21', text_color='#FFFFFF'),
                  sg.Push(background_color='#181c21')],
                 [sg.Graph(
                     canvas_size=canvas_size,
                     graph_bottom_left=(0, canvas_size[1]),
                     graph_top_right=(canvas_size[0], 0),
                     key="-GRAPH-",
                     enable_events=True,
                     background_color='#181c21',
                     drag_submits=True,
                     motion_events=True,
                     right_click_menu=[[], ['Erase attribute', 'Erase all attributes']])],
                 [sg.Text(key='-INFO-', background_color='#181c21')],
                 [sg.Frame('Current attributes', curr_att, visible=True, key='-CURR ATT-',
                           background_color='#181c21')]]

    layout = [[sg.Col(left_col, key='-LEFT COL-', background_color='#181c21'),
               sg.VSeparator(),
               sg.Col(right_col, key='-RIGHT COL-', background_color='#181c21')
               ]]
    # GUI object
    window = sg.Window("Bright Machines Annotation (BMA) tool", layout, icon='bm_logo.ico',
                       finalize=True, return_keyboard_events=True, background_color='#181c21')
    images: list = []  # will hold "Image" class objects
    im_dir_l: list = []  # will hold all the dir for all the images
    im_name_l: list = []  # will hold all the file names for all the images
    im_dic: dict = {}  # dict that connect file name to dir
    new_images = False  # indicates if a new set of images was added
    # get the graph element for ease of use later
    graph = window["-GRAPH-"]  # For easy use
    anno_shape = 'Rectangle'  # holds the shap of the annotation the user want to make
    bypass = False  # if 'Esc' key had been pressed switch to True
    dragging: bool = False  # This will indicate if an annotation is being drawn at the moment
    stretching: bool = False  # This will indicate if an annotation is being stretches at the moment
    point = None  # the selected red point from stretching function
    prior = None  # this will hold the current anno_obj
    first_line = False  # Indicate if the first line of the polygon was made
    poly_points = []  # all the points that define the polygon
    start_p = end_p = None, None
    cur_im = None  # Hold's the current image object

    # debugger
    """
    sg.change_look_and_feel('BlueMono')
    layout = [
        [sg.T('A typical PSG application')],
        [sg.In(key='_IN_')],
        [sg.T('        ', key='_OUT_', size=(30, 1))],
        [sg.Radio('a', 1, key='_R1_'), sg.Radio('b', 1, key='_R2_'), sg.Radio('c', 1, key='_R3_')],
        [sg.Combo(['c1', 'c2', 'c3'], size=(6, 3), key='_COMBO_')],
        [sg.Output(size=(50, 6))],
        [sg.Ok(), sg.Exit(), sg.Button('Debug'), sg.Button('Popout')],
    ]

    windowd = sg.Window('This is your Application Window', layout)

    counter = 0
    timeout = 100
    """
    att: bool = False  # this is fixing the jump of the presented image when the first annotations is being pressed
    while True:
        event, values = window.read(timeout=20)
        if event == sg.WIN_CLOSED:
            break
        elif event == '-SAVE-':
            if len(images) == 0:
                sg.popup_error('Pleas enter images and annotations before clicking the save button',
                               background_color='#181c21')
            elif values['-COCO DIR-'] == '' or values['-COCO NAME-'] == '':
                sg.popup_error('Pleas enter export location an name for export file', background_color='#181c21')
            else:
                export_coco(images, values['-COCO DIR-'], values['-COCO NAME-'], list(im_dic.values()))
        elif event == '-IMPORT-':
            if values['-IMPORT DIR-'] == '':
                sg.popup_error('Please enter a Coco (json format) file direction', background_color='#181c21')
            else:
                graph.erase()  # clear graph
                # 'import_coco' will return all is needed to load all data from coco
                # images, im_name_l, im_dir_l, im_dic, category['supercategory'], \
                # category['Part'], category['Status'] = import_coco(values['-IMPORT DIR-'], canvas_size)
                n_images, n_im_name_l, n_im_dir_l, n_im_dic, n_super_class, \
                n_part, n_status = import_coco(values['-IMPORT DIR-'], canvas_size)
                # add values to current values
                images = n_images + images
                im_name_l = n_im_name_l + im_name_l
                im_dir_l = n_im_dir_l + im_dir_l
                im_dic.update(n_im_dic)
                # only unique values need to be added
                category['Supercategory'] = list(set(n_super_class + category['Supercategory']))
                category['Name'] = list(set(n_part + category['Name']))
                category['Status'] = list(set(n_status + category['Status']))
                window["-IMAGE LIST-"].update(  # updating the image list
                    values=['[' + str(int(i + 1)) + ']  ' + im_name_l[i] for i in range(0, len(im_name_l))],
                    set_to_index=0)
                new_images = True  # will present the first image in the new images set
                for cat in list(category.keys()):  # updating the categories that are in use in coco
                    window[cat].update(values=category[cat], set_to_index=0)
                    window['cur ' + cat].update(values=category[cat], set_to_index=0)
                sg.popup('Import is done', background_color='#181c21')

        elif event == '-AUTO ANNO-':
            # Lets export only the current image as coco format for easy loading in Auto Annotator
            # check if the image is in a folder inside images folder
            # if not than AA should not be preformed
            if not cur_im:
                sg.popup_error("please select an image for Auto Annotation to do the magic")
            elif not cur_im.anno:
                sg.popup_error("please annotate image before Auto Annotation")
            elif cur_im.path.split('\\')[-2] == 'images':
                sg.popup_error("Auto Annotation will be preformed only in sub folder inside images folder")
            else:
                export_coco([cur_im], os.path.dirname(os.path.dirname(cur_im.path)), 'temp_coco')
                temp_coco_dir: str = os.path.join(os.path.dirname(os.path.dirname(cur_im.path)), 'temp_coco.json')
                AA = AutoAnnotator(temp_coco_dir)  # create Auto Annotator class
                # name of the folder with all the images to be annotated
                images_folder_name: str = os.path.basename(os.path.dirname(cur_im.path))
                message = AA.process(images_folder_name)
                print(Fore.GREEN if message == "All good" else Fore.RED + "Annotations message - {}".format(message))
                if message == 'All good': # Make Auto Annotation
                    graph.erase()  # clear graph
                    n_images, n_im_name_l, n_im_dir_l, n_im_dic, n_super_class, \
                    n_part, n_status = import_coco(file_dir=AA.coco.original_coco_dir, resize=canvas_size)
                    os.remove(AA.coco.original_coco_dir)  # removing temp file, not needed anymore
                    # add values to current values
                    images = n_images + images
                    im_name_l = n_im_name_l + im_name_l
                    im_dir_l = n_im_dir_l + im_dir_l
                    im_dic.update(n_im_dic)
                    # only unique values need to be added
                    category['Supercategory'] = list(set(n_super_class + category['Supercategory']))
                    category['Name'] = list(set(n_part + category['Name']))
                    category['Status'] = list(set(n_status + category['Status']))
                    window["-IMAGE LIST-"].update(  # updating the image list
                        values=['[' + str(int(i + 1)) + ']  ' + im_name_l[i] for i in range(0, len(im_name_l))],
                        set_to_index=0)
                    new_images = True  # will present the first image in the new images set
                    sg.popup("Auto Annotation was done, please check outcome")
                else:
                    sg.popup_error(message)

                for cat in list(category.keys()):  # updating the categories that are in use in coco
                    window[cat].update(values=category[cat], set_to_index=0)
                    window['cur ' + cat].update(values=category[cat], set_to_index=0)
                sg.popup('Import is done', background_color='#181c21')

        elif event == "-FOLDER-":  # Show image names in the "-IMAGE LIST-" list box
            t_im_dir_l: list = values["-FOLDER-"]  # Gets all wanted image dir as string
            t_im_dir_l: list = t_im_dir_l.rsplit(";")  # Convert string into list

            t_im_dir_l = [s if s.count('/') == 0 else s.replace("/", "\\") for s in t_im_dir_l]  # replace / to \\.

            i = 0
            while i < len(t_im_dir_l):  # Check if dir all ready exist if it does, we don't want it
                if t_im_dir_l[i] in im_dir_l:
                    t_im_dir_l.remove(t_im_dir_l[i])
                    i -= 1
                else:  # Adding new images to image list Image object, scale is not set...
                    img = PIL.Image.open(t_im_dir_l[i])
                    images.append(Image(path=t_im_dir_l[i],
                                        name=os.path.basename(t_im_dir_l[i]),
                                        width=img.size[0], height=img.size[1]))
                    # Only if new images are in
                    new_images = True  # will present the first image in the new images set
                i += 1
            im_dir_l += t_im_dir_l
            im_name_l += [os.path.basename(i) for i in t_im_dir_l]  # Gets only the file name
            im_dic.update(dict(zip(im_name_l, im_dir_l)))  # dict that connect file name to dir
            # will put list of file names in the Listbox with  numbering in the beginning
            window["-IMAGE LIST-"].update(
                values=['[' + str(int(i + 1)) + ']  ' + im_name_l[i] for i in range(0, len(im_name_l))])
            window['-IMAGE LIST-'].update(set_to_index=len(im_dir_l) - len(t_im_dir_l))
            """  Making an image appear in the beginning
            images.append(Image(im_dic[im_name_l[0]], im_name_l[0]))  # Adding the first im images list : type Image
            cur_im = images[-1]  # This is the current image we are working on
            # Scale = scale factor from original to new
            # Converts the incoming im to PIL file and resize to fit canvas size
            im_show, scale = Image.show_im(cur_im, resize=canvas_size)
            image_el = graph.draw_image(data=im_show, location=(0, 0))  # Put's the images on the graph
            """
        elif event == '-DEL FILE-':  # Deleting image from image list
            if 'cur_im' in locals():
                images.remove(cur_im)
                im_dir_l.remove(cur_im.path)
                im_name_l.remove(cur_im.name)
                print('"{}" was removed from image list'.format(cur_im.name))
                im_dic.pop(cur_im.name)
                window["-IMAGE LIST-"].update(
                    values=['[' + str(int(i + 1)) + ']  ' + im_name_l[i] for i in range(0, len(im_name_l))])
                graph.delete_figure(image_el)
                for an in cur_im.anno:  # deleting annotations from graph
                    graph.delete_figure(an.anno_obj)
        elif event == '-DEL ALL FILE-':  # Delete all images from image list:
            images = []
            im_dir_l = []
            im_name_l = []
            im_dic = {}
            window["-IMAGE LIST-"].update(values='')
            graph.erase()
        elif event == '-IMAGE LIST-' or new_images:  # A file was chosen from the listbox
            new_images = False
            for im in images:  # Finding the wanted image in the Image object list
                if im.name == values['-IMAGE LIST-'][0].split(None, 1)[1]:  # if the wanted image
                    cur_im = im
                    # Scale = scale factor from original to new
                    # Converts the incoming im to PIL file and resize to fit canvas size
                    im_show, scale = Image.show_im(cur_im, resize=canvas_size)
                    cur_im.scale = scale  # Saving images scale ratio to graph
                    image_el = Image.load_im(cur_im, graph, canvas_size)  # load up the image and annotations
                    print('"{}" image was selected'.format(cur_im.name))
                    break  # exit
        # Selecting annotation shape - gives the 'anno_shape' variable the right string and colors the right shape in
        # red by uploading the right image
        elif event == '-RECTANGLE-':
            window['-RECTANGLE-'].update(image_filename='AnnotationAssistant/images/RedRectangle.PNG',
                                         image_subsample=11)
            anno_shape = 'Rectangle'
            window['-CIRCLE-'].update(image_filename='AnnotationAssistant/images/Circle.PNG',
                                      image_subsample=11)
            window['-POLYGON-'].update(image_filename='AnnotationAssistant/images/Polygon.PNG',
                                       image_subsample=11)
        elif event == '-CIRCLE-':
            window['-CIRCLE-'].update(image_filename='AnnotationAssistant/images/RedCircle.PNG',
                                      image_subsample=11)
            anno_shape = 'Circle'
            window['-RECTANGLE-'].update(image_filename='AnnotationAssistant/images/Rectangle.PNG',
                                         image_subsample=11)
            window['-POLYGON-'].update(image_filename='AnnotationAssistant/images/Polygon.PNG',
                                       image_subsample=11)
        elif event == '-POLYGON-':
            window['-POLYGON-'].update(image_filename='AnnotationAssistant/images/RedPolygon.PNG',
                                       image_subsample=11)
            anno_shape = 'Polygon'
            window['-RECTANGLE-'].update(image_filename='AnnotationAssistant/images/Rectangle.PNG',
                                         image_subsample=11)
            window['-CIRCLE-'].update(image_filename='AnnotationAssistant/images/Circle.PNG',
                                      image_subsample=11)
        # if we want the graph to be active
        if 'image_el' in locals():
            if values[list(category.keys())[0]] and \
                    values[list(category.keys())[1]] and values[list(category.keys())[2]]:
                if 'no_classi' in locals():  # dismiss warning of no classification
                    graph.delete_figure(no_classi)
                # check if not is the middle of making an annotation
                if not dragging and event == "-GRAPH-" and not stretching:
                    # if an annotation had been selected showing its info and returning the annotation object
                    cur_im.anno_info(window, graph, values['-GRAPH-'], category)
                    if not cur_im.cur_anno:
                        cur_im.colord_anno(graph=graph, color_m='black')  # color all annotation in black
                # if an all ready exist annotation was selected enter and don't go to next if's
                # and if the mouse is moving, move the figure
                if cur_im.cur_anno and event != 'Erase attribute':
                    # prior = new_anno
                    stretching, start_p, point, prior = cur_im.stretching_anno(graph, values['-GRAPH-'],
                                                                               stretching, start_p, event, point, prior)
                    # check if an annotation is being moved
                    if cur_im.moving_anno and not stretching:
                        # if an annotation was selected let's check if its moving' and if it does let's move it
                        dragging, start_p = cur_im.move_anno(window, graph, event,
                                                             values, dragging, start_p)
                elif anno_shape == 'Polygon' and 'image_el' in locals():  # if Polygon anno_obj
                    # if enter in keyboard was pressed
                    if event in ('\r', 'special 16777220', 'special 16777221') and 'priors' in locals():
                        for prior in priors:
                            graph.delete_figure(prior)  # delete all the lines from old polygon
                        if len(poly_points) > 2:  # Check that polygon has more than two points
                            priors = cur_im.drawpoly(graph, poly_points)  # Makes the realtime polygon

                            cur_im.add_anno(values['Supercategory'], values['Name'],
                                            values['Status'], anno_shape, priors, poly_points)
                            print('new polygon annotation in position {}'.format(poly_points))
                        else:
                            sg.popup_error("Polygon need's more the two points to exist", background_color='#181c21')
                        # restart variables
                        start_p = end_p = prior = None
                        del priors
                        dragging = first_line = False
                        poly_points = []
                    if event == "-GRAPH-" and not dragging:  # the first point
                        x, y = values['-GRAPH-']
                        start_p = [x, y]
                        poly_points.append(start_p)  # Adding first point to point's list
                        dragging = True  # first point is set
                        # second point is being set by mouse press
                    elif event == "-GRAPH-" and dragging and prior and not first_line:
                        poly_points.append(end_p)
                        end_p = None
                        first_line = True  # this will indicate second point is in polygon
                    # from third point (including) and on by mouse press
                    elif event == "-GRAPH-" and dragging and first_line and end_p:
                        if end_p != poly_points[-1]:
                            poly_points.append(end_p)  # Add point to polygon
                            print(end_p)
                    elif event.endswith('+MOVE') and dragging:  # Real time polygon mouse is not being pressed
                        x, y = values['-GRAPH-']  # Gets current point mouse is havering on
                        end_p = [x, y]
                        window["-INFO-"].update(  # put's pixel location under graph
                            value='mouse x:{} , y:{}'.format(int(x / cur_im.scale), int(y / cur_im.scale)))
                        if None not in (start_p[0], end_p[0], start_p[1], end_p[1]):
                            if prior:  # for first line
                                graph.delete_figure(prior)  # delete the old line
                            if 'priors' in locals():  # for polygon
                                for prior in priors:
                                    graph.delete_figure(prior)  # delete all the lines from old polygon
                            if first_line:
                                priors = cur_im.drawpoly(graph, poly_points + [end_p])  # Makes the realtime polygon
                            else:
                                prior = graph.draw_line(start_p, end_p, width=5)
                    # 'esc' key was pressed, erase the anno in building
                    if event == 'Escape:27' and 'start_p' in locals():
                        if 'priors' in locals():  # if there is more than one line
                            for prior in priors:
                                graph.delete_figure(prior)  # delete all the lines from old polygon
                            del priors
                        else:
                            graph.delete_figure(prior)
                        # restart variables
                        start_p = end_p = prior = None
                        dragging = first_line = False
                        poly_points = []
                    if event in ('\r', 'special 16777220', 'special 16777221'):
                        print(poly_points)

                # if there's a "Graph" event, then it's a mouse
                elif event == "-GRAPH-" and 'image_el' in locals() and not bypass and not stretching:
                    x, y = values["-GRAPH-"]
                    if not dragging:  # The mouse was pressed just now
                        start_p = [x, y]  # Beginning point
                        dragging = True  # Next points are not beginning anymore
                    else:
                        end_p = [x, y]  # In the end will hold the end point
                        if prior:  # there is a rec all ready
                            graph.delete_figure(prior)  # delete the last rec
                        if None not in (start_p[0], end_p[0], start_p[1], end_p[1]):
                            if prior:  # there is a rec all ready
                                graph.delete_figure(prior)  # delete the last rec
                            if anno_shape == 'Rectangle':
                                prior = graph.draw_rectangle(start_p, end_p, line_color='black', line_width=5)
                            elif anno_shape == 'Circle':
                                prior = graph.draw_circle(start_p,
                                                          ((start_p[0] - end_p[0]) ** 2 + (
                                                                  start_p[1] - end_p[1]) ** 2) ** 0.5,
                                                          line_color='black', line_width=5)
                        # Show's location of mouse real time when left click is pressed
                    window["-INFO-"].update(
                        value='mouse x:{} , y:{}'.format(int(x / cur_im.scale), int(y / cur_im.scale)))
                # if the mouse is now longer being pressed not for Polygon
                elif event.endswith('+UP') and 'image_el' in locals() and anno_shape != 'Polygon' and dragging:
                    graph.delete_figure(prior)
                    # drawing the anno_obj
                    if anno_shape == 'Circle':
                        radius = ((start_p[0] - end_p[0]) ** 2 + (start_p[1] - end_p[1]) ** 2) ** 0.5
                        if radius < anno_min_size[1]:  # minimum size for circle
                            sg.popup_error('Enter a bigger size boundary box', background_color='#181c21')
                            bypass = True
                        else:
                            bbox = (start_p, radius)
                            anno_obj = graph.draw_circle(start_p, radius, line_color='black', line_width=5)
                    elif anno_shape == 'Rectangle':
                        if start_p[0] >= end_p[0]:
                            temp = end_p[0]
                            end_p[0] = start_p[0]
                            start_p[0] = temp
                        if start_p[1] > end_p[1]:
                            temp = end_p[1]
                            end_p[1] = start_p[1]
                            start_p[1] = temp
                        # minimum size for rectangle
                        if (end_p[0] - start_p[0]) * (end_p[1] - start_p[1]) < anno_min_size[0]:
                            sg.popup_error('Enter a bigger size boundary box', background_color='#181c21')
                            bypass = True
                        else:
                            bbox = (start_p, end_p)
                            anno_obj = graph.draw_rectangle(start_p, end_p, line_color='black', line_width=5)
                    if not bypass:  # if size of bbox is OK
                        cur_im.add_anno(values['Supercategory'], values['Name'],
                                        values['Status'], anno_shape, anno_obj, bbox)
                        print('new {} annotation in position {}'.format(anno_shape, bbox))
                    dragging, bypass = False, False  # restart variable
                    start_p, end_p, prior = None, None, None
                if event == 'Escape:27' and dragging:  # 'esc' key was pressed, erase the anno in building
                    graph.delete_figure(prior)
                    dragging = False  # restart variable
                    start_p, end_p, prior = None, None, None
                    bypass = True
                if event.endswith('+UP') and bypass:  # the mouse is up and al back to normal
                    bypass = False
            elif event == '-GRAPH-' and 'no_classi' not in locals():  # write on image warning if no classification was
                # entered
                no_classi = graph.draw_text('please select attributes classification \n before drawing annotation'
                                            ' on image', color='red',
                                            location=(canvas_size[0] / 2, canvas_size[1] / 2), font=[25], angle=30)

        elif not att:  # this is fixing the jump of the presented image when the first annotations is being pressed
            window['-CURR ATT-'].update(visible=False)
            att = True
        # deleting annotation - need to be if and not elif because event == 'graph' is happening as well
        if event == 'Erase attribute':  # Deleting annotation with right click
            cur_im.del_anno(canvas_size, graph, values['-GRAPH-'])
            window['-CURR ATT-'].update(visible=False)
        if event == 'Erase all attributes':  # Deleting all annotation with right click
            for i in range(0, len(cur_im.anno)):  # Running on all list of annotations
                cur_im.anno.pop(0)
            cur_im.load_im(graph, canvas_size)

        ### Attributes ###
        if event == '-PLUS-' and values['-CATEGORY-'] and values['-ADDING NAME-']:  # Adding a new classification
            if values['-ADDING NAME-'] not in category[values['-CATEGORY-']]:  # if name doesn't exist
                # Adding the name to selecting attributes
                category[values['-CATEGORY-']].append(values['-ADDING NAME-'])
                window[values['-CATEGORY-']].update(values=category[values['-CATEGORY-']], set_to_index=0)
                # Adding the name to current attributes
                window['cur ' + values['-CATEGORY-']].update(values=category[values['-CATEGORY-']], set_to_index=0)
        if event == '-MINOS-' and values['-CATEGORY-'] and values['-ADDING NAME-']:  # Deleting a new classification
            if values['-ADDING NAME-'] in category[values['-CATEGORY-']]:
                category[values['-CATEGORY-']].remove(values['-ADDING NAME-'])
                window[values['-CATEGORY-']].update(values=category[values['-CATEGORY-']], set_to_index=0)
                name_index = 0
        if event == '-UPDATE CUR ATT-' and cur_im.cur_anno:
            cur_im.cur_anno.superclass = values['cur ' + list(category.keys())[0]]
            cur_im.cur_anno.part = values['cur ' + list(category.keys())[1]]
            cur_im.cur_anno.status = values['cur ' + list(category.keys())[2]]
            print('Update current annotation to {}, {}, {}'.format(values['cur ' + list(category.keys())[0]],
                                                                   values['cur ' + list(category.keys())[1]],
                                                                   values['cur ' + list(category.keys())[2]]))

        # debugger
        """
        event, values = windowd.read(timeout=timeout)
        if event in (None, 'Exit'):
            break
        elif event == 'Ok':
            print('You clicked Ok.... this is where print output goes')
        elif event == 'Debug':
            imwatchingyou.show_debugger_window()  # STEP 2
        elif event == 'Popout':
            imwatchingyou.show_debugger_popout_window()  # STEP 2
        counter += 1
        # to prove window is operating, show the input in another area in the window.
        windowd['_OUT_'].update(values['_IN_'])

        # don't worry about the "state" of things, just call this function "frequently"
        imwatchingyou.refresh_debugger()  # STEP 3 - refresh debugger
        """


if __name__ == '__main__':
    gui()
