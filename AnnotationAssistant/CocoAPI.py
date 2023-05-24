from termcolor import colored
from typing import Union


class CocoInfo:
    """
    Struct object for storing Info section of Coco file:

    year: int
    version: str
    description: str
    contributor: str
    url: str
    date_created : str

    """

    def __init__(self, data: Union[None, dict] = None):
        # print(f'Initiating {self.__class__.__name__} object...', end=" ")

        self.year: int = 0  # data['year']
        self.version: str = "0.0"  # data['version']
        self.description: str = ""  # data['description']
        self.contributor: str = ""  # data['contributor']
        self.url: str = ""  # data['url']
        self.date_created: str = ""  # data['date_created']
        # print(self)
        if data is not None:
            self.init(data)

    def __str__(self):
        s:str = f'{self.__class__.__name__} Obj: '
        for k, v in self.__dict__.items():
            s += f'{k}={v}, '
        return s

    def init(self, data: dict):
        self.year = data['year']
        self.version = data['version']
        self.description = data['description']
        self.contributor = data['contributor']
        self.url = data['url']
        self.date_created = data['date_created']


class CocoImage:
    """
    Struct object for storing Image section of Coco file:

    id: int
    width: int
    height: int
    file_name: str
    license: int
    date_captured: str
    """

    def __init__(self, data: dict):
        self.id: int = data['id']  # 1
        self.width: int = data['width']  # 3088
        self.height: int = data['height']  # 2064
        self.file_name: str = data['file_name'].split("\\")[-1]  # "LB001.jpg"
        self.license: int = data['license']  # 0
        self.date_captured: str = data['date_captured']  # ""


class CocoAnnotation:
    """
    Struct object for storing Annotation section of Coco file:

    segmentation: list[float]
    area: int
    bbox: list[int]
    iscrowd: int
    id: int
    image_id: int
    category_id: int
    """

    def __init__(self, data: dict):
        self.id: int = data["id"]  # 1
        self.category_id: int = data["category_id"]  # 1
        self.image_id: int = data["image_id"]  # 1
        self.bbox: list[int] = data["bbox"]  # [663, 439, 142, 134]
        self.area: int = data["area"]  # 19028
        self.segmentation: list = data["segmentation"]  # [[663, 439, 805, 439, 805, 573, 663, 573]]
        self.iscrowd: int = data["iscrowd"]  # 0


    def to_dict(self) -> dict:
        d: dict = {
                    "id": self.id,
                    "category_id": self.category_id,
                    "image_id": self.image_id,
                    "bbox": self.bbox,
                    "area": self.area,
                    "segmentation": self.segmentation,
                    "iscrowd": self.iscrowd
                  }
        return d


class CocoCategory:
    """
    Struct object for storing Category section of Coco file:

    supercategory: str
    id: int
    name: str
    """

    def __init__(self, data: dict):

        self.id: int = data['id']
        self.supercategory: str = data['supercategory']
        self.name: str = data['name']
        self.status: str = data['status']


class CocoData:
    """
    Struct object for storing all parsed data from Coco file. ordered in classes:
    (1) info: CocoInfo -> for Info Section
    (2) images: list[CocoImage] -> for Image Section. stored in list.
    (3) annotations: list[CocoAnnotation]  -> for Annotation Section. stored in list.
    (4) categories: list[CocoCategory]   -> for Category Section. stored in list.
    (5) file: str = file -> the coco.json file name directory. parsed from.

    """

    def __init__(self, data: Union[None, dict] = None, file: str = ""):
        print(colored(f"Initiating {self.__class__.__name__} object...", 'cyan'))

        self.info: CocoInfo = CocoInfo()
        self.images: list[CocoImage] = []
        self.annotations: list[CocoAnnotation] = []
        self.categories: list[CocoCategory] = []
        self.file: str = ""

        if data is not None:
            self.init(data, file)

    def init(self, data: dict, file: str = ""):
        self.info.init(data['info'])  # = CocoInfo(data['info'])
        print('\t', self.info)

        self.images = [CocoImage(d) for d in data['images']]
        print('\t', colored('Images', 'yellow'), f' = {len(self.images)}')

        self.annotations = [CocoAnnotation(d) for d in data['annotations']]
        print('\t', colored('Annotations', 'yellow'), f' = {len(self.annotations)}')

        self.categories = [CocoCategory(d) for d in data['categories']]
        print('\t', colored('Categories', 'yellow'), f' = {len(self.categories)}')

        self.file: str = file
