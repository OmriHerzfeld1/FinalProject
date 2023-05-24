import numpy as np
import cv2 as cv
import utilities
from termcolor import colored


class DataSet:
    _img_idx: int = 0

    def __init__(self, name: str, dataPath: str, suffix: str = ""):
        print(colored(f"Initiate {self.__class__.__name__} Obj...", 'cyan'), end=" ")
        self.name: str = name
        self.data_path: str = dataPath
        self.files: list[str] = utilities.load_files(dataPath, suffix)
        self.n_images: int = len(self.files)
        self.img_idx: int = 0
        print(self)

    def get_next(self) -> np.ndarray:
        img: np.ndarray = cv.imread(self.files[self.img_idx])
        self.img_idx += 1
        return img


    def __str__(self):
        return f"{self.__class__.__name__} Info: " \
               f"Name= {colored(self.name, 'yellow')}  " \
               f"Path= {colored(self.data_path, 'yellow')}, " \
               f"#Images= {colored(str(self.n_images), 'yellow')}"




    @property
    def img_idx(self) -> int:
        return self._img_idx

    @img_idx.setter
    def img_idx(self, i: int):
        assert type(i) == int, TypeError(f"index have to be integer, got{type(i)}")
        assert 0 <= i, ValueError(f"Index have to be non-negative, got {i}.")
        self._img_idx = i if i < self.n_images else 0



class DataSetsIterator:
    """ Iterator class """

    def __init__(self, dataset):
        self._dataset: DataSets = dataset  # Team object reference
        self._index: int = 0  # member variable to keep track of current index

    def __next__(self):
        """Returns the next value from team object's lists """
        if self._index < self._dataset.size:
            result = self._dataset.sets[self._index]
            self._index += 1
            return result
        # End of Iteration
        raise StopIteration


class DataSets:

    def __init__(self, config: dict):
        self.sets: list[DataSet] = [DataSet(ds["Name"], ds["Path"]) for ds in config["DataSetList"]]
        self._index: int = 0
        self.size: int = len(self.sets)

    def __iter__(self):
        return DataSetsIterator(self)


if __name__ == "__main__":
    data_path: str = r"C:\BM\Data\WW\Data\images\train"
    data = DataSet("MySet", data_path)

    # Test cycling over files:
    for _ in range(2 * data.n_images):
        print(data.files[data.img_idx])
        data.get_next()
