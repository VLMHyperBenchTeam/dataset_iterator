from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeVar


@dataclass
class AbstractSample(ABC):
    """ Реализует абстрактный класс одного объекта датасета. По умолчанию пуст. """
    id: int


# Определяем TypeVar с ограничением на AbstractAnswer и его наследников
TSample = TypeVar('T', bound=AbstractSample)


class AbstractIterator(ABC):
    """Класс абстрактного итератора для получения сэмпла из датасета. """

    def __init__(self, task_name:str, dataset_name:str, start=0, filter_doc_class=None, filter_question_type=None, 
                 dataset_dir_path='/data', csv_file='annotation.csv'):
        self.dataset_name = dataset_name
        self.row_index = start
        self.task_name = task_name
        self.filter_doc_class = filter_doc_class
        self.filter_question_type = filter_question_type
        self.dataset_dir_path = dataset_dir_path
        self.csv_file = csv_file
        self._read_data()

    @abstractmethod
    def _read_data(self):
        """Загружает таблицу с аннотацией данных. """
        pass

    def __iter__(self):
        return self

    @abstractmethod
    def __next__(self) -> TSample:
        pass
