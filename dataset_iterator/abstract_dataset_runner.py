from abc import ABC, abstractmethod
from typing import TypeVar

from .abstract_iterator import AbstractIterator, TSample

TIterator = TypeVar('T', bound=AbstractIterator)


class AbstractDatasetRunner(ABC):
    """ Абстрактный класс, реализующий логику прогона модели по датасету. """
    def __init__(self, dataset: TIterator, model, dataset_dir_path="/workspace/data",
                 answers_dir_path="/workspace/answers", csv_name="annotation.csv"):
        
        self.dataset = dataset
        self.model = model
        
        self.model_answers = []
        self.dataset_dir_path = dataset_dir_path 
        self.answers_dir_path = answers_dir_path 
        self.csv_name = csv_name 

    @abstractmethod
    def run(self):
        """ Осуществляет прогон модели по датасету, собирает ответы и записывает их на диск. """
        pass

    @abstractmethod
    def add_answer(self, sample: TSample, answer):
        """ Добавляет ответ модели в список ответов """
        pass

    @abstractmethod
    def save_answers(self):
        """ Сохраняет ответы в CSV-файл по пути self.answers_dir_path. """
        pass
