from abc import ABC, abstractmethod
from typing import TypeVar, Any

from .abstract_iterator import AbstractIterator, TSample

TIterator = TypeVar('TIterator', bound=AbstractIterator)


class AbstractDatasetRunner(ABC):
    """Абстрактный класс, реализующий логику прогона модели по датасету.

    Атрибуты:
        iterator (TIterator): Итератор, который предоставляет доступ к данным датасета.
        model (ModelInterface): VLM-модель, которая будет использоваться для получения ответа.
        model_answers (list): Список для хранения ответов модели.
        dataset_dir_path (str): Путь к директории с датасетом. По умолчанию "/workspace/data".
        answers_dir_path (str): Путь к директории для сохранения ответов. По умолчанию "/workspace/answers".
        csv_name (str): Имя CSV-файла для сохранения ответов. По умолчанию "annotation.csv".
    """

    def __init__(self, iterator: TIterator, model: Any, dataset_dir_path: str = "/workspace/data",
                 answers_dir_path: str = "/workspace/answers", csv_name: str = "annotation.csv") -> None:
        """Инициализирует экземпляр AbstractDatasetRunner.

        Аргументы:
            iterator (TIterator): Итератор, который предоставляет доступ к данным датасета.
            model (ModelInterface): VLM-модель, которая будет использоваться для получения ответа.
            dataset_dir_path (str): Путь к директории с датасетом. По умолчанию "/workspace/data".
            answers_dir_path (str): Путь к директории для сохранения ответов. По умолчанию "/workspace/answers".
            csv_name (str): Имя CSV-файла для сохранения ответов. По умолчанию "annotation.csv".
        """
        self.iterator = iterator
        self.model = model
        self.model_answers = []
        self.dataset_dir_path = dataset_dir_path
        self.answers_dir_path = answers_dir_path
        self.csv_name = csv_name

    @abstractmethod
    def run(self) -> None:
        """Осуществляет прогон модели по датасету, собирает ответы и записывает их на диск.

        Этот метод должен быть реализован в подклассах.
        """
        pass

    @abstractmethod
    def add_answer(self, sample: TSample, answer: Any) -> None:
        """Добавляет ответ модели в список ответов.

        Аргументы:
            sample (TSample): Образец данных из датасета.
            answer (Any): Ответ модели на данный образец.

        Этот метод должен быть реализован в подклассах.
        """
        pass

    @abstractmethod
    def save_answers(self) -> None:
        """Сохраняет ответы в CSV-файл под названием csv_name по пути self.answers_dir_path.

        Этот метод должен быть реализован в подклассах.
        """
        pass