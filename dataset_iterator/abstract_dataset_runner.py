import os
from datetime import datetime

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
        answers_dir_path (str): Путь к директории для сохранения ответов. По умолчанию "/workspace/answers".
        csv_name (str): Имя CSV-файла для сохранения ответов. По умолчанию "annotation.csv".
    """

    def __init__(self, iterator: TIterator, model: Any, answers_dir_path: str = "/workspace/answers", 
                 csv_name: str = None) -> None:
        """Инициализирует экземпляр AbstractDatasetRunner.

        Аргументы:
            iterator (TIterator): Итератор, который предоставляет доступ к данным датасета.
            model (ModelInterface): VLM-модель, которая будет использоваться для получения ответа.
            answers_dir_path (str): Путь к директории для сохранения ответов. По умолчанию "/workspace/answers".
            csv_name (str): Имя CSV-файла для сохранения ответов. По умолчанию None и задаётся динамически согласно атрибутам класса.
        """
        self.iterator = iterator
        self.model = model
        self.model_answers = []
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

    def get_answer_filename(self) -> str:
        """Генерирует путь до файла с ответами модели.
        """
        # TODO: получить Modelframework из свойств модели 
        os.makedirs(self.answers_dir_path, exist_ok=True)  # Создаем директорию, если её нет
        timestamp = datetime.now().strftime(r"%Y%m%d_%H%M%S")  # Формат: ГГГГММДД_ЧЧММСС
        save_path = os.path.join(
            self.answers_dir_path,
            f"{self.iterator.dataset_name}_MODELFRAMEWORK_{self.model.model_name}_{self.iterator.task_name}_answers_{timestamp}.csv"
        )
        return save_path

    @abstractmethod
    def save_answers(self) -> str:
        """Сохраняет ответы в CSV-файл под названием csv_name по пути self.answers_dir_path.
        Возвращает путь до сохранённого файла с ответами модели.
        Этот метод должен быть реализован в подклассах.
        """
        pass