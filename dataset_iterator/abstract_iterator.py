from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeVar, Optional


@dataclass
class AbstractSample(ABC):
    """Абстрактный класс, представляющий один объект датасета.

    Атрибуты:
        id (int): Уникальный идентификатор объекта датасета.
    """
    id: int


# Определяем TypeVar с ограничением на AbstractSample и его наследников
TSample = TypeVar('TSample', bound=AbstractSample)


class AbstractIterator(ABC):
    """Абстрактный класс итератора для получения сэмплов из датасета.

    Атрибуты:
        dataset_name (str): Название датасета.
        row_index (int): Текущий индекс строки в данных. По умолчанию 0.
        task_name (str): Название задачи.
        filter_doc_class (Optional[str]): Фильтр для класса документа. По умолчанию None.
        filter_question_type (Optional[str]): Фильтр для типа вопроса. По умолчанию None.
        dataset_dir_path (str): Путь к директории с датасетом. По умолчанию '/data'.
        csv_name (str): Имя CSV-файла с аннотацией данных. По умолчанию 'annotation.csv'.
    """

    def __init__(self, task_name: str, dataset_name: str, start: int = 0, 
                 filter_doc_class: Optional[str] = None, filter_question_type: Optional[str] = None, 
                 dataset_dir_path: str = '/data', csv_name: str = 'annotation.csv') -> None:
        """Инициализирует экземпляр AbstractIterator.

        Аргументы:
            task_name (str): Название задачи.
            dataset_name (str): Название датасета.
            start (int): Начальный индекс строки для итерации. По умолчанию 0.
            filter_doc_class (Optional[str]): Фильтр для класса документа. По умолчанию None.
            filter_question_type (Optional[str]): Фильтр для типа вопроса. По умолчанию None.
            dataset_dir_path (str): Путь к директории с датасетом. По умолчанию '/data'.
            csv_name (str): Имя CSV-файла с аннотацией данных. По умолчанию 'annotation.csv'.
        """
        self.dataset_name = dataset_name
        self.row_index = start
        self.task_name = task_name
        self.filter_doc_class = filter_doc_class
        self.filter_question_type = filter_question_type
        self.dataset_dir_path = dataset_dir_path
        self.csv_name = csv_name
        self._read_data()

    @abstractmethod
    def _read_data(self) -> None:
        """Загружает таблицу с аннотацией данных.

        Этот метод должен быть реализован в подклассах.
        """
        pass

    def __iter__(self) -> 'AbstractIterator':
        """Возвращает итератор для обхода датасета.

        Возвращает:
            AbstractIterator: Текущий экземпляр итератора.
        """
        return self

    @abstractmethod
    def __next__(self) -> TSample:
        """Возвращает следующий сэмпл из датасета.

        Возвращает:
            TSample: Следующий сэмпл из датасета.

        Выбрасывает:
            StopIteration: Если достигнут конец датасета.

        Этот метод должен быть реализован в подклассах.
        """
        pass
