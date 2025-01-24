from typing import Optional, TypeVar

from .vqa_iterator import VQADatasetIterator
from .vqa_dataset_runner import VQADatasetRunner
from .abstract_dataset_runner import TIterator, AbstractDatasetRunner

# Определяем TypeVar с ограничением на AbstractDatasetRunner и его наследников
TRunner = TypeVar('TRunner', bound=AbstractDatasetRunner)


class IteratorFabric:
    """Фабрика для получения итератора и объекта для прогона модели по датасету.

    Атрибуты:
        _VQAName (str): Название задачи VQA.
        _RPOName (str): Название задачи RPO.
        _iterators (dict): Словарь, сопоставляющий название задачи с классом итератора.
        _runers (dict): Словарь, сопоставляющий название задачи с классом для прогона.
        _tasks (set): Множество названий задач, для которых реализованы итераторы и прогоны.
    """

    _VQAName = "VQA"
    _RPOName = "RPO"

    _iterators = {
        _VQAName: VQADatasetIterator,
    }

    _runers = {
        _VQAName: VQADatasetRunner,
    }

    _tasks = _iterators.keys()

    @classmethod
    def get_dataset_iterator(cls, task_name: str, dataset_name: str, start: int = 0, 
                             filter_doc_class: Optional[str] = None, filter_question_type: Optional[str] = None, 
                             dataset_dir_path: str = '/data', csv_name: str = 'annotations.csv', *args, **kwargs) -> TIterator:
        """Возвращает итератор по датасету для указанной задачи.

        Аргументы:
            task_name (str): Название задачи (например, "VQA").
            dataset_name (str): Название датасета.
            start (int): Начальный индекс строки для итерации. По умолчанию 0.
            filter_doc_class (Optional[str]): Фильтр для класса документа. По умолчанию None.
            filter_question_type (Optional[str]): Фильтр для типа вопроса. По умолчанию None.
            dataset_dir_path (str): Путь к директории с датасетом. По умолчанию '/data'.
            csv_name (str): Имя CSV-файла с аннотацией данных. По умолчанию 'annotations.csv'.
            **kwargs: Дополнительные аргументы для инициализации итератора.

        Возвращает:
            TIterator: Итератор по датасету.

        Выбрасывает:
            ValueError: Если задача не реализована.
        """
        if task_name not in cls._tasks:
            raise ValueError(f"Task '{task_name}' is not implemented!")
        return cls._iterators[task_name](
            task_name=task_name,
            dataset_name=dataset_name,
            start=start,
            filter_doc_class=filter_doc_class,
            filter_question_type=filter_question_type,
            dataset_dir_path=dataset_dir_path,
            csv_name=csv_name,
            *args,
            **kwargs
        )

    @classmethod
    def get_runner(cls, iterator: TIterator, model, **kwargs) -> TRunner:
        """Возвращает объект для запуска прогона модели по датасету.

        Аргументы:
            iterator (TIterator): Итератор по датасету.
            model (ModelInterface): Модель, которая будет использоваться для обработки данных.
            **kwargs: Дополнительные аргументы для инициализации объекта прогона.

        Возвращает:
            AbstractDatasetRunner: Объект для прогона модели по датасету.
        """
        return cls._runers[iterator.task_name](iterator, model, **kwargs)
