import os
import pandas as pd
from typing import Optional

from .abstract_iterator import AbstractIterator, AbstractSample
from prompt_adapter.prompt_adapter import PromptAdapter


class VQASample(AbstractSample):
    """Dataclass для описания одного объекта датасета в задаче VQA.

    Атрибуты:
        id (int): Уникальный идентификатор объекта датасета.
        image_path (str): Путь к изображению.
        question (str): Вопрос, связанный с изображением.
        answer (str): Ответ на вопрос.
        doc_class (str): Класс документа.
        question_type (str): Тип вопроса.
    """
    image_path: str
    question: str
    answer: str
    doc_class: str
    question_type: str

    def __init__(self, id: int, image_path: str, question: str, answer: str, doc_class: str, question_type: str) -> None:
        """Инициализирует экземпляр VQASample.

        Аргументы:
            id (int): Уникальный идентификатор сэмпла.
            image_path (str): Путь к изображению.
            question (str): Вопрос, связанный с изображением.
            answer (str): Ответ на вопрос.
            doc_class (str): Класс документа.
            question_type (str): Тип вопроса.
        """
        super().__init__(id=id)
        self.image_path = image_path
        self.question = question
        self.answer = answer
        self.doc_class = doc_class
        self.question_type = question_type


class VQADatasetIterator(AbstractIterator):
    """Класс итератора для работы с датасетом задачи VQA.

    Атрибуты:
        prompt_adapter (Optional[PromptAdapter]): Адаптер для работы с промптами. Если не задан, равен None.
    """

    def __init__(self, prompt_collection_filename: Optional[str] = None, *args, **kwargs) -> None:
        """Инициализирует экземпляр VQADatasetIterator.

        Аргументы:
            prompt_collection_filename (Optional[str]): Путь к файлу с коллекцией промптов. По умолчанию None.
            *args: Аргументы для базового класса.
            **kwargs: Ключевые аргументы для базового класса.
        """
        super().__init__(*args, **kwargs)

        if prompt_collection_filename:
            self.prompt_adapter = PromptAdapter(prompt_collection_filename)
        else:
            self.prompt_adapter = None

    def _read_data(self) -> None:
        """Загружает таблицу с аннотацией данных.

        Считывает CSV-файл с аннотацией, применяет фильтры (если заданы) и инициализирует итератор по данным.
        """
        annot_path = os.path.join(self.dataset_dir_path, self.csv_name)
        # Считываем названия столбцов
        dataframe_header = pd.read_csv(annot_path, sep=";", nrows=1)

        # Считываем содержание таблицы, начиная с self.row_index
        if self.row_index > 0:
            self.row_index -= 1

        dataframe = pd.read_csv(annot_path, sep=";", skiprows=self.row_index)

        if self.filter_doc_class:
            dataframe = dataframe[
                (dataframe["doc_class"] == self.filter_doc_class) &
                (dataframe["question_type"] == self.filter_question_type)
            ]

        # Записываем названия столбцов в dataframe из dataframe_header
        dataframe.columns = dataframe_header.columns
        self.iterator = dataframe.iterrows()

    def __next__(self) -> VQASample:
        """Возвращает следующий сэмпл из датасета.

        Возвращает:
            VQASample: Сэмпл, содержащий путь к изображению, вопрос, ответ, класс документа и тип вопроса.

        Выбрасывает:
            StopIteration: Если достигнут конец датасета.
        """
        # Итерируемся по Dataframe
        index, row = next(self.iterator)

        # Получаем только нужные поля и склеиваем путь до изображения
        image_path, question, answer, doc_class, question_type = row[["image_path", "question", "answer", "doc_class", "question_type"]]

        # Получаем оптимальный промпт из промпт адаптера
        if self.prompt_adapter:
            question = self.prompt_adapter.get_prompt(doc_class, question_type)

        image_path = os.path.join(self.dataset_dir_path, image_path)

        return VQASample(
            id=index,
            image_path=image_path,
            question=question,
            answer=answer,
            doc_class=doc_class,
            question_type=question_type
        )
