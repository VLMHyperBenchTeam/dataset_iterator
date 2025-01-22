import os
import pandas as pd
from abstract_iterator import AbstractIterator, AbstractSample


class VQASample(AbstractSample):
    """ Dataclass для описания одного объекта датасета в задаче VQA """
    image_path: str
    question: str
    answer: str


class VQADatasetIterator(AbstractIterator):

    def _read_data(self):
        """Загружает таблицу с аннотацией данных. """

        annot_path = os.path.join(self.dataset_dir_path, self.csv_name)
        # считываем названия столбцов
        dataframe_header = pd.read_csv(annot_path, sep=";", nrows=1)

        # считываем содержание таблицы, начиная с self.row_index
        if self.row_index > 0:
            self.row_index -= 1

        dataframe = pd.read_csv(annot_path, sep=";", skiprows=self.row_index)

        if self.filter_doc_class:
            dataframe = dataframe[
                (dataframe["doc_class"] == self.doc_class_filter) & 
                (dataframe["question_type"] == self.question_type_filter)
            ]

        # Записываем названия столбцов в dataframe из dataframe_header
        dataframe.columns = dataframe_header.columns
        self.iterator = dataframe.iterrows()

    def __next__(self) -> VQASample:
        """Возвращаем склееный путь до изображения, вопрос и ответ на него. """

        # итерируемся по Dataframe
        index, row = next(self.iterator)
        self.row_index += 1

        # получаем только нужные поля и склеиваем путь до изображения
        id, image_path, question, answer = row[["id", "image_path", "question", "answer"]]
        image_path = os.path.join(self.dataset_dir_path, image_path)

        return VQASample(id, image_path, question, answer)
