import os
import csv
import pandas as pd

from tqdm import tqdm
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Any, Dict
from collections import Counter

from .abstract_dataset_runner import AbstractDatasetRunner, TIterator
from .rpo_iterator import RPOSample


@dataclass
class SortingModelAnswer:
    """Класс, представляющий один ответ модели для задачи сортировки на датасете RPO.

    Атрибуты:
        id (int): Уникальный идентификатор ответа, соответствующий идентификатору сэмпла.
        answer (str): Текст ответа модели.
    """
    sample_id: int
    answer: str


class SortingRunner(AbstractDatasetRunner):
    """Класс, реализующий прогон модели по датасету задачи этапа сортировки в RPO.

    Атрибуты:
        iterator (TIterator): Итератор, который предоставляет доступ к данным датасета.
        model (ModelInterface): VLM-модель, которая будет использоваться для получения ответа.
        model_answers (list[ModelAnswer]): Список ответов модели.
        classification_answers (Dict[int, str]): Ответы на задачу классификации. Ключом является индекс сэмпа, значением - ответ модели классификации.
        answers_dir_path (str): Путь к директории для сохранения ответов. По умолчанию "/workspace/answers".
        csv_name (str): Имя CSV-файла для сохранения ответов. По умолчанию None и задаётся динамически согласно атрибутам класса.
    """

    def __init__(self, iterator: TIterator, model: Any, answers_dir_path: str = "/workspace/answers", 
                 csv_name: str = None, classification_answers_path: str = "/workspace/answers/cls_ans.csv",) -> None:
        """Инициализирует экземпляр SortingRunner.

        Аргументы:
            task_name (str): Название задачи.
            dataset_name (str): Название датасета.
            classification_answers_path (str): Путь до 
            start (int): Начальный индекс строки для итерации. По умолчанию 0.
            filter_doc_class (Optional[str]): Фильтр для класса документа. По умолчанию None.
            filter_question_type (Optional[str]): Фильтр для типа вопроса. По умолчанию None.
            dataset_dir_path (str): Путь к директории с датасетом. По умолчанию '/data'.
            csv_name (str): Имя CSV-файла с аннотацией данных. По умолчанию 'annotation.csv'.
        """
        super().__init__(iterator, model, answers_dir_path, csv_name)
        self.classification_answers = self._read_classification_answers(classification_answers_path)
        self.model_answers = []

    def _read_classification_answers(self, classification_answers_path: str) -> Dict[int, str]:
        """Читает CSV-файл с ответами модели и возвращает словарь соответствия индекса сэмпла и ответу модели.

        Аргументы:
            classification_answers_path (str): Путь к CSV-файлу с ответами модели.

        Возвращает:
            Dict[int, str]: Список ответов модели.
        """
        classification_answers = {}

        with open(classification_answers_path, mode='r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=';',)
            for row in reader:
                sample_id = int(row['sample_id'])
                answer = row['model_answer']
                classification_answers[sample_id] = answer

        return classification_answers

    def run(self) -> None:
        """Осуществляет прогон модели по датасету RPO и проводит сортировку внутри документа.

        Проходит по всем сэмплам в итераторе, получает ответы модели и сохраняет их.
        """

        row: RPOSample
        for row in tqdm(self.iterator):
            # получаем ответ модели классификации для данного сэмла, содержит в себе 12554
            answer_cls = self.classification_answers.get(row.id) 

            if answer_cls != None:
                # получаем все классы и оставляем те, что встречаются более 1 раза
                doc_classes = Counter(answer_cls) 
                doc_classes = {k: v for k, v in doc_classes.items() if v > 1}

                for key in doc_classes.keys():
                    # достаем все картинки нужного класса: сначала получили все индексы, а затем все пути до файлов
                    class_images_idx = [i for i, val in enumerate(answer_cls) if val == key]
                    class_images = [row.images[i] for i in class_images_idx]

                    page_order = self.model.predict_on_images(class_images, row.prompt)
                    # ответ в формате  "2,2,5,5,3", убираем запятые
                    page_order = page_order.replace(",", "")
                    self.add_answer(row, page_order)

    def add_answer(self, sample: RPOSample, answer: str) -> None:
        """Добавляет ответ модели в список ответов.

        Аргументы:
            sample (RPOSample): Сэмпл из датасета RPO.
            answer (str): Ответ модели на вопрос из сэмпла.
        """
        self.model_answers.append(
            SortingModelAnswer(
                sample.id,
                answer
            )
        )

    def get_answer_filename(self) -> str:
        """Генерирует путь до файла с ответами модели.
        """
        # TODO: получить Modelframework из свойств модели ModelAnswersClassification
        os.makedirs(self.answers_dir_path, exist_ok=True)  # Создаем директорию, если её нет
        timestamp = datetime.now().strftime(r"%Y%m%d_%H%M%S")  # Формат: ГГГГММДД_ЧЧММСС
        save_path = os.path.join(
            self.answers_dir_path,
            f"{self.iterator.dataset_name}_MODELFRAMEWORK_{self.model.model_name}_{self.iterator.task_name}_sorting_answers_{timestamp}.csv"
        )
        return save_path

    def save_answers(self) -> str:
        """Сохраняет ответы в CSV-файл по пути self.answers_dir_path с добавлением timestamp в название файла.

        Если список ответов пуст, выводит предупреждение и не сохраняет файл.
        """
        if not self.model_answers:
            print("Нет ответов для сохранения.")
            return

        # Преобразуем список ответов в DataFrame
        answers_df = pd.DataFrame([asdict(answer) for answer in self.model_answers])

        # Создаем путь для сохранения файла с timestamp
        save_path = self.get_answer_filename()

        # Сохраняем DataFrame в CSV
        answers_df.to_csv(save_path, index=False, sep=" ;")
        return save_path
