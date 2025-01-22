import os
import datetime

from dataclasses import dataclass, asdict
import pandas as pd

from abstract_dataset_runner import AbstractDatasetRunner
from vqa_iterator import VQASample 


@dataclass
class ModelAnswer:
    """ Реализует класс одного ответа модели. По умолчанию пуст. """
    id: int
    answer: str


class VQADatasetRunner(AbstractDatasetRunner):
    """ Класс, реализующий прогон модели по датасету задачи VQA. """

    def run(self):
        """ Осуществляет прогон модели по датасету VQA, собирает ответы и записывает их на диск. """
        row: VQASample
        for row in self.dataset:
            answer = self.model.predict_on_image(row.image_path, row.question)
            self.add_answer(row, answer)
            

    def add_answer(self, sample: VQASample, answer: str):
        """ Добавляет ответ модели в список ответов """
        self.model_answers.append(
            ModelAnswer(
                sample.id,
                answer
            )
        )

    def save_answers(self):
        """ Сохраняет ответы в CSV-файл по пути self.answers_dir_path с добавлением timestamp в название файла. """
        if not self.model_answers:
            print("Нет ответов для сохранения.")
            return

        # Преобразуем список ответов в DataFrame
        answers_df = pd.DataFrame([asdict(answer) for answer in self.model_answers])

        # Создаем путь для сохранения файла с timestamp
        os.makedirs(self.answers_dir_path, exist_ok=True)  # Создаем директорию, если её нет
        timestamp = datetime.now().strftime(r"%Y%m%d_%H%M%S")  # Формат: ГГГГММДД_ЧЧММСС
        save_path = os.path.join(self.answers_dir_path, 
                                 f"{self.dataset.dataset_name}_MODELFRAMEWORK_MODELNAME_{self.task_name}_answers_{timestamp}.csv")

        # Сохраняем DataFrame в CSV
        answers_df.to_csv(save_path, index=False, sep=";")
        print(f"Ответы сохранены в файл: {save_path}")
