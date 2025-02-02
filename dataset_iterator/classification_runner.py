import os
from datetime import datetime
from dataclasses import dataclass, asdict
import pandas as pd

from .abstract_dataset_runner import AbstractDatasetRunner
from .rpo_iterator import RPOSample


@dataclass
class ClassificationModelAnswer:
    """Класс, представляющий один ответ модели для задачи классификации на датасете RPO.

    Атрибуты:
        id (int): Уникальный идентификатор ответа, соответствующий идентификатору сэмпла.
        model_answer (str): Текст ответа модели.
    """
    sample_id: int
    model_answer: str


class ClassificationRunner(AbstractDatasetRunner):
    """Класс, реализующий прогон модели по датасету задачи этапа классификации в RPO.

    Атрибуты:
        iterator (TIterator): Итератор, который предоставляет доступ к данным датасета.
        model (ModelInterface): VLM-модель, которая будет использоваться для получения ответа.
        model_answers (list[ModelAnswer]): Список ответов модели.
        answers_dir_path (str): Путь к директории для сохранения ответов. По умолчанию "/workspace/answers".
        csv_name (str): Имя CSV-файла для сохранения ответов. По умолчанию None и задаётся динамически согласно атрибутам класса.
    """

    def run(self) -> None:
        """Осуществляет прогон модели по датасету RPO и проводит классификацию модели.

        Проходит по всем сэмплам в итераторе, получает ответы модели и сохраняет их.
        """
        row: RPOSample
        for row in self.iterator:
            answer_cls = self.model.predict_on_images(row.images, row.prompt)
            # ответ в формате  "2,2,5,5,3", убираем запятые
            answer_cls = answer_cls.replace(",", "")
            self.add_answer(row, answer_cls)

    def add_answer(self, sample: RPOSample, answer: str) -> None:
        """Добавляет ответ модели в список ответов.

        Аргументы:
            sample (ClassificationModelAnswer): Сэмпл из датасета RPO.
            answer (str): Ответ модели на вопрос из сэмпла.
        """
        self.model_answers.append(
            ClassificationModelAnswer(
                sample.id,
                answer
            )
        )

    def get_answer_filename(self) -> str:
        """Генерирует путь до файла с ответами модели.
        """
        # TODO: получить Modelframework из свойств модели
        os.makedirs(self.answers_dir_path, exist_ok=True)  # Создаем директорию, если её нет
        timestamp = datetime.now().strftime(r"%Y%m%d_%H%M%S")  # Формат: ГГГГММДД_ЧЧММСС
        save_path = os.path.join(
            self.answers_dir_path,
            f"{self.iterator.dataset_name}_MODELFRAMEWORK_{self.model.model_name}_{self.iterator.task_name}_classification_answers_{timestamp}.csv"
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
        answers_df.to_csv(save_path, index=False, sep=";")
        return save_path
