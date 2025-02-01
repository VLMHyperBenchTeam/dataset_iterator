from dataclasses import dataclass, asdict
import pandas as pd

from .abstract_dataset_runner import AbstractDatasetRunner
from .vqa_iterator import VQASample


@dataclass
class VQAModelAnswer:
    """Класс, представляющий один ответ модели для задачи VQA.

    Атрибуты:
        id (int): Уникальный идентификатор ответа, соответствующий идентификатору сэмпла.
        answer (str): Текст ответа модели.
    """
    id: int
    answer: str


class VQADatasetRunner(AbstractDatasetRunner):
    """Класс, реализующий прогон модели по датасету задачи VQA.

    Атрибуты:
        iterator (TIterator): Итератор, который предоставляет доступ к данным датасета.
        model (ModelInterface): VLM-модель, которая будет использоваться для получения ответа.
        model_answers (list[ModelAnswer]): Список ответов модели.
        answers_dir_path (str): Путь к директории для сохранения ответов. По умолчанию "/workspace/answers".
        csv_name (str): Имя CSV-файла для сохранения ответов. По умолчанию None и задаётся динамически согласно атрибутам класса.
    """

    def run(self) -> None:
        """Осуществляет прогон модели по датасету VQA и собирает ответы.

        Проходит по всем сэмплам в итераторе, получает ответы модели и сохраняет их.
        """
        row: VQASample
        for row in self.iterator:
            answer = self.model.predict_on_image(row.image_path, row.question)
            self.add_answer(row, answer)

    def add_answer(self, sample: VQASample, answer: str) -> None:
        """Добавляет ответ модели в список ответов.

        Аргументы:
            sample (VQASample): Сэмпл из датасета VQA.
            answer (str): Ответ модели на вопрос из сэмпла.
        """
        self.model_answers.append(
            VQAModelAnswer(
                sample.id,
                answer
            )
        )

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
