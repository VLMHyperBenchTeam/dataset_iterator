import os
import json
from typing import Optional, List

from .abstract_iterator import AbstractIterator, AbstractSample
from prompt_adapter.prompt_adapter import PromptAdapter


class RPOSample(AbstractSample):
    """Dataclass для описания одного объекта датасета в задаче RPO. 
    Один объект представляет собой пачку документов. 
    Пачка документов состоит из набора изображений и json-описания правильного ответа. 

    Атрибуты:
        id (int): Уникальный идентификатор объекта датасета.
        images (List[str]): Список путей к изображению одного семпла.
        answer (str): Ответ на вопрос.
    """
    images: List[str]
    answer: dict

    def __init__(self, id: int, images: List[str], answer: str) -> None:
        """Инициализирует экземпляр VQASample.

        Аргументы:
            id (int): Уникальный идентификатор объекта датасета.
            images (List[str]): Список путей к изображению одного семпла.
            answer (str): Ответ на вопрос.
        """
        super().__init__(id=id)
        self.answer = answer
        self.images = images


class RPODatasetIterator(AbstractIterator):
    """Класс итератора для работы с датасетом задачи RPO.

    Атрибуты:
        prompt_adapter (Optional[PromptAdapter]): Адаптер для работы с промптами. Если не задан, равен None.
        samples (List[RPOSample]): Список всех элементов датасета
    """

    def __init__(self, prompt_collection_filename: Optional[str] = None, *args, **kwargs) -> None:
        """Инициализирует экземпляр RPODatasetIterator.

        Аргументы:
            prompt_collection_filename (Optional[str]): Путь к файлу с коллекцией промптов. По умолчанию None.
            *args: Аргументы для базового класса.
            **kwargs: Ключевые аргументы для базового класса.
        """
        super().__init__(*args, **kwargs)
        self.samples = []
        self.index = 0

        if prompt_collection_filename:
            self.prompt_adapter = PromptAdapter(prompt_collection_filename)
        else:
            self.prompt_adapter = None

        self._read_data()


    def _read_data(self) -> None:
        """Собирает все пути до файлов и создает список объектов RPOSample.
        """
        images_dir = os.path.join(self.dataset_dir_path, 'images')
        jsons_dir = os.path.join(self.dataset_dir_path, 'jsons')

        # Проходим по всем поддиректориям в images
        for dir_name in os.listdir(images_dir):
            dir_path = os.path.join(images_dir, dir_name)
            if os.path.isdir(dir_path):
                # Собираем все изображения в текущей директории
                images = [os.path.join(dir_path, img) for img in os.listdir(dir_path) if img.endswith('.jpg')]
                
                # Получаем путь до json-файла
                json_path = os.path.join(jsons_dir, f'{dir_name}.json')
                
                # Проверяем, существует ли json-файл
                if os.path.exists(json_path):
                    with open(json_path, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)

                    sample = RPOSample(id=int(dir_name), 
                                       images=images, 
                                       answer=json_data)
                    self.samples.append(sample)
        
    def __next__(self) -> RPOSample:
        """Возвращает следующий сэмпл из датасета.

        Возвращает:
            RPOSample: текущий сэмпл датасета.

        Выбрасывает:
            StopIteration: Если достигнут конец датасета.
        """
        if self.index < len(self.samples):
            sample = self.samples[self.index]
            self.index += 1
            return sample
        else:
            raise StopIteration
