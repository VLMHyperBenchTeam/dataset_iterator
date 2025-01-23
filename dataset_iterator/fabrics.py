from .vqa_iterator import VQADatasetIterator
from .vqa_dataset_runner import VQADatasetRunner
from .abstract_dataset_runner import TIterator


class IteratorFabric:
    """ Класс для получения класса итератора и класса для прогона """

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
    def get_dataset_iterator(cls, task_name:str, dataset_name:str, start=0, filter_doc_class=None, filter_question_type=None, 
                             dataset_dir_path='/data', csv_name='annotations.csv', **kwargs):
        """ Возвращает итератор по датасету """
        if task_name not in cls._tasks:
            raise ValueError(f"Task '{task_name}' is implemented!")
        return cls._iterators[task_name](
                                         task_name=task_name, 
                                         dataset_name=dataset_name, 
                                         start=start, 
                                         filter_doc_class=filter_doc_class, 
                                         filter_question_type=filter_question_type, 
                                         dataset_dir_path=dataset_dir_path, 
                                         csv_name=csv_name, 
                                         **kwargs)
    

    @classmethod
    def get_runner(cls, iterator: TIterator, model, **kwargs):
        """ Возвращает объект для запуска прогона по датасету """
        
        return cls._runers[iterator.task_name](iterator, model, **kwargs)
