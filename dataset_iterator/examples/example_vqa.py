from dataset_iterator.fabrics import IteratorFabric
from model import ModelInterface


if __name__ == "__main__":
    model = ModelInterface("Cool2-VL", "FrameworkFace")

    print("VQA SECTION")

    # получаем итератор по датесету 
    vqa_iterator = IteratorFabric.get_dataset_iterator(task_name="VQA", 
                                                       dataset_name="pass",
                                                       start=0,
                                                       filter_doc_class=None,
                                                       filter_question_type=None,
                                                       dataset_dir_path=r".\datasets\data",
                                                       csv_name = "annotations.csv")
    # Получаем раннер
    vqa_runner = IteratorFabric.get_runner(iterator=vqa_iterator, 
                                           model=model,
                                           answers_dir_path="/workspace/answers",
                                           csv_name="answers.csv")
    # Совершаем прогон по датасету
    vqa_runner.run()
    print("Раннер закончил работу")
    # Сохраняем ответы
    save_path = vqa_runner.save_answers()
    print("Ответы сохранены в", save_path)
