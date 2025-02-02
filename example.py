from dataset_iterator.fabrics import IteratorFabric
from dataset_iterator.rpo_iterator import RPOSample

class ModelInterface:
    def __init__(self, model_name) -> None:
        self.model_name = model_name

    def predict_on_image(self, image, question) -> str:
        print("predict!")
        return "predict!"
    
    def predict_on_images(self, images, question) -> str:
        print("predict on multiple images!")
        return "44132"




if __name__ == "__main__":
    model = ModelInterface("Cool2-VL")

    # VQA SECTION

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
    # Сохраняем ответы
    save_path = vqa_runner.save_answers()
    print("Ответы сохранены в", save_path)


    # RPO SECTION

    # Classification
    # получаем итератор по датесету 
    rpo_iterator = IteratorFabric.get_dataset_iterator(
                                                       dataset_dir_path=r".\datasets\rpo",
                                                       task_name="RPOClassification", 
                                                       dataset_name="small-dataset",
                                                       start=0,
                                                       filter_doc_class=None,
                                                       filter_question_type=None,
                                                       prompt_file_dir=r".\datasets",
                                                       prompt_file_name=r"my_prompt.txt",
                                                       )
    # Пример работы итератора
    # sample: RPOSample
    # for sample in rpo_iterator:
    #     print("ID:", sample.id)
    #     print(sample.answer.keys())
    #     for img in sample.images:
    #         print(img, end=" ")
    #     print()

    # Получаем раннер
    rpo_runner = IteratorFabric.get_runner(iterator=rpo_iterator, 
                                           model=model,
                                           answers_dir_path="/workspace/answers",
                                           csv_name="cls_answers.csv")

    # Совершаем прогон по датасету
    rpo_runner.run()
    # Сохраняем ответы
    save_path_classification = rpo_runner.save_answers()
    print("Ответы сохранены в", save_path_classification)

    # Sorting
    # создаем новый итератор
    rpo_iterator = IteratorFabric.get_dataset_iterator(task_name="RPOSorting",
                                                       dataset_name="small-dataset",
                                                       dataset_dir_path=r".\datasets\rpo",
                                                       prompt_file_dir=r".\datasets",
                                                       prompt_file_name=r"my_prompt.txt",
                                                       )
     

    # Получаем раннер
    rpo_runner = IteratorFabric.get_runner(iterator=rpo_iterator, 
                                           model=model,
                                           answers_dir_path="/workspace/answers",
                                           csv_name="sort_answers.csv",
                                           classification_answers_path=save_path_classification)

    # Совершаем прогон по датасету
    rpo_runner.run()
    # Сохраняем ответы
    save_path = rpo_runner.save_answers()
    print("Ответы сохранены в", save_path)
