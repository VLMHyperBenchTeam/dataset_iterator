from dataset_iterator.fabrics import IteratorFabric


class ModelInterface:
    def __init__(self, model_name) -> None:
        self.model_name = model_name

    def predict_on_image(self, image, question) -> str:
        print("predict!")
        return "predict!"



if __name__ == "__main__":
    model = ModelInterface("Cool2-VL")

    # получаем итератор по датесету параметры 
    iterator = IteratorFabric.get_dataset_iterator(task_name="VQA", 
                                                   dataset_name="pass",
                                                   start=0,
                                                   filter_doc_class=None,
                                                   filter_question_type=None,
                                                   dataset_dir_path=r".\datasets\data",
                                                   csv_name = "annotations.csv")
    # Получаем раннер
    runner = IteratorFabric.get_runner(iterator=iterator, 
                                       model=model,
                                       dataset_dir_path="/workspace/data",
                                       answers_dir_path="/workspace/answers",
                                       csv_name="answers.csv")
    # Совершаем прогон по датасету
    runner.run()
    # Сохраняем ответы
    save_path = runner.save_answers()
    print("Ответы сохранены в", save_path)
