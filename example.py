from dataset_iterator.fabrics import IteratorFabric


class ModelInterface:
    def __init__(self, model_name) -> None:
        self.model_name = model_name

    def predict_on_image(self, image, question) -> str:
        print("predict!")
        return "predict!"



if __name__ == "__main__":
    model = ModelInterface("Cool2-VL")

    # Совершаем прогон по датасету
    iterator = IteratorFabric.get_dataset_iterator(task_name="VQA", 
                                                   dataset_name="pass",
                                                   dataset_dir_path=r".\data")
    runner = IteratorFabric.get_runner(iterator, model)
    runner.run()
    runner.save_answers()
