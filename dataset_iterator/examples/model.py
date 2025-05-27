class ModelInterface:
    def __init__(self, model_name, model_framework) -> None:
        self.model_name = model_name
        self.framework = model_framework

    def predict_on_image(self, image, question) -> str:
        print("predict!")
        return "predict!"
    
    def predict_on_images(self, images, question) -> str:
        print("predict on multiple images!")
        return "44132"