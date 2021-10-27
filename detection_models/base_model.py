from abc import ABC, abstractmethod


class BaseModel(ABC):
    @abstractmethod
    def run_recognition(self, image):
        """
        Processes the image using the implemented model
        :param image: A cv2 image
        :return: A list or map of the model results
        """
        pass

    @abstractmethod
    def decorate_image(self, image, result):
        """
        Debug method - Draw the model results onto the source image
        :param image: the image interpreted by the model
        :param result: the result returned by run_recognition
        :return: a cv2 image with results drown on for testing
        """
        pass
