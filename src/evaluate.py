from keras.preprocessing.image import ImageDataGenerator
import os


DIRNAME = os.path.dirname(__file__)
TEST_DIR = os.path.join(DIRNAME, "../data/test/")


def evaluate(model, batch_size=2):
    datagen = ImageDataGenerator()
    test_generator = datagen.flow_from_directory(TEST_DIR,
                                                 (300, 250),
                                                 batch_size=batch_size
                                                 )
    print("Loss, Acurácia: ",
          model.evaluate_generator(test_generator, verbose=1,
                                   steps=test_generator.samples // batch_size))
