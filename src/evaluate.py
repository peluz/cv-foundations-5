from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusionMatrix, classification_report
import matplotlib.pyplot as plt
import os
import numpy as np


DIRNAME = os.path.dirname(__file__)
TEST_DIR = os.path.join(DIRNAME, "../data/test/")


def evaluate(model, batch_size=2, confusion_matrix=True):
    datagen = ImageDataGenerator()
    test_generator = datagen.flow_from_directory(TEST_DIR,
                                                 (300, 250),
                                                 batch_size=batch_size,
                                                 shuffle=False
                                                 )
    results = model.evaluate_generator(test_generator, verbose=1,
                                       steps=test_generator.samples // batch_size + 1)
    print("Loss, Acurácia: ", results)

    if confusion_matrix:
        Y_pred = model.predict_generator(test_generator, steps=test_generator.samples // batch_size + 1)
        y_pred = np.argmax(Y_pred, axis=1)
        target_names = ["Bedroom", "Coast", "Forest", "Highway", "Industrial",
                        "InsideCity", "Kitchen", "LivingRoom", "Mountain",
                        "Office", "OpenCountry", "Store", "Street", "Suburb",
                        "TallBuilding"]
        cm = confusionMatrix(test_generator.classes, y_pred)
        cm = cm / np.sum(cm.astype(np.float), axis=1)[:, np.newaxis]
        print(classification_report(test_generator.classes, y_pred,
                                    target_names=target_names))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(cm)
        fig.colorbar(cax)
        ax.set_xticklabels([''] + target_names)
        ax.set_yticklabels([''] + target_names)
        plt.xlabel('Predição')
        plt.ylabel('Verdade')
        plt.show()

    return results
