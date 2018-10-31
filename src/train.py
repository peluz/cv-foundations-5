from keras.preprocessing.image import ImageDataGenerator
from keras.applications.xception import Xception
from keras.layers import Dense, Dropout
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import SGD
from keras import Model
import os


DIRNAME = os.path.dirname(__file__)
TRAIN_DIR = os.path.join(DIRNAME, "../data/train/")


def train(pooling="avg", num_units=1024, batch_size=2, name="test", drop_prob=0.):
    model_dir = os.path.join(DIRNAME, "models/{}".format(name))
    os.makedirs(model_dir, exist_ok=True)

    datagen = ImageDataGenerator(horizontal_flip=True,
                                 validation_split=0.2)

    train_generator = datagen.flow_from_directory(TRAIN_DIR,
                                                  (300, 250),
                                                  batch_size=batch_size,
                                                  subset='training')

    valid_generator = datagen.flow_from_directory(TRAIN_DIR,
                                                  (300, 250),
                                                  batch_size=batch_size,
                                                  subset='validation')

    base_model = Xception(include_top=False, weights="imagenet",
                          input_shape=(300, 250, 3),
                          pooling=pooling)

    x = base_model.output
    x = Dense(num_units, activation="relu")(x)
    x = Dropout(drop_prob)(x)
    predictions = Dense(15, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # for layer in base_model.layers:
    #     layer.trainable = False

    optimizer = SGD(momentum=0.9, clipnorm=5.)

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    tensorboard = TensorBoard(log_dir=model_dir,
                              batch_size=batch_size, update_freq="batch")
    saver = ModelCheckpoint("{}/model.hdf5".format(model_dir), verbose=1,
                            save_best_only=True, monitor="val_acc",
                            mode="max")
    stopper = EarlyStopping(patience=6, verbose=1, monitor="val_acc",
                            mode="max")
    reduce_lr = ReduceLROnPlateau(monitor="loss", factor=0.5,
                                  patience=4, verbose=1, min_lr=0.001)

    model.fit_generator(train_generator, steps_per_epoch=train_generator.samples // batch_size,
                        validation_data=valid_generator,
                        validation_steps=valid_generator.samples // batch_size,
                        verbose=2,
                        epochs=20,
                        callbacks=[tensorboard, saver, stopper, reduce_lr])
    print("Modelo {} treinado!".format(name))
