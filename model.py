import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Tuple, Union, List, Dict

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import compute_class_weight

from keras import backend as K
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

import logging

level = logging.INFO
logging.basicConfig(level=level)
logger = logging.getLogger(__name__)


class Data:
    """
    Data Pipeline contains reading data into memory, label encoding, and splitting on train/test sets.
    """

    def __init__(self):
        self.link = "customer_chat_sample.csv"

    def read_data(self) -> pd.DataFrame:
        data = pd.read_csv(self.link)
        logger.info("There are {} entries in the dataframe".format(data.shape[0]))
        labels = data['label'].unique()
        logger.info("There are {} labels in the dataframe".format(len(labels)))
        return data

    def encode_label(self, label_in: str = 'label', label_out: str = 'label_enc') -> pd.DataFrame:
        data = self.read_data()
        le = LabelEncoder()
        data[label_out] = le.fit_transform(data[label_in])
        label_dict = (data[[label_in, label_out]].drop_duplicates()
                      .sort_values(by=label_out)
                      .reset_index(drop=True)[label_in]
                      .to_dict())
        return data

    def split_data(self) -> Tuple[pd.Series, pd.Series, np.ndarray, np.ndarray]:
        data = self.encode_label()

        y = tf.keras.utils.to_categorical(data["label_enc"].values, num_classes=8)

        x_train, x_test, y_train, y_test = train_test_split(data['text'],
                                                            y,
                                                            test_size=0.3,
                                                            shuffle=True,
                                                            random_state=5,
                                                            stratify=data['label'])
        return x_train, x_test, y_train, y_test


class Metrics:
    """
    Contains set of perfomance metrics averaged across the classes.
    """

    @staticmethod
    def balanced_recall(y_true: np.ndarray, y_pred: np.ndarray):
        """Calculates the balanced recall metric.
        recall = TP / (TP + FN)
        """
        recall_by_class = 0
        # iterate over each predicted class to get class-specific metric
        for i in range(y_pred.shape[1]):
            y_pred_class = y_pred[:, i]
            y_true_class = y_true[:, i]
            true_positives = K.sum(K.round(K.clip(y_true_class * y_pred_class, 0, 1)))
            possible_positives = K.sum(K.round(K.clip(y_true_class, 0, 1)))
            recall = true_positives / (possible_positives + K.epsilon())
            recall_by_class = recall_by_class + recall
        return recall_by_class / y_pred.shape[1]

    @staticmethod
    def balanced_precision(y_true: np.ndarray, y_pred: np.ndarray):
        """Calculates the balanced precision metric.
        precision = TP / (TP + FP)
        """
        precision_by_class = 0
        # iterate over each predicted class to get class-specific metric
        for i in range(y_pred.shape[1]):
            y_pred_class = y_pred[:, i]
            y_true_class = y_true[:, i]
            true_positives = K.sum(K.round(K.clip(y_true_class * y_pred_class, 0, 1)))
            predicted_positives = K.sum(K.round(K.clip(y_pred_class, 0, 1)))
            precision = true_positives / (predicted_positives + K.epsilon())
            precision_by_class = precision_by_class + precision

        return precision_by_class / y_pred.shape[1]

    @staticmethod
    def balanced_f1_score(y_true: np.ndarray, y_pred: np.ndarray):
        """Calculates the F1 score metric."""
        precision = Metrics.balanced_precision(y_true, y_pred)
        recall = Metrics.balanced_recall(y_true, y_pred)
        return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


class Classifier:
    """
    Multiclass Text Classifier Pipeline.
    """

    def __init__(self, num_classes: int = 8) -> None:
        self.batch_size = 30
        self.epochs = 1
        self.metrics = ['accuracy',
                        Metrics.balanced_recall,
                        Metrics.balanced_precision,
                        Metrics.balanced_f1_score]

        self.num_classes = num_classes
        self.preprocessor = hub.KerasLayer(
            "https://tfhub.dev/google/universal-sentence-encoder-cmlm/multilingual-preprocess/2")
        self.encoder = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder-cmlm/multilingual-base/1")

    def compute_weights(self, y_train: np.ndarray) -> Dict[int, float]:

        class_weight = compute_class_weight(class_weight="balanced",
                                            classes=list(range(self.num_classes)),
                                            y=list(np.argmax(y_train, axis=1)))

        class_weight = dict(zip(list(range(self.num_classes)), class_weight))

        return class_weight

    def init_model(self) -> tf.keras.Model:

        text_input = tf.keras.layers.Input(shape=(),
                                           dtype=tf.string,
                                           name='text')

        encoder_outputs = self.encoder(self.preprocessor(text_input))
        embedding_layer = encoder_outputs['pooled_output']

        dropout1 = tf.keras.layers.Dropout(0.1)(embedding_layer)
        dense = tf.keras.layers.Dense(50,
                                      activation='relu',
                                      name="dense")(dropout1)
        dropout2 = tf.keras.layers.Dropout(0.1)(dense)
        net = tf.keras.layers.Dense(self.num_classes,
                                    activation="softmax",
                                    name="classifier")(dropout2)

        return tf.keras.Model(inputs=text_input, outputs=net)

    def train(self, x_train: pd.Series, y_train: np.ndarray,
              x_test: pd.Series, y_test: np.ndarray, save: bool = False, weighted: bool = False) -> List[
        Union[tf.keras.Model, tf.keras.callbacks.History]]:

        model = self.init_model()
        model.compile(loss='categorical_crossentropy', optimizer='Nadam', metrics=self.metrics)

        model_checkpoint = ModelCheckpoint(filepath=f'classifier/saved_weights/',
                                           save_weights_only=True)

        early_stopping = EarlyStopping(patience=3,
                                       monitor='val_loss',
                                       min_delta=0,
                                       mode='min',
                                       restore_best_weights=False,
                                       verbose=1)

        reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                      min_lr=0.000001,
                                      patience=1,
                                      mode='min',
                                      factor=0.1,
                                      min_delta=0.01,
                                      verbose=1)

        if weighted:
            class_weight = self.compute_weights(y_train)
        else:
            class_weight = dict(zip(list(range(self.num_classes)),
                                    [1 / self.num_classes] * self.num_classes))  # equally weighted

        hist = model.fit(x_train, y_train,
                         epochs=self.epochs,
                         batch_size=self.batch_size,
                         validation_data=(x_test, y_test),
                         callbacks=[reduce_lr, early_stopping],
                         class_weight=class_weight)

        if save:
            tf.keras.models.save_model(model, f'classifier/saved_models/model',
                                       overwrite=True,
                                       signatures=None)

        return [model, hist]


if __name__ == '__main__':
    logger.info('Started')
    data = Data()
    logger.info('Loading data...')
    x_train, x_test, y_train, y_test = data.split_data()
    model = Classifier()
    logger.info('Training...')
    model.train(x_train, y_train, x_test, y_test, save=True)
