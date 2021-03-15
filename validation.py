import gc
import tensorflow as tf
from tensorflow.keras import backend as K
from sklearn.metrics import log_loss, accuracy_score
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
class Sparse_Validation_Callback(Callback):

    def __init__(self, validation_data = (), interval = 1):

        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs ={}):

        y_pred = self.model.predict(self.X_val, verbose = 0)
        scce = SparseCategoricalCrossentropy(from_logits=True)
        score = scce(self.y_val, y_pred).numpy()
        acc = SparseCategoricalAccuracy()
        acc.update_state(self.y_val, y_pred)
        print("\n")
        print("The loss is : {}, the accuracy is: {}".format(score, acc.result().numpy()))
        gc.collect()
        K.clear_session()

class Validation_Callback(Callback):

    def __init__(self, validation_data = (), interval = 1):

        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs ={}):

        y_pred = self.model.predict(self.X_val, verbose = 0)
        score = log_loss(self.y_val, y_pred)
        y_classes = y_pred.round() 
        accuracy = accuracy_score(self.y_val, y_classes)
        print("\n")
        print("The loss is : {}, the accuracy is: {}".format(score, accuracy))
        gc.collect()
        K.clear_session()

