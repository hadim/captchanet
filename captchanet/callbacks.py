import tensorflow as tf


class LRTensorBoard(tf.keras.callbacks.TensorBoard):
  def on_epoch_end(self, epoch, logs=None):
    logs = logs or {}
    logs['lr'] = tf.keras.backend.get_value(self.model.optimizer.lr)
    super().on_epoch_end(epoch, logs)
