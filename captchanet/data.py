import numpy as np
import tensorflow as tf

PADDING_VALUE = '#'


def bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def bytes_features(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_features(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def int64_features(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def encode_data(image, word, tokenizer, max_len_word):
  image = np.asarray(image)
  image_string = tf.image.encode_png(image)
  token = tokenizer.texts_to_sequences([word])[0]
  assert len(token) == len(word)
  token = tf.pad(token, [[0, max_len_word - len(word)]])

  feature = {}
  feature['width'] = int64_feature(image.shape[0])
  feature['height'] = int64_feature(image.shape[1])
  feature['depth'] = int64_feature(image.shape[2])

  feature['image_raw'] = bytes_feature(image_string)
  feature['word'] = bytes_feature(word.encode())
  feature['token'] = int64_features(token)

  return tf.train.Example(features=tf.train.Features(feature=feature))


def decode_data(tokenizer, max_len_word, image_size=None, input_as_dict=False, normalize=True):
  def _decode(example):
    feature_description = {}
    feature_description['width'] = tf.io.FixedLenFeature([], tf.int64)
    feature_description['height'] = tf.io.FixedLenFeature([], tf.int64)
    feature_description['depth'] = tf.io.FixedLenFeature([], tf.int64)
    feature_description['image_raw'] = tf.io.FixedLenFeature([], tf.string)
    feature_description['word'] = tf.io.FixedLenFeature([], tf.string)
    feature_description['token'] = tf.io.FixedLenFeature([max_len_word], tf.int64)
    data = tf.io.parse_single_example(example, feature_description)

    # Decode image.
    data['image'] = tf.image.decode_png(data['image_raw'], dtype=tf.uint8)
    data['original_image'] = data['image']

    # Normalize
    data['image'] = tf.image.per_image_standardization(tf.cast(data['image'], 'float32'))

    if image_size:
      data['image'] = tf.image.resize(data['image'], image_size)

    # Remove unneeded image string.
    data.pop('image_raw')

    # Embed th token.
    vocabulary_size = len(tokenizer.index_word) + 1
    data['label'] = tf.one_hot(data['token'], depth=vocabulary_size)

    if input_as_dict:
      return data

    return data['image'], data['label']

  return _decode
