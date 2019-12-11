#!/usr/bin/env python
# coding: utf-8
# Tensorflow Version == tf 1.14
# tf.Keras Version == 2.2.4-tf
# cleverhans Version == 3.0.1


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

import tensorflow as tf
from tensorflow import keras
import numpy as np

from cleverhans.attacks import FastGradientMethod
from cleverhans.compat import flags
from cleverhans.dataset import MNIST, CIFAR10

from cleverhans.loss import CrossEntropy
from cleverhans.train import train
from cleverhans.utils import AccuracyReport
from cleverhans.utils_keras import cnn_model,betterCNN
from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.utils_tf import model_eval
import timeit

FLAGS = flags.FLAGS

NB_EPOCHS = 50
BATCH_SIZE = 128 #old batch size 128
LEARNING_RATE = .001 #.001
TRAIN_DIR = 'train_dir'
FILENAME = 'mnist.ckpt'
LOAD_MODEL = False
f= open("logFile.txt","a+")
f.write('Adversarial Trainging Results on FGSM attack. MNIST')


#old train_end is 60000
#old test_end is 10000
def mnist_tutorial(train_start=0, train_end=10000, test_start=0,
                   test_end=4000, nb_epochs=NB_EPOCHS, batch_size=BATCH_SIZE,
                   learning_rate=LEARNING_RATE, train_dir=TRAIN_DIR,
                   filename=FILENAME, load_model=LOAD_MODEL,
                   testing=False, label_smoothing=0.1):
  """
  MNIST CleverHans tutorial
  :param train_start: index of first training set example
  :param train_end: index of last training set example
  :param test_start: index of first test set example
  :param test_end: index of last test set example
  :param nb_epochs: number of epochs to train model
  :param batch_size: size of training batches
  :param learning_rate: learning rate for training
  :param train_dir: Directory storing the saved model
  :param filename: Filename to save model under
  :param load_model: True for load, False for not load
  :param testing: if true, test error is calculated
  :param label_smoothing: float, amount of label smoothing for cross entropy
  :return: an AccuracyReport object
  """
  tf.keras.backend.set_learning_phase(0)

# Object used to keep track of (and return) key accuracies
  report = AccuracyReport()

# Set TF random seed to improve reproducibility
  tf.compat.v1.set_random_seed(1234)

  if keras.backend.image_data_format() != 'channels_last':
      raise NotImplementedError("this tutorial requires keras to be configured to channels_last format")

# Create TF session and set as Keras backend session
  sess = tf.compat.v1.Session()
  tf.compat.v1.keras.backend.set_session(sess)

  def grayscale(data, dtype='float32'):
        r, g, b = np.asarray(.3, dtype=dtype), np.asarray(.59, dtype=dtype), np.asarray(.11, dtype=dtype)
        rst = r * data[:, :, :, 0] + g * data[:, :, :, 1] + b * data[:, :, :, 2]
        # add channel dimension
        rst = np.expand_dims(rst, axis=3)
        return rst

  mnist = CIFAR10(train_start=train_start, train_end=train_end,
            test_start=test_start, test_end=test_end)
  x_train, y_train = mnist.get_set('train')
  x_test, y_test = mnist.get_set('test')
  x_train = grayscale(x_train)
  x_test = grayscale(x_test)  


# Obtain Image Parameters
  img_rows, img_cols, nchannels = x_train.shape[1:4]
  nb_classes = y_train.shape[1]

# Define input TF placeholder
  x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols,
                                      nchannels))
  y = tf.placeholder(tf.float32, shape=(None, nb_classes))

  # Define TF model graph
  model = betterCNN(img_rows=img_rows, img_cols=img_cols,
                  channels=nchannels, nb_filters=64,
                  nb_classes=nb_classes)
  #model = cnn_model(img_rows=img_rows, img_cols=img_cols,
                  #channels=nchannels, nb_filters=64,
                  #nb_classes=nb_classes)


  preds = model(x)
  print("Defined TensorFlow model graph.")

  def evaluate():
  # Evaluate the accuracy of the MNIST model on legitimate test examples
      eval_params = {'batch_size': batch_size}
      acc = model_eval(sess, x, y, preds, x_test, y_test, args=eval_params)
      report.clean_train_clean_eval = acc
      #        assert X_test.shape[0] == test_end - test_start, X_test.shape
      print('Test accuracy on legitimate examples: %0.4f' % acc)
      f.write('Test accuracy on legitimate examples: %0.4f' % acc)
      

# Train an MNIST model
  train_params = {
    'nb_epochs': nb_epochs,
    'batch_size': batch_size,
    'learning_rate': learning_rate,
    'train_dir': train_dir,
    'filename': filename
  }
  quick_train = {  
    'nb_epochs': 5,
    'batch_size': batch_size,
    'learning_rate': learning_rate,
    'train_dir': train_dir,
    'filename': filename
  }

  rng = np.random.RandomState([2017, 8, 30])
  if not os.path.exists(train_dir):
      os.mkdir(train_dir)

  ckpt = tf.train.get_checkpoint_state(train_dir)
  print(train_dir, ckpt)
  ckpt_path = False if ckpt is None else ckpt.model_checkpoint_path
  wrap = KerasModelWrapper(model)

  if load_model and ckpt_path:
      saver = tf.train.Saver()
      print(ckpt_path)
      saver.restore(sess, ckpt_path)
      print("Model loaded from: {}".format(ckpt_path))
      evaluate()
  else:
      print("Model was not loaded, training from scratch.")
      loss = CrossEntropy(wrap, smoothing=label_smoothing)
      train(sess, loss, x_train, y_train, evaluate=evaluate,
            args=train_params, rng=rng) #args=train_params, rng=rng)

# Calculate training error
  if testing:
      eval_params = {'batch_size': batch_size}
      acc = model_eval(sess, x, y, preds, x_train, y_train, args=eval_params)
      report.train_clean_train_clean_eval = acc

# Initialize the Fast Gradient Sign Method (FGSM) attack object and graph
  fgsm = FastGradientMethod(wrap, sess=sess)
  fgsm_params = {'eps': 0.3,
                'clip_min': 0.,
                'clip_max': 1.}
  adv_x = fgsm.generate(x, **fgsm_params)
  # Consider the attack to be constant
  adv_x = tf.stop_gradient(adv_x)
  preds_adv = model(adv_x)

# Evaluate the accuracy of the MNIST model on adversarial examples
  eval_par = {'batch_size': batch_size}
  acc = model_eval(sess, x, y, preds_adv, x_test, y_test, args=eval_par)
  print('Test accuracy on adversarial examples: %0.4f\n' % acc)
  f.write('Test accuracy on adversarial examples: %0.4f\n' % acc)
  report.clean_train_adv_eval = acc

# Calculating train error
  if testing:
      eval_par = {'batch_size': batch_size}
      acc = model_eval(sess, x, y, preds_adv, x_train,
                        y_train, args=eval_par)
      report.train_clean_train_adv_eval = acc

  print("Repeating the process, using adversarial training")
  f.write("Results using Adversarial Training")
  # Redefine TF model graph
  model_2 = cnn_model(img_rows=img_rows, img_cols=img_cols,
                    channels=nchannels, nb_filters=64,
                    nb_classes=nb_classes)

  #model_2 = cnn_model(img_rows=img_rows, img_cols=img_cols,
                    #channels=nchannels, nb_filters=64,
                    #nb_classes=nb_classes)
  wrap_2 = KerasModelWrapper(model_2)
  preds_2 = model_2(x)
  fgsm2 = FastGradientMethod(wrap_2, sess=sess)

  def attack(x):
      return fgsm2.generate(x, **fgsm_params)

  preds_2_adv = model_2(attack(x))
  loss_2 = CrossEntropy(wrap_2, smoothing=label_smoothing, attack=attack)

  def evaluate_2():
  # Accuracy of adversarially trained model on legitimate test inputs
      eval_params = {'batch_size': batch_size}
      accuracy = model_eval(sess, x, y, preds_2, x_test, y_test,
                            args=eval_params)
      print('Test accuracy on legitimate examples: %0.4f' % accuracy)
      f.write('Test accuracy on legitimate examples: %0.4f' % accuracy)
      report.adv_train_clean_eval = accuracy

      # Accuracy of the adversarially trained model on adversarial examples
      accuracy = model_eval(sess, x, y, preds_2_adv, x_test,
                            y_test, args=eval_params)
      print('Test accuracy on adversarial examples: %0.4f' % accuracy)
      f.write('Test accuracy on adversarial examples: %0.4f' % accuracy)
      report.adv_train_adv_eval = accuracy

# Perform and evaluate adversarial training
  train(sess, loss_2, x_train, y_train, evaluate=evaluate_2,
      args=train_params, rng=rng)

  # Calculate training errors
  if testing:
      eval_params = {'batch_size': batch_size}
      accuracy = model_eval(sess, x, y, preds_2, x_train, y_train,
                        args=eval_params)
      report.train_adv_train_clean_eval = accuracy
      accuracy = model_eval(sess, x, y, preds_2_adv, x_train,
                        y_train, args=eval_params)
      report.train_adv_train_adv_eval = accuracy

  return report


def main(argv=None):
  from cleverhans_tutorials import check_installation
#check_installation(__file__)

  mnist_tutorial(nb_epochs=FLAGS.nb_epochs,
                batch_size=FLAGS.batch_size,
                learning_rate=FLAGS.learning_rate,
                train_dir=FLAGS.train_dir,
                filename=FLAGS.filename,
                load_model=FLAGS.load_model)


if __name__ == '__main__':
  try:  
      flags.DEFINE_integer('nb_epochs', NB_EPOCHS,
                            'Number of epochs to train model')
      flags.DEFINE_integer('batch_size', BATCH_SIZE, 'Size of training batches')
      flags.DEFINE_float('learning_rate', LEARNING_RATE,
                        'Learning rate for training')
      flags.DEFINE_string('train_dir', TRAIN_DIR,
                        'Directory where to save model.')
      flags.DEFINE_string('filename', FILENAME, 'Checkpoint filename.')
      flags.DEFINE_boolean('load_model', LOAD_MODEL,
                          'Load saved model or train.')
  except:
      print("already Defined")
  tf.compat.v1.app.run()
f.close


''' Citation for Cleverhans toolbox
@article{papernot2018cleverhans,
  title={Technical Report on the CleverHans v2.1.0 Adversarial Examples Library},
  author={Nicolas Papernot and Fartash Faghri and Nicholas Carlini and
  Ian Goodfellow and Reuben Feinman and Alexey Kurakin and Cihang Xie and
  Yash Sharma and Tom Brown and Aurko Roy and Alexander Matyasko and
  Vahid Behzadan and Karen Hambardzumyan and Zhishuai Zhang and
  Yi-Lin Juang and Zhi Li and Ryan Sheatsley and Abhibhav Garg and
  Jonathan Uesato and Willi Gierke and Yinpeng Dong and David Berthelot and
  Paul Hendricks and Jonas Rauber and Rujun Long},
  journal={arXiv preprint arXiv:1610.00768},
  year={2018}
}
'''
