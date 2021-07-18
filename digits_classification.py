
# coding: utf-8

# # MNIST digits classification with TensorFlow

# <img src="images/mnist_sample.png" style="width:30%">




import numpy as np
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')
import tensorflow as tf
print("We're using TF", tf.__version__)





import sys
sys.path.append("../..")
import grading

import matplotlib_utils
from importlib import reload
reload(matplotlib_utils)

import grading_utils
reload(grading_utils)

import keras_utils
from keras_utils import reset_tf_session





grader = grading.Grader(assignment_key="XtD7ho3TEeiHQBLWejjYAA", 
                        all_parts=["9XaAS", "vmogZ", "RMv95", "i8bgs", "rE763"])




COURSERA_TOKEN = "fSbSxtPy4ugjxYnA"
COURSERA_EMAIL = "matthew.dicicco38@myhunter.cuny.edu"


import preprocessed_mnist
X_train, y_train, X_val, y_val, X_test, y_test = preprocessed_mnist.load_dataset_from_file()



print("X_train [shape %s] sample patch:\n" % (str(X_train.shape)), X_train[1, 15:20, 5:10])
print("A closeup of a sample patch:")
plt.imshow(X_train[1, 15:20, 5:10], cmap="Greys")
plt.show()
print("And the whole sample:")
plt.imshow(X_train[1], cmap="Greys")
plt.show()
print("y_train [shape %s] 10 samples:\n" % (str(y_train.shape)), y_train[:10])


# # Linear model



X_train_flat = X_train.reshape((X_train.shape[0], -1))
print(X_train_flat.shape)

X_val_flat = X_val.reshape((X_val.shape[0], -1))
print(X_val_flat.shape)





import keras

y_train_oh = keras.utils.to_categorical(y_train, 10)
y_val_oh = keras.utils.to_categorical(y_val, 10)

print(y_train_oh.shape)
print(y_train_oh[:3], y_train[:3])


s = reset_tf_session()



W = tf.get_variable("W", dtype = tf.float32, shape=(784, 10), trainable = True) ### YOUR CODE HERE ### tf.get_variable(...) with shape[0] = 784
b = tf.get_variable("b", dtype = tf.float32, shape=(10, )) ### YOUR CODE HERE ### tf.get_variable(...)


input_X = tf.placeholder(tf.float32, shape=(None, 784)) ### YOUR CODE HERE ### tf.placeholder(...) for flat X with shape[0] = None for any batch size
input_y = tf.placeholder(tf.float32, shape=(None, 10)) ### YOUR CODE HERE ### tf.placeholder(...) for one-hot encoded true labels





logits = input_X @ W + b
probas = tf.nn.softmax(logits)
classes = tf.argmax(probas, 1) 


loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = input_y))


step = tf.train.AdamOptimizer().minimize(loss) 






s.run(tf.global_variables_initializer())

BATCH_SIZE = 512
EPOCHS = 40


simpleTrainingCurves = matplotlib_utils.SimpleTrainingCurves("cross-entropy", "accuracy")

for epoch in range(EPOCHS):  
    
    batch_losses = []
    for batch_start in range(0, X_train_flat.shape[0], BATCH_SIZE): #the data is already shuffled
        _, batch_loss = s.run([step, loss], {input_X: X_train_flat[batch_start:batch_start+BATCH_SIZE], 
                                             input_y: y_train_oh[batch_start:batch_start+BATCH_SIZE]})
        # collect batch losses, this is almost free as we need a forward pass for backprop anyway
        batch_losses.append(batch_loss)

    train_loss = np.mean(batch_losses)
    val_loss = s.run(loss, {input_X: X_val_flat, input_y: y_val_oh}) 
    train_accuracy = accuracy_score(y_train, s.run(classes, {input_X: X_train_flat}))  #usually skipped
    valid_accuracy = accuracy_score(y_val, s.run(classes, {input_X: X_val_flat}))  
    simpleTrainingCurves.add(train_loss, val_loss, train_accuracy, valid_accuracy)





## GRADED PART, DO NOT CHANGE!
 
grader.set_answer("9XaAS", grading_utils.get_tensors_shapes_string([W, b, input_X, input_y, logits, probas, classes]))
# Validation loss
grader.set_answer("vmogZ", s.run(loss, {input_X: X_val_flat, input_y: y_val_oh}))
# Validation accuracy
grader.set_answer("RMv95", accuracy_score(y_val, s.run(classes, {input_X: X_val_flat})))



grader.submit(COURSERA_EMAIL, COURSERA_TOKEN)


# # MLP with hidden layers

# Previously we've coded a dense layer with matrix multiplication by hand. 
# But this is not convenient, you have to create a lot of variables and your code becomes a mess. 
# In TensorFlow there's an easier way to make a dense layer:
# ```python
# hidden1 = tf.layers.dense(inputs, 256, activation=tf.nn.sigmoid)


hidden1 = tf.layers.dense(input_X, 256, activation = tf.nn.sigmoid)
hidden2 = tf.layers.dense(hidden1, 256, activation = tf.nn.sigmoid)
logits = tf.layers.dense(hidden2, 10)

probas = tf.nn.softmax(logits)
classes = tf.argmax(probas, 1)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = input_y)) 
step = tf.train.AdamOptimizer().minimize(loss)

s.run(tf.global_variables_initializer())

BATCH_SIZE = 512
EPOCHS = 40

# for logging the progress right here in Jupyter (for those who don't have TensorBoard)
simpleTrainingCurves = matplotlib_utils.SimpleTrainingCurves("cross-entropy", "accuracy")

for epoch in range(EPOCHS):  
    
    batch_losses = []
    for batch_start in range(0, X_train_flat.shape[0], BATCH_SIZE): 
        _, batch_loss = s.run([step, loss], {input_X: X_train_flat[batch_start:batch_start+BATCH_SIZE], 
                                             input_y: y_train_oh[batch_start:batch_start+BATCH_SIZE]})
        
        batch_losses.append(batch_loss)

    train_loss = np.mean(batch_losses)
    val_loss = s.run(loss, {input_X: X_val_flat, input_y: y_val_oh}) 
    train_accuracy = accuracy_score(y_train, s.run(classes, {input_X: X_train_flat}))  
    valid_accuracy = accuracy_score(y_val, s.run(classes, {input_X: X_val_flat}))  
    simpleTrainingCurves.add(train_loss, val_loss, train_accuracy, valid_accuracy)




## GRADED PART, DO NOT CHANGE!
# Validation loss for MLP
grader.set_answer("i8bgs", s.run(loss, {input_X: X_val_flat, input_y: y_val_oh}))
# Validation accuracy for MLP
grader.set_answer("rE763", accuracy_score(y_val, s.run(classes, {input_X: X_val_flat})))





# you can make submission with answers so far to check yourself at this stage
grader.submit(COURSERA_EMAIL, COURSERA_TOKEN)

