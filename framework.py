# %% [markdown]
# # Fernando Jimenez Pereyra
#     A01734609

# %%
# !pip install pandas numpy matplotlib dataprep sklearn
# !pip install tensorflow-gpu

# %%
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

np.set_printoptions(precision=3, suppress=True)

# %%
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers

# %%
# !pip install git+https://github.com/tensorflow/docs

import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots

# %%
print('Tensorflow: ', tf.__version__)

# %%
df = pd.read_csv('./clean_fish.csv')
df = df.drop(columns=['Unnamed: 0'])
df

# %%
train, test = train_test_split(df, test_size=0.33)

# %%
y_train = train['Weight']
y_test = test['Weight']
x_train = train.drop(columns=['Weight'])
x_test = test.drop(columns=['Weight'])

# %%
test_results = {}

# %%
def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.xlabel('Epoch')
  plt.ylabel('Error [MPG]')
  plt.legend()
  plt.grid(True)
  plt.show()

# %%
normalizer = tf.keras.layers.Normalization(axis=-1)

# %%
normalizer.adapt(np.array(x_train))

# %%
print(normalizer.mean.numpy())

# %%
linear_model = tf.keras.Sequential([
    normalizer,
    layers.Dense(units=1)
])

# %%
x_train[:10]

# %%
linear_model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss='mean_absolute_error',
    metrics=[tf.keras.metrics.Accuracy()])

# %%
history = linear_model.fit(
    x_train,
    y_train,
    epochs=100,
    # Suppress logging.
    verbose=0,
    # Calculate validation results on 20% of the training data.
    validation_split = 0.2)

# %%
plot_loss(history)

# %%
test_results['linear_model'] = linear_model.evaluate(
    x_test, y_test, verbose=0)

# %%
test_predictions = linear_model.predict(x_test).flatten()

a = plt.axes(aspect='equal')
plt.scatter(y_test, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
lims = [0, 2000]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
plt.show()

# %%
print(test_results['linear_model'])

# %% [markdown]
# # Neural network

# %%
def compile_model(model):

  model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(0.001), 
                metrics=[
                  tf.keras.losses.BinaryCrossentropy(
                      from_logits=True, name='binary_crossentropy'),
                  'accuracy'])
  return model

# %%
tiny_model = keras.Sequential([
        normalizer,
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
tiny_dnn_model = compile_model(tiny_model)

# %%
history = tiny_dnn_model.fit(
    x_train,
    y_train,
    validation_split=0.2,
    verbose=0, epochs=100)

# %%
historys = {}
historys['tiny'] = history  
plot_loss(history)

# %%
try:
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
except KeyError:
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
plt.title('Accuracy vs. epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='lower right')
plt.show() 

# %%
test_results['tiny_dnn_model'] = tiny_dnn_model.evaluate(x_test, y_test, verbose=0)

# %%
test_predictions = tiny_dnn_model.predict(x_test).flatten()

a = plt.axes(aspect='equal')
plt.scatter(y_test, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
lims = [0, 2000]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
plt.show()


# %%
print(r2_score(y_test, test_predictions))

# %%
error = test_predictions - y_test
plt.hist(error, bins=25)
plt.xlabel('Prediction Error [MPG]')
_ = plt.ylabel('Count')
plt.show()


# %%
small_model = keras.Sequential([
        normalizer,
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
small_dnn_model = compile_model(small_model)

# %%
history = small_dnn_model.fit(
    x_train,
    y_train,
    validation_split=0.2,
    verbose=0, epochs=100)

# %%
historys['small'] = history
plot_loss(history)

# %%
test_results['small_bdnn_model'] = small_dnn_model.evaluate(x_test, y_test, verbose=0)

# %%
test_predictions = small_dnn_model.predict(x_test).flatten()

a = plt.axes(aspect='equal')
plt.scatter(y_test, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
lims = [0, 2000]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
plt.show()

# %%
print(r2_score(y_test, test_predictions))

# %%
error = test_predictions - y_test
plt.hist(error, bins=25)
plt.xlabel('Prediction Error [MPG]')
_ = plt.ylabel('Count')
plt.show()

# %%
big_model = keras.Sequential([
        normalizer,
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(5, activation='relu'),
        layers.Dense(5, activation='relu'),
        layers.Dense(1)
    ])
big_dnn_model = compile_model(big_model)

# %%
history = big_dnn_model.fit(
    x_train,
    y_train,
    validation_split=0.2,
    verbose=0, epochs=100)
historys['big'] = history

# %%
plot_loss(history)

# %%
test_results['big_bdnn_model'] = big_dnn_model.evaluate(x_test, y_test, verbose=0)

# %%
test_predictions = big_dnn_model.predict(x_test).flatten()

a = plt.axes(aspect='equal')
plt.scatter(y_test, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
lims = [0, 2000]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
plt.show()

# %%
print(r2_score(y_test, test_predictions))

# %%
error = test_predictions - y_test
plt.hist(error, bins=25)
plt.xlabel('Prediction Error [MPG]')
_ = plt.ylabel('Count')
plt.show()

# %%
# plot_loss(historys)
for i in historys:
    plt.plot(historys[i].history['loss'], label= i + ' loss')
    plt.plot(historys[i].history['val_loss'], label= i + ' val_loss')
plt.xlabel('Epoch')
plt.ylabel('Error [MPG]')
plt.legend()
plt.grid(True)
plt.show()

# %%
plotter = tfdocs.plots.HistoryPlotter(metric = 'binary_crossentropy', smoothing_std=10)
plotter.plot(historys)
a = plt.xscale('log')
plt.xlabel("Epochs [Log Scale]")
plt.show()

# %%
big_model_r = keras.Sequential([
        normalizer,
        layers.Dense(64, kernel_regularizer=regularizers.l2(0.001), activation='relu'),
        layers.Dense(64, kernel_regularizer=regularizers.l2(0.001), activation='relu'),
        layers.Dense(64, kernel_regularizer=regularizers.l2(0.001), activation='relu'),
        layers.Dense(5, kernel_regularizer=regularizers.l2(0.001), activation='relu'),
        layers.Dense(5, kernel_regularizer=regularizers.l2(0.001), activation='relu'),
        layers.Dense(1)
    ])
big_dnn_model_r = compile_model(big_model_r)

# %%
history = big_dnn_model_r.fit(
    x_train,
    y_train,
    validation_split=0.2,
    verbose=0, epochs=100)
historys['big_r'] = history

# %%
plot_loss(history)

# %%
test_results['big_bdnn_model_r'] = big_dnn_model_r.evaluate(x_test, y_test, verbose=0)

# %%
test_predictions = big_dnn_model_r.predict(x_test).flatten()

a = plt.axes(aspect='equal')
plt.scatter(y_test, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
lims = [0, 2000]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
plt.show()

# %%
print(r2_score(y_test, test_predictions))

# %%
error = test_predictions - y_test
plt.hist(error, bins=25)
plt.xlabel('Prediction Error [MPG]')
_ = plt.ylabel('Count')
plt.show()

# %%
print(test_results)

# %%
big_model_r = keras.Sequential([
        normalizer,
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),    
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),    
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),    
        layers.Dense(5, activation='relu'),
        layers.Dropout(0.5), 
        layers.Dense(5, activation='relu'),
        layers.Dropout(0.5), 
        layers.Dense(1)
    ])
big_dnn_model_r = compile_model(big_model_r)

# %%
history = big_dnn_model_r.fit(
    x_train,
    y_train,
    validation_split=0.2,
    verbose=0, epochs=100)
historys['big_r'] = history

# %%
plot_loss(history)

# %%
test_results['big_bdnn_model_r'] = big_dnn_model_r.evaluate(x_test, y_test, verbose=0)

# %%
test_predictions = big_dnn_model_r.predict(x_test).flatten()

a = plt.axes(aspect='equal')
plt.scatter(y_test, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
lims = [0, 2000]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
plt.show()
# %%
print(r2_score(y_test, test_predictions))

# %%
error = test_predictions - y_test
plt.hist(error, bins=25)
plt.xlabel('Prediction Error [MPG]')
_ = plt.ylabel('Count')
plt.show()

# %%
# tf.keras.losses.BinaryCrossentropy(
                    #   from_logits=True, name='binary_crossentropy'),
                #   'accuracy']

# %%
print(test_results)


