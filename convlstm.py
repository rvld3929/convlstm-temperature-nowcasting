from netCDF4 import Dataset, num2date
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import keras
import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt

# Reading the data
ncfile = "java-temperature-24h-2021-2023"
f1 = Dataset(ncfile)
print(f1.variables.keys())
lats = f1.variables["latitude"][:]
lons = f1.variables["longitude"][:]
time = f1.variables["time"]
dates = num2date(time[:], time.units)
times = [date.strftime("%Y-%m-%d %H:%M:%S") for date in dates]
t2m = f1.variables["t2m"][:]
t2m = t2m[:, 0,:, :] - 273.15
ds = xr.Dataset(
    {
"t2m": (("times","lat", "lon"), t2m),
    },
    {
"times": times,
"lat": lats,
"lon": lons,
    },
)
df = ds.to_dataframe()
df.reset_index(inplace=True)

# Sorting, reshaping, and normalizing the data
df = df.sort_values(by=['times','lat','lon'], ascending=[True, False, True])
dataset = df["t2m"].to_numpy().reshape(-1,len(lats),len(lons),1)
dataset = dataset[-2000:]
max_ds = np.max(dataset)
min_ds = np.min(dataset)
dataset = (dataset-min_ds)/(max_ds-min_ds)

# Divide data into training, validation, and testing
num_train_samples = int(0.7 * dataset.shape[0])
num_val_samples = int(0.15 * dataset.shape[0])
num_test_samples = dataset.shape[0] - num_train_samples - num_val_samples

sequences_length = 24

train_dataset = keras.utils.timeseries_dataset_from_array(
    dataset,
    None,
    sequence_length=sequences_length,
    start_index=0,
    end_index=num_train_samples)
val_dataset = keras.utils.timeseries_dataset_from_array(
    dataset,
    None,
    sequence_length=sequences_length,
    start_index=num_train_samples,
    end_index=num_train_samples + num_val_samples)
test_dataset = keras.utils.timeseries_dataset_from_array(
    dataset,
    None,
    sequence_length=sequences_length,
    start_index=num_train_samples + num_val_samples)

dataset_train = np.empty([0,sequences_length,15, 45, 1])
dataset_val = np.empty([0,sequences_length,15, 45, 1])
dataset_test = np.empty([0,sequences_length,15, 45, 1])

for batch_train in train_dataset:
    dataset_train = np.concatenate((dataset_train, batch_train), axis=0)
    
for batch_val in val_dataset:
    dataset_val = np.concatenate((dataset_val, batch_val), axis=0)
    
for batch_test in test_dataset:
    dataset_test = np.concatenate((dataset_test, batch_test), axis=0)

def create_shifted_frames(data):
    x = data[:, 0 : data.shape[1] - 1, :, :]
    y = data[:, 1 : data.shape[1], :, :]
    return x, y

x_train, y_train = create_shifted_frames(dataset_train)
x_val, y_val = create_shifted_frames(dataset_val)
x_test, y_test = create_shifted_frames(dataset_test)

# Create and compile the model
inp = layers.Input(shape=(None, *x_train.shape[2:]))

x = layers.BatchNormalization()(inp)
x = layers.ConvLSTM2D(
    filters=16,
    kernel_size=(5, 5),
    padding="same",
    return_sequences=True,
    activation="relu",
)(x)
x = layers.BatchNormalization()(x)
x = layers.ConvLSTM2D(
    filters=32,
    kernel_size=(3, 3),
    padding="same",
    return_sequences=True,
    activation="relu",
)(x)
x = layers.BatchNormalization()(x)
x = layers.BatchNormalization()(x)
x = layers.ConvLSTM2D(
    filters=32,
    kernel_size=(1, 1),
    padding="same",
    return_sequences=True,
    activation="relu",
)(x)
x = layers.BatchNormalization()(x)
x = layers.Conv3D(
    filters=1, kernel_size=(3, 3, 3), activation="sigmoid", padding="same"
)(x)

model = keras.models.Model(inp, x)
model.compile(
    loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam(),
)

# Fit the model
early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=3)

epochs = 40
batch_size = 8

model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_val, y_val),
    callbacks=[early_stopping, reduce_lr]
)

# Save the model
model.save("model.h5")