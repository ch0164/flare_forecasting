# Custom Imports
import pandas as pd

from source.utilities import *

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder

# Disable Warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

experiment = "neural_network"
experiment_caption = experiment.title().replace("_", " ")
now_string, figure_directory, metrics_directory, other_directory = \
        build_experiment_directories(experiment)
# tf.compat.v1.enable_eager_execution()
print(tf.compat.v1.executing_eagerly())

def get_tss(y_true, y_pred, convert=True):
    if convert:
        y_true = y_true.numpy()
        y_pred = y_pred.numpy()
    y_pred = np.where(y_pred > 0.5, 1, 0)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    detection_rate = tp / float(tp + fn)
    false_alarm_rate = fp / float(fp + tn)
    tss = detection_rate - false_alarm_rate
    return tss

def get_hss(y_true, y_pred, convert=True):
    if convert:
        y_true = y_true.numpy()
        y_pred = y_pred.numpy()
    y_pred[np.nonzero(y_pred)] = 1
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    p = tp + fn
    n = fp + tn
    numerator = 2 * ((tp * tn) - (fn * fp))
    denominator = p * (fn + tn) + n * (tp + fp)
    hss = numerator / denominator
    return hss

data_dir = r"C:\Users\youar\PycharmProjects\flare_forecasting\results\time_series_goodness_of_fit\other/"
# df = pd.DataFrame()
# for index, param in enumerate(FLARE_PROPERTIES):
#     temp_df = pd.read_csv(data_dir + param.lower() + ".csv")
#     if index == 0:
#         df = temp_df[["FLARE_TYPE", "COINCIDENCE"]]
#     temp_df.drop(["FLARE_TYPE", "COINCIDENCE", "Unnamed: 0"], axis=1, inplace=True)
#     df = pd.concat([df, temp_df], axis=1)
# df["LABEL"] = df["FLARE_TYPE"].apply(get_ar_class)
#
# train_ratio = 0.7
# test_ratio = 0.2
# validation_ratio = 0.1
# train_df, test_df = train_test_split(df, random_state=10,
#                                                         test_size=test_ratio,
#                                                     stratify=df["FLARE_TYPE"])
# train_df, validation_df = train_test_split(train_df,
#                                           stratify=train_df["FLARE_TYPE"],
#                                           random_state=10,
#                                           test_size=validation_ratio / (
#                                                   train_ratio + test_ratio))
# for index, param in enumerate(FLARE_PROPERTIES):
#     # print(train_df[[f"{param}_{i}" for i in range(1, 121)]])
#     # print(train_df[[f"{param}_{i}" for i in range(1, 121)]].mean())
#     # print(train_df[[f"{param}_{i}" for i in range(1, 121)]].mean(axis=1))
#     # exit(1)
#     train_mean = train_df[[f"{param}_{i}" for i in range(1, 121)]].mean()
#     train_std = train_df[[f"{param}_{i}" for i in range(1, 121)]].std()
#     # print(train_mean)
#     # print(train_std)
#
#     for i in range(1, 121):
#         train_df[f"{param}_{i}"] = (train_df[
#                                         f"{param}_{i}"] - train_mean[i-1]) / train_std[i-1]
#         validation_df[f"{param}_{i}"] = (validation_df[
#                                         f"{param}_{i}"] - train_mean[i-1]) / train_std[i-1]
#         test_df[f"{param}_{i}"] = (test_df[
#                                         f"{param}_{i}"] - train_mean[i-1]) / train_std[i-1]
#
# for label, df in zip(["train", "validation", "test"], [train_df, validation_df, test_df]):
#     label_df = df[["FLARE_TYPE", "COINCIDENCE", "LABEL"]]
#     label_df.to_csv(f"{other_directory}{label}_flare_labels.csv", index=False)
# for label, df in zip(["train", "validation", "test"], [train_df, validation_df, test_df]):
#     df.drop(["FLARE_TYPE", "COINCIDENCE", "LABEL"], axis=1, inplace=True)
#     df.to_csv(f"{other_directory}flare_{label}_normalized.csv", index=False)

# labels_df = pd.read_csv(f"{other_directory}flare_labels.csv")
train_X_df = pd.read_csv(f"{other_directory}flare_train_normalized.csv")
test_X_df = pd.read_csv(f"{other_directory}flare_test_normalized.csv")
validation_X_df = pd.read_csv(f"{other_directory}flare_validation_normalized.csv")
train_y_df = pd.read_csv(f"{other_directory}flare_train_labels.csv")
test_y_df = pd.read_csv(f"{other_directory}flare_test_labels.csv")
validation_y_df = pd.read_csv(f"{other_directory}flare_validation_labels.csv")

# kept_params = ["SAVNCPP", "TOTUSJH", "TOTUSJZ", "R_VALUE", "ABSNJZH", "USFLUX", "AREA_ACR", "SHRGT45", "g_s"]
kept_params = FLARE_PROPERTIES
dfs = [train_X_df, test_X_df, validation_X_df]
train_X_df = train_X_df[[f"{param}_{i}" for param in kept_params for i in range(1, 121)]]
test_X_df = test_X_df[[f"{param}_{i}" for param in kept_params for i in range(1, 121)]]
validation_X_df = validation_X_df[[f"{param}_{i}" for param in kept_params for i in range(1, 121)]]



# (batch_size, input_dim, window)
batch_size = 128
epochs = 15
input_dim = len(kept_params)
window = 120
feature_length = window * input_dim
input_shape = (feature_length, 1)

y_train = np.asarray(train_y_df["LABEL"].values).astype('float32').reshape((-1, 1))
y_val = np.asarray(validation_y_df["LABEL"].values).astype('float32').reshape((-1, 1))
y_test = np.asarray(test_y_df["LABEL"].values).astype('float32').reshape((-1, 1))

model_name = "resnet"

# n_hidden_1 = 120 # 1st layer number of neurons
# n_hidden_2 = 60 # 2nd layer number of neurons
# n_hidden_3 = 30 # 2nd layer number of neurons
# n_hidden_4 = 10 # 2nd layer number of neurons
# n_input = 2400
# n_classes = 1
#
# weights = {
#     'h1': tf.Variable(tf.random.normal([n_input, n_hidden_1])),
#     'h2': tf.Variable(tf.random.normal([n_hidden_1, n_hidden_2])),
#     'h3': tf.Variable(tf.random.normal([n_hidden_2, n_hidden_3])),
#     'h4': tf.Variable(tf.random.normal([n_hidden_3, n_hidden_4])),
#     'out': tf.Variable(tf.random.normal([n_hidden_4, n_classes]))
# }
# biases = {
#     'b1': tf.Variable(tf.random.normal([n_hidden_1])),
#     'b2': tf.Variable(tf.random.normal([n_hidden_2])),
#     'b3': tf.Variable(tf.random.normal([n_hidden_3])),
#     'b4': tf.Variable(tf.random.normal([n_hidden_4])),
#     'out': tf.Variable(tf.random.normal([n_classes]))
# }
#
# def multilayer_perceptron(x):
#     # Hidden fully connected layer with 256 neurons
#     layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
#     layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
#     layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
#     layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
#     # Output fully connected layer with a neuron for each class
#     out_layer = tf.matmul(layer_4, weights['out']) + biases['out']
#     return out_layer
#
# # Construct model
# X = tf.compat.v1.placeholder("float", [None, n_input])
# logits = multilayer_perceptron(X)

# expected input data shape: (batch_size, timesteps, data_dim)
def build_model():
    model = keras.Sequential()
    model.add(layers.Reshape((120, 20), input_shape=(2400,)))
    # model.add(layers.Dense())
    model.add(layers.LSTM(units=256,
                          # return_sequences=True,
                          batch_input_shape=(batch_size, window, input_dim)))
    # model.add(layers.LSTM(units=64,
    #                       return_sequences=True,
    #                       batch_input_shape=(batch_size, window, input_dim)))
    # model.add(layers.LSTM(units=64,
    #                       # return_sequences=True,
    #                       batch_input_shape=(batch_size, window, input_dim)))
    model.add(layers.Dense(1, activation='relu'))
    model.summary()
    return model


def residual_block(x, filters, stride=1, projection=False):
    shortcut = x
    if projection:
        shortcut = layers.Conv1D(filters, 1, strides=stride, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    x = layers.Conv1D(filters, 1, strides=stride, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv1D(filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv1D(filters, 1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    return x


def resnet50(input_tensor):
    x = layers.ZeroPadding1D(padding=3)(input_tensor)
    x = layers.Conv1D(64, 7, strides=2, padding='valid')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(3, strides=2, padding='same')(x)

    x = residual_block(x, filters=64, projection=True)
    for _ in range(2):
        x = residual_block(x, filters=64)

    x = residual_block(x, filters=128, stride=2, projection=True)
    for _ in range(3):
        x = residual_block(x, filters=128)

    x = residual_block(x, filters=256, stride=2, projection=True)
    for _ in range(5):
        x = residual_block(x, filters=256)

    x = residual_block(x, filters=512, stride=2, projection=True)
    for _ in range(2):
        x = residual_block(x, filters=512)

    x = layers.MaxPooling1D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(1, activation='sigmoid')(x)

    model = keras.Model(inputs=input_tensor, outputs=x, name='resnet50')
    model.summary()
    return model


def build_ocdcnn_model():
    # Define the model architecture
    model = keras.Sequential()

    # model.add(layers.Reshape((120, 20), input_shape=(2400,)))

    # Add a 1D convolutional layer with 5 filters of length 3 and input size of 2400
    model.add(layers.Conv1D(filters=5, kernel_size=3, activation='relu',
                     input_shape=input_shape))
    # model.add(layers.MaxPool1D(pool_size=2, strides=2))
    model.add(layers.BatchNormalization())
    # model.add(layers.Activation(activation="relu"))
    # model.add(layers.BatchNormalization())
    # model.add(layers.Activation("relu"))
    # model.add(layers.Conv1D(filters=5, kernel_size=3, activation='relu',
    #                         input_shape=input_shape))
    # model.add(layers.Add())
    # model.add(layers.BatchNormalization())
    # model.add(layers.Activation("relu"))
    # model.add(layers.Conv1D(filters=5, kernel_size=3, activation='relu',
    #                         input_shape=input_shape))
    # model.add(layers.Add())
    # model.add(layers.BatchNormalization())
    # model.add(layers.Activation("relu"))
    # model.add(layers.MaxPool1D(pool_size=2, strides=2))



    model.add(layers.Conv1D(filters=5, kernel_size=3, activation='relu',
                            input_shape=input_shape))
    # model.add(layers.MaxPool1D(pool_size=2, strides=2))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv1D(filters=5, kernel_size=3, activation='relu',
                            input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool1D(pool_size=2, strides=2))

    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='relu'))
    # model.add(layers.Dense(64, activation='relu'))
    # model.add(layers.Dense(1))

    # Print the model summary
    model.summary()

    return model

def train_model(model):
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=[get_tss, get_hss],
                  run_eagerly=True)
    history = model.fit(train_X_df.values, y_train,
                        validation_data=(validation_X_df.values, y_val),
                        batch_size=batch_size,
                        epochs=epochs)
    model.save(f"{other_directory}{model_name}")
    # Gather history parameters.
    history_dict = {
        "tss": history.history["get_tss"],
        "validation_tss": history.history['val_get_tss'],
        "hss": history.history["get_hss"],
        "validation_hss": history.history['val_get_hss'],
        "loss": history.history['loss'],
        "validation_loss": history.history['val_loss'],
    }

    # Save to a CSV in the model directory.
    if os.path.exists(f"{other_directory}{model_name}/history.csv"):
        df = pd.read_csv(f"{other_directory}{model_name}/history.csv")
        history_df = pd.DataFrame(data=history_dict)
        df = pd.concat([df, history_df], axis=0)
    else:
        df = pd.DataFrame(data=history_dict)
    df.to_csv(f"{other_directory}{model_name}/history.csv")


def plot_history(filename):
    # Read file.
    df = pd.read_csv(filename)
    epochs_count = len(df["tss"])
    epochs_np = np.array(list(range(1, epochs_count + 1)))

    fig, ax = plt.subplots(2)

    # Plot training vs. validation accuracy.
    print(df["tss"].shape)
    ax[0].plot(epochs_np, df["tss"], 'b', label='Training TSS')
    ax[0].plot(epochs_np, df["validation_tss"], 'r',
             label='Validation TSS')
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("TSS")
    ax[0].set_title('Training and Validation TSS')
    ax[0].legend()

    # Plot training vs. validation loss.
    ax[1].plot(epochs_np, df["loss"], 'b', label='Training Loss')
    ax[1].plot(epochs_np, df["validation_loss"], 'r', label='Validation Loss')
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Binary Crossentropy Loss")
    ax[1].set_title('Training and Validation Loss')
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(f"{figure_directory}{model_name}_tss_loss.png")
# # Generate dummy training data
# x_train = np.random.random((1000, timesteps, data_dim))
# y_train = np.random.random((1000, num_classes))
#
# # Generate dummy validation data
# x_val = np.random.random((100, timesteps, data_dim))
# y_val = np.random.random((100, num_classes))
#
# model.fit(x_train, y_train,
#           batch_size=64, epochs=5,
#           validation_data=(x_val, y_val))


def test_model(filename):
    model = keras.models.load_model(filename, custom_objects={"get_tss": get_tss,
                                                              "get_hss": get_hss})
    y_pred = model.predict(test_X_df.values)
    y_true = test_y_df["LABEL"].values
    tss = get_tss(y_true, y_pred, convert=False)
    hss = get_hss(y_true, y_pred, convert=False)

    print()
    print(model_name)
    print("-" * 50)
    print("TSS:", tss)
    print("HSS:", hss)


def mlp_classfifier():
    global train_X_df, train_y_df, test_X_df, test_y_df
    train_X_df = pd.concat([train_X_df, validation_X_df])
    train_y_df = pd.concat([train_y_df, validation_y_df])
    X = pd.concat([train_X_df, test_X_df])
    y = pd.concat([train_y_df, test_y_df])
    df = pd.concat([X, y], axis=1)
    # df = df.loc[df["COINCIDENCE"] == False]

    best_triple = [f"AREA_ACR_{i}" for i in range(1, 121)] + [f"SAVNCPP_{i}" for i in range(1, 121)] + [f"TOTUSJH_{i}" for i in range(1, 121)]

    tss_list = []
    for train_index in range(100):
        print(f"{train_index}/100")
        train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["FLARE_TYPE"])
        # train_X_df = train_df.drop(["FLARE_TYPE", "COINCIDENCE", "LABEL"], axis=1)
        train_X_df = train_df[best_triple]
        train_y_df = train_df[["FLARE_TYPE", "COINCIDENCE", "LABEL"]]
        test_X_df = test_df[best_triple]
        test_y_df = test_df[["FLARE_TYPE", "COINCIDENCE", "LABEL"]]
        model = MLPClassifier(hidden_layer_sizes=(120, 60, 30, 10),
                        solver="adam",
                        activation='relu',
                        learning_rate_init=0.0001,
                        batch_size=64,
                        max_iter=300,
                        n_iter_no_change=50,
                        learning_rate='adaptive')
        model.fit(train_X_df.values, train_y_df["LABEL"].values)
        y_pred = model.predict(test_X_df.values)
        y_true = test_y_df["LABEL"].values
        tss = get_tss(y_true, y_pred, convert=False)
        tss_list.append(tss)
        print(tss)
    print(tss_list)
    print(f"TSS: {np.mean(tss_list):.4f} +/- {np.std(tss_list):.4f}")

def main() -> None:
    # resnet = keras.models.load_model(
    #     r"C:\Users\youar\PycharmProjects\flare_forecasting\resnet_mnist_digits\resnet_mnist_digits.hdf5")
    # resnet.summary()
    # exit(1)
    # model = build_ocdcnn_model()
    # exit(1)
    model = resnet50(keras.Input(shape=(2400, 1)))
    train_model(model)
    test_model(f"{other_directory}{model_name}")
    # plot_history(f"{other_directory}{model_name}/history.csv")


if __name__ == "__main__":
    main()
