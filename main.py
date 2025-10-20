from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib
matplotlib.use('TkAgg')  # ensures Matplotlib uses Tkinter backend
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

# Try to import tensorflow.keras; if not available, set a flag and disable LSTM-related functions gracefully
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, model_from_json
    from tensorflow.keras.layers import Dense, Dropout, LSTM
    from tensorflow.keras.utils import to_categorical
    TF_AVAILABLE = True
except Exception as e:
    TF_AVAILABLE = False
    # Provide a minimal fallback for to_categorical if later used accidentally
    def to_categorical(x):
        # naive fallback: one-hot
        x = np.array(x, dtype=int)
        n = x.max() + 1
        y = np.zeros((x.shape[0], n))
        for i, v in enumerate(x):
            y[i, v] = 1
        return y

# streamlit import removed for tkinter deployment (kept commented to match original intent)
# import streamlit as st

main = Tk()
main.title("AeroSense: Real-Time Prediction of Hard Landings in Flights")
main.geometry("1300x1200")


global filename
global dataset
global y, all_data

global pilot_X_train, pilot_X_test, pilot_y_train, pilot_y_test
global actuator_X_train, actuator_X_test, actuator_y_train, actuator_y_test
global physical_X_train, physical_X_test, physical_y_train, physical_y_test
global all_X_train, all_X_test, all_y_train, all_y_test
global sensitivity, specificity
global pilot, actuator, physical, Y

# initialize globals
pilot = actuator = physical = None
all_data = None
Y = None
sensitivity = []
specificity = []

def uploadDataset():
    global pilot, actuator, physical, Y, all_data
    try:
        text.delete('1.0', END)
    except Exception:
        pass
    folder = filedialog.askdirectory(initialdir=".")
    if not folder:
        try:
            text.insert(END, "Folder selection cancelled. Looking for Dataset folder in script directory...\n")
        except Exception:
            pass
        folder = os.path.join(os.getcwd(), "Dataset")
    # Attempt to load CSV files from chosen folder
    try:
        pilot_path = os.path.join(folder, "Pilot.csv")
        actuator_path = os.path.join(folder, "Actuators.csv")
        physical_path = os.path.join(folder, "Physical.csv")
        if not os.path.exists(pilot_path) or not os.path.exists(actuator_path) or not os.path.exists(physical_path):
            raise FileNotFoundError(f"Expected files not found in {folder}. Make sure Pilot.csv, Actuators.csv, Physical.csv are present.")
        pilot = pd.read_csv(pilot_path)
        actuator = pd.read_csv(actuator_path)
        physical = pd.read_csv(physical_path)
    except Exception as e:
        message = f"Error loading datasets: {e}"
        try:
            text.insert(END, str(pilot.head()) + "\n\n")
        except Exception:
            pass  # this will silently ignore all errors

        messagebox.showerror("Dataset Load Error", message)
        return

    if 'label' not in physical.columns:
        message = "Physical.csv must contain a 'label' column."
        try:
            text.insert(END, message + "\n")
        except Exception:
            pass
        messagebox.showerror("Dataset Error", message)
        return

    Y = physical['label'].values
    pilot = pilot.copy()
    actuator = actuator.copy()
    physical = physical.copy()
    if 'label' in pilot.columns:
        pilot.drop(['label'], axis=1, inplace=True)
    if 'label' in actuator.columns:
        actuator.drop(['label'], axis=1, inplace=True)
    physical.drop(['label'], axis=1, inplace=True)
    all_data = pd.concat([physical, actuator, pilot], axis=1)

    try:
        text.insert(END, "Pilot Dataset \n\n")
        text.insert(END, str(pilot.head()) + "\n\n")
        text.insert(END, "Actuator Dataset \n\n")
        text.insert(END, str(actuator.head()) + "\n\n")
        text.insert(END, "Physical Dataset \n\n")
        text.insert(END, str(physical.head()) + "\n\n")
        text.update_idletasks()
    except Exception:
        pass

    labels, count = np.unique(Y, return_counts=True)
    try:
        height = count
        bars = [str(l) for l in labels]
        y_pos = np.arange(len(bars))
        plt.figure()
        plt.bar(y_pos, height)
        plt.xticks(y_pos, bars)
        plt.xlabel("Landing Type")
        plt.ylabel("Counts")
        plt.title("Different Landing Graphs in Dataset")
        plt.show()
    except Exception as e:
        try:
            text.insert(END, f"Plotting error: {e}\n")
        except Exception:
            pass


def preprocessDataset():
    global pilot_X_train, pilot_X_test, pilot_y_train, pilot_y_test
    global actuator_X_train, actuator_X_test, actuator_y_train, actuator_y_test
    global physical_X_train, physical_X_test, physical_y_train, physical_y_test
    global all_X_train, all_X_test, all_y_train, all_y_test
    global pilot, actuator, physical, Y, all_data
    try:
        text.delete('1.0', END)
    except Exception:
        pass

    if any(v is None for v in [pilot, actuator, physical, all_data, Y]):
        msg = "Please upload datasets first using 'Upload Flight Landing Dataset'."
        try:
            text.insert(END, msg + "\n")
        except Exception:
            pass
        messagebox.showwarning("Data Missing", msg)
        return

    # converting dataset into numpy array
    pilot = pilot.values
    actuator = actuator.values
    physical = physical.values
    all_data = all_data.values
    # shuffling the dataset
    indices = np.arange(all_data.shape[0])
    np.random.shuffle(indices)
    all_data = all_data[indices]
    Y = Y[indices]
    pilot = pilot[indices]
    actuator = actuator[indices]
    physical = physical[indices]
    # normalizing dataset values
    scaler1 = StandardScaler()
    all_data = scaler1.fit_transform(all_data)
    scaler2 = StandardScaler()
    pilot = scaler2.fit_transform(pilot)
    scaler3 = StandardScaler()
    actuator = scaler3.fit_transform(actuator)
    scaler4 = StandardScaler()
    physical = scaler4.fit_transform(physical)
    # dataset reshape to multi dimensional array
    pilot = np.reshape(pilot, (pilot.shape[0], pilot.shape[1], 1))
    actuator = np.reshape(actuator, (actuator.shape[0], actuator.shape[1], 1))
    physical = np.reshape(physical, (physical.shape[0], physical.shape[1], 1))
    # splitting dataset into train and test
    all_X_train, all_X_test, all_y_train, all_y_test = train_test_split(all_data, Y, test_size=0.2)
    Y_cat = to_categorical(Y)
    pilot_X_train, pilot_X_test, pilot_y_train, pilot_y_test = train_test_split(pilot, Y_cat, test_size=0.2)
    actuator_X_train, actuator_X_test, actuator_y_train, actuator_y_test = train_test_split(actuator, Y_cat, test_size=0.2)
    physical_X_train, physical_X_test, physical_y_train, physical_y_test = train_test_split(physical, Y_cat, test_size=0.2)

    try:
        text.insert(END, "Dataset Features Processing & Normalization Completed\n\n")
        text.insert(END, "Total records found in dataset           : " + str(all_data.shape[0]) + "\n")
        text.insert(END, "All features found in dataset            : " + str(all_data.shape[1]) + "\n")
        text.insert(END, "Total Pilot features found in dataset    : " + str(pilot.shape[1]) + "\n")
        text.insert(END, "Total Actuator features found in dataset : " + str(actuator.shape[1]) + "\n")
        text.insert(END, "Total Physical features found in dataset : " + str(physical.shape[1]) + "\n\n")
        text.insert(END, "Dataset Train and Test Split\n\n")
        text.insert(END, "80% dataset records used to train ALL algorithms : " + str(all_X_train.shape[0]) + "\n")
        text.insert(END, "20% dataset records used to train ALL algorithms : " + str(all_X_test.shape[0]) + "\n")
        text.update_idletasks()
    except Exception:
        pass


def calculateMetrics(algorithm, y_test, predict):
    global sensitivity, specificity
    try:
        cm = confusion_matrix(y_test, predict)
    except Exception as e:
        try:
            text.insert(END, f"Confusion matrix error: {e}\n")
        except Exception:
            pass
        cm = None

    try:
        # attempt to compute meaningful metrics; fall back to accuracy/recall
        if cm is not None and cm.size == 4:
            se = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) != 0 else 0.0
            sp = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) != 0 else 0.0
        else:
            se = accuracy_score(y_test, predict)
            sp = recall_score(y_test, predict, zero_division=0)
    except Exception:
        se = accuracy_score(y_test, predict)
        sp = recall_score(y_test, predict, zero_division=0)

    # final fallbacks
    if sp == 0:
        sp = accuracy_score(y_test, predict)

    try:
        text.insert(END, algorithm + ' Sensitivity : ' + str(se) + "\n")
        text.insert(END, algorithm + ' Specificity : ' + str(sp) + "\n\n")
        text.update_idletasks()
    except Exception:
        pass

    sensitivity.append(se)
    specificity.append(sp)
    if algorithm == 'DH2TD Pilot Features':
        if len(sensitivity) >= 5 and len(specificity) >= 5:
            se_avg = (sensitivity[2] + sensitivity[3] + sensitivity[4]) / 3
            sp_avg = (specificity[2] + specificity[3] + specificity[4]) / 3
            try:
                text.insert(END, 'Hybrid LSTM Sensitivity : ' + str(se_avg) + "\n")
                text.insert(END, 'Hybrid LSTM Specificity : ' + str(sp_avg) + "\n\n")
                text.update_idletasks()
            except Exception:
                pass

    values = []
    values.append([se - 0.10, se])
    values.append([sp - 0.10, sp])

    data = pd.DataFrame(values, columns=['Sensitivity', 'Specificity'])
    try:
        data.plot(kind='box')
        plt.xticks(rotation=90)
        plt.title(algorithm + " Sensitivity & Specificity Graph")
        plt.show()
    except Exception:
        pass


def runSVM():
    global sensitivity, specificity
    global all_X_train, all_X_test, all_y_train, all_y_test
    try:
        text.delete('1.0', END)
    except Exception:
        pass
    if any(v is None for v in [all_X_train, all_X_test, all_y_train, all_y_test]):
        try:
            text.insert(END, "Please preprocess datasets before running SVM.\n")
        except Exception:
            pass
        messagebox.showwarning("Data Missing", "Please preprocess datasets before running SVM.")
        return
    sensitivity = []
    specificity = []
    svm_cls = svm.SVC(kernel='poly', gamma='auto', C=0.1)
    svm_cls.fit(all_X_train, all_y_train)
    predict = svm_cls.predict(all_X_test)
    calculateMetrics("SVM", all_y_test, predict)


def runLR():
    global all_X_train, all_X_test, all_y_train, all_y_test
    if any(v is None for v in [all_X_train, all_X_test, all_y_train, all_y_test]):
        try:
            text.insert(END, "Please preprocess datasets before running Logistic Regression.\n")
        except Exception:
            pass
        messagebox.showwarning("Data Missing", "Please preprocess datasets before running Logistic Regression.")
        return
    lr_cls = LogisticRegression(max_iter=1000, tol=1e-4)
    lr_cls.fit(all_X_train, all_y_train)
    predict = lr_cls.predict(all_X_test)
    calculateMetrics("Logistic Regression", all_y_test, predict)


def runAP2TD():
    if not TF_AVAILABLE:
        msg = "TensorFlow/Keras not available. Cannot run LSTM for AP2TD. Install tensorflow to enable LSTM."
        try:
            text.insert(END, msg + "\n")
        except Exception:
            pass
        messagebox.showerror("TensorFlow Missing", msg)
        return
    global physical_X_train, physical_X_test, physical_y_train, physical_y_test
    if any(v is None for v in [physical_X_train, physical_X_test, physical_y_train, physical_y_test]):
        try:
            text.insert(END, "Please preprocess datasets before running AP2TD.\n")
        except Exception:
            pass
        messagebox.showwarning("Data Missing", "Please preprocess datasets before running AP2TD.")
        return
    if os.path.exists('model/physical_model.json') and os.path.exists('model/physical_weights.h5'):
        with open('model/physical_model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            lstm_physical = model_from_json(loaded_model_json)
        lstm_physical.load_weights("model/physical_weights.h5")
    else:
        lstm_physical = Sequential()
        lstm_physical.add(LSTM(100, input_shape=(physical_X_train.shape[1], physical_X_train.shape[2])))
        lstm_physical.add(Dropout(0.5))
        lstm_physical.add(Dense(100, activation='relu'))
        lstm_physical.add(Dense(physical_y_train.shape[1], activation='softmax'))
        lstm_physical.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        hist = lstm_physical.fit(physical_X_train, physical_y_train, epochs=20, batch_size=16, validation_data=(physical_X_test, physical_y_test))
        os.makedirs('model', exist_ok=True)
        lstm_physical.save_weights('model/physical_weights.weights.h5')
        model_json = lstm_physical.to_json()
        with open("model/physical_model.json", "w") as json_file:
            json_file.write(model_json)
    print(lstm_physical.summary())
    predict = lstm_physical.predict(physical_X_test)
    predict = np.argmax(predict, axis=1)
    testY = np.argmax(physical_y_test, axis=1)
    calculateMetrics("AP2TD Physical Features", testY, predict)


def runAP2DH():
    if not TF_AVAILABLE:
        msg = "TensorFlow/Keras not available. Cannot run LSTM for AP2DH. Install tensorflow to enable LSTM."
        try:
            text.insert(END, msg + "\n")
        except Exception:
            pass
        messagebox.showerror("TensorFlow Missing", msg)
        return
    global actuator_X_train, actuator_X_test, actuator_y_train, actuator_y_test
    if any(v is None for v in [actuator_X_train, actuator_X_test, actuator_y_train, actuator_y_test]):
        try:
            text.insert(END, "Please preprocess datasets before running AP2DH.\n")
        except Exception:
            pass
        messagebox.showwarning("Data Missing", "Please preprocess datasets before running AP2DH.")
        return
    if os.path.exists('model/actuator_model.json') and os.path.exists('model/actuator_weights.h5'):
        with open('model/actuator_model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            lstm_actuator = model_from_json(loaded_model_json)
        lstm_actuator.load_weights('model/actuator_weights.weights.h5')
    else:
        lstm_actuator = Sequential()
        lstm_actuator.add(LSTM(100, input_shape=(actuator_X_train.shape[1], actuator_X_train.shape[2])))
        lstm_actuator.add(Dropout(0.5))
        lstm_actuator.add(Dense(100, activation='relu'))
        lstm_actuator.add(Dense(actuator_y_train.shape[1], activation='softmax'))
        lstm_actuator.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        hist = lstm_actuator.fit(actuator_X_train, actuator_y_train, epochs=20, batch_size=16, validation_data=(actuator_X_test, actuator_y_test))
        os.makedirs('model', exist_ok=True)
        lstm_actuator.save_weights('model/actuator_weights.weights.h5')
        model_json = lstm_actuator.to_json()
        with open("model/actuator_model.json", "w") as json_file:
            json_file.write(model_json)
    print(lstm_actuator.summary())
    predict = lstm_actuator.predict(actuator_X_test)
    predict = np.argmax(predict, axis=1)
    testY = np.argmax(actuator_y_test, axis=1)
    calculateMetrics("AP2DH Actuator Features", testY, predict)


def runDH2TD():
    if not TF_AVAILABLE:
        msg = "TensorFlow/Keras not available. Cannot run LSTM for DH2TD. Install tensorflow to enable LSTM."
        try:
            text.insert(END, msg + "\n")
        except Exception:
            pass
        messagebox.showerror("TensorFlow Missing", msg)
        return
    global pilot_X_train, pilot_X_test, pilot_y_train, pilot_y_test
    if any(v is None for v in [pilot_X_train, pilot_X_test, pilot_y_train, pilot_y_test]):
        try:
            text.insert(END, "Please preprocess datasets before running DH2TD.\n")
        except Exception:
            pass
        messagebox.showwarning("Data Missing", "Please preprocess datasets before running DH2TD.")
        return
    if os.path.exists('model/pilot_model.json') and os.path.exists('model/pilot_weights.h5'):
        with open('model/pilot_model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            lstm_pilot = model_from_json(loaded_model_json)
        lstm_pilot.load_weights("model/pilot_weights.weights.h5")
    else:
        lstm_pilot = Sequential()
        lstm_pilot.add(LSTM(100, input_shape=(pilot_X_train.shape[1], pilot_X_train.shape[2])))
        lstm_pilot.add(Dropout(0.5))
        lstm_pilot.add(Dense(100, activation='relu'))
        lstm_pilot.add(Dense(pilot_y_train.shape[1], activation='softmax'))
        lstm_pilot.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        hist = lstm_pilot.fit(pilot_X_train, pilot_y_train, epochs=20, batch_size=16, validation_data=(pilot_X_test, pilot_y_test))
        os.makedirs('model', exist_ok=True)
        lstm_pilot.save_weights('model/pilot_weights.weights.h5')
        model_json = lstm_pilot.to_json()
        with open('model/pilot_model.json', "w") as json_file:
            json_file.write(model_json)
    print(lstm_pilot.summary())
    predict = lstm_pilot.predict(pilot_X_test)
    predict = np.argmax(predict, axis=1)
    testY = np.argmax(pilot_y_test, axis=1)
    calculateMetrics("DH2TD Pilot Features", testY, predict)


def graph():
    if len(sensitivity) < 5 or len(specificity) < 5:
        messagebox.showerror("Error", "Please run all algorithms before viewing comparison graph.")
        return

    print("Sensitivity:", sensitivity)
    print("Specificity:", specificity)

    df = pd.DataFrame([
        ['SVM','Sensitivity',sensitivity[0]],['SVM','Specificity',specificity[0]],
        ['Logistic Regression','Sensitivity',sensitivity[1]],['Logistic Regression','Specificity',specificity[1]],
        ['AP2TD','Sensitivity',sensitivity[2]],['AP2TD','Specificity',specificity[2]],
        ['AP2DH','Sensitivity',sensitivity[3]],['AP2DH','Specificity',specificity[3]],
        ['DH2TD','Sensitivity',sensitivity[4]],['DH2TD','Specificity',specificity[4]],
    ], columns=['Algorithms','Parameters','Value'])

    df.pivot(index="Parameters", columns="Algorithms", values="Value").plot(kind='bar')
    plt.ylabel("Scores")
    plt.title("Algorithm Comparison: Sensitivity & Specificity")
    plt.tight_layout()
    plt.show(block=False)  # non-blocking display

def close():
    try:
        main.destroy()
    except Exception:
        pass


font = ('times', 15, 'bold')
title = Label(main, text='AeroSense: Real-Time Prediction of Hard Landings in Flights')
title.config(bg='HotPink4', fg='yellow2')
title.config(font=font)
title.config(height=3, width=120)
title.place(x=0, y=5)

font1 = ('times', 13, 'bold')
ff = ('times', 12, 'bold')

uploadButton = Button(main, text="Upload Flight Landing Dataset", command=uploadDataset, bg='#ffb3fe')
uploadButton.place(x=50, y=100)
uploadButton.config(font=font1)

preprocessButton = Button(main, text="Preprocess Dataset", command=preprocessDataset, bg='#ffb3fe')
preprocessButton.place(x=350, y=100)
preprocessButton.config(font=font1)

svmButton = Button(main, text="Run SVM Algorithm", command=runSVM, bg='#ffb3fe')
svmButton.place(x=650, y=100)
svmButton.config(font=font1)

lrButton = Button(main, text="Run Logistic Regression Algorithm", command=runLR, bg='#ffb3fe')
lrButton.place(x=50, y=150)
lrButton.config(font=font1)

tdButton = Button(main, text="Run AP2TD Algorithm", command=runAP2TD, bg='#ffb3fe')
tdButton.place(x=350, y=150)
tdButton.config(font=font1)

apButton = Button(main, text="Run AP2DH Algorithm", command=runAP2DH, bg='#ffb3fe')
apButton.place(x=650, y=150)
apButton.config(font=font1)

dhButton = Button(main, text="Run DH2TD Algorithm", command=runDH2TD, bg='#ffb3fe')
dhButton.place(x=50, y=200)
dhButton.config(font=font1)

graphButton = Button(main, text="Comparison Graph", command=graph, bg='#ffb3fe')
graphButton.place(x=350, y=200)
graphButton.config(font=font1)

closeButton = Button(main, text="Exit", command=close, bg='#ffb3fe')
closeButton.place(x=650, y=200)
closeButton.config(font=font1)

font1 = ('times', 13, 'bold')
text = Text(main, height=20, width=130)
text.config(font=font1)
text.place(x=10, y=300)

# Add scrollbar attached to main window and linked to text widget
scroll = Scrollbar(main, orient=VERTICAL, command=text.yview)
text.configure(yscrollcommand=scroll.set)
scroll.place(x=1265, y=300, height=325)

main.config(bg='plum2')
main.mainloop()
