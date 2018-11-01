import matplotlib.pyplot as plt
import pickle

experiment_name = 'experiment_4'
base_path = '/data/models/'

file = open(base_path + 'history_' + experiment_name + '.pkl', 'rb')
complete_history = pickle.load(file)
file.close()

# summarize history for accuracy
plt.plot(complete_history['categorical_accuracy'])
plt.plot(complete_history['val_categorical_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')
plt.show()

# summarize history for loss
plt.plot(complete_history['loss'])
plt.plot(complete_history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()