from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# validation
test_batch_size = 25

test_datagen = ImageDataGenerator()
test_generator = test_datagen.flow_from_directory(
    '/data/full/validation/',
    target_size=(224, 224),
    batch_size=test_batch_size,
    shuffle=False,
    class_mode='categorical')

experiment_name = 'experiment_2'
base_path = '/data/models/'

test_model = load_model(base_path + 'model_' + experiment_name + '.h5')

# matplotlib.use('Agg')
# plt.ion()

print(test_generator.class_indices)
tp = 0
for _ in range(int(400 / test_batch_size)):
    test_x, test_y = test_generator.next()
    test_pred = test_model.predict_on_batch(test_x)

    _tp = sum([np.argmax(test_pred[i]) == np.argmax(test_y[i]) for i in range(test_batch_size)])
    print(_tp)
    tp += _tp

    for i in range(25):
        is_tp = np.argmax(test_pred[i]) == np.argmax(test_y[i])
        if not is_tp:
            plt.imshow(test_x[i].astype(int))
            cSo = "{0:.2f}".format(round(test_pred[i, 0], 2))
            cSt = "{0:.2f}".format(round(test_pred[i, 1], 2))
            pred_class = 'Soft ' if np.argmax(test_pred[i]) == 0 else 'Stiff'
            plt.suptitle(str(is_tp) + '  ' + pred_class + '  (C so: ' + str(cSo) + ' st: ' + cSt + ')')

            plt.show()

print(matplotlib.rcParams['interactive'] == True)
print(tp / 400)
