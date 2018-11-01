from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

# validation
test_batch_size = 25

test_datagen = ImageDataGenerator()
test_generator = test_datagen.flow_from_directory(
        '/data/full/validation/',
        target_size=(224, 224),
        batch_size=test_batch_size,
        shuffle=True,
        class_mode='categorical')

experiment_name = 'experiment_2'
base_path = '/data/models/'

test_model = load_model(base_path + 'model_' + experiment_name + '.h5')

results = test_model.evaluate_generator(test_generator, steps=(400/test_batch_size), verbose=1)
print(test_model.metrics_names)
print(results)