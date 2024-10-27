import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2

train_data_gen = ImageDataGenerator(rescale=1./255)
validation_data_gen = ImageDataGenerator(rescale=1./255)

train_generator = train_data_gen.flow_from_directory(
    'data/train',
    target_size = (48,48),
    batch_size = 64,
    color_mode = 'grayscale',
    class_mode = 'categorical')

validation_generator = validation_data_gen.flow_from_directory(
    'data/test',
    target_size = (48,48),
    batch_size = 64,
    color_mode = 'grayscale',
    class_mode = 'categorical')


#model structure!

emotion_model = keras.Sequential()
emotion_model.add(keras.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
emotion_model.add(keras.Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(keras.MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(keras.Dropout(0.25))

emotion_model.add(keras.Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(keras.MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(keras.Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(keras.MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(keras.Dropout(0.25))

emotion_model.add(keras.Flatten())
emotion_model.add(keras.Dense(1024, activation='relu'))
emotion_model.add(keras.Dropout(0.5))
emotion_model.add(keras.Dense(7, activation='softmax'))

cv2.ocl.setUseOpenCL(False)

emotion_model.compile(loss = 'categorical_crossentropy', optimizer = keras.Adam(lr = 0.0001, decay = 1e-6), metrics = ['accuracy'])

# Train the neural network/model
emotion_model_info = emotion_model.fit_generator(
        train_generator,
        steps_per_epoch = 28709 // 64,
        epochs = 50,
        validation_data = validation_generator,
        validation_steps = 7178 // 64)

emotion_model.save('emotion_model.keras')

model_json = emotion_model.to_json()
with open("emotion_model.json", "w") as json_file:
    json_file.write(model_json)

# save trained model weight in .h5 file
emotion_model.save_weights('emotion_model.h5')
