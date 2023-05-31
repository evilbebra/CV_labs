Установка слоёв для модели

model = keras.Sequential()
model.add(keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)))
model.add(keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)))
model.add(keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)))
model.add(keras.layers.MaxPooling2D((2, 2)))

Компиляция модели

model.compile(
              optimizer='Nadam',
              loss='mean_squared_error',
              metrics=['accuracy'])

Обучение модели

Epoch 1/10
625/625 [==============================] - 431s 686ms/step - loss: 0.0693 - accuracy: 0.4268 - val_loss: 0.0545 - val_accuracy: 0.5861
Epoch 2/10
625/625 [==============================] - 416s 666ms/step - loss: 0.0480 - accuracy: 0.6410 - val_loss: 0.0425 - val_accuracy: 0.6839
Epoch 3/10
625/625 [==============================] - 424s 678ms/step - loss: 0.0389 - accuracy: 0.7157 - val_loss: 0.0385 - val_accuracy: 0.7224
Epoch 4/10
625/625 [==============================] - 403s 645ms/step - loss: 0.0333 - accuracy: 0.7621 - val_loss: 0.0360 - val_accuracy: 0.7396
Epoch 5/10
625/625 [==============================] - 422s 676ms/step - loss: 0.0291 - accuracy: 0.7951 - val_loss: 0.0357 - val_accuracy: 0.7396
Epoch 6/10
625/625 [==============================] - 422s 676ms/step - loss: 0.0255 - accuracy: 0.8225 - val_loss: 0.0356 - val_accuracy: 0.7436
Epoch 7/10
625/625 [==============================] - 398s 637ms/step - loss: 0.0222 - accuracy: 0.8469 - val_loss: 0.0325 - val_accuracy: 0.7667
Epoch 8/10
625/625 [==============================] - 412s 659ms/step - loss: 0.0196 - accuracy: 0.8665 - val_loss: 0.0328 - val_accuracy: 0.7753
Epoch 9/10
625/625 [==============================] - 410s 655ms/step - loss: 0.0173 - accuracy: 0.8842 - val_loss: 0.0337 - val_accuracy: 0.7717
Epoch 10/10
625/625 [==============================] - 410s 656ms/step - loss: 0.0154 - accuracy: 0.8971 - val_loss: 0.0341 - val_accuracy: 0.7697
313/313 [==============================] - 24s 78ms/step - loss: 0.0353 - accuracy: 0.7638

Итоговый результат

Test loss: 0.03526553511619568
Test accuracy: 0.7638000249862671