# Food Image Classification with Calorie Estimation

This project uses a Convolutional Neural Network (CNN) to classify images of different food items and estimate their calorie content. The model is trained and evaluated using a dataset of food images, and predictions are made on a test set with corresponding calorie content displayed for each predicted food item.

## Project Structure

- `Task5.py`: Python script to train, evaluate, and make predictions using the CNN model.
- `README.md`: This file.
- `food_classification_model.h5`: The saved model after training (generated after running the script).

## Requirements

- Python 3.6+
- numpy
- tensorflow
- opencv-python
- seaborn
- matplotlib
- Pillow

You can install the required libraries using pip:

```sh
pip install numpy tensorflow opencv-python seaborn matplotlib Pillow
```

## Code Overview

The script `Task5.py` performs the following steps:

1. **Define Paths and Parameters**: Set the paths to the train and test directories, and define image dimensions and batch size.

    ```python
    train_dir = "path_to_train_directory"
    test_dir = "path_to_test_directory"
    img_width, img_height = 224, 224
    batch_size = 32
    ```

2. **Data Augmentation and Preprocessing**: Use `ImageDataGenerator` for data augmentation and preprocessing.

    ```python
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    ```

3. **Load and Augment Training Images**: Load and augment the training images.

    ```python
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical'
    )
    ```

4. **Load Validation Images**: Load the validation images.

    ```python
    validation_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical'
    )
    ```

5. **Define the Model Architecture**: Build the CNN model using TensorFlow Keras.

    ```python
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    ```

6. **Compile and Train the Model**: Compile the model with the Adam optimizer and categorical cross-entropy loss, and train the model with a validation split.

    ```python
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=20,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size
    )
    ```

7. **Save the Model**: Save the trained model to a file.

    ```python
    model.save("food_classification_model.h5")
    ```

8. **Plot Training History**: Plot the training and validation accuracy and loss.

    ```python
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    plt.show()

    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.show()
    ```

9. **Evaluate the Model**: Generate predictions on the test set and print the confusion matrix and classification report.

    ```python
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    Y_pred = model.predict(test_generator)
    y_pred = np.argmax(Y_pred, axis=1)
    cm = confusion_matrix(test_generator.classes, y_pred)
    print("Confusion Matrix:")
    print(cm)

    print("Classification Report:")
    target_names = list(test_generator.class_indices.keys())
    print(classification_report(test_generator.classes, y_pred, target_names=target_names))
    ```

10. **Calorie Estimation**: Use a dictionary to map each food class to its estimated calorie content and display predictions for random images from the test set.

    ```python
    calorie_mapping = {
        "apple_pie": 300,
        "baby_back_ribs": 700,
        ...
    }

    random_indices = np.random.choice(len(test_generator.filenames), size=5, replace=False)
    for index in random_indices:
        image_path = os.path.join(test_dir, test_generator.filenames[index])
        image = Image.open(image_path)
        image = image.resize((img_width, img_height))
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)
        prediction = model.predict(image)
        predicted_class = target_names[np.argmax(prediction)]

        print(f'Predicted Class: {predicted_class}')
        calorie_content = calorie_mapping.get(predicted_class, None)
        if calorie_content is not None:
            print(f'Calorie Content: {calorie_content} calories')
        else:
            print('Calorie content information not available')

        plt.imshow(image.squeeze())
        plt.axis('off')
        plt.title(f'Predicted Class: {predicted_class}')
        plt.show()
    ```

## Running the Code

1. Ensure that the train and test directories are correctly set up with images organized into subdirectories for each food class.
2. Update the `train_dir` and `test_dir` variables with the correct paths to your data.
3. Run the script `Task5.py`:

    ```sh
    python Task5.py
    ```

4. The script will train the model, save it, evaluate it, and display predictions with calorie estimates.

## Acknowledgements

This project is inspired by various food image classification datasets and methodologies.


