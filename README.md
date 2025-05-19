# CNN-Crack-Detection
This is a simple 4 layers convolutional neural network to classify cracks on a surface. The data is sourced from [Surface Crack Detection](https://www.kaggle.com/datasets/arunrk7/surface-crack-detection)

# Results
> Accuracy: 0.9921

# Data Preprocessing
1. Read the input data and convert it into grayscale.
> img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
2. Resize the image to 120 * 120 for more consistent input
> resized_arr = cv2.resize(img_arr, (120, 120))
3. Transform the data to increase amount of training data.
> datagen = ImageDataGenerator(
> 
>    rotation_range=20,
> 
>    width_shift_range=0.2,
> 
>    height_shift_range=0.2,
> 
>    shear_range=0.2,
> 
>    zoom_range=0.2,
> 
>    horizontal_flip=True,
> 
>    fill_mode='nearest' )

# Model Architecture

```python
model = Sequential([
    Input(shape=(img_size, img_size, 1)),
    
    # First convolutional block
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    # Second convolutional block
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    # Third convolutional block
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    # Fourth convolutional block
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    # Fully connected layers
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Output layer
])
