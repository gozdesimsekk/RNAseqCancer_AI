from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Model oluşturma
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

# Modeli derleme
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Veri setini yükleyin ve ön işleme yapın
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Eğitim ve test verileri
train_set = train_datagen.flow_from_directory('./train', target_size=(64, 64), batch_size=32, class_mode='binary')
test_set = test_datagen.flow_from_directory('./test', target_size=(64, 64), batch_size=32, class_mode='binary')

# Modeli eğitme
model.fit(train_set, epochs=25, validation_data=test_set)


# Modeli değerlendirme
predictions = model.predict(test_set)
y_pred = np.round(predictions) 


y_true = test_set.classes
class_indices_train = train_set.class_indices
class_indices_test = test_set.class_indices

print("Eğitim seti sınıf indeksleri:", class_indices_train)
print("Test seti sınıf indeksleri:", class_indices_test)

# Print confusion matrix and classification report
conf_matrix = confusion_matrix(y_true, y_pred)
class_report = classification_report(y_true, y_pred)

print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

