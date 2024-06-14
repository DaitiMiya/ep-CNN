#######################################################################
#                Inteligencia Artificial - ACH2016                    #
#                                                                     #
#  Gandhi Daiti Miyahara 11207773                                     #
#  Lucas Tatsuo Nishida 11208270                                      #
#  Juan Kineipe 11894610                                              #
#  Leonardo Ken Matsuda Cancela 11207665                              #
#  João de Araújo Barbosa da Silva 11369704                           #
#                                                                     #
#######################################################################

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import utils
from keras import models 
from keras import optimizers 

def load_images(folder_path, img_height=12, img_width=10):
    images = []
    labels = []
    label_map = {chr(i + 65): i for i in range(26)}  # Map A-Z to 0-25

    for idx, filename in enumerate(sorted(os.listdir(folder_path), key=lambda x: int(x.split('.')[0]))):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
            label = chr(65 + (idx % 26))  # A-Z repetido
            labels.append(label_map[label])
    
    # images = np.array(images).reshape(-1, img_height, img_width, 1).astype('float32') / 255
    images = np.array(images)
    labels = utils.to_categorical(np.array(labels), num_classes=26)
    return images, labels

# Carregar as imagens
folder_path = 'data/X_png/'  # Substitua pelo caminho da sua pasta
images, labels = load_images(folder_path)

# Dividir em conjunto de treinamento e teste
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

# Definir o modelo
model = models.Sequential()
model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(12, 10, 1)))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(26, activation='softmax'))

# Compilar o modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Treinar o modelo
model.fit(train_images, train_labels, epochs=10, batch_size=32, validation_split=0.2)

# Avaliar o modelo
loss, accuracy = model.evaluate(test_images, test_labels)
print(f'Acurácia do modelo: {accuracy * 100:.2f}%')

# Plotar a imagem original e a detecção de borda


# plt.show()
# print(images[0])
# plt.figure(figsize=(12, 10))
# for i in range(25):  # Printar as primeiras 25 imagens
#     # Aplicar filtro de Sobel para detecção de borda
#     sobelx = cv2.Sobel(images[i], cv2.CV_64F, 1, 0, ksize=3)  # Derivada em x (bordas verticais)
#     sobely = cv2.Sobel(images[i], cv2.CV_64F, 0, 1, ksize=3)  # Derivada em y (bordas horizontais)
#     edges = cv2.magnitude(sobelx, sobely)  # Magnitude das derivadas (detecção de borda)
#     plt.subplot(1, 3, 1)
#     plt.imshow(images[i], cmap='gray')
#     plt.title('Imagem Original')
#     plt.axis('off')

#     plt.subplot(1, 3, 2)
#     plt.imshow(sobelx, cmap='gray')
#     plt.title('Sobel X (Bordas Verticais)')
#     plt.axis('off')

#     plt.subplot(1, 3, 3)
#     plt.imshow(sobely, cmap='gray')
#     plt.title('Sobel Y (Bordas Horizontais)')
#     plt.axis('off')

#     plt.show()
# # Dividir em conjuntos de treino e teste
# train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

# # Converter labels para categórico (one-hot encoding)
# train_labels_categorical = to_categorical(train_labels, num_classes=26)
# test_labels_categorical = to_categorical(test_labels, num_classes=26)

# # Construção do modelo CNN
# model = Sequential()

# # Primeira camada convolucional
# model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(12, 10, 1)))
# model.add(MaxPooling2D((2, 2)))

# # Segunda camada convolucional
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D((2, 2)))

# # Terceira camada convolucional
# model.add(Conv2D(128, (3, 3), activation='relu'))
# model.add(MaxPooling2D((2, 2)))

# # Camada Flatten
# model.add(Flatten())

# # Camada densa totalmente conectada
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))

# # Camada de saída
# model.add(Dense(26, activation='softmax'))

# # Compilação do modelo
# model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# # Treinamento do modelo
# history = model.fit(train_images, train_labels_categorical, epochs=10, batch_size=32, validation_split=0.2)

# # Avaliação do modelo
# test_loss, test_accuracy = model.evaluate(test_images, test_labels_categorical)
# print(f'Test accuracy: {test_accuracy * 100:.2f}%')