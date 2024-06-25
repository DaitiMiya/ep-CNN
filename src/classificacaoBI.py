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
import random
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.utils import shuffle
from keras import layers
from keras import utils
from keras import models 
from keras import optimizers 
from sklearn.metrics import confusion_matrix
import seaborn as sns

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def load_images(folder_path, img_height=12, img_width=10):
    images = []
    labels = []

    file_list = sorted(os.listdir(folder_path), key=lambda x: int(x.split('.')[0]))

    for idx, filename in enumerate(file_list):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
            label = chr(65 + (idx % 26))  # A-Z repetido
            if label in 'ABCDEFGHIJKLM':
                labels.append(1)  # Letras do começo do alfabeto (A-M)
            else:
                labels.append(0)  # Letras do fim do alfabeto (N-Z)
    
    images = np.array(images).reshape(-1, img_height, img_width, 1).astype('float32') / 255
    labels = np.array(labels)
    labels = utils.to_categorical(labels, num_classes=2)
    
    # Separar as últimas 200 imagens para teste
    test_images = images[-200:]
    test_labels = labels[-200:]

    # O restante para treino
    train_images = images[:-200]
    train_labels = labels[:-200]

    # Aleatorizar os dados de treino
    indices = np.arange(train_images.shape[0])
    np.random.shuffle(indices)
    train_images = train_images[indices]
    train_labels = train_labels[indices]

    return train_images, train_labels, test_images, test_labels

# Carregar as imagens
folder_path = 'data/X_png/'  # Substitua pelo caminho da sua pasta
train_images, train_labels, test_images, test_labels = load_images(folder_path)

# Definir o modelo
model = models.Sequential()
model.add(layers.Conv2D(8, kernel_size=(3, 3), activation='relu', input_shape=(10, 12, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.0))  # Adicionar Dropout após a camada convolucional


model.add(layers.Conv2D(16, kernel_size=(3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.0))  # Adicionar Dropout após a camada convolucional


model.add(layers.Flatten())
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dropout(0.0))  # Adicionar Dropout após a camada densa
model.add(layers.Dense(2, activation='softmax'))  # A saída deve ter 2 neurônios para classes binárias

# Ajustar a taxa de aprendizado
sgd = optimizers.SGD(learning_rate=0.03)  # Reduzir a taxa de aprendizado

# Compilar o modelo
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# Treinar o modelo
history = model.fit(train_images, train_labels, epochs=300, verbose=1, batch_size=256, validation_split=0.2)

# Extrair dados do histórico
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs_range = range(1, len(acc) + 1)

# Plotar o gráfico de acurácia
plt.figure(figsize=(10, 6))
plt.plot(epochs_range, acc, label='Acurácia de Treinamento')
plt.plot(epochs_range, val_acc, label='Acurácia de Validação')
plt.title('Acurácia de Treinamento e Validação por Época')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend()
plt.grid(True)
plt.show()

# Avaliar o modelo
loss, accuracy = model.evaluate(test_images, test_labels)
print(f'Acurácia do modelo: {accuracy * 100:.2f}%')

# Fazer previsões no conjunto de teste
predictions = model.predict(test_images)
predictions_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(test_labels, axis=1)

# Gerar a matriz de confusão
conf_matrix = confusion_matrix(true_labels, predictions_labels)

# Plotar a matriz de confusão
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Fim do Alfabeto', 'Começo do Alfabeto'], yticklabels=['Fim do Alfabeto', 'Começo do Alfabeto'])
plt.xlabel('Previsões')
plt.ylabel('Verdadeiros')
plt.title('Matriz de Confusão')
plt.show()
