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
import numpy as np
import cv2
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.utils import shuffle
from keras import layers
from keras import utils
from keras import models
from keras import optimizers
from keras import callbacks
from sklearn.metrics import confusion_matrix
import seaborn as sns

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def load_images_and_extract_hog(folder_path, img_height=12, img_width=10):
    images = []
    labels = []
    label_map = {chr(i + 65): i for i in range(26)}  # Map A-Z to 0-25

    file_list = sorted(os.listdir(folder_path), key=lambda x: int(x.split('.')[0]))

    for idx, filename in enumerate(file_list):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            resized_img = cv2.resize(img, (img_width, img_height))
            hog_features = hog(resized_img, pixels_per_cell=(2, 2), cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)
            images.append(hog_features)
            label = chr(65 + (idx % 26))  # A-Z repetido
            labels.append(label_map[label])
    
    images = np.array(images).astype('float32')
    labels = utils.to_categorical(np.array(labels), num_classes=26)

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

# Carregar as imagens e extrair características HOG
folder_path = 'data/X_png/'  # Substitua pelo caminho da sua pasta
train_images, train_labels, test_images, test_labels = load_images_and_extract_hog(folder_path)

# Definir o modelo
model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_shape=(train_images.shape[1],)))
model.add(layers.Dropout(0.0))

model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.0))

model.add(layers.Dense(26, activation='softmax'))

sgd = optimizers.SGD(learning_rate=0.03)

# Compilar o modelo
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# Salvar hiperparâmetros da arquitetura da rede neural e hiperparâmetros de inicialização
hyperparameters = {
    "input_shape": train_images.shape[1],
    "layers": [
        {"type": "Dense", "units": 256, "activation": "relu"},
        {"type": "Dropout", "rate": 0.5},
        {"type": "Dense", "units": 128, "activation": "relu"},
        {"type": "Dropout", "rate": 0.5},
        {"type": "Dense", "units": 26, "activation": "softmax"}
    ],
    "optimizer": "SGD",
    "learning_rate": 0.03,
    "loss": "categorical_crossentropy",
    "metrics": ["accuracy"],
    "epochs": 300,
    "batch_size": 128,
    "validation_split": 0.2,
    "early_stopping_patience": 30
}

with open('outputs/HOG/hyperparameters.txt', 'w') as f:
    for key, value in hyperparameters.items():
        f.write(f"{key}: {value}\n")

# Salvar pesos iniciais da rede
initial_weights = model.get_weights()
with open('outputs/HOG/initial_weights.txt', 'w') as f:
    for weight in initial_weights:
        np.savetxt(f, weight)

# Definir EarlyStopping callback
early_stopping = callbacks.EarlyStopping(
    monitor='val_accuracy',  # Métrica a ser monitorada
    patience=30,             # Número de épocas sem melhoria após o qual o treinamento será interrompido
    restore_best_weights=True  # Restaurar os pesos do modelo para a melhor época
)

# Treinar o modelo
history = model.fit(
    train_images,
    train_labels,
    epochs=300,
    verbose=1,
    batch_size=128,
    validation_split=0.1,
    callbacks=[early_stopping]
)

# Salvar pesos finais da rede
final_weights = model.get_weights()
with open('outputs/HOG/final_weights.txt', 'w') as f:
    for weight in final_weights:
        np.savetxt(f, weight)

# Salvar o erro cometido pela rede neural em cada iteração do treinamento
training_errors = {
    "accuracy": history.history['accuracy'],
    "val_accuracy": history.history['val_accuracy'],
    "loss": history.history['loss'],
    "val_loss": history.history['val_loss']
}

with open('outputs/HOG/training_errors.txt', 'w') as f:
    for key, value in training_errors.items():
        f.write(f"{key}: {value}\n")

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
plt.savefig('outputs/HOG/accuracy_plot.png')  # Salvar o gráfico como imagem
plt.show()

# Avaliar o modelo
loss, accuracy = model.evaluate(test_images, test_labels)
print(f'Acurácia do modelo: {accuracy * 100:.2f}%')

# Fazer previsões no conjunto de teste
predictions = model.predict(test_images)
predictions_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(test_labels, axis=1)

# Salvar as saídas produzidas pela rede neural para cada um dos dados de teste
test_outputs = {
    "predictions": predictions.tolist(),
    "true_labels": true_labels.tolist()
}

with open('outputs/HOG/test_outputs.txt', 'w') as f:
    for key, value in test_outputs.items():
        f.write(f"{key}: {value}\n")

# Gerar a matriz de confusão
conf_matrix = confusion_matrix(true_labels, predictions_labels)

# Plotar a matriz de confusão
plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=[chr(i) for i in range(65, 91)], yticklabels=[chr(i) for i in range(65, 91)])
plt.xlabel('Previsões')
plt.ylabel('Verdadeiros')
plt.title('Matriz de Confusão')
plt.savefig('outputs/HOG/confusion_matrix.png') 
plt.show()