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

# Função para salvar as imagens individualmente
def save_images(images, folder_path):
    os.makedirs(folder_path, exist_ok=True)
    for i, img in enumerate(images):
        file_path = os.path.join(folder_path, f'image_{i}.txt')
        np.savetxt(file_path, img.reshape(-1), fmt='%.5f')

# Função para salvar os pesos da rede em um único arquivo
def save_weights(weights, file_path):
    with open(file_path, 'w') as f:
        for weight in weights:
            np.savetxt(f, weight.flatten(), fmt='%.5f')

def load_images(folder_path, img_height=12, img_width=10):
    images = []
    labels = []
    label_map = {chr(i + 65): i for i in range(26)}  # Map A-Z to 0-25

    file_list = sorted(os.listdir(folder_path), key=lambda x: int(x.split('.')[0]))

    for idx, filename in enumerate(file_list):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
            label = chr(65 + (idx % 26))  # A-Z repetido
            labels.append(label_map[label])
    
    images = np.array(images).reshape(-1, img_height, img_width, 1).astype('float32') / 255
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

# Carregar as imagens
folder_path = 'data/X_png/'  # Substitua pelo caminho da sua pasta
train_images, train_labels, test_images, test_labels = load_images(folder_path)

# Definir o modelo
model = models.Sequential()
model.add(layers.Conv2D(8, kernel_size=(3, 3), activation='relu', input_shape=(10, 12, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(26, activation='softmax'))

sgd = optimizers.SGD(learning_rate=0.03)

# Compilar o modelo
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# Salvar hiperparâmetros da arquitetura da rede neural e hiperparâmetros de inicialização
hyperparameters = {
    "input_shape": train_images.shape[1:],
    "layers": [
        {"type": "Conv2D", "filters": 8, "kernel_size": (3, 3), "activation": "relu"},
        {"type": "MaxPooling2D", "pool_size": (2, 2)},
        {"type": "Conv2D", "filters": 32, "kernel_size": (3, 3), "activation": "relu"},
        {"type": "MaxPooling2D", "pool_size": (2, 2)},
        {"type": "Flatten"},
        {"type": "Dense", "units": 64, "activation": "relu"},
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

with open('outputs/multiclasses/hyperparameters.txt', 'w') as f:
    for key, value in hyperparameters.items():
        f.write(f"{key}: {value}\n")

# Salvar pesos iniciais da rede
initial_weights = model.get_weights()
save_weights(initial_weights, 'outputs/multiclasses/initial_weights.txt')

# Definir EarlyStopping callback
early_stopping = callbacks.EarlyStopping(
    monitor='val_accuracy',  # Métrica a ser monitorada
    patience=50,             # Número de épocas sem melhoria após o qual o treinamento será interrompido
    restore_best_weights=True  # Restaurar os pesos do modelo para a melhor época
)

# Treinar o modelo
history = model.fit(
    train_images, 
    train_labels, 
    epochs=3000, 
    verbose=1,
    batch_size=128, 
    validation_split=0.2,
    callbacks=[early_stopping]
)

# Salvar pesos finais da rede
final_weights = model.get_weights()
save_weights(final_weights, 'outputs/multiclasses/final_weights.txt')

# Salvar o erro cometido pela rede neural em cada iteração do treinamento
training_errors = {
    "accuracy": history.history['accuracy'],
    "val_accuracy": history.history['val_accuracy'],
    "loss": history.history['loss'],
    "val_loss": history.history['val_loss']
}

with open('outputs/multiclasses/training_errors.txt', 'w') as f:
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
plt.savefig('outputs/multiclasses/accuracy_plot.png')  # Salvar o gráfico como imagem
plt.show()

# Avaliar o modelo
loss, accuracy = model.evaluate(test_images, test_labels)
print(f'Acurácia do modelo: {accuracy * 100:.2f}%')

with open('outputs/multiclasses/model_accuracy.txt', 'w') as f:
    f.write(f'Acurácia do modelo: {accuracy * 100:.2f}%\n')

# Fazer previsões no conjunto de teste
predictions = model.predict(test_images)
predictions_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(test_labels, axis=1)

# Salvar as saídas produzidas pela rede neural para cada um dos dados de teste
test_outputs = {
    "predictions": predictions.tolist(),
    "true_labels": true_labels.tolist()
}

with open('outputs/multiclasses/test_outputs.txt', 'w') as f:
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
plt.savefig('outputs/multiclasses/confusion_matrix.png')  # Salvar o gráfico como imagem
plt.show()
