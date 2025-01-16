import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from tensorflow import keras
import numpy as np
from sklearn.metrics import accuracy_score
import time
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from keras import layers
from keras.models import Sequential
from keras.layers import Dense, Conv2D,MaxPooling2D,Flatten,Dropout
import matplotlib.pyplot as plt

def cargarMNISTClase(clase=None,multiclase=False,mlp=False,cnn=False,adaboostclassifier=False):
    
    #Prepara los datos MNIST para la red MLP
    if mlp:
        (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Normalizar y aplanar los datos
        X_train = X_train.reshape(60000, 784).astype('float32') / 255
        X_test = X_test.reshape(10000, 784).astype('float32') / 255

        # Convertir etiquetas a formato one-hot
        Y_train = keras.utils.to_categorical(y_train, 10)
        Y_test = keras.utils.to_categorical(y_test, 10)
    elif cnn: #Prepara los datos de MNIST para las redes CNN
        num_classes = 10

        # Load the data and split it between train and test sets
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

        # Scale images to the [0, 1] range
        x_train = x_train.astype("float32") / 255
        x_test = x_test.astype("float32") / 255
        # Make sure images have shape (28, 28, 1)
        X_train = np.expand_dims(x_train, -1)
        X_test = np.expand_dims(x_test, -1)


        # convert class vectors to binary class matrices
        Y_train = keras.utils.to_categorical(y_train, num_classes)
        Y_test = keras.utils.to_categorical(y_test, num_classes)
    #Prepara los datos de MNIST para los clasificadores multiclase
    elif multiclase or adaboostclassifier:
        (X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()

        # Formatear imágenes a vectores de floats y normalizar
        X_train = X_train.reshape((X_train.shape[0], 28*28)).astype("float32") / 255.0
        X_test = X_test.reshape((X_test.shape[0], 28*28)).astype("float32") / 255.0

        
    
    else:#Prepara los datos de MNIST para el clasificador binario, recibe un parámetro que determina la clase a la que el clasificador va a clasificar.

        # Cargar los datos de entrenamiento y test (MNIST de Yann Lecun)
        (X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()

        # Formatear imágenes a vectores de floats y normalizar
        X_train = X_train.reshape((X_train.shape[0], 28*28)).astype("float32") / 255.0
        X_test = X_test.reshape((X_test.shape[0], 28*28)).astype("float32") / 255.0

        # Formatear las clases a enteros con signo para aceptar clase -1
        Y_train = Y_train.astype("int8")
        Y_test = Y_test.astype("int8")
        
    
        # Convertir el problema a clasificación binaria: 'clase_binaria' vs 'no clase_binaria'
        Y_train = np.where(Y_train == clase, 1, -1)
        Y_test = np.where(Y_test == clase, 1, -1)
    return X_train, Y_train, X_test, Y_test

class DecisionStump:
    def __init__(self,n_features):
        self.caracteristica = np.random.randint(n_features)
        self.polaridad = 1 if np.random.rand() > 0.5 else -1
        self.umbral = np.random.uniform(0, 1)

    def predict(self,X):
        n_samples = X.shape[0]
        predicciones = np.ones(n_samples)
        if self.polaridad == 1:
            predicciones[X[:, self.caracteristica] < self.umbral] = -1
        else:
            predicciones[X[:, self.caracteristica] > self.umbral] = -1
        return predicciones
            
class Adaboost:
    def __init__(self,T=5,A=20):
        self.clasificadores = []
        self.A = A
        self.T = T

    def fit(self,X,Y,verbose=False):
        observaciones = len(X)
        caracteristicas = len(X[0])
        pesos = np.ones(observaciones) / observaciones

        if verbose:
            print("Entrenando clasificadores de umbral (con dimensión, umbral, dirección y error):")    
        
        for t in range(self.T):
            mejorError = float('inf')
            mejorClasificador = None
            for a in range(self.A):
                clasificadorDebil = DecisionStump(caracteristicas)
                predicciones = clasificadorDebil.predict(X)
                error = np.sum(pesos[predicciones != Y])

                if error < mejorError:
                    mejorError = error
                    mejorClasificador = clasificadorDebil
            
            if verbose:
                print(f"Añadido clasificador {t+1}: {mejorClasificador.caracteristica}, {mejorClasificador.umbral}, {mejorClasificador.polaridad}, {mejorError}")
            at = 0.5 * np.log((1 - mejorError) / max(mejorError, 1e-10))            
            self.clasificadores.append((mejorClasificador,at))

            predicciones = mejorClasificador.predict(X)

            pesos *= np.exp(-at * Y * predicciones)
            pesos /= np.sum(pesos)


    def predict(self, X,multiclase=False):
        final_pred = np.zeros(len(X))
        for clasificador_debil, at in self.clasificadores:
            predicciones = clasificador_debil.predict(X)
            final_pred += at * predicciones
        if multiclase:
            return final_pred
        
        return np.sign(final_pred)

    


class AdaboostMulticlase():
    def __init__(self,clases = 10,T=20,A=50):
        self.clasificadores = [Adaboost(T=T,A=A) for _ in range(clases)]
    
    def fit(self,X,Y,verbose):
        for i, clasificador in enumerate(self.clasificadores):
            if verbose:
                print(f"Entrenando clasificador Adaboost para el dígito {i}, T={clasificador.T},A={clasificador.A}")
            Y_binario = (Y==i).astype(int) * 2 - 1
            clasificador.fit(X,Y_binario,verbose)
    
    def predict(self,X):
        predicciones = np.array([clasificador.predict(X,multiclase=True) for clasificador in self.clasificadores])
        return np.argmax(predicciones,axis = 0)



def test_decisionStump_init(X_train):
    print("---Test de inicialización DecisionStump---")
    n_features = X_train.shape[1]
    stump = DecisionStump(n_features)
    print(f"Característica: 0 <= {stump.caracteristica} < {n_features}")
    print(f"Polaridad: {stump.polaridad}")
    print(f"Umbral: 0 <= {stump.umbral} <= 1")

def test_decisiontStump_prediction(X_train):
    print("---Test de prediccion de DecisionStump---")
    stump = DecisionStump(X_train.shape[1])
    predictions = stump.predict(X_train)
    print("Número de predicciones: ",len(predictions))
    assert set(predictions).issubset({-1, 1}), "Valores de predicción incorrectos"

def test_adaboost_init():
    print("---Test de inicialización de Adaboost---")
    adaboost = Adaboost()
    print("T = ",adaboost.T)
    print("A = ",adaboost.A)

def test_adaboost_fit(X_train,Y_train):
    print("---Test de entrenamiento de Adaboost---")
    adaboost = Adaboost()
    adaboost.fit(X_train, Y_train)
    print("Número de clasificadores: ", len(adaboost.clasificadores))

def test_adaboost_prediction(X_train,Y_train,X_test):
    print("---Test de prediccion de Adaboost---")
    adaboost = Adaboost()
    adaboost.fit(X_train, Y_train)
    predictions = adaboost.predict(X_test)
    print("Número de predicciones: ",len(predictions))
    assert set(predictions).issubset({-1, 1}), "Valores de predicción incorrectos"

def tests_DecisionStump():
    X_train,Y_train,X_test,Y_test = cargarMNISTClase(clase=9)
    test_decisionStump_init(X_train)
    test_decisiontStump_prediction(X_train)

def tests_Adaboost():
    X_train,Y_train,X_test,Y_test = cargarMNISTClase(clase=9)
    test_adaboost_init()
    test_adaboost_fit(X_train,Y_train)
    test_adaboost_prediction(X_train,Y_train,X_test)

def tareas_1A_y_1B_adaboost_binario(clase,T,A,verbose):
    if T >= 1 and A >= 1:
        print(f"Adaboost Binario para la clase: {clase}")
        X_train,Y_train,X_test,Y_test = cargarMNISTClase(clase)
        if verbose:
            print(f"Entrenando clasificador Adaboost para el dígito {clase}, T={T},A={A}")

        start = time.time()
        ada = Adaboost(T=T,A=A)
        ada.fit(X_train,Y_train,verbose=verbose)
        y_pred_test = ada.predict(X_test)
        y_pred_train = ada.predict(X_train)
        end = time.time()
        tasaAciertoTest = accuracy_score(Y_test,y_pred_test) * 100
        tasaAciertoTrain = accuracy_score(Y_train,y_pred_train) * 100
        tiempo = end-start
        print(f"Tasas acierto (train, test) y tiempo: {tasaAciertoTrain}%, {tasaAciertoTest}%, {tiempo} s.")
        return ((tasaAciertoTrain,tasaAciertoTest),tiempo)
    else:
        print("El valor de A debe ser mayor a 0, sin embargo, el valor de T debe ser mayor que 0")
        return None

def test_adaboost_multiclase_init():
    print("---Inicialización de Adaboost Multiclase---")
    adaboost_multiclase = AdaboostMulticlase(clases=10, T=5, A=10)
    print("Número de clases: ", len(adaboost_multiclase.clasificadores))

def test_adaboost_multiclase_fit(X_train,Y_train):
    print("---Entrenamiento de Adaboost Multiclase---")
    adaboost_multiclase = AdaboostMulticlase(clases=10, T=5, A=10)
    adaboost_multiclase.fit(X_train, Y_train, verbose=False)
    for i,clasificador in enumerate(adaboost_multiclase.clasificadores):
        print(f"Número de clasificadores para el clasificador {i+1}: {len(clasificador.clasificadores)}")

def test_adaboost_multiclase_predict(X_train,Y_train,X_test,Y_test):
    print("---Predicción de Adaboost Multiclase---")
    adaboost_multiclase = AdaboostMulticlase(clases=10, T=5, A=10)
    adaboost_multiclase.fit(X_train, Y_train, verbose=False)
    predictions = adaboost_multiclase.predict(X_test)
    print(f"Número de predicciones: ",len(predictions))

def test_AdaboostMulticlase():
    X_train,Y_train,X_test,Y_test = cargarMNISTClase(multiclase=True)
    test_adaboost_multiclase_init()
    test_adaboost_multiclase_fit(X_train,Y_train)
    test_adaboost_multiclase_predict(X_train,Y_train,X_test,Y_test)

def tarea_1D_adaboost_multiclase(T,A,verbose):
    if T >= 1 and A >= 1:
        print("Adaboost Multiclase")
        time.sleep(5)
        X_train,Y_train,X_test,Y_test = cargarMNISTClase(multiclase=True)

        adaMulti = AdaboostMulticlase(T=T,A=A)
        start = time.time()
        adaMulti.fit(X_train,Y_train,verbose)
        y_pred_test = adaMulti.predict(X_test)
        y_pred_train = adaMulti.predict(X_train)
        end = time.time()
        tasaAciertoTest = accuracy_score(Y_test,y_pred_test) * 100
        tasaAciertoTrain = accuracy_score(Y_train,y_pred_train) * 100
        tiempo = end - start
        print(f"Tasas acierto (train, test) y tiempo: {tasaAciertoTrain}%, {tasaAciertoTest}%, {tiempo} s.")

        return ((tasaAciertoTrain,tasaAciertoTest),tiempo)
    else:
        print("El valor de A debe ser mayor a 0, sin embargo, el valor de T debe ser mayor que 0")
        return None

# n_Estimators equivale a A, indica cuantos clasificadores débiles se van a entrenar.
def tarea_2A_AdaBoostClassifier_default():
    print("Adaboost Classifier Default")
    
    X_train,Y_train,X_test,Y_test = cargarMNISTClase(adaboostclassifier=True)
    ada = AdaBoostClassifier()
    start = time.time()
    ada.fit(X_train,Y_train)
    y_pred_test = ada.predict(X_test)
    y_pred_train = ada.predict(X_train)
    end = time.time()
    tasaAciertoTest = accuracy_score(Y_test,y_pred_test) * 100
    tasaAciertoTrain = accuracy_score(Y_train,y_pred_train) * 100
    tiempo = end - start
    print(f"Tasas de acierto (train,test) y tiempo: {tasaAciertoTrain}%, {tasaAciertoTest}%, {tiempo} s.")
    return ((tasaAciertoTrain,tasaAciertoTest),tiempo)

def tarea_2B_graficas_rendimiento(rend_1A,rend_1D,rend_2A):
    tasas_1A, tiempos_1A = rend_1A
    tasas_1D, tiempos_1D = rend_1D
    tasas_2A, tiempos_2A = rend_2A

    # Precisión y tiempos de entrenamiento para cada método
    precisiones = [tasas_1A[1],tasas_1D[1], tasas_2A[1]]
    tiempos = [tiempos_1A,tiempos_1D, tiempos_2A]
    metodos = ['1A','1D', '2A']

    # Configuración de la gráfica
    fig, ax1 = plt.subplots()

    # Graficar la precisión
    color = 'tab:blue'
    ax1.set_xlabel('Tareas')
    ax1.set_ylabel('Precisión (%)', color=color)
    ax1.bar(metodos, precisiones, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # Utilizar un eje y adicional para el tiempo de entrenamiento
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Tiempo de Entrenamiento (s)', color=color)
    ax2.plot(metodos, tiempos, label='Tiempo de Entrenamiento', color=color, marker='o')
    ax2.tick_params(axis='y', labelcolor=color)

    # Mostrar la gráfica
    plt.title('Comparación de Rendimiento y Tiempo de Entrenamiento')
    fig.tight_layout()
    plt.show()

    
def tarea_2C_AdaBoostClassifier_DecisionTree(profundidad):
    if profundidad >= 1:
        print(f"Adaboost Classifier con Árboles de decision de profundidad={profundidad}")
        ada = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=profundidad))
        X_train,Y_train,X_test,Y_test = cargarMNISTClase(adaboostclassifier=True)
        start = time.time()
        ada.fit(X_train,Y_train)
        y_pred_test = ada.predict(X_test)
        y_pred_train = ada.predict(X_train)
        end = time.time()
        tasaAciertoTest = accuracy_score(Y_test,y_pred_test) * 100
        tasaAciertoTrain = accuracy_score(Y_train,y_pred_train) * 100
        tiempo = end - start
        print(f"Tasas de acierto (train,test) y tiempo: {tasaAciertoTrain}%, {tasaAciertoTest}%, {tiempo} s.")
        return ((tasaAciertoTrain,tasaAciertoTest),tiempo)
    else:
        print("La profundidad del árbol debe ser mayor o igual a 1.")
        return None
    
def tarea_2D_MLP_Keras(n_hid_lyrs,n_nrns_lyr,verbose,activacion):
    print("Red neuronal MLP")
    X_train,Y_train, X_test,Y_test = cargarMNISTClase(mlp=True)
    modelo = Sequential()

    #Añadimos la capa inicial con 784 neuronas, 1 por cada característica.
    modelo.add(Dense(784,activation=activacion,input_shape=(784,)))
    
    #Añadimos tantas capas ocultas como hayamos indicado
    for _ in range(n_hid_lyrs):
        #Cada capa tendrá 'n_nrns_lyr' neuronas
        modelo.add(Dense(n_nrns_lyr,activation=activacion))
    
    #Añadimos la capa final, que tendrá 10 neuronas, cada neurona indicará la característica.
    modelo.add(Dense(10,activation='softmax'))

    modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    start = time.time()
    modelo.fit(X_train,Y_train,epochs=10, batch_size=128,verbose=verbose)
    score = modelo.evaluate(X_test,Y_test,verbose=verbose)
    end = time.time()
    tiempo = end-start
    print(f"Tasas (pérdida, acierto): {score[0]*100}%, {score[1]*100}% y tiempo = {end - start} s.")
    return (score,tiempo)

def tarea_2E_CNN_Keras(verbose,activation):
    
    print("Red neuronal CNN")
    X_train,Y_train, X_test,Y_test = cargarMNISTClase(cnn=True)
    modelo = Sequential()

    #Añade la capa de entrada al modelo. La forma de entrada es (28,28,1), correspondiente a imágenes de 28x28 píxeles con un solo canal de color (escala de grises).
    modelo.add(keras.Input(shape=(28,28,1)))

    #Añade una capa convolucional con 32 filtros (kernels), cada uno de tamaño 3x3. La función de activación se especifica con el parámetro activation.
    modelo.add(Conv2D(32, kernel_size=(3,3), activation=activation))

    #Añade una capa de agrupamiento máximo (Max Pooling) con un tamaño de ventana de 2x2. Esto reduce las dimensiones espaciales (altura y anchura) de los mapas de características.
    modelo.add(MaxPooling2D(pool_size=(2,2)))

    #Añade otra capa convolucional, esta vez con 64 filtros, manteniendo el mismo tamaño de kernel y función de activación.
    modelo.add(Conv2D(64, kernel_size=(3,3), activation=activation))

    #Añade otra capa de agrupamiento máximo con el mismo tamaño de ventana para reducir aún más las dimensiones de los mapas de características.
    modelo.add(MaxPooling2D(pool_size=(2,2)))

    # Aplana los mapas de características para convertirlos en un vector unidimensional.
    modelo.add(Flatten())

    #Añade una capa de Dropout con una tasa de 0.5 para reducir el sobreajuste. Durante el entrenamiento, el 50% de las unidades de la capa anterior se desactivarán aleatoriamente.
    modelo.add(Dropout(0.5))

    #Añade una capa densa (Fully Connected) con 10 unidades (una por cada clase en MNIST) y una función de activación softmax para la clasificación multiclase.
    modelo.add(Dense(10,activation="softmax"))
    batch_size = 128
    epochs = 10
    modelo.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    start = time.time()
    modelo.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1,verbose=verbose)
    score = modelo.evaluate(X_test, Y_test, verbose=verbose)
    end = time.time()
    tiempo = end-start
    print(f"Tasas (pérdida, acierto): {score[0] * 100}%, {score[1] * 100}% y tiempo = {end - start} s.")
    return (score,tiempo)

def comparativaRedesNeuronales(rend_2D, rend_2E):
    """
    Función para comparar el rendimiento de dos configuraciones de redes neuronales.

    Args:
    rend_2D (tuple): Contiene el rendimiento (score, tiempo) de la primera configuración.
    rend_2E (tuple): Contiene el rendimiento (score, tiempo) de la segunda configuración.
    """
    # Extracción de datos de rendimiento
    tasas_2D, tiempos_2D = rend_2D[0], rend_2D[1]  # score, tiempo
    tasas_2E, tiempos_2E = rend_2E[0], rend_2E[1]  # score, tiempo

    # Precisión y tiempos de entrenamiento para cada método
    precisiones = [tasas_2D[1] * 100, tasas_2E[1] * 100]
    tiempos = [tiempos_2D, tiempos_2E]

    # Configuración de la gráfica
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Gráfica de Precisión
    ax1.bar(['Configuración 2D', 'Configuración 2E'], precisiones, color=['blue', 'green'])
    ax1.set_title('Comparación de Precisión')
    ax1.set_ylabel('Precisión (%)')
    ax1.set_ylim([0, 100])

    # Gráfica de Tiempos de Entrenamiento
    ax2.bar(['Configuración 2D', 'Configuración 2E'], tiempos, color=['red', 'orange'])
    ax2.set_title('Comparación de Tiempos de Entrenamiento')
    ax2.set_ylabel('Tiempo (s)')

    # Mostrar gráficas
    plt.tight_layout()
    plt.show()

def tarea_2F_graficas_rendimiento(rend_1D,rend_2A,rend_2C,rend_2D,rend_2E):
    tasas_1D, tiempos_1D = rend_1D
    tasas_2A, tiempos_2A = rend_2A
    
    tasas_2C, tiempos_2C = rend_2C
    tasas_2D, tiempos_2D = rend_2D[0], rend_2D[1] # score, tiempo
    tasas_2E, tiempos_2E = rend_2E[0], rend_2E[1] # score, tiempo

    # Precisión y tiempos de entrenamiento para cada método
    precisiones = [tasas_1D[1], tasas_2A[1], tasas_2C[1], tasas_2D[1] * 100, tasas_2E[1] * 100]
    tiempos = [tiempos_1D, tiempos_2A, tiempos_2C, tiempos_2D, tiempos_2E]
    metodos = ['1D', '2A','2C', '2D', '2E']

    # Configuración de la gráfica
    fig, ax1 = plt.subplots()

    # Graficar la precisión
    color = 'tab:blue'
    ax1.set_xlabel('Método')
    ax1.set_ylabel('Precisión (%)', color=color)
    ax1.bar(metodos, precisiones, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # Utilizar un eje y adicional para el tiempo de entrenamiento
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Tiempo de Entrenamiento (s)', color=color)
    ax2.plot(metodos, tiempos, label='Tiempo de Entrenamiento', color=color, marker='o')
    ax2.tick_params(axis='y', labelcolor=color)

    # Mostrar la gráfica
    plt.title('Comparación de Rendimiento y Tiempo de Entrenamiento')
    fig.tight_layout()
    plt.show()


def main():
    
    #tests_DecisionStump()
    #tests_Adaboost()
   
    rend_1A = tareas_1A_y_1B_adaboost_binario(clase=9,T=100,A=9,verbose=False)
   
    #test_AdaboostMulticlase()
    rend_1D = tarea_1D_adaboost_multiclase(T=50,A=20,verbose=False)
    #tarea_1D_graficas_rendimiento(rend_1D,T,A)

    rend_2A = tarea_2A_AdaBoostClassifier_default()
    #tarea_2B_graficas_rendimiento(rend_1A,rend_1D,rend_2A)

    
    
    rend_2C = tarea_2C_AdaBoostClassifier_DecisionTree(profundidad=3)
    

    
    rend_2D = tarea_2D_MLP_Keras(n_hid_lyrs=2,n_nrns_lyr=16,verbose=0,activacion='sigmoid')
    rend_2E = tarea_2E_CNN_Keras(verbose=0,activation='sigmoid')
    #comparativaRedesNeuronales(rend_2D,rend_2E)

    tarea_2F_graficas_rendimiento(rend_1D,rend_2A,rend_2C,rend_2D,rend_2E)

if __name__ == "__main__":
    main()

    




