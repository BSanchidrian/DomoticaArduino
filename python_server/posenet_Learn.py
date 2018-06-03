import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

class posenet_learn(object):
    def __init__(self, learning_rate=0.2):
        self.learning_rate = learning_rate
        self.y = [0, 1, 1, 0]
        a0 = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
              [0, 0], [0, 0], [0, 0], [0, 0]]
        a1 = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
              [0, 0], [0, 0], [0, 0], [0, 0]]
        a2 = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
              [0, 0], [0, 0], [0, 0], [0, 0]]
        a3 = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
              [0, 0], [0, 0], [0, 0], [0, 0]]
        self.X = [a0, a1, a2, a3]
        self.weights1 = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
                         [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]  # 0-16 1ªcapa, 17-21 2ªcapa, 22-27 bias
        self.weights2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.weights1 = 2 * np.random.random((21, 2)) - 1
        self.weights2 = 2 * np.random.random((10, 1)) - 1
        self.Bias = 1

    # Metodo que recorre la red neuronal y devuelve su resultado
    def feedFordward(self, X):
        self.cabeza = self.sigmoide(
            self.weights[0][0] * X[0][0] + self.weights[0][1] * X[0][1] + self.weights[1][0] * X[1][0] +
            self.weights[1][1] * X[1][1] + self.weights[2][0] * X[2][0] + self.weights[2][1] * X[2][1] +
            self.weights[3][0] * X[3][0] + self.weights[3][1] * X[3][1] + self.weights[4][0] * X[4][0] +
            self.weights[4][1] * X[4][1] + self.weights2[5] * self.Bias)
        self.brazoIzq = self.sigmoide(
            self.weights[5][0] * X[5][0] + self.weights[5][1] * X[5][1] + self.weights[7][0] * X[7][0] +
            self.weights[7][1] * X[7][1] + self.weights[9][0] * X[9][0] + self.weights[9][1] * X[9][1] + self.weights2[
                6] * self.Bias)
        self.brazoDch = self.sigmoide(
            self.weights[6][0] * X[6][0] + self.weights[6][1] * X[6][1] + self.weights[8][0] * X[8][0] +
            self.weights[8][1] * X[8][1] + self.weights[10][0] * X[10][0] + self.weights[10][1] * X[10][1] +
            self.weights2[7] * self.Bias)
        self.pierzaIzq = self.sigmoide(
            self.weights[11][0] * X[11][0] + self.weights[11][1] * X[11][1] + self.weights[13][0] * X[13][0] +
            self.weights[13][1] * X[13][1] + self.weights[15][0] * X[15][0] + self.weights[15][1] * X[15][1] +
            self.weights2[8] * self.Bias)
        self.pierzaDch = self.sigmoide(
            self.weights[12][0] * X[12][0] + self.weights[12][1] * X[12][1] + self.weights[14][0] * X[14][0] +
            self.weights[14][1] * X[14][1] + self.weights[16][0] * X[16][0] + self.weights[16][1] * X[16][1] +
            self.weights2[9] * self.Bias)
        salida = self.sigmoide(
            self.cabeza * self.weights2[0] + self.brazoIzq * self.weights2[1] + self.brazoDch * self.weights2[
                2] + self.piernaIzq * self.weights2[3] + self.piernaDch * self.weights2[4] + self.weights2[
                10] * self.Bias)
        return salida

    # Funcion de activacion sigmoide
    def sigmoide(self, x, sigmoide=True):
        return 1 / (1 + np.exp(-x))

    # Metodo para saber si se han alcanzado los margenes de % de acierto
    def errorAlcanzado(self, yHat, i):
        value = 0
        if yHat >= 0.8:
            value = 1
        elif yHat <= 0.2:
            value = 0
        else:
            return False
        if value == self.y[i]: return True
        return False

    # Metodo para actualizar los valores ed los pesos
    def Backpropagation(self):
        i = 0  # Contador de iteraciones para evitar bucles infinitos
        yAux = [0, 0, 0, 0]  # Vector auxiliar que almacena los resultados obtenidos de y en una iteracion del bucle
        errorAlcanzado = False  # bool que nos indica si se han alcanzado los margenes de error
        while errorAlcanzado == False:
            for j in range(len(self.X)):
                i += 1
                yHat = self.feedFordward(self.X[j])
                yAux[j] = yHat
                error = yHat * (1 - yHat) * (self.y[j] - yHat)  # Error de la capa de salida
                self.weights2[0] += self.learning_rate * error * self.cabeza
                self.weights2[1] += self.learning_rate * error * self.brazoIzq
                self.weights2[2] += self.learning_rate * error * self.brazoDch
                self.weights2[3] += self.learning_rate * error * self.pierzaIzq
                self.weights2[4] += self.learning_rate * error * self.pierzaDch
                self.weights[20] += self.learning_rate * error * self.Bias
                error_cabeza = error * self.weights[17] * self.cabeza * (1 - self.cabeza)
                self.weights[0][0] += self.learning_rate * error_cabeza * self.X[0][0]
                self.weights[0][1] += self.learning_rate * error_cabeza * self.X[0][1]
                self.weights[1][0] += self.learning_rate * error_cabeza * self.X[1][0]
                self.weights[1][1] += self.learning_rate * error_cabeza * self.X[1][1]
                self.weights[2][0] += self.learning_rate * error_cabeza * self.X[2][0]
                self.weights[2][1] += self.learning_rate * error_cabeza * self.X[2][1]
                self.weights[3][0] += self.learning_rate * error_cabeza * self.X[3][0]
                self.weights[3][1] += self.learning_rate * error_cabeza * self.X[3][1]
                self.weights[4][0] += self.learning_rate * error_cabeza * self.X[4][0]
                self.weights[4][1] += self.learning_rate * error_cabeza * self.X[4][1]
                self.weights2[5] += self.learning_rate * error_cabeza * self.Bias
                error_brazoIzq = error * self.weights[18] * self.brazoIzq * (1 - self.brazoIzq)
                self.weights[5][0] += self.learning_rate * error_brazoIzq * self.X[5][0]
                self.weights[5][1] += self.learning_rate * error_brazoIzq * self.X[5][1]
                self.weights[7][0] += self.learning_rate * error_brazoIzq * self.X[7][0]
                self.weights[7][1] += self.learning_rate * error_brazoIzq * self.X[7][1]
                self.weights[9][0] += self.learning_rate * error_brazoIzq * self.X[9][0]
                self.weights[9][1] += self.learning_rate * error_brazoIzq * self.X[9][1]
                self.weights2[6] += self.learning_rate * error_brazoIzq * self.Bias
                error_brazoDch = error * self.weights[19] * self.brazoDch * (1 - self.brazoDch)
                self.weights[6][0] += self.learning_rate * error_brazoDch * self.X[6][0]
                self.weights[6][1] += self.learning_rate * error_brazoDch * self.X[6][1]
                self.weights[8][0] += self.learning_rate * error_brazoDch * self.X[8][0]
                self.weights[8][1] += self.learning_rate * error_brazoDch * self.X[8][1]
                self.weights[10][0] += self.learning_rate * error_brazoDch * self.X[10][0]
                self.weights[10][1] += self.learning_rate * error_brazoDch * self.X[10][1]
                self.weights2[7] += self.learning_rate * error_brazoDch * self.Bias
                error_piernaIzq = error * self.weights[20] * self.piernaIz * (1 - self.piernaIz)
                self.weights[11][0] += self.learning_rate * error_piernaIzq * self.X[11][0]
                self.weights[11][1] += self.learning_rate * error_piernaIzq * self.X[11][1]
                self.weights[13][0] += self.learning_rate * error_piernaIzq * self.X[13][0]
                self.weights[13][1] += self.learning_rate * error_piernaIzq * self.X[13][1]
                self.weights[15][0] += self.learning_rate * error_piernaIzq * self.X[15][0]
                self.weights[15][1] += self.learning_rate * error_piernaIzq * self.X[15][1]
                self.weights2[8] += self.learning_rate * error_piernaIzq * self.Bias
                error_piernaDch = error * self.weights[21] * self.piernaDch * (1 - self.piernaDch)
                self.weights[12][0] += self.learning_rate * error_piernaDch * self.X[12][0]
                self.weights[12][1] += self.learning_rate * error_piernaDch * self.X[12][1]
                self.weights[14][0] += self.learning_rate * error_piernaDch * self.X[14][0]
                self.weights[14][1] += self.learning_rate * error_piernaDch * self.X[14][1]
                self.weights[16][0] += self.learning_rate * error_piernaDch * self.X[16][0]
                self.weights[16][1] += self.learning_rate * error_piernaDch * self.X[16][1]
                self.weights2[9] += self.learning_rate * error_piernaDch * self.Bias

            for k in range(len(yAux)):  # bucle para comprobar que todos
                resultado = self.errorAlcanzado(yAux[k], k)
                if resultado == False: break
            errorAlcanzado = resultado
            if i == 100000: errorAlcanzado = True  # En el caso de que se hagan 100000 iteraciones se cierra el bucle
            # print("Iteraciones: %d"%i)

    def fit(self):
        self.Backpropagation()

    def predict(self, x):
        feed = self.feedFordward(x)
        print(feed)
        return '1' if (feed >= 0.5) else '0'  # Redondeamos los datos para mostrarlos en pantalla

    def train(self, buffer):
        positions = []
        for entry in buffer:
            for part in entry:
                positions.append(part["position"])
        print(positions)