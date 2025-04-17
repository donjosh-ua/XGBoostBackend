import numpy as np
import torch
import sys
import pandas as pd
from app.XGBoostRNA.redneuronal_bay.Layers.layers import *
from app.XGBoostRNA.redneuronal_bay.utils import *
from app.XGBoostRNA.redneuronal_bay.funcion_costo import *
from collections import OrderedDict, defaultdict
from app.XGBoostRNA.redneuronal_bay.Optimizers.optimizers import *
from app.XGBoostRNA.redneuronal_bay.metricas_eva import *
from app.XGBoostRNA.redneuronal_bay.Div_Datos import *
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from builtins import input

output_folder = "app/XGBoostRNA/rna_models/"

_optimizers = {
    "SGD": SGD,
    "Nesterov": Nesterov,
    "Adagrad": Adagrad,
    "Adam": Adam,
    "RMSProp": RMSProp,
}
_Criteria = {
    "SSE": SSE,
    "cross_entropy": cross_entropy,
    "Log_neg": NLLLoss,
    "entropy_bin": BCELoss,
}
# _FA_extra = {"softmax": softmax}
# _metrics = {'MSE':mean_squared_error, 'AS':accuracy_score}


class RedNeuBay(object):
    """
    Clase principal de la red bayesiana
    """

    def __init__(
        self,
        tol=0.001,
        alpha=0.01,
        epoch=100,
        criteria="cross_entropy",
        optimizer="SGD",
        image_size=1,
        verbose=True,
        Lambda=0.005,
        decay=0.0,
        momentum=0.0,
        image=True,
        FA_ext=None,
        Bay=False,
        save_mod="ModR",
        pred_hot=True,
        test_size=0.2,
        batch_size=64,
        cv=False,
        Kfold=5,
    ):
        """
        :tol: tolerance del error 0.0001.
        :alpha: learning rate.
        :epoch: epochs
        :criteria: Funcion de perdida.
        :image_size: Si trabaja con imagenes colocar el tamaño por ejemplo con una de 28x 28 colocar 784. Si no trabaja con imagenes "None"
        :verbose: Si es True imprime el error de cada epoch, por default es True.
        :decay: Decaimiento de la tasa de learning rate
        :momentum: Rango de valores [0,1]. Se recomienda usar un learning rate bajo con un momentum alto y viceversa.
        :image: permite aplanar las matrices en caso que sea imagenes
        : ojo lambda puede ser 0.005
        """

        "Compruebo que los arguentos de la red esten correctos."
        assert criteria in [
            "SSE",
            "cross_entropy",
            "Log_neg",
            "entropy_bin",
        ], "{0} no es compatible la funcion".format(criteria)
        assert epoch > 0, "El epoch debe ser mayor a 0"
        assert (
            verbose == True or verbose == False
        ), "{0} no es compatible, coloque True o False".format(verbose)
        assert (
            momentum >= 0 and momentum <= 1
        ), "Precaucion el momento debe estar en el rango [0,1]"
        assert optimizer in _optimizers, "{0} el colocado no es compatible".format(
            optimizer
        )
        # assert (metric in _metrics), '{0} no es compatible '.format(metric)

        self.layernum = -1
        self.layersObject = defaultdict()  # Para que me almacene las capas de la red
        self.criteria = _Criteria[criteria]
        self.image_size = image_size
        self.learningRate = alpha
        self.epoch = epoch
        self.verbose = verbose
        self.tol = tol
        self.Lambda = Lambda
        self.momentum = momentum
        self.Bay = Bay
        self.decay = decay
        self.image = image
        self.save_mod = save_mod
        self.pred_hot = pred_hot
        self.batch_size = batch_size
        self.test_size = test_size
        self.cv = cv
        self.Kfold = Kfold
        self.optimizer = _optimizers[optimizer](
            learning_rate=self.learningRate,
            decay=self.decay,
            momentum=self.momentum,
            Bay=self.Bay,
            img=self.image,
            image_size=self.image_size,
        )
        # self.metric = _metrics[metric]

    "Iteracion sobre las capas de la red"

    def __iter__(self):
        return [i for i in self.layersObject]

    def __str__(self):
        """
        Anulo la funcion de impresion de python.
        :regresaa: Especificacion de la red.
        """

        return (
            "******************RedNeuBay summary******************" + "\n"
            "{:>10s}{:>10d}\n".format("Layers", len(self.layersObject))
            + "{:>12s}{:>18s}\n".format("Criteria", self.criteria.__name__)
            + "{:>13s}{:>8s}\n".format("Optimizer", self.optimizer.__class__.__name__)
            + "{:>9s}{:>12d}\n".format("Epoch", self.epoch)
            + "{:>12s}{:>10f}\n".format(
                "Learning rate",
                self.learningRate,
            )
            + "{:>9s}{:>14f}\n".format("Decay", self.decay)
            + "{:>12s}{:>10.2f}\n".format("Momentum", self.momentum)
        )

    def add(self, layer_object):
        """
        Se v agregando una capa a la red y se cmprueba si la dimension de salida
        concuerda con la de entrada de la siguiente capa

        :layer_object: Capa que se agregara a la red neuronal.
        :regresa: None
        """
        if self.layernum != -1:
            oldLayer = self.layersObject[self.layernum]
            assert oldLayer.output_dim == layer_object.input_dim, (
                "La dimension de entrada de la capa actual no coincide con la dimension de salida de la capa anterior,"
                "Dimension de entrada de la capa actual = {0}, Dimension de salida d ela capa anterior = {1}.".format(
                    layer_object.input_dim, oldLayer.output_dim
                )
            )

        self.layernum += 1
        self.layersObject[self.layernum] = layer_object

    def train(
        self, df_cla=False, X_cla=False, Y_cla=False, train_set=False, test_set=False
    ):  # ojo revisar como ingresar
        """
        Funcion para el entrenamiento de la red.
        :xy: training - archivo en formato Dataloader (1 solo archivo con inputs y targets)
        :inputs en formato FloatTensor
        :outputs en formato LongTensor
        :retorna: Valores de Loss y accuracy en cada epoch
        """

        # Primero toma los datos y los divide en entrenamiento y test (solo en datos normales) y en imagenes se debe ingresar
        # la base de entrenamiento y test. Además en los dos casos hace el batch size
        if self.image == True:
            train_loader, X_test, Y_test = trat_Imag(
                train_set=train_set,
                test_set=test_set,
                batch_size=self.batch_size,
                test_size=self.test_size,
            )
        if self.image == False:
            train_loader, X_test, Y_test = trat_Dat(
                df_cla=df_cla,
                X_cla=X_cla,
                Y_cla=Y_cla,
                batch_size=self.batch_size,
                test_size=self.test_size,
            )

        print("-----------------------------------")
        print("----Iniciando de entrenamiento-----")
        self.optimizer.backprop(
            train_loader,
            self,
            save_mod=self.save_mod,
            verbose=self.verbose,
            cv=self.cv,
            k=self.cv,
        )

        # -------------------------------------
        # Para predecir la base de datos test en caliente, caso contrario guarda la base de datos X y Y test formato csv

        if self.pred_hot == True:
            if test_set == False and self.image == True:
                print(
                    "No se puede predecir ya que no a ingresado la base de datos para el test"
                )
            else:
                nam = "best_" + self.save_mod
                ee = torch.load(output_folder + nam)
                self.predict(
                    mod=ee,
                    x=X_test,
                    y=Y_test,
                    img=self.image,
                    image_size=self.image_size,
                    target=True,
                )
        else:

            if self.image == True and X_test == 0:
                print("")
            else:
                nam_test = input("With what name do you want to save data test: ")
                nam_test = nam_test + ".csv"
                # print(X_test.shape)
                # print(Y_test.shape)
                Test = np.insert(X_test, X_test.shape[1], Y_test, 1)
                # print(Test.shape)
                Test = pd.DataFrame(Test)  # Data frame dpara guardar base d test
                Test.to_csv(
                    nam_test, sep=","
                )  # Se guardan los resultados para metricas
        # ---------------------------------------

        return self.layersObject

    def predict(self, mod, x, y=0, img=True, image_size=1, target=True):
        """
        Funcion para predecir
        :x: test
        :regresa: valores predichos
        """

        print("---------------------------------------")
        print("-------Iniciando predicción------------")
        print("---------------------------------------")
        outputs = []

        if img == True:
            enput = x.view(-1, image_size)
        else:
            enput = torch.FloatTensor(x)
            y = torch.Tensor(y)

        for i in range(len(mod)):
            layer = mod[i]
            # print('entrada',input.shape)
            # print('pesos',layer.weights.shape)
            # print('sesgo',layer.bias.shape)

            a = 1
            Output = layer.funcion_activacion(
                torch.add(torch.matmul(enput, layer.weights), layer.bias), a
            )
            Output = torch.FloatTensor(Output)
            outputs = Output
            enput = Output
            # print(outputs)
            # print('vuelta',i)

        _, pred = torch.max(outputs, 1)

        if target == False:
            print(pred)
        else:
            # print(pred)
            n_total_row = len(y)
            ac = torch.sum(pred == y).float() / n_total_row
            ac = np.asarray(ac)
            print("---------------------------------------")
            print("--------Rendimiento del modelo---------")
            print(f"Accuracy test: {np.round(ac*100.0,2)}%")
            print("---------------------------------------")

            # Guardar resultados para analizar con librera de metricas diseñada
            name_res = input("With what name do you want to save results: ")
            name_res = name_res + ".csv"
            resultados = {"True": y, "Predicted": pred}
            resultados = pd.DataFrame(
                resultados, columns=["True", "Predicted"]
            )  # Data frame de resultados para utilizar metricas
            resultados.to_csv(
                name_res, sep=","
            )  # Se guardan los resultados para metricas
            print("---------------------------------------")
            print("----10 predicciones iniciales----------")
            print(resultados.head(10))
            print("---------------------------------------")
            print("-------matriz de confusión-------------")
            print("---------------------------------------")

            # Matriz de confusion
            print("antes")
            cm = confusion_matrix(pred, y)
            tar = np.unique(y, return_counts=True)
            targ = np.array(tar[0])
            ncat = len(targ)
            # ncat = int(input('How many classes are you going to predict='))

            b = listofzeros = [0] * ncat
            for i in range(ncat):
                b[i] = i

            cm_test = pd.DataFrame(cm, index=[i for i in b], columns=[i for i in b])
            plt.figure(figsize=(10, 7))
            sns.set(font_scale=1)
            sns.heatmap(
                cm_test, annot=True, cmap=plt.cm.Reds, fmt="d"
            )  # color azul cmap="YlGnBu"
            plt.title("Confusion Matrix - Test")
            plt.ylabel("Classes")

        return mod

    def cv_train(
        self, df_cla=False, X_cla=False, Y_cla=False, train_set=False, test_set=False
    ):  # ojo revisar como ingresar
        """
        Funcion para el entrenamiento de la red.
        :xy: training - archivo en formato Dataloader (1 solo archivo con inputs y targets)
        :inputs en formato FloatTensor
        :outputs en formato LongTensor
        :retorna: Valores de Loss y accuracy en cada epoch
        """

        # Primero toma los datos y los divide en entrenamiento y test (solo en datos normales) y en imagenes se debe ingresar
        # la base de entrenamiento y test. Además en los dos casos hace el batch size
        if self.image == True:
            train_loader, X_test, Y_test = trat_Imag(
                train_set=train_set,
                test_set=test_set,
                batch_size=self.batch_size,
                test_size=self.test_size,
            )
        if self.image == False:
            X, Y = cv_prepros(
                df_cla=df_cla, X_cla=X_cla, Y_cla=Y_cla
            )  # funcion de libreria Div_Dat - me da dataset
            inicio = 0
            K_tra_val = (
                len(X)
            ) // self.Kfold  # las dos barras me hacen division entera - numero de elementos para cada divison k-fold
            k_acc_p = np.array([])
            K_acc = np.zeros(shape=(self.Kfold, self.epoch))
            K_loss = np.zeros(shape=(self.Kfold, self.epoch))

            for i in range(self.Kfold):
                ite = i
                train_loader, X_test, Y_test, fin = cv_trat_Dat(
                    X=X,
                    Y=Y,
                    batch_size=self.batch_size,
                    Kfold=self.Kfold,
                    inicio=inicio,
                    tra_val=K_tra_val,
                    ite=ite,
                )
                inicio = fin

                print("----------------------------------------")
                print("--Iniciando de entrenamiento Kfold=", (ite + 1), "--")
                mode, k_ac, k_los = self.optimizer.backprop(
                    train_loader,
                    self,
                    save_mod=self.save_mod,
                    verbose=self.verbose,
                    cv=self.cv,
                    k=(ite + 1),
                )

                # -------------------------------------------------------------
                # Guardo accuracys y loss de cada modelo por epoch paa graficar al final
                K_acc[i] = k_ac
                K_loss[i] = k_los
                # -------------------------------------------------------------
                k_item = str(ite + 1)
                nam = "best_" + self.save_mod + "_K" + k_item

                # ee = torch.load(nam)
                ee = torch.load(output_folder + nam, weights_only=False)

                ac = self.cv_predict(
                    mod=ee,
                    x=X_test,
                    y=Y_test,
                    img=self.image,
                    image_size=self.image_size,
                    target=True,
                    k=(ite + 1),
                )
                k_acc_p = np.append(
                    k_acc_p, ac
                )  # Guardo los accuracys de prediccion para sacar la media al final

            K_accuracy = np.mean(k_acc_p)

            print("---------------------------------------")
            print("-------Rendimiento final por CV--------")
            print("----------------------------------------")
            print(f"Accuracy test: {np.round(K_accuracy*100.0,2)}%")
            print("---------------------------------------")

            # ------------------------------------------------------------
            # Graficas de la CV de todos los modelo Acc y loss

            # Grafica de Loss sin batch solo con epochs
            plt.figure()
            ### cambio por actualizacion de la nueva libreria ###
            # plt.style.use('seaborn-whitegrid')
            sns.set_theme(style="whitegrid")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.title("Loss - Train")
            for i in range(len(K_loss)):
                plt.plot(np.arange(self.epoch), K_loss[i], label="k" + str(i + 1))

            plt.legend()
            plt.grid()
            plt.show()

            # Grafica de Accuracy sin batch solo con epochs
            plt.figure()
            ### cambio por actualizacion de la nueva libreria ###
            # plt.style.use('seaborn-whitegrid')
            sns.set_theme(style="whitegrid")
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
            plt.title("Accuracy - Train")
            for i in range(len(K_loss)):
                plt.plot(np.arange(self.epoch), K_acc[i], label="k" + str(i + 1))

            plt.legend()
            plt.grid()
            plt.show()
            # -------------------------------------------------------------

        return self.layersObject

    def cv_predict(self, mod, x, y=0, img=True, image_size=1, target=True, k=0):
        """
        Funcion para predecir
        :x: test
        :regresa: valores predichos
        """

        print("----------------------------------------")
        print("--Iniciando predicción Kfold=", (k), "--")
        print("----------------------------------------")
        outputs = []

        if img == True:
            enput = x.view(-1, image_size)
        else:
            enput = torch.FloatTensor(x)
            y = torch.Tensor(y)

        for i in range(len(mod)):
            layer = mod[i]
            # print('entrada',input.shape)
            # print('pesos',layer.weights.shape)
            # print('sesgo',layer.bias.shape)

            a = 1
            Output = layer.funcion_activacion(
                torch.add(torch.matmul(enput, layer.weights), layer.bias), a
            )
            Output = torch.FloatTensor(Output)
            outputs = Output
            enput = Output
            # print(outputs)
            # print('vuelta',i)

        _, pred = torch.max(outputs, 1)

        if target == False:
            print(pred)
        else:
            # print(pred)
            n_total_row = len(y)
            ac = torch.sum(pred == y).float() / n_total_row
            ac = np.asarray(ac)

            # --------

            print("---------------------------------------")
            print("--Rendimiento del modelo-Kfold=", (k), "--")
            print("----------------------------------------")
            print(f"Accuracy test: {np.round(ac*100.0,2)}%")
            print("---------------------------------------")

            print("---------------------------------------")
            print("-------matriz de confusión-------------")
            print("---------------------------------------")

            # Matriz de confusion
            cm = confusion_matrix(pred, y)

            # Para ver el numero de targets
            tar = np.unique(y, return_counts=True)
            targ = np.array(tar[0])
            ncat = len(targ)
            # ncat = int(input('How many classes are you going to predict='))

            b = listofzeros = [0] * ncat
            for i in range(ncat):
                b[i] = i

            cm_test = pd.DataFrame(cm, index=[i for i in b], columns=[i for i in b])
            # plt.figure(figsize = (10,7))
            plt.figure()
            sns.set(font_scale=1)
            sns.heatmap(
                cm_test, annot=True, cmap=plt.cm.Reds, fmt="d"
            )  # color azul cmap="YlGnBu"
            plt.title("Confusion Matrix - Test")
            plt.ylabel("Classes")
            plt.show()

        return ac
