from __future__ import division
from abc import abstractmethod, ABCMeta
import numpy as np
import torch
import time as t
import pytensor.tensor as tt
import pymc as pm
from torch.autograd import Variable
from app.XGBoostRNA.redneuronal_bay.utils import *
from app.XGBoostRNA.redneuronal_bay.metricas_eva import accuracy_score
from app.XGBoostRNA.redneuronal_bay.utils import cambia_labels
from app.XGBoostRNA.redneuronal_bay.funcion_activacion import softmax
import matplotlib.pyplot as plt
import logging
import seaborn as sns

output_folder = "app/XGBoostRNA/rna_models/"

"Clase base para todos los optimizadores"


class BaseOptimizer:
    """
    Es una clase base y no debe ser instanciada
    """

    __metaclass__ = ABCMeta

    def __init__(self, learning_rate, decay, momentum, Bay, img, image_size):

        self.learning_rate = learning_rate
        self.decay = decay
        self.momentum = momentum
        self.Bay = Bay
        self.img = img
        self.image_size = image_size
        self._gradients = {}

    def _forwardprop(self, Rn, x, run, trace=True):
        """
        Forward propogation metodo para la red neuronal.
        :nn: red neuronal tipo objeto
        :x: datos de entrada
        :trace: (Binario)por default es True
        :regresa: Si trace es true, la funcion devuelve lla entrada de cada capa y la salida de la ultima capa.
        Si trace es False, La funcion devuelve la salida de la ultima capa.

        'Ejemplo: Si tenemos una entrada y dos capas ocultas la salida de forward con trace = True es' \
        '[ (input, None),(input,None),(input,output)]'
        """
        # print('ingreso',x.shape)
        if trace == True:
            outputs = []

        if self.img == True:
            enput = x.view(-1, self.image_size)
            # print('despues de aplanado',enput.shape)
        else:
            enput = x

        # print(f'El numero de instancias son: {len(x)}')
        # print(f'El tamaño de entrada es {x.shape}')

        for i in range(len(Rn.layersObject)):

            layer = Rn.layersObject[i]

            ##print(f'Tamaño de pesos de capa {i}: {layer.weights.shape}')
            # print(input)
            # print(input.shape)
            # print(input.type)
            # print(input.dtype)
            # print(layer.weights)
            ##print(f'Tamaño de sesgos de capa {i}: {layer.bias.shape}')
            # print(i)
            # print(layer.weights.shape)
            # print(layer.weights.dtype)
            # print(layer.bias)
            # print(layer.bias.shape)
            # print(layer.bias.dtype)
            # print('Salida:',i)
            # print(layer.weights.dtype)
            a = 0
            if Rn.epoch == (run + 1):
                a = 1
            # print('capa=',i)
            # print('entrada de x=',enput.shape,'Entrada de pesos',layer.weights.shape, 'entrada de sesgos',layer.bias.shape)
            Output = layer.funcion_activacion(
                torch.add(torch.matmul(enput, layer.weights), layer.bias), a
            )
            Output = Variable(torch.FloatTensor(Output), requires_grad=True)
            ##print(Output)
            # Output = layer.funcion_activacion(np.add(np.dot(input, layer.weights), layer.bias))
            # print (f'Salida de la capa {i}: {Output.shape}')
            # print(Output)

            if trace == True:
                if i == len(Rn.layersObject) - 1:
                    outputs.append((enput, Output))

                else:
                    outputs.append((enput, None))
            enput = Output
            # print(Output)

        if trace == False:
            outputs = Output

        return outputs

    def backprop(self, xy, Rn, save_mod, verbose, cv, k):

        # name_model = input("With what name do you want to save the model: ")
        name_model = save_mod

        # Para que trabaje si tengo cuda osea GPU
        # -----------------------------------------------------
        CUDA = False
        # CUDA = torch.cuda.is_available()

        if CUDA == True:
            self.Rn.cuda()
        # ------------------------------------------------------

        timein = t.time()

        """
        Algoritmo de Backpropogation 
        :nn: Red neuronal tipo  object
        :x: imagenes transformadas de entrenamiento
        :y: targets de entrenamiento
        :regresa: None
        """
        # np.random.seed(Rn.random_seed)
        np.random.seed()

        "La clase m agilitara determinar el accuracy y loss en el caso batch size especialmente"

        class RunningMetric:  # Esta funcion me permite que se vaya obteniendo valores promedio del traajo de la red,

            def __init__(self):
                self.S = 0
                self.N = 0

            def update(self, val, size):
                self.S += val
                self.N += size

            def __call__(self):
                return self.S / float(
                    self.N
                )  # float para tenre un resultado flotante y no una division entera

        "Aqui creo el diccionario que me almacenara los gradientes para ocuparlos en la actualizacion"
        for i in range(len(Rn.layersObject)):
            self._gradients[i] = 0

        self.set_smooth_gradients(Rn.layersObject)

        # y_real = copy.copy(y)

        # if Rn.layersObject[Rn.layernum].output_dim != 1:
        #    y = cambia_labels(y) -- ya no lo necesito pues lo hago en formato dataloader

        "----------------------------------------------------------"
        "Genera lotes de datos dependiendo del tamaño del lote."
        "----------------------------------------------------------"

        # batches = genera_batches(x.shape[0], batch_size=Rn.batch_size)
        # num_batches = len(batches)
        # print(batches[0,:])
        # print(f'batches es: {len(batches)} La dimensiones es: {batches.shape}')
        loss_b = np.array([])  # guarada la perdida en cada iteración
        loss_b2 = np.array([])
        ac_b = np.array([])
        ac_b2 = np.array([])

        run = 0
        while run < Rn.epoch:

            self.decay_learning_rate(run)

            # optimizer.zero_grad()

            ite_act = 0
            # iteraciones = 0
            num_batches = len(xy)
            running_loss = RunningMetric()  # Perdida
            running_acc = RunningMetric()  # Accuracy

            for inputs, targets in xy:
                x, y = inputs, targets

                # Para el caso CUDA ------------------------------------------
                if CUDA == True:
                    x = x.cuda()
                    y = y.cuda()
                # ------------------------------------------------------------

                n_total_row = len(y)
                # print(xtrain.shape)
                # print(ytrain.shape)

                each_layer_output = self._forwardprop(Rn, x, run)

                ##print('Termino el forward propagation')
                # print(f'Tamaño final de salida: {len(each_layer_output)}')

                # print(type(each_layer_output))
                # print((each_layer_output[0]))
                # print((each_layer_output[0][0]).shape)
                # print((each_layer_output[1]))
                # print((each_layer_output[1][0]).shape)
                # print((each_layer_output[2]))
                # print((each_layer_output[2][0]).shape)
                # print((each_layer_output[3]))
                # print((each_layer_output[3][0]).shape)
                # print((each_layer_output[4]))
                # print((each_layer_output[4][0]).shape)
                # print((each_layer_output[4][1]))
                # print((each_layer_output[4][1]).shape)

                reversedlayers = list(range(len(Rn.layersObject))[::-1])
                # print(reversedlayers)
                outputlayerindex = reversedlayers[0]
                imgcurr = self.img  # Nuevo

                for i in reversedlayers:

                    if i == outputlayerindex:
                        # print((Rn.layersObject[i].derivative(each_layer_output[i][1]).shape))
                        # print(ytrain)
                        # print(ytrain.shape)
                        # print(ytrain.dtype)
                        # print(each_layer_output[i][1])
                        # print((each_layer_output[i][1]).shape)
                        # print((each_layer_output[i][1]).dtype)
                        # print(Rn.criteria(ytrain, each_layer_output[i][1]).shape)

                        ##print('inicia loss y  gradientes')

                        actbay = 1  # Nuevo

                        ent = Variable(each_layer_output[i][1], requires_grad=True)

                        # ent = each_layer_output[i][1]
                        ##print('salida')
                        ##print(ent)
                        ##print(ent.dtype)
                        layerout, targ = ent, y

                        F_sigmoide = 0
                        if len(layerout[1]) == 1:
                            pred = torch.round(layerout)
                            F_sigmoide = 1
                        else:
                            _, pred = torch.max(layerout, 1)

                        ac1 = torch.sum(pred == targ)
                        ac = torch.sum(pred == targ).float() / n_total_row
                        # ac_b = np.append(ac_b,np.round(np.asscalar(ac),4)) # Para graficar funcion de perdida
                        ac_b = np.append(ac_b, ac)  # Para graficar funcion de accuracy
                        # print(ac1)
                        # print('accuarcy =', ac)

                        Losgr = Rn.criteria(targ, layerout)

                        Losgr.backward()
                        # loss_b = np.append(loss_b,np.round(np.asscalar(Losgr.data),4)) #Para graficar funcion de perdida
                        # loss_b = np.append(loss_b,(Losgr.data))
                        loss_b = np.append(
                            loss_b, (Losgr.data / num_batches)
                        )  # Para graficar funcion de perdida

                        # iteraciones += 1

                        running_loss.update(Losgr * n_total_row, n_total_row)
                        running_acc.update(
                            ac1.float(), n_total_row
                        )  # Actualizacion y casteo a flotante

                        # print('Loss=', Losgr.data)

                        ##print('termina loss y  gradientes')

                        gradien = ent.grad.data
                        # print('derivadas')
                        # print(gradien)
                        gradient_out = gradien
                        ##print('gradientes de capa final=',i)
                        ##print(gradien)
                        # gradient_out = self._gradients[i][1]

                        ##print('Valor de delta de capa = ', i)
                        delta = self.calculate_delta(gradient_out, Losgr)
                        ##print(delta)
                        ##print(delta.shape)
                        ##print(delta.dtype)

                        # delta = self.calculate_delta(Rn.layersObject[i].derivative(each_layer_output[i][1])
                        #                                  , Rn.criteria(ytrain, each_layer_output[i][1]))
                        # delta = delta.mean(axis=0)

                    else:
                        # print('inicia las siguiente capa=', i)
                        # ents = each_layer_output[i+1][0]
                        # self._gradients[i+1][0] = ents.grad.data
                        # gradient_outs = self._gradients[i+1][0]
                        # print(gradient_outs)
                        # print(gradient_outs.shape)

                        # print('Empieza a calcular delta')
                        # delta2 = self.calculate_delta(gradient_outs, torch.matmul(delta, Rn.layersObject[i+1].weights.T)
                        # print(torch.matmul(delta, Rn.layersObject[i+1].weights.T))
                        # if i==0:
                        #    print(i)
                        #    print(delta)
                        #    print('pesos transpuestos')
                        #    print(Rn.layersObject[i+1].weights.T)
                        # print(i)
                        # print(Rn.layersObject[i].derivative(each_layer_output[i+1][0])
                        # ----------------nuevo-------------------------------
                        if imgcurr == True:
                            actbay = 0
                        else:
                            actbay = 1
                        # -----------------------------------------------------

                        delta = self.calculate_delta(
                            Rn.layersObject[i].derivative(each_layer_output[i + 1][0]),
                            torch.matmul(delta, Rn.layersObject[i + 1].weights.T),
                        )

                    # print(i)
                    # print(delta.shape)
                    # print(delta)
                    # print(each_layer_output[i][0].shape)
                    # print(Rn.layersObject[i].weights.shape)

                    Rn.layersObject[i].weights = self.update_weights(
                        delta,
                        layer_input=each_layer_output[i][0],
                        layer_index=i,
                        curr_weights=Rn.layersObject[i].weights,
                        Lambda=Rn.Lambda,
                        Bay=self.Bay,
                        img=self.img,
                        image_size=self.image_size,
                        ite_act=ite_act,
                        total=num_batches,
                        acbay=actbay,
                    )
                    ##print(f'Ajustado peso: {i}')
                    ##print(Rn.layersObject[i].weights)
                    ##print(Rn.layersObject[i].weights.shape)

                    Rn.layersObject[i].bias = self.update_bias(
                        delta,
                        curr_bias=Rn.layersObject[i].bias,
                        Bay=self.Bay,
                        img=self.img,
                        ite_act=ite_act,
                        total=num_batches,
                        layer_index=i,
                        acbay=actbay,
                    )
                    ##print(f'Ajustado sesgo: {i}')
                    ##print(Rn.layersObject[i].bias)
                    ##print(Rn.layersObject[i].bias.shape)

                ite_act += 1
                if Rn.verbose == True:
                    progreso_epo(
                        ite_act=ite_act,
                        total=num_batches,
                        total_epoch=Rn.epoch,
                        act_epoch=run + 1,
                        prefix="Progress:",
                    )

            # para graficas almacenas perdidas y accuracy por batchs
            "Calcula la perdida de entrenamiento loss(SSE)"
            loss_b2 = np.append(loss_b2, running_loss().detach().numpy())
            ac_b2 = np.append(ac_b2, running_acc().detach().numpy())
            ### Cambio porque asscalar ya no se usa###
            # loss = np.round(np.asscalar(Losgr.data),4)
            loss = np.round(Losgr.data.item(), 4)
            # Para guardar los mejores modelos y accuracy
            k_nam = str(k)

            if cv == True:
                name2 = "best_" + name_model + "_K" + k_nam
            else:
                name2 = "best_" + name_model

            if run == 0:
                best_model = Rn.layersObject
                acc_ini = running_acc().data
            else:
                if running_acc().data >= acc_ini:
                    ep = run
                    best_model = Rn.layersObject
                    torch.save(best_model, output_folder + name2)
                    best_acc = running_acc().data
                    acc_ini = best_acc

            # Para imprimir accuracy y loss si verbse ==True
            ru_los = np.asarray(running_loss().data)
            ru_acc = np.asarray(running_acc().data)

            if Rn.verbose == True:

                # print('learning rate:{0}, loss:{1}'.format(self.learning_rate,(running_loss().data:,.4f)))
                print(
                    f"learning rate:{self.learning_rate} loss:{np.round(ru_los*1.0,4)}"
                )

                if F_sigmoide == 1:
                    print(f"Accuracy: {np.round(ru_acc*1,2)}%")
                else:
                    print(f"Accuracy: {np.round(ru_acc*100.0,2)}%")
            # ---------------------------------------------------------------------------------

            # Para luego graficar el accuracy normal versus el bayesiano grabo los valores
            # Bay=self.Bay
            # if Bay==False:
            #    n1 = 'Acc'
            #    n2 = 'Loss'
            # else:
            #    n1 = 'Acc_Bay'
            #    n2 = 'Loss_Bay'

            # torch.save(ac_b2, n1)
            # torch.save(loss_b2, n2)
            # torch.save(run,'epoch')
            # -------------------------------------------------------------------------------

            run += 1

        timeout = t.time()

        # Para hacer las graficas de Loss y accuracy

        # ---------------------------------------------------------------------------------
        # Para luego graficar el accuracy normal versus el bayesiano grabo los valores
        Bay = self.Bay
        if Bay == False:
            if cv == True:
                n1 = "Acc" + name_model + "_K" + k_nam
                n2 = "Loss" + name_model + "_K" + k_nam
            else:
                n1 = "Acc"
                n2 = "Loss"

        else:
            if cv == True:
                n1 = "Acc_Bay" + name_model + "_K" + k_nam
                n2 = "Loss_Bay" + name_model + "_K" + k_nam
            else:
                n1 = "Acc_Bay"
                n2 = "Loss_Bay"

        # ojo esto solo me sirve para las pruebas para yo poder graficar, guardo los acc y lost
        torch.save(ac_b2, output_folder + n1)
        torch.save(loss_b2, output_folder + n2)
        torch.save(run, output_folder + "epoch")  # Se podria quitar ojo revisar bien
        # -------------------------------------------------------------------------------
        # Para grafica, si no hay cv hace todas las graficas caso contrario no

        if cv == True:
            print("")

        else:

            # Grafica de Loss tomando en cuenta los batch
            ### cambio por actualizacion de la nueva libreria ###
            # plt.style.use('seaborn-whitegrid')
            sns.set_theme(style="whitegrid")

            # plt.style.use('seaborn-whitegrid')
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.title("With batch: Loss - Train")
            plt.plot(np.arange(len(loss_b)), loss_b)

            # Grafica de Loss sin batch solo con epochs
            plt.figure()
            ### cambio por actualizacion de la nueva libreria ###
            # plt.style.use('seaborn-whitegrid')
            sns.set_theme(style="whitegrid")
            # plt.style.use('seaborn-whitegrid')
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.title("Loss - Train")
            plt.plot(np.arange(Rn.epoch), loss_b2)
            plt.show()

            # Grafica de Accuracy tomando en cuenta los batch
            ### cambio por actualizacion de la nueva libreria ###
            # plt.style.use('seaborn-whitegrid')
            sns.set_theme(style="whitegrid")
            # plt.style.use('seaborn-whitegrid')
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
            plt.title("With batch: Accuracy - Train")
            plt.plot(np.arange(len(ac_b)), ac_b)

            # Grafica de Accuracy sin batch solo con epochs
            plt.figure()
            ### cambio por actualizacion de la nueva libreria ###
            # plt.style.use('seaborn-whitegrid')
            sns.set_theme(style="whitegrid")
            # plt.style.use('seaborn-whitegrid')
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
            plt.title("Accuracy - Train")
            plt.plot(np.arange(Rn.epoch), ac_b2)
            plt.show()

        # return (loss_b, ac_b,Rn) # Revisar porque no me retorna los datos
        # print(predicted)
        # print(np.round(predicted,3))

        # Tiempo de ejecución
        print("---------------------------------------")
        print("The process took = ", round((timeout - timein) / 60, 4), "minutes")
        print("---------------------------------------")

        # Guardar el modelo

        print("---------------------------------------")
        torch.save(Rn.layersObject, output_folder + name_model)
        print("Final model saved as:", name_model)
        if F_sigmoide == 1:
            print(f"Final accuracy = {np.round(ru_acc*1,2)}%")
        else:
            print(f"Final accuracy = {np.round(ru_acc*100.0,2)}%")
        print("---------------------------------------")

        # name2 = 'best_'+name_model
        print("---------------------------------------")
        # torch.save(best_model, name2)
        print("Best model saved as:", name2)
        best_ru_acc = np.asarray(best_acc)
        if F_sigmoide == 1:
            print(f"Best accuracy = {np.round(best_ru_acc*1,2)}%")
        else:
            print(f"Best accuracy = {np.round(best_ru_acc*100.0,2)}% in epoch:{ep}")
        print("---------------------------------------")

        print("---------------------------------------")
        print("-------Entrenamiento terminado---------")
        print("---------------------------------------")
        # Para probar que esta grabando los ultimos valores obtenidos de pesos y sesgos
        # ------------

        # for i in range(len(Rn.layersObject)):

        #    layer = Rn.layersObject[i]
        #    print(type(layer))

        #    print(f'Tamaño de pesos de capa {i}: {layer.weights.shape}')
        #    print(layer.weights)
        # ------------

        # loss_b = np.array([]) # guarada la perdida en cada iteración
        # loss_b2 = np.array([])
        # ac_b = np.array([])
        # ac_b2 = np.array([])

        for i in range(len(Rn.layersObject)):
            self._gradients[i] = 0

        # sys.stdout.flush()

        # --------------------------
        # Esto eliminar solo utilizar en el caso que en backpro se coloque cv y tambien en RedNeuBay en def train
        # if cv==False:
        #    return Rn.layersObject
        # else:
        #    salida = Rn.layersObject # Se realiza esto para que guarde el modelo que tendra como salida y aplicar la funcion para
        #                          que los pesos regresen a un valor aleatorio (esto es para CV) - en utils act_Weig_bias
        #    act_Weig_bias(Rn.layersObject)

        #    return salida
        # -------------------------------

        salid = (
            Rn.layersObject
        )  # Se realiza esto para que guarde el modelo que tendra como salida y aplicar la funcion para
        #                          que los pesos regresen a un valor aleatorio (esto es para CV) - en utils act_Weig_bias
        act_Weig_bias(Rn)
        salida = act_Weig_bias(Rn)

        return salida, ac_b2, loss_b2

    @abstractmethod
    def update_weights(
        self, delta, layer_input, layer_index, curr_weights, Lambda, Bay, rseed
    ):
        """
        Al ser una funcion abstracta se divide para cada clase derivada
        :delta:
        :layer_input: entrada de cada capa
        :layer_index:Indices de capas a ser actualizadas
        :curr_weights: Capa actual de pesos
        :Lambda: Parámetro de regularizacion.
        :regreso: Pesos actualizados
        """
        pass

    def update_bias(
        self, delta, curr_bias, Bay, img, ite_act, total, layer_index, acbay
    ):
        """
        Al ser una funcion abstracta se divide para cada clase derivada
        :delta:
        :curr_bias: current bias of layer.
        :regreso: Sesgos actualizados
        """
        # delta = np.asarray(delta)

        # --------Proceso bayesiano-------------------------------------------------------
        img = self.img
        # np.random.seed()

        # if Bay==True: # Para que haga el bayesiano en cada batch de cada epoch
        if (
            Bay == True and ite_act == (total - 1) and acbay == 1
        ):  # Para que haga el bayesiano en el ultimo batch de cada epoch

            # --------------------------------------------
            # Para que no presente en consola el proceso
            logger = logging.getLogger("pymc3")
            # logger.propagate = False
            logger.setLevel(logging.CRITICAL)
            # --------------------------------------------

            # Analisis Bayesiano----------------------------------------------------------

            delta = delta.data.numpy()

            # Para hacer que el tunning varie por capas mas numerosas
            mues = layer_index  # es el numero de capa
            tunb = 10

            # paraloizado al momento por acbay---------------------------------
            # if img==True:
            #    if mues==1:
            #        tun = 200
            #    elif mues==0:
            #        tun= 100
            #    else:
            #        tun = 300
            # ------------------------------------------------------------------

            with pm.Model() as NNBb:
                # Defino priors

                # sd_b = pm.HalfNormal('sd_b', sigma=1)
                delt_b = pm.Normal("delt_b", mu=delta, sigma=0.01, shape=delta.shape)
                sd_b = pm.Uniform("sd_b", 0, 100)

                # Defino valores en funcion de las priors
                delta_b = pm.Deterministic(
                    "delta_b", (tt.sum(delt_b, axis=0, keepdims=True))
                )
                # --delta_b = tt.sum(delta, axis=0, keepdims=True)

                # Defino likelihood en funcion de los valores establecidos y priors
                obs_pos_b = pm.Normal(
                    "obs_pos_b", mu=delta_b, sigma=sd_b, observed=delta_b
                )
                # obs_pos = pm.Normal('obs_pos', mu=delta_w, sigma=sd,observed=delta_w)
                # obs_pos = pm.Normal('obs_pos', mu=delta_w, sigma=sd,observed=False)

                # Elijo el modelo
                # --start = pm.find_MAP() # esta opcion permite que encuentre los valores iniciales para optimizacion
                # step = pm.NUTS(target_accept=.95)
                # step = pm.HamiltonianMC()
                # --step = pm.NUTS(state=start)
                step = pm.NUTS()
                # Obtengo las muestras posteriores
                # trace = pm.sample(500, step=step, tune=2500, cores=1, chains=1, compute_convergence_checks=True, progressbar=True,  random_seed=42)
                # trace = pm.sample(500, step=step, tune=400, cores=2, chains=2, compute_convergence_checks=False, progressbar=True)
                trace = pm.sample(
                    5,
                    step=step,
                    tune=tunb,
                    cores=4,
                    chains=1,
                    compute_convergence_checks=False,
                    progressbar=True,
                )

            db = trace["delta_b"]
            db = db.mean(axis=0)

            # -------------------------------------------------------------------------------

            db = Variable(torch.FloatTensor(db), requires_grad=True)
            gradient = db  # - Lambda * curr_weights
            # ------------------------------------------------------------------------------

        else:
            gradient = torch.sum(delta, dim=0, keepdim=True)

        # -----------------------------------------------------------

        # gradient = torch.sum(delta, dim=0, keepdim=True)
        new_bias = curr_bias - self.learning_rate * gradient
        pass
        return new_bias

    def calculate_delta(self, derivative, loss):
        """
        Calcula los valores de delta
        :derivative: deriva la capa.
        :param loss: loss de cada capa.
        :regresa: delta
        """
        # print('loss')
        # print(loss)
        # print('derivada')
        # print(derivative)
        # epss = 1e-10
        return torch.mul(derivative, loss)
        # return (torch.mul(derivative, loss)+epss)

    def decay_learning_rate(self, run):
        """
        Perite que la tasa de aprendizaje decaiga dependiendo
        del numero de epochs
        :run:
        :regreso:
        """
        pass

    def set_smooth_gradients(self, Rn):
        """
        Solo se usa en Adam.
        :nn: red neuronal
        :regreso: None
        """
        pass
