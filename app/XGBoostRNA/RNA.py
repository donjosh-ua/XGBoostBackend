import torchvision.datasets as dset
import torchvision
import torchvision.transforms as transformsb
from redneuronal_bay.RedNeuBay import RedNeuBay
from redneuronal_bay.Layers.layers import *
from redneuronal_bay.preprocesamiento import *
from redneuronal_bay.metricas_eva import *
from redneuronal_bay.funcion_activacion import *

import pandas as pd
import numpy as np

filename = "XGBoostRNA/pima-indians-diabetes.data.csv"
names = ["preg", "plas", "pres", "skin", "test", "mass", "pedi", "age", "class"]
df_cla = pd.read_csv(filename, names=names)  # Base de datos tipo data frame

# Red 1

Red_Bay = RedNeuBay(
    alpha=0.001,
    epoch=20,
    criteria="cross_entropy",
    optimizer="SGD",
    image_size=None,
    verbose=True,
    decay=0.0,
    momentum=0.9,
    image=False,
    FA_ext=None,
    Bay=False,
    save_mod="ModiR",
    pred_hot=True,
    test_size=0.2,
    batch_size=64,
    cv=True,
    Kfold=5,
)

Red_Bay.add(Tanh_Layer(8, 13))  # Capa de entrada
Red_Bay.add(Tanh_Layer(13, 8))  # Capa oculta
# Red_Bay.add(SoftmaxBay_Layer(10,2))     #Capa final bayesiana
Red_Bay.add(Softmax_Layer(8, 2))
# Red_Bay.add(Sigmoid_Layer(8,1))  # Capa final
# Si deseara aplicar una funcion exttra ala salida de las capas por ejemplo una softmax - colocar en funcion
print(Red_Bay)
# out = Red_Bay.train(df_cla=df_cla) #Sin cross validacion
out = Red_Bay.cv_train(df_cla=df_cla)  # Con cross validacion
out

# torch.manual_seed(123) #fijamos la semilla
transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,)),
    ]
)

root = "./XGBoostRNA/data"
train_set = dset.MNIST(root=root, train=True, transform=transforms, download=False)
test_set = dset.MNIST(root=root, train=False, transform=transforms)

# Red 1    image_size=784 porque imagen es de 28x28

# Modelo  Tanh y Softmax
# ------------------------SIN BAYESIANO VALIDACION 50 epochs--------------------------------
Red_Bay = RedNeuBay(
    alpha=0.001,
    epoch=50,
    criteria="cross_entropy",
    optimizer="SGD",
    image_size=784,
    verbose=True,
    decay=0.0,
    momentum=0.9,
    image=True,
    FA_ext=None,
    Bay=False,
    save_mod="Img_ori2",
    pred_hot=True,
    test_size=None,
    batch_size=64,
    cv=False,
    Kfold=5,
)

Red_Bay.add(Tanh_Layer(784, 1000))  # Capa de entrada
Red_Bay.add(Tanh_Layer(1000, 50))  # Capa oculta
# Red_Bay.add(SoftmaxBay_Layer(50,10))     #Capa final
Red_Bay.add(Softmax_Layer(50, 10))

print(Red_Bay)
out = Red_Bay.train(train_set=train_set, test_set=test_set)
out
