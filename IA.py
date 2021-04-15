import numpy as np

from data_prep import features,targets, features_test, targets_test

def sigmoide(x):
    return 1/(1 + np.exp(-x))



#Hyperparametres
n_hidden = 2 # number of units in the layer
epochs = 1000 # numero de iteraciones
alpha = 0.05 #learning rate

m,k = features.shape

#Initialization of weights
entrada_escondida = np.random.normal(scale = 1/k**0.5,
                                      size = (k,n_hidden) 
                                    )
escondida_salida = np.random.normal(scale = 1/k**0.5,
                                    size = n_hidden
                                   )  

#Training  

for e in range(epochs):

    #variables for the gradient
    gradiente_entrada_escondida = np.zeros(entrada_escondida.shape)
    gradiente_escondida_salida = np.zeros(escondida_salida.shape)  

    #iterates over the training set

    for x,y in zip(features.values,targets):
        #Forward pass

        z=sigmoide(np.matmul(x, entrada_escondida))
        y_ =sigmoide(np.matmul(escondida_salida))

        #backward pass
        salida_error = (y - y_) *y_ *(1- y_)

        escondida_error = np.dot(salida_error,escondida_salida) *z * (1 -z)

        gradiente_entrada_escondida += escondida_error * x[:,None]
        gradiente_escondida_salida += salida_error * z



    entrada_escondida +=alpha *gradiente_entrada_escondida / m
    escondida_salida +=alpha * gradiente_escondida_salida / m

    if e % (epochs / 10) == 0:
        z= sigmoide(np.dot(features.value,entrada_escondida))
        y_ = sigmoide(np.dot(z, escondida_salida))

        costo = np.mean((y_ - targets)**2)

        if ult_costo and ult_costo < costo:
            print("Costo de entrenamiento: ", costo,"ADVERTENCIA")
        else:
            print("Costo de entrenamiento: ",costo)

        ult_costo = costo

        #accuracy of test data

        z = sigmoide(np.dot(features_test,entrada_escondida))
        y_ = sigmoide(np.dot(z, escondida_salida))

        predicciones = y_ > 0.5
        precision = np.mean(predicciones == targets_test)
        print("Presicion: {:.3f}".format(precision))

  