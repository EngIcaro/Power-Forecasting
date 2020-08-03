# Author: Ícaro Gabriel Paiva Bastos
# last update: 27 /07/2020
# e-mail: igpb@ic.ufal.br
#%%
import numpy as np
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import  MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.externals import joblib
import json
#%%
# function:       - getting a loss function and plotting learning curve .
#
# Arguments:      - Number of lstm cell in the first layer 
#                 - Number of lstm cell in the second layer 
# Return:         - Null                
layer1 = "[120]"
layer2 = "[0]"
loss     = []
val_loss = []

for j in range(1,9):
    fold= j
    aux_loss = []
    aux_val_loss = []
    
    with open(r'./../saidas/history_lstm'+layer1+layer2+'['+str(fold)+']') as f:
        js = json.load(f)
        aux = js.get("loss")
        aux = aux.split(", ")
        aux_loss.append(float(aux[0].replace("[", "")))
        for i in range(1, len(aux)-1):
            aux_loss.append(float(aux[i]))
        aux_loss.append(float(aux[i+1].replace("]", "")))
            
        aux = js.get("val_loss")
        aux = aux.split(", ")
        aux_val_loss.append(float(aux[0].replace("[", "")))
        for i in range(1, len(aux)-1):
            aux_val_loss.append(float(aux[i]))
        aux_val_loss.append(float(aux[i+1].replace("]", "")))
    loss.append(aux_loss)
    val_loss.append(aux_val_loss)

max = 0
for i in range(len(loss)):
    if max < len(loss[i]):
        max = len(loss[i])

for i in range(len(loss)):
    for j in range(len(loss[i]), max):
        loss[i].append(loss[i][len(loss[i])-1])
        val_loss[i].append(val_loss[i][len(val_loss[i])-1])
#def creat_graph():
mean_loss = []
mean_val_loss = []
for i in range(0,max):
    mean_loss.append(((loss[0][i] + loss[1][i] + loss[2][i] + loss[3][i] + loss[4][i] + loss[5][i] + loss[6][i] + loss[7][i])/8))
    mean_val_loss.append(((val_loss[0][i] + val_loss[1][i] + val_loss[2][i] + val_loss[3][i] + val_loss[4][i] + val_loss[5][i] + val_loss[6][i] + val_loss[7][i])/8))

# [20]   loss:  0.000214753843919419 val_loss:  0.00035803289523708557
# [40]   loss:  0.00018430440494407308 val_loss:  0.0003527437538747795
# [60]   loss:  0.00017096278356158772 val_loss:  0.0003448472319629745 
# [80]   loss:  0.0001761295428544066 val_loss:  0.0003979237774712584
# [100]  loss:  0.0001708861832771089 val_loss:  0.0003666537458593332
# [120]  loss:  0.00016974340823302673 val_loss:  0.0003572943074155811
#%%
all_loss =     []
all_val_loss = []
x = []
all_loss.append(0.000214753843919419); all_loss.append(0.00018430440494407308); all_loss.append(0.00017096278356158772); all_loss.append(0.0001761295428544066);all_loss.append(0.0001708861832771089); all_loss.append(0.00016974340823302673)
all_val_loss.append(0.00035803289523708557); all_val_loss.append(0.0003527437538747795); all_val_loss.append(0.0003448472319629745); all_val_loss.append(0.0003979237774712584); all_val_loss.append(0.0003666537458593332); all_val_loss.append(0.0003572943074155811)
x.append(20); x.append(40); x.append(60); x.append(80); x.append(100); x.append(120)
print("loss: ", mean_loss[-1], "val_loss: ", mean_val_loss[-1])
fig = plt.figure(figsize=(8,6))
ax1=fig.add_subplot(1, 1, 1)
ax1.plot(x,all_loss,'o-',label = 'Erro de treinamento')
ax1.plot(x,all_val_loss, 'o-',label = 'Erro da validação')
plt.legend()
plt.grid()
ax1.set_ylabel("Erro quadrático médio", fontsize = 13)
ax1.set_xlabel("Neurônios na camada LSTM", fontsize = 13)
