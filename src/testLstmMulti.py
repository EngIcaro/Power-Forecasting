# Author: Ícaro Gabriel Paiva Bastos
# last update: 27 /07/2020
# e-mail: igpb@ic.ufal.br
#%%
import numpy as np
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import  MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import dates
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.externals import joblib
import json
import datetime
from sklearn.metrics import mean_squared_error
from pylab import *
from sklearn.metrics import mean_squared_error, mean_absolute_error
#%%
# function:       - forecasting five seconds of the power 
#
# Arguments:      - Null 
#                 - Null 
# Return:         - Null
# Abrindo o arquivo que contém a estrutura da rede
arquivo = open(r'./../saidasmulti/regressor_lstm.json', 'r')
estrutura_rede = arquivo.read()
# Fechando o arquivo
arquivo.close()
# Pegando a estrutura da rede
classificador = model_from_json(estrutura_rede)
# Lendo os pesos salvos e colocando na rede neural
classificador.load_weights(r'./../saidasmulti/pesos_lstm.h5')
#%%
# Lendo o arquivo de entrada de teste
base_teste = pd.read_csv(r'./../kfold/testeInput.csv')
# Pegando todas as variáveis de entrada
base_teste = base_teste.iloc[:,0:5].values
# Carregando o normalizador dos dados de entrada da base de treinamento. 
# Será usada para normalizar nossos dados de entrada do teste
normalizador = joblib.load(r'./../saidas/normTrainingInput.save')
base_teste_normalizada = normalizador.transform(base_teste)

# Carregando o normalizador dos dados de saída da base de treinamento.
# Será utilizado para inverter as previsoes geradas para os valores reais de leitura.
normalizador_output = joblib.load(r'./../saidas/normTrainingOutput.save')

# Lendo a base de saída
base_teste_output = pd.read_csv(r'./../kfold/testeOutput.csv')
#Pegando todos os valores de saídas
base_teste_output_treinamento = base_teste_output.iloc[:,0:1].values
#%%
entradas = []
saidas   = []
# esse for vai criar as matrizes de três dimensões. Dos dados de entrada de teste
# E os valores reais dos dados de saída para ser comparado com os valores previstos pela rede
for i in range(20, (np.size(base_teste_normalizada[:,0]) - 5)):
    # pega os 20 anteriores
    entradas.append(base_teste_normalizada[i-20:i,:])
    # pega o 21 para ser a saída correspondente as 90 leituras anteriores 
    saidas.append(base_teste_output_treinamento[i:i+5,0])
entradas, saidas = np.array(entradas), np.array(saidas)
#%%
# Fazendo a predição dos valores
previsoes = classificador.predict(entradas)
print(previsoes)
previsoes = normalizador_output.inverse_transform(previsoes)
saidas = np.reshape(saidas,(saidas.shape[0], 5))
#%%

# step =  1 MAE =  6.213466809788959
# step =  2 MAE =  6.7602272528973915
# step =  3 MAE =  7.516553516908826
# step =  4 MAE =  8.387587813918728
# step =  5 MAE =  9.231599965038306
mae = []
step = []
for i in range(5):
    mae.append(mean_absolute_error(saidas[:,i], previsoes[:,i]))
    print("step = ", (i+1), "MAE = ", mean_absolute_error(saidas[:,i], previsoes[:,i]))
    step.append(i+1)
plt.plot(step, mae, 'bo')
plt.xticks(range(1, 6))
#%%
import statistics 
dp = []
sub = saidas[:,0] - previsoes[:,0]
for i in range(5):
    dp.append(statistics.stdev(saidas[:,i] - previsoes[:,i]))
    print("step = ", (i+1), "dp = ", statistics.stdev(saidas[:,i] - previsoes[:,i]))
#%%
diff = []
for i in range(len(saidas[:,0])):
   diff.append(abs(previsoes[i,:] - saidas[i,:]))
diff = np.array(diff)
#%%
print(np.percentile(diff[:,0], 75))
print(np.percentile(diff[:,1], 75))
print(np.percentile(diff[:,2], 75))
print(np.percentile(diff[:,3], 75))
print(np.percentile(diff[:,4], 75))

# Create a figure instance
fig = plt.figure(1, figsize=(9, 6))
# Create an axes instanc8
ax = fig.add_subplot(111)
# Remove outliers showfliers=False
bp_dict = ax.boxplot((diff[:,0],diff[:,1], diff[:,2], diff[:,3], diff[:,4]), showfliers=False)
plt.title("Erro absoluto da potência", fontsize = 13)
plt.legend()
plt.ylabel("Watts", fontsize = 13)
array = [np.median(diff[:,0]),np.median(diff[:,1]),np.median(diff[:,2]), np.median(diff[:,3]), np.median(diff[:,4])]
print(array)
aux = 0
for line in bp_dict['medians']:
    # get position data for median line
    value = array[aux]
    x, y = line.get_xydata()[1] # top of median line
    # overlay median value
    text(x, y, '%.3f' % value,
         verticalalignment='center', size=10) # draw above, centered
    aux = aux + 1

ax.set_xticklabels(['Um-segundo','Dois-segundos', 'Três-segundos', 'Quatro-segundos', 'Cinco-segundos'], fontsize = 13)

show()
#%%
t = np.arange(0, len(saidas), 1)
times=np.array([datetime.datetime(2019, 9, 27, int(p/3600), int((p/60)%60), int(p%60)) for p in t])
fmtr = dates.DateFormatter("%H:%M")

fig = plt.figure(figsize=(11,5))
ax1 = fig.add_subplot(1, 1, 1)
ax1.plot(times,saidas[:,4],linestyle='-',color= 'red',label = 'Potência Real', linewidth=1)
ax1.plot(times,previsoes[:,4],linestyle='--', color= 'royalblue', label = 'Potência Prevista', linewidth=1,dashes=(1, 2))
ax1.xaxis.set_major_formatter(fmtr)
ax1.set_ylabel("Watts", fontsize = 14)
ax1.set_xlabel("Hora", fontsize = 14)
plt.grid()
plt.legend()
plt.show()
#%%
plt.scatter(saidas[:,0], previsoes[:,0], s=1)
plt.title("Um segundo", fontsize = 13)
plt.ylabel("Potência Prevista", fontsize = 13)
plt.xlabel("Potência Real",fontsize = 13)
plt.plot(saidas[:,0], saidas[:,0], color = 'r')
