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
# function:       - training neural networking with cross-validation, k-fold.
#
# Arguments:      - Number of lstm cell in the first layer 
#                 - Number of lstm cell in the second layer 
# Return:         - Null
layer1 = "[60]"
layer2 = "[0]"
for j in range(1,9):
    fold = j
    base_input = pd.read_csv(r'./../kfold/trainingInputFold['+str(fold)+'].csv')
    # pegando todas as colunas
    base_input_treinamento = base_input.iloc[:,0:5].values
    # normalizando os valores
    normalizador = MinMaxScaler(feature_range=(0,1))
    base_input_normalizada = normalizador.fit(base_input_treinamento)
    base_input_normalizada = normalizador.transform(base_input_treinamento)
    #salvando o normalizador
    joblib.dump(normalizador, r'./../saidasmulti/normTrainingInput'+layer1+layer2+'['+str(fold)+'].save')
    # Lendo a base de saída
    base_output = pd.read_csv(r'./../kfold/trainingOutputFold['+str(fold)+'].csv')
    # Pegando todas as colunas
    base_output_treinamento = base_output.iloc[:,0:1].values
    # normalizando os valores
    normalizador_output = MinMaxScaler(feature_range=(0,1))
    base_output_normalizada = normalizador_output.fit(base_output_treinamento)
    base_output_normalizada = normalizador_output.transform(base_output_treinamento)
    joblib.dump(normalizador_output, r'./../saidasmulti/normTrainingOutput'+layer1+layer2+'['+str(fold)+'].save')
    
    
    base_teste = pd.read_csv(r'./../kfold/testeInputFold'+'['+str(fold)+'].csv') 
    base_teste = base_teste.iloc[:,0:5].values  
    base_teste_normalizada = normalizador.transform(base_teste)
    base_teste_output = pd.read_csv(r'./../kfold/testeOutputFold'+'['+str(fold)+'].csv')
    base_teste_output_treinamento = base_teste_output.iloc[:,0:1].values
    base_teste_output_treinamento_norm = normalizador_output.transform(base_teste_output_treinamento)
    
    entradas_teste = []
    saidas_teste   = []
    for i in range(20, (np.size(base_teste_normalizada[:,0])-5)):
        # pega os 20 anteriores
        entradas_teste.append(base_teste_normalizada[i-20:i,:])
        # pega o 21 para ser a saída correspondente as 90 leituras anteriores 
        saidas_teste.append(base_teste_output_treinamento_norm[i:i+5,0])

    entradas_teste, saidas_teste = np.array(entradas_teste), np.array(saidas_teste)
    
    
    
    # vamos gerar os numpy array. o primeiro vai conter as 20 leituras das cinco variáveis  
    # de dimensão base_input_normalizada. e o segundo vai conter a próxima leitura da variável potência
    previsores = []
    valor_real = []

    for i in range(20, (np.size(base_input_normalizada[:,0])-5)):
        # pega 20 anteriores
        previsores.append(base_input_normalizada[i-20:i,:])
        # pega as 21:25 para ser a saída correspondente as 20 leituras anteriores 
        valor_real.append(base_output_normalizada[i:i+5,0])
    
    # É necessários transformar em numpy arrays pois a rede so aceita esse tipo de entrada
    previsores, valor_real = np.array(previsores), np.array(valor_real)
    # A rede vai ser do tipo sequencial. alimentada para frente
    regressor = Sequential()
    # Vamos acrescentar uma primeira camada LSTM com 30 camadas "enroladas" na camada escondida
    # o parâmetro return_sequences significa que ele vai passar o resultado para frente  para as próximas camadas
    # no input_shape dizemos como é a nossa entrada. temos seis entradas atrasadas ou amostrada em 20 segundos
    # return_sequences retornam a saída do estado oculto para cada etapa do tempo de entrada.
    regressor.add(LSTM(units = 60, input_shape = (previsores.shape[1],5)))
    # Vamos criar a camada de saída 
    regressor.add(Dense(units = 5, activation = 'linear'))
    # Vamos compilar a rede
    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['mean_absolute_error'])
    # função early stop vai para de treinar a rede se algum parâmetro monitorado parou de melhorar
    es = EarlyStopping(monitor ='loss', min_delta = 1e-10, patience = 10, verbose = 1)
    # ele vai reduzir a taxa de aprendizagem quando uma metrica parou de melhorar
    rlr = ReduceLROnPlateau(monitor = 'loss', factor = 0.2, patience = 5, verbose = 1)
    mcp =  ModelCheckpoint(filepath=r'./../saidasmulti/pesos_lstm'+layer1+layer2+'['+str(fold)+'].h5', monitor = 'loss', save_best_only= True)
    history = regressor.fit(x = previsores,
                            y= valor_real, 
                            validation_data = (entradas_teste, saidas_teste),
                            epochs = 100, 
                            batch_size = 500, 
                            callbacks = [es,rlr,mcp])
    regressor_json = regressor.to_json()
    hist = {'loss': str(history.history['loss']),
            'val_loss': str(history.history['val_loss']),
            'mae': str(history.history['mean_absolute_error']),
            'val_mae': str(history.history['val_mean_absolute_error'])
            }
    j_hist = json.dumps(hist)
    with open(r'./../saidasmulti/history_lstm'+layer1+layer2+'['+str(fold)+']', 'w') as json_file:
        json_file.write(j_hist)
    #j_hist = json.dumps(hist)
    with open(r'./../saidasmulti/regressor_lstm'+layer1+layer2+'['+str(fold)+'].json', 'w') as json_file:
        json_file.write(regressor_json)
    
#%%
# function:       - getting a loss function and plotting learning curve .
#
# Arguments:      - Number of lstm cell in the first layer 
#                 - Number of lstm cell in the second layer 
# Return:         - Null                
layer1 = "[60]"
layer2 = "[0]"
loss     = []
val_loss = []

for j in range(1,9):
    fold= j
    aux_loss = []
    aux_val_loss = []
    
    with open(r'./../saidasmulti/history_lstm'+layer1+layer2+'['+str(fold)+']') as f:
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


#def creat_graph():
mean_loss = []
mean_val_loss = []
for i in range(54):
    mean_loss.append(((loss[0][i] + loss[1][i] + loss[2][i] + loss[3][i] + loss[4][i] + loss[5][i] + loss[6][i] + loss[7][i])/8))
    mean_val_loss.append(((val_loss[0][i] + val_loss[1][i] + val_loss[2][i] + val_loss[3][i] + val_loss[4][i] + val_loss[5][i] + val_loss[6][i] + val_loss[7][i])/8))

fig = plt.figure(figsize=(8,6))
ax1=fig.add_subplot(1, 1, 1)
ax1.plot(mean_loss, label = "Treinamento")
ax1.plot(mean_val_loss, label = "Validação")
ax1.set_ylabel("Média do Erro Quadrático Médio", fontsize = 13)
ax1.set_xlabel("Épocas", fontsize = 13)
plt.legend()
#%%
# function:       - 
# Arguments:      - 
# Return:         - 
def saveHist(path,history):
    new_hist = {}
    for key in list(history.history.keys()):
        if type(history.history[key]) == np.ndarray:
            new_hist[key] == history.history[key].tolist()
        elif type(history.history[key]) == list:
           if  type(history.history[key][0]) == np.float64:
               new_hist[key] = list(map(float, history.history[key]))

    print(new_hist)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(new_hist, f, separators=(',', ':'), sort_keys=True, indent=4)
