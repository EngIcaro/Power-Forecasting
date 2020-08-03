# Author: Ícaro Gabriel Paiva Bastos
# last update: 26/07/2020
# e-mail: igpb@ic.ufal.br
#%% 
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import dates
import datetime
# function:       - Remove negative numbers 
#                 - Create a folder for the K-fold validation 
#                   where inputs and outputs as separated
#
# Arguments:      - the folder path where .csv files are
#                 - list of string telling which input variables
#                 - list of string telling which output variables
#                 - name of files 
# Return:         - Null
def creatDataBase1(path, variablesInput,variablesOutput,testDay, fold):
    all_file = glob.glob(path + "*.csv")
    # Criando uma lista que vai contar todos os csv
    trainingCsv = []
    testCsv = []
    # Pecorrendo todos os arquivos .csv
    for filename in all_file:
        # Pega o arquivo atual
        csv_file = pd.read_csv(filename)
        # Retirando as linhas com valores nulos
        csv_file = csv_file.dropna()
        # Verificando se existe algum valor nulo
        ##print("passando aqui,", csv_file.isnull().any())
        # Vai retirar valores negativos de todas as variáveis e setar para 0
        indexAux = csv_file[(csv_file['TENSAO'] < 0)].index
        print('TEN', indexAux)
        csv_file['TENSAO'][indexAux] = 0
    
        indexAux = csv_file[(csv_file['CORRENTE'] < 0)].index
        print('CORR', indexAux)
        csv_file['CORRENTE'][indexAux] = 0
        
        indexAux = csv_file[(csv_file['TEMPERATURA'] < 0)].index
        print('TEMP', indexAux)
        csv_file['TEMPERATURA'][indexAux] = 0
    
        indexAux = csv_file[(csv_file['PRESSAO'] < 0)].index  
        print('PRES', indexAux)
        csv_file['PRESSAO'][indexAux] = 0 

        indexAux = csv_file[(csv_file['IRRADIANCIA'] < 0)].index
        print('IRRA', indexAux)
        csv_file['IRRADIANCIA'][indexAux] = 0 
    
        indexAux = csv_file[(csv_file['TEMP_PAINEL'] < 0)].index 
        print('TEMP_PAI', indexAux)
        csv_file['TEMP_PAINEL'][indexAux] = 0
    
        indexAux = csv_file[(csv_file['VELOCIDADE'] < 0)].index 
        print('VELO', indexAux)
        csv_file['VELOCIDADE'][indexAux] = 0
        
        indexAux = csv_file[(csv_file['UMIDADE'] < 0)].index 
        print('UMIDA', indexAux)
        csv_file['UMIDADE'][indexAux] = 0
        # Criando a coluna potência
        csv_file['POTENCIA'] = csv_file['TENSAO']*csv_file['CORRENTE']
        # Verificando se o csv atual faz parte do testDay
        # Se fizer coloco o csv tratado em uma fila
        aux = 1
        for i in(testDay):
            day = path + i
            if(day == filename):
                testCsv.append(csv_file)
                aux = 0
        # Se não coloco o csv tratado em outra fila
        if(aux):
            trainingCsv.append(csv_file)
    # Junto todos os csv de treino em um único csv
    frame = pd.concat(trainingCsv, axis=0, ignore_index = True)
    # Salvo o arquivo das variáveis de entrada da base de treinamento
    frame.to_csv((r'./../kfold/trainingInputFold'+fold+'.csv'),columns = variablesInput ,index = False)
    # Salvo o arquivo das variáveis de saída
    frame.to_csv((r'./../kfold/trainingOutputFold'+fold+'.csv'),columns = variablesOutput ,index = False)
    
    # Junto todos os csv de teste em um único csv
    frame = pd.concat(testCsv, axis=0, ignore_index = True)
    # Salvo o arquivo das variáveis de entrada da base de treinamento
    frame.to_csv((r'./../kfold/testeInputFold'+fold+'.csv'),columns = variablesInput ,index = False)
    # Salvo o arquivo das variáveis de saída
    frame.to_csv((r'./../kfold/testeOutputFold'+fold+'.csv'),columns = variablesOutput ,index = False)
    return 0
        
creatDataBase1(r'./../dados/', ["TEMPERATURA", "TEMP_PAINEL", "IRRADIANCIA", "UMIDADE", "VELOCIDADE"], ["POTENCIA"],["3_9.csv" , "4_9.csv"],  '[1]')
creatDataBase1(r'./../dados/', ["TEMPERATURA", "TEMP_PAINEL", "IRRADIANCIA", "UMIDADE", "VELOCIDADE"], ["POTENCIA"],["4_10.csv", "5_9.csv" , "6_8.csv"],  '[2]')
creatDataBase1(r'./../dados/', ["TEMPERATURA", "TEMP_PAINEL", "IRRADIANCIA", "UMIDADE", "VELOCIDADE"], ["POTENCIA"],["7_8.csv" , "7_9.csv" , "8_9.csv"],  '[3]')
creatDataBase1(r'./../dados/', ["TEMPERATURA", "TEMP_PAINEL", "IRRADIANCIA", "UMIDADE", "VELOCIDADE"], ["POTENCIA"],["10_8.csv", "12_9.csv", "13_9.csv"], '[4]')
creatDataBase1(r'./../dados/', ["TEMPERATURA", "TEMP_PAINEL", "IRRADIANCIA", "UMIDADE", "VELOCIDADE"], ["POTENCIA"],["14_9.csv", "15_9.csv", "16_9.csv"], '[5]')
creatDataBase1(r'./../dados/', ["TEMPERATURA", "TEMP_PAINEL", "IRRADIANCIA", "UMIDADE", "VELOCIDADE"], ["POTENCIA"],["17_9.csv", "18_9.csv", "21_8.csv"], '[6]')
creatDataBase1(r'./../dados/', ["TEMPERATURA", "TEMP_PAINEL", "IRRADIANCIA", "UMIDADE", "VELOCIDADE"], ["POTENCIA"],["21_9.csv", "22_8.csv", "27_9.csv"], '[7]')
creatDataBase1(r'./../dados/', ["TEMPERATURA", "TEMP_PAINEL", "IRRADIANCIA", "UMIDADE", "VELOCIDADE"], ["POTENCIA"],["28_9.csv", "29_9.csv", "30_9.csv"], '[8]')
#%%
# function      : - Remove negative numbers 
#                 - Creates folders for the test and training data sets 
#                   where inputs and outputs as separated
#
# Arguments:      - the folder path where .csv files are
#                 - list of string telling which input variables
#                 - list of string telling which output variables
#                 - name of files 
# Return:         - Null
def creatDataBase1(path, variablesInput,variablesOutput,testDay):
    all_file = glob.glob(path + "*.csv")
    # Criando uma lista que vai contar todos os csv
    trainingCsv = []
    testCsv = []
    # Pecorrendo todos os arquivos .csv
    for filename in all_file:
        # Pega o arquivo atual
        csv_file = pd.read_csv(filename)
        # Retirando as linhas com valores nulos
        csv_file = csv_file.dropna()
        # Verificando se existe algum valor nulo
        ##print("passando aqui,", csv_file.isnull().any())
        # Vai retirar valores negativos de todas as variáveis e setar para 0
        indexAux = csv_file[(csv_file['TENSAO'] < 0)].index
        csv_file['TENSAO'][indexAux] = 0
    
        indexAux = csv_file[(csv_file['CORRENTE'] < 0)].index
        csv_file['CORRENTE'][indexAux] = 0
        
        indexAux = csv_file[(csv_file['TEMPERATURA'] < 0)].index
        csv_file['TEMPERATURA'][indexAux] = 0
    
        indexAux = csv_file[(csv_file['PRESSAO'] < 0)].index  
        csv_file['PRESSAO'][indexAux] = 0 

        indexAux = csv_file[(csv_file['IRRADIANCIA'] < 0)].index
        csv_file['IRRADIANCIA'][indexAux] = 0 
    
        indexAux = csv_file[(csv_file['TEMP_PAINEL'] < 0)].index 
        csv_file['TEMP_PAINEL'][indexAux] = 0
    
        indexAux = csv_file[(csv_file['VELOCIDADE'] < 0)].index 
        csv_file['VELOCIDADE'][indexAux] = 0
        
        indexAux = csv_file[(csv_file['UMIDADE'] < 0)].index 
        csv_file['UMIDADE'][indexAux] = 0
        # Criando a coluna potência
        csv_file['POTENCIA'] = csv_file['TENSAO']*csv_file['CORRENTE']
        # Verificando se o csv atual faz parte do testDay
        # Se fizer coloco o csv tratado em uma fila
        aux = 1
        for i in(testDay):
            day = path + i
            if(day == filename):
                testCsv.append(csv_file)
                aux = 0
        # Se não coloco o csv tratado em outra fila
        if(aux):
            trainingCsv.append(csv_file)
    # Junto todos os csv de treino em um único csv
    frame = pd.concat(trainingCsv, axis=0, ignore_index = True)
    # Salvo o arquivo das variáveis de entrada da base de treinamento
    frame.to_csv((r'./../kfold/trainingInput.csv'),columns = variablesInput ,index = False)
    # Salvo o arquivo das variáveis de saída
    frame.to_csv((r'./../kfold/trainingOutput.csv'),columns = variablesOutput ,index = False)
    
    # Junto todos os csv de teste em um único csv
    frame = pd.concat(testCsv, axis=0, ignore_index = True)
    # Salvo o arquivo das variáveis de entrada da base de treinamento
    frame.to_csv((r'./../kfold/testeInput.csv'),columns = variablesInput ,index = False)
    # Salvo o arquivo das variáveis de saída
    frame.to_csv((r'./../kfold/testeOutput.csv'),columns = variablesOutput ,index = False)
    return 0
        
creatDataBase1(r'./../dados/', ["TEMPERATURA", "TEMP_PAINEL", "IRRADIANCIA", "UMIDADE", "VELOCIDADE"], ["POTENCIA"],["1_10.csv"])
#%%
# function      : - Creates a pearson correlation matrix 
# Arguments:      - Null 
# Return:         - Null
def heatmap():
    base = pd.read_csv(r'./../dados/1_10.csv')
    base = base.drop('DIRECAO', 1)
    base = base.drop('CHUVA', 1)
    base['POTENCIA'] = base['TENSAO']*base['CORRENTE']
    base = base.drop('TENSAO', 1)
    base = base.drop('CORRENTE', 1)
    base.columns = ['DATA', 'TEMP_AMBIENTE', 'PRESSÃO', 'UMIDADE', 'IRRADIÂNCIA', 'VELOCIDADE', 'TEMP_PAINEL', 'POTÊNCIA']
    plt.figure(figsize=(8, 6)) 
    mask = np.zeros_like(base.corr())
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(base.corr(),
                annot = True,
                fmt = '.2f',
                mask=mask,
                cmap=sns.diverging_palette(0, 250, as_cmap=True))
    plt.title('Correlação entre Variáveis', fontsize = 12)
    plt.show()
    
heatmap()

#%%
# function      : - plot a voltage graph  
# Arguments:      - Null 
# Return:         - Null
base = pd.read_csv(r'./../dados/1_10.csv')
val = base.iloc[:,0:1].values
t = np.arange(0, len(val), 1)
times=np.array([datetime.datetime(2019, 9, 27, int(p/3600), int((p/60)%60), int(p%60)) for p in t])
fmtr = dates.DateFormatter("%H:%M")

fig = plt.figure(figsize=(7,3))
ax1=fig.add_subplot(1, 1, 1)
ax1.plot(times, base['TENSAO'])
ax1.set_ylabel("Volts", fontsize = 12)
ax1.xaxis.set_major_formatter(fmtr)
ax1.set_xlabel("Hora", fontsize = 12)
plt.title("Tensão", fontsize = 12)
plt.grid()
#%%
# function      : - plot a current graph  
# Arguments:      - Null 
# Return:         - Null
fig = plt.figure(figsize=(7,3))
ax2=fig.add_subplot(1, 1, 1)
ax2.plot(times, base['CORRENTE'])
ax2.set_ylabel("Ampere", fontsize = 12)
ax2.xaxis.set_major_formatter(fmtr)
ax2.set_xlabel("Hora", fontsize = 12)
plt.title("Corrente", fontsize = 12)
plt.grid()
#%%
# function      : - plot a power graph  
# Arguments:      - Null 
# Return:         - Null
fig = plt.figure(figsize=(7,3))
ax3=fig.add_subplot(1, 1, 1)
ax3.plot(times, (base['TENSAO']*base['CORRENTE']))
ax3.set_ylabel("Watts", fontsize = 12)
ax3.xaxis.set_major_formatter(fmtr)
ax3.set_xlabel("Hora", fontsize = 12)
plt.title("Potência", fontsize = 12)
plt.grid()
#%%
# function      : - plot a power, voltage and current graph  
# Arguments:      - Null 
# Return:         - Null
fig = plt.figure(figsize=(8,5))
ax1=fig.add_subplot(3, 1, 1)
ax1.plot(times, base['TENSAO'])
ax1.set_ylabel("Volts", fontsize = 12)
ax1.xaxis.set_major_formatter(fmtr)
plt.title("Tensão", fontsize = 12)
plt.grid()
ax2=fig.add_subplot(3, 1, 2)
ax2.plot(times, base['CORRENTE'])
ax2.set_ylabel("Ampere", fontsize = 12)
ax2.xaxis.set_major_formatter(fmtr)
plt.title("Corrente", fontsize = 12)
plt.grid()
ax3=fig.add_subplot(3, 1, 3)
ax3.plot(times, (base['TENSAO']*base['CORRENTE']))
ax3.set_ylabel("Watts", fontsize = 12)
ax3.xaxis.set_major_formatter(fmtr)
ax3.set_xlabel("Hora", fontsize = 12)
plt.title("Potência", fontsize = 12)
plt.grid()