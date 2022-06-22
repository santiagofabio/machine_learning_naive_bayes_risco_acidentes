import pandas as pd
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, scale
from yellowbrick.classifier import ConfusionMatrix
from sklearn.metrics import accuracy_score,confusion_matrix ,ConfusionMatrixDisplay, classification_report
from sklearn.preprocessing import StandardScaler
import pickle
from aplica_label_enconder import aplica_label_encoder
from sklearn.naive_bayes import GaussianNB
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from yellowbrick.classifier import ConfusionMatrix
import sys
import os
import seaborn as sns
os.system ('cls')

##Os dados referente as esta base de  dados já foram previamente tratados



dataset = pd.read_excel("base_de_dados.xlsx")


#Imprime os cinco primeiros elementos do cabeçalho
print(f'{dataset.head(5)}')

print(f'Dimensoes da base de dados:  {dataset.shape}')

#Descrição dos dados
print(dataset.describe())


#Mostra  as conlunas com qual vamos trabalhar
print(f'{dataset.columns }')

# Previsores e classe.
x_dataset =dataset.iloc[:,0:6].values
y_dataset =dataset.iloc[:,6].values


#Aplica o LabelEnconder
label_enconder_fumante = LabelEncoder()
label_enconder_bebidas = LabelEncoder()
label_enconder_drogas = LabelEncoder()
x_dataset[:,0] = label_enconder_fumante.fit_transform(x_dataset[:,0])
x_dataset[:,1]= label_enconder_bebidas.fit_transform(x_dataset[:,1])
x_dataset[:,2]= label_enconder_drogas.fit_transform(x_dataset[:,2])

#Aplica Onehotenconder
onehotencoder_dataset = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(),[0,1,2])], remainder='passthrough')
x_dataset =  onehotencoder_dataset.fit_transform(x_dataset)

# Escalona os dados
scale_dataset = StandardScaler()
x_dataset = scale_dataset.fit_transform(x_dataset)


#Divisão enetre treinamento e precisores
x_dataset_treinamento, x_dataset_teste,y_dataset_treinamento, y_dataset_teste = train_test_split(x_dataset, y_dataset, test_size =0.15, random_state= 0)

print(f'x_treinamento{x_dataset_treinamento.shape}')
print(f'x_teste{x_dataset_teste.shape} ')
print(f'y_treinamento{y_dataset_treinamento.shape}')
print(f'y_teste{y_dataset_teste.shape} ')

# Realização do tratamento dos dados categoricos



grafico  = sns.heatmap(dataset.corr(), annot = True, fmt=".1f", linewidths=.6)
plt.show()

#Base treinamento
print(f'Base treinamento')

#base previsora
print(f'Base previsora')
print(f'{x_dataset_teste.shape}')
print(f'{y_dataset_teste.shape} ')

naive_census =GaussianNB()
naive_census.fit(x_dataset_treinamento, y_dataset_treinamento)
previsoes = naive_census.predict(x_dataset_teste)

print(f'{previsoes}')
print(f'{y_dataset_teste}')
precisao = accuracy_score(y_dataset_teste,previsoes)
print(f'Taxa de precisão:  {precisao}')
cm =ConfusionMatrix(naive_census)
cm.fit( x_dataset_treinamento, y_dataset_treinamento, rownames =['Real'], colnames =['predicto'], margins = True)
score_cm = cm.score(x_dataset_teste, y_dataset_teste)
print(f'Score Confusion Matriz {score_cm}')
plt.savefig("matriz_de_confusao.png", dpi =300, format='png') 
cm.show()
