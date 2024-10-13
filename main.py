# Importando as bibliotecas necessárias
import pandas as pd

# Carregar o dataset
insurance_data = pd.read_csv('C:/Users/ventu/Downloads/archive/insurance.csv')

# Verificar os primeiros registros para entender a estrutura
#print(insurance_data.head())

# Verificar se há valores ausentes
missing_values = insurance_data.isnull().sum()
#print("Valores ausentes por coluna:")
#print(missing_values)

# One-Hot Encoding para as colunas categóricas
insurance_data_encoded = pd.get_dummies(insurance_data, columns=['sex', 'smoker', 'region'], drop_first=True)

# Exibir as primeiras linhas do dataset transformado
print(insurance_data_encoded.head(5))

# Variável target => charges
# 
