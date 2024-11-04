import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import statsmodels.api as sm

# Introdução
insurance_data = pd.read_csv('datasets/insurance.csv')
insurance_data.head(10)
print(insurance_data.info())
print(insurance_data.describe())


# Matriz de Correlação
# Filtrar apenas as colunas numéricas para gerar a matriz de correlação
insurance_data_numeric = insurance_data.select_dtypes(include=['float64', 'int64'])
# Gerar a matriz de correlação
correlation_matrix = insurance_data_numeric.corr()
# Plotar o heatmap da matriz de correlação
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Matriz de Correlação')
plt.show()


# Boxplot de Outliers
# Boxplot para verificar a presença de outliers em 'charges'
sns.boxplot(insurance_data['charges'])
plt.show()


# Tratamento de Valores Nulos
insurance_data_cleaned = insurance_data.dropna()


# Tratamento de Valores Textuais
insurance_data_encoded = pd.get_dummies(insurance_data_cleaned, columns=['sex', 'smoker', 'region'], drop_first=True)
insurance_data_encoded = insurance_data_encoded.astype(float)
print(insurance_data_encoded.head())


# Estratificação da Amostra com Base em Determinada Variável
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(insurance_data_encoded, insurance_data_encoded["age"]):
    strat_train_set = insurance_data_encoded.loc[train_index]
    strat_test_set = insurance_data_encoded.loc[test_index]


# Separação das Variáveis Independentes e a Variável Dependente
# Para os treinos
# Variáveis independentes (X) e
X_train = strat_train_set.drop("charges", axis=1)
# Variável dependente/target (y)
y_train = strat_train_set["charges"].copy()
# Para os testes
# Variáveis independentes (X) e
X_test = strat_test_set.drop("charges", axis=1)
# Variável dependente/target (y)
y_test = strat_test_set["charges"].copy()


# Padronização dos Dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Modelagem
model = LinearRegression()


# Treino
model.fit(X_train_scaled, y_train)


# Teste
y_pred = model.predict(X_test_scaled)
print(y_pred)


# Gráfico de Dispersão: Valores Reais x Valores Previstos 
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='b')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Valores Reais')
plt.ylabel('Valores Preditos')
plt.title('Valores Reais vs. Valores Preditos')
plt.show()


# Gráfico de Resíduos
residuos = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuos, alpha=0.6, color='purple')
plt.hlines(y=0, xmin=y_pred.min(), xmax=y_pred.max(), color='red', linestyles='dashed')
plt.xlabel('Valores Preditos')
plt.ylabel('Resíduos')
plt.title('Gráfico de Resíduos')
plt.show()


# Erro Quadrático Médio
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error (MSE): {mse}')


# Erro Médio Absoluto
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error (MAE): {mae}')


# R² Score
r2 = r2_score(y_test, y_pred)
print(f'R² Score: {r2}')


# Validações Estatísticas
# Adicionar uma constante para o termo de interceptação
X_train_sm = sm.add_constant(X_train)
X_train_sm.info()


# Ajuste e Resumo estatístico
model_sm = sm.OLS(y_train, X_train_sm).fit()
# Resumo estatístico do modelo
print(model_sm.summary())