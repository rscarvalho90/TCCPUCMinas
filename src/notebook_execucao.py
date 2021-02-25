from src.DownloadDados import DownloadDados
from src.Util import CorrigeValores
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

dd = DownloadDados()

# Gera um Pandas Dataframe com os dados de arrecadação diária de todos os tributos
pd_arrecad_diaria = DownloadDados.download_arrecadacao_sefaz_rs()

arrecad_diaria = {}

# Gera dataframes de arrecadação diária para cada tributo, sem a correção pela inflação
for tributo in pd_arrecad_diaria['Tributo'].unique():
    arrecad_diaria[tributo] = pd_arrecad_diaria[pd_arrecad_diaria['Tributo'] == tributo].reset_index()
    arrecad_diaria[tributo] = arrecad_diaria[tributo].drop(['index'], axis=1)

# Plota os gráficos das séries temporais dos tributos, sem a correção pela inflação
for tributo in pd_arrecad_diaria['Tributo'].unique():
    sns.lineplot(arrecad_diaria[tributo]['Data'], arrecad_diaria[tributo]['Valor']).set_title(tributo)
    plt.show()

# Corrige os valores pela inflação
igp = dd.download_igp()
pd_arrecad_diaria = CorrigeValores.corrige_inflacao(pd_arrecad_diaria, igp)

# Gera dataframes de arrecadação diária para cada tributo, após a correção pela inflação
for tributo in pd_arrecad_diaria['Tributo'].unique():
    arrecad_diaria[tributo] = pd_arrecad_diaria[pd_arrecad_diaria['Tributo'] == tributo].reset_index()
    arrecad_diaria[tributo] = arrecad_diaria[tributo].drop(['index'], axis=1)

# Plota os gráficos das séries temporais dos tributos, após a correção pela inflação
for tributo in pd_arrecad_diaria['Tributo'].unique():
    sns.lineplot(arrecad_diaria[tributo]['Data'], arrecad_diaria[tributo]['Valor']).set_title(tributo)
    plt.show()

# Cria modelo univariado utilizando o Facebook Prophet
from fbprophet import Prophet
from src.ModelosUtil import ProphetUtil
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer

for tributo in pd_arrecad_diaria['Tributo'].unique():
    # Calcula os valores em termos absolutos
    prophet = Prophet(daily_seasonality=True)
    pd_prophet = ProphetUtil.transforma_dataframe(arrecad_diaria[tributo], ['Data', 'Valor'])
    df_treino, df_teste = ProphetUtil.divide_treino_teste(pd_prophet)
    df_teste.reset_index(drop=True, inplace=True)
    prophet.fit(df_treino)
    predito = prophet.predict(pd.DataFrame(df_teste['ds']))
    rmse = mean_squared_error(pd.DataFrame(df_teste['y']).values, predito['yhat'].values) ** (1 / 2)
    mae = mean_absolute_error(pd.DataFrame(df_teste['y']).values, predito['yhat'].values)
    fig, (sub1) = plt.subplots(1, 1, sharex=True)
    sub1.fill_between(df_teste['ds'], predito['yhat_upper'], predito['yhat_lower'], facecolor='dodgerblue')
    pred, = plt.plot(df_teste['ds'], predito['yhat'], c='blue', label='Predito')
    pred_sup, = plt.plot(df_teste['ds'], predito['yhat_upper'], c='royalblue')
    pred_inf, = plt.plot(df_teste['ds'], predito['yhat_lower'], c='royalblue')
    real = plt.scatter(df_teste['ds'], df_teste['y'], s=3, c='orange')
    plt.legend([pred, pred_sup,  real],
               ['Predito', 'Predito (limites superior e inferior)', 'Real'],
               fontsize=8)
    fig.autofmt_xdate()
    plt.xlabel('Data')
    plt.ylabel('Valor (R$)')
    plt.title(tributo)
    plt.show()

    # Calcula os valores em desvios-padrões
    prophet = Prophet(daily_seasonality=True)
    scaler = StandardScaler()
    pd_prophet = ProphetUtil.transforma_dataframe(arrecad_diaria[tributo], ['Data', 'Valor'])
    df_treino, df_teste = ProphetUtil.divide_treino_teste(pd_prophet)
    df_treino['y'] = scaler.fit_transform(df_treino['y'].values.reshape(-1, 1))
    df_teste['y'] = scaler.transform(df_teste['y'].values.reshape(-1, 1))
    prophet.fit(df_treino)
    predito = prophet.predict(pd.DataFrame(df_teste['ds']))
    rmse_dp = mean_squared_error(pd.DataFrame(df_teste['y']).values, predito['yhat'].values) ** (1 / 2)
    mae_dp = mean_absolute_error(pd.DataFrame(df_teste['y']).values, predito['yhat'].values)

    print('Tributo '+tributo+' - Início DF teste : '+str(df_teste.reset_index().loc[0, 'ds'])+' Fim DF teste : '+str(df_teste.reset_index().loc[len(df_teste)-1, 'ds']))
    print('Para o tributo '+tributo+' o MAE foi de '+str(mae)+' ('+str(mae_dp)+' DP) e o RMSE foi de '+str(rmse)+' ('+str(rmse_dp)+' DP)')

# Cria modelo univariado utilizando LSTM
from src.ModelosUtil import LSTMUtil
from src.ModelosNN import LSTMUnivariada

for tributo in pd_arrecad_diaria['Tributo'].unique():

    # Utiliza o mesmo método do Prophet para tornar os resultados comparáveis
    df_treino, df_teste = ProphetUtil.divide_treino_teste(arrecad_diaria[tributo])
    print('Tributo '+tributo+' - Início DF teste : '+str(df_teste.reset_index().loc[0, 'Data'])+' Fim DF teste : '+str(df_teste.reset_index().loc[len(df_teste)-1, 'Data']))
    df_treino = LSTMUtil.transforma_dataframe(df_treino, 'Data')
    df_teste = LSTMUtil.transforma_dataframe(df_teste, 'Data')

    # Plota a distribuição de probabilidade dos valores da arrecadação diária dos tributos
    sns.distplot(df_treino['Valor']).set_title(tributo)
    plt.show()

    # Plota os boxplots dos valores da arrecadação diária dos tributos
    sns.boxplot(x=df_treino['Valor']).set_title(tributo)
    plt.show()

    # Faz os testes com diversos "scalers" para verificar o com menor erro

    # Standard Scaler
    std_scaler = StandardScaler()
    valor_treino_std = std_scaler.fit_transform(df_treino['Valor'].values.reshape(-1, 1))
    valor_teste_std = std_scaler.transform(df_teste['Valor'].values.reshape(-1, 1))

    model = LSTMUnivariada()
    model.fit()

    # Robust Scaler
    rbt_scaler = RobustScaler()
    valor_treino_rbt = rbt_scaler.fit_transform(df_treino['Valor'].values.reshape(-1, 1))
    valor_teste_rbt = rbt_scaler.transform(df_teste['Valor'].values.reshape(-1, 1))

    # Power Transformer (yeo-johnson)
    pwr_scaler = PowerTransformer()
    valor_treino_rbt = pwr_scaler.fit_transform(df_treino['Valor'].values.reshape(-1, 1))
    valor_teste_rbt = pwr_scaler.transform(df_teste['Valor'].values.reshape(-1, 1))

