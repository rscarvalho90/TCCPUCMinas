from src.DownloadDados import DownloadDados
from src.Util import CorrigeValores
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

dd = DownloadDados()

# Gera um Pandas Dataframe com os dados de arrecadação diária de todos os tributos
pd_arrecad_diaria = DownloadDados.download_arrecadacao_sefaz_rs()

arrecad_diaria = {}
pd.set_option('display.float_format', lambda x: '%.5f' % x)

# Gera dataframes de arrecadação diária para cada tributo, sem a correção pela inflação
for tributo in pd_arrecad_diaria['Tributo'].unique():
    arrecad_diaria[tributo] = pd_arrecad_diaria[pd_arrecad_diaria['Tributo'] == tributo].reset_index()
    arrecad_diaria[tributo] = arrecad_diaria[tributo].drop(['index'], axis=1)
    
for tributo in pd_arrecad_diaria['Tributo'].unique():
    print(tributo)
    print(arrecad_diaria[tributo]['Valor'].describe())

# Plota os gráficos das séries temporais dos tributos, sem a correção pela inflação
for tributo in pd_arrecad_diaria['Tributo'].unique():
    sns.scatterplot(arrecad_diaria[tributo]['Data'], arrecad_diaria[tributo]['Valor'], size=3, legend=False).set_title(
        tributo)
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
    '''# Plota os scatterplots arrecadação diária dos tributos
    sns.scatterplot(arrecad_diaria[tributo]['Data'], arrecad_diaria[tributo]['Valor'], size=3, legend=False).set_title(tributo)
    plt.show()
    # Plota os boxplots arrecadação diária dos tributos
    sns.boxplot(x=arrecad_diaria[tributo]['Valor']).set_title(tributo)
    plt.show()'''
    # Plota a distribuição de probabilidade dos valores da arrecadação diária dos tributos
    sns.distplot(arrecad_diaria[tributo]['Valor']).set_title(tributo)
    plt.show()

# Imprime a descrição dos dados    
for tributo in pd_arrecad_diaria['Tributo'].unique():
    print(tributo)
    print(arrecad_diaria[tributo]['Valor'].describe())

# Cria modelo com única variável quantitativa utilizando o Facebook Prophet
from fbprophet import Prophet
from src.ModelosUtil import ProphetUtil
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer, LabelEncoder

pd_datas_testes = pd.DataFrame(columns=['Inicio', 'Fim'])
pd_datas_treinos = pd.DataFrame(columns=['Inicio', 'Fim'])
pd_performance = pd.DataFrame(columns=['MAE', 'RMSE'])
pd_stats = pd.DataFrame(columns=['dp'])

# Predição com a remoção de 'outliers'
for tributo in pd_arrecad_diaria['Tributo'].unique():
    # Calcula os valores em termos absolutos
    prophet = Prophet(daily_seasonality=True)    
     
    pd_prophet = ProphetUtil.transforma_dataframe(arrecad_diaria[tributo], ['Data', 'Valor'])
    df_treino, df_teste = ProphetUtil.divide_treino_teste(pd_prophet)
    
    # Remove os 'outliers' do dataframe de treino
    primeiro_quartil = df_treino['y'].quantile(.25)
    terceiro_quartil = df_treino['y'].quantile(.75)
    desvio_quartil = (terceiro_quartil-primeiro_quartil)/2
    df_treino = df_treino[(df_treino['y']<terceiro_quartil+1.5*desvio_quartil) & (df_treino['y']>primeiro_quartil-1.5*desvio_quartil)]
   
    df_teste.reset_index(drop=True, inplace=True)
    prophet.fit(df_treino)
    predito = prophet.predict(pd.DataFrame(df_teste['ds']))
    
    # Grava os erros
    rmse = mean_squared_error(pd.DataFrame(df_teste['y']).values, predito['yhat'].values) ** (1 / 2)
    mae = mean_absolute_error(pd.DataFrame(df_teste['y']).values, predito['yhat'].values)
    
    # Plota as predições
    fig, (sub1) = plt.subplots(1, 1, sharex=True)
    sub1.fill_between(df_teste['ds'], predito['yhat_upper'], predito['yhat_lower'], facecolor='dodgerblue')
    pred, = plt.plot(df_teste['ds'], predito['yhat'], c='blue', label='Predito')
    pred_sup, = plt.plot(df_teste['ds'], predito['yhat_upper'], c='royalblue')
    pred_inf, = plt.plot(df_teste['ds'], predito['yhat_lower'], c='royalblue')
    real = plt.scatter(df_teste['ds'], df_teste['y'], s=3, c='orange')
    plt.legend([pred, pred_sup, real],
               ['Predito', 'Predito (limites superior e inferior)', 'Real'],
               fontsize=8)
    fig.autofmt_xdate()
    plt.xlabel('Data')
    plt.ylabel('Valor (R$)')
    plt.title(tributo)
    plt.show()
    
    pd_datas_testes.loc[tributo+' - Prophet - Univariável - Com Remoção de Outliers', 'Inicio'] = df_teste.reset_index().loc[0, 'ds']
    pd_datas_testes.loc[tributo+' - Prophet - Univariável - Com Remoção de Outliers', 'Fim'] = df_teste.reset_index().loc[len(df_teste) - 1, 'ds']
    pd_datas_treinos.loc[tributo+' - Prophet - Univariável - Com Remoção de Outliers', 'Inicio'] = df_treino.reset_index().loc[0, 'ds']
    pd_datas_treinos.loc[tributo+' - Prophet - Univariável - Com Remoção de Outliers', 'Fim'] = df_treino.reset_index().loc[len(df_treino) - 1, 'ds']
    pd_performance.loc[tributo+' - Prophet - Univariável - Com Remoção de Outliers', 'MAE'] = mae
    pd_performance.loc[tributo+' - Prophet - Univariável - Com Remoção de Outliers', 'RMSE'] = rmse
    pd_stats.loc[tributo+' - DP', 'dp'] = df_teste['y'].std()
    
    # Plota os componentes    
    prophet.plot_components(predito)

    print('Tributo ' + tributo + ' - Início DF teste : ' + str(
        df_teste.reset_index().loc[0, 'ds']) + ' Fim DF teste : ' + str(
        df_teste.reset_index().loc[len(df_teste) - 1, 'ds']))
    print(
        'Para o tributo ' + tributo + ' o MAE foi de ' + str(mae) + '  e o RMSE foi de ' + str(
            rmse))
    
# Predição sem a remoção de 'outliers'
for tributo in pd_arrecad_diaria['Tributo'].unique():
    # Calcula os valores em termos absolutos
    prophet = Prophet(daily_seasonality=True)    
     
    pd_prophet = ProphetUtil.transforma_dataframe(arrecad_diaria[tributo], ['Data', 'Valor'])
    df_treino, df_teste = ProphetUtil.divide_treino_teste(pd_prophet)    
    
    df_teste.reset_index(drop=True, inplace=True)
    prophet.fit(df_treino)
    predito = prophet.predict(pd.DataFrame(df_teste['ds']))
    
    # Grava os erros
    rmse = mean_squared_error(pd.DataFrame(df_teste['y']).values, predito['yhat'].values) ** (1 / 2)
    mae = mean_absolute_error(pd.DataFrame(df_teste['y']).values, predito['yhat'].values)
    
    # Plota as predições
    fig, (sub1) = plt.subplots(1, 1, sharex=True)
    sub1.fill_between(df_teste['ds'], predito['yhat_upper'], predito['yhat_lower'], facecolor='dodgerblue')
    pred, = plt.plot(df_teste['ds'], predito['yhat'], c='blue', label='Predito')
    pred_sup, = plt.plot(df_teste['ds'], predito['yhat_upper'], c='royalblue')
    pred_inf, = plt.plot(df_teste['ds'], predito['yhat_lower'], c='royalblue')
    real = plt.scatter(df_teste['ds'], df_teste['y'], s=3, c='orange')
    plt.legend([pred, pred_sup, real],
               ['Predito', 'Predito (limites superior e inferior)', 'Real'],
               fontsize=8)
    fig.autofmt_xdate()
    plt.xlabel('Data')
    plt.ylabel('Valor (R$)')
    plt.title(tributo)
    plt.show()
    
    pd_datas_testes.loc[tributo+' - Prophet - Univariável - Sem Remoção de Outliers', 'Inicio'] = df_teste.reset_index().loc[0, 'ds']
    pd_datas_testes.loc[tributo+' - Prophet - Univariável - Sem Remoção de Outliers', 'Fim'] = df_teste.reset_index().loc[len(df_teste) - 1, 'ds']
    pd_datas_treinos.loc[tributo+' - Prophet - Univariável - Sem Remoção de Outliers', 'Inicio'] = df_treino.reset_index().loc[0, 'ds']
    pd_datas_treinos.loc[tributo+' - Prophet - Univariável - Sem Remoção de Outliers', 'Fim'] = df_treino.reset_index().loc[len(df_treino) - 1, 'ds']
    pd_performance.loc[tributo+' - Prophet - Univariável - Sem Remoção de Outliers', 'MAE'] = mae
    pd_performance.loc[tributo+' - Prophet - Univariável - Sem Remoção de Outliers', 'RMSE'] = rmse
    pd_stats.loc[tributo+' - DP', 'dp'] = df_teste['y'].std()
    
    # Plota os componentes    
    prophet.plot_components(predito)

    print('Tributo ' + tributo + ' - Início DF teste : ' + str(
        df_teste.reset_index().loc[0, 'ds']) + ' Fim DF teste : ' + str(
        df_teste.reset_index().loc[len(df_teste) - 1, 'ds']))
    print(
        'Para o tributo ' + tributo + ' o MAE foi de ' + str(mae) + '  e o RMSE foi de ' + str(
            rmse))

# Cria modelo com única variável quantitativa utilizando LSTM
from src.ModelosUtil import LSTMUtil
from src.ModelosNN import LSTMUnivariada
import tensorflow.keras.optimizers as ko
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

comparativo = pd.DataFrame(columns=['StandardScaler', 'RobustScaler', 'PowerTransformer'])

for tributo in pd_arrecad_diaria['Tributo'].unique():
    # Utiliza o mesmo método do Prophet para tornar os resultados comparáveis
    df_treino, df_teste = ProphetUtil.divide_treino_teste(arrecad_diaria[tributo])
    print('Tributo ' + tributo + ' - Início DF teste : ' + str(
        df_teste.reset_index().loc[0, 'Data']) + ' Fim DF teste : ' + str(
        df_teste.reset_index().loc[len(df_teste) - 1, 'Data']))
    df_treino = LSTMUtil.transforma_dataframe(df_treino, 'Data')
    df_teste = LSTMUtil.transforma_dataframe(df_teste, 'Data')   

    # Faz o Label Encoder do dia, dia da semana e mês (apesar de dia e mês serem numéricos, o Label Encoder inicia a contagem em 0 ao invés de 1)
    encoder_dia = LabelEncoder()
    dia_treino_enc = encoder_dia.fit_transform(df_treino['Dia'].values)
    dia_teste_enc = encoder_dia.transform(df_teste['Dia'].values)
    encoder_mes = LabelEncoder()
    mes_treino_enc = encoder_mes.fit_transform(df_treino['Mes'].values)    
    mes_teste_enc = encoder_mes.transform(df_teste['Mes'].values)
    encoder_dia_semana = LabelEncoder()
    dia_semana_treino_enc = encoder_dia_semana.fit_transform(df_treino['Dia_Semana'].values)    
    dia_semana_teste_enc = encoder_dia_semana.transform(df_teste['Dia_Semana'].values)
    encoder_dia_semana = LabelEncoder()
    dia_semana_treino_enc = encoder_dia_semana.fit_transform(df_treino['Dia_Semana'].values)    
    dia_semana_teste_enc = encoder_dia_semana.transform(df_teste['Dia_Semana'].values)
    
    np_dia_mes_treino = np.concatenate((dia_treino_enc.reshape(-1, 1), mes_treino_enc.reshape(-1, 1), dia_semana_treino_enc.reshape(-1, 1)), axis=1)[5:]
    np_dia_mes_teste = np.concatenate((dia_teste_enc.reshape(-1, 1), mes_teste_enc.reshape(-1, 1), dia_semana_teste_enc.reshape(-1, 1)), axis=1)[5:]
    
    # Faz os testes com diversos "scalers" para verificar o com menor erro   

    # Standard Scaler
    std_scaler = StandardScaler()
    valor_treino_std = std_scaler.fit_transform(df_treino['Valor'].values.reshape(-1, 1))
    valor_teste_std = std_scaler.transform(df_teste['Valor'].values.reshape(-1, 1))
    
    # A saída (label) é a arrecadação do dia seguinte ao último dia da sequência
    saida_treino = valor_treino_std[5:]
    saida_teste = valor_teste_std[5:]

    valor_arrecadacao_serie_temporal_lstm_treino = LSTMUtil.cria_intervalos_temporais(valor_treino_std)
    valor_arrecadacao_serie_temporal_lstm_teste = LSTMUtil.cria_intervalos_temporais(valor_teste_std)

    model = LSTMUnivariada(df_treino)
    checkpoint = ModelCheckpoint('checkpoint_regressor_'+tributo+'_teste_standard_scaler.hdf5', monitor='loss', verbose=2,
                                save_best_only=True, save_weights_only=False,
                                mode='auto', period=1)
    model.compile(optimizer=ko.Adam(lr=0.1), loss='mse')
    model.fit([np_dia_mes_treino, valor_arrecadacao_serie_temporal_lstm_treino], saida_treino, validation_data=([np_dia_mes_teste, valor_arrecadacao_serie_temporal_lstm_teste], saida_teste), 
              epochs=100, batch_size=50, callbacks=[checkpoint])
    
    # Carrega o melhor modelo salvo pelo Checkpoint
    model.load_weights('checkpoint_regressor_'+tributo+'_teste_standard_scaler.hdf5')
    
    std_pred = model.predict([np_dia_mes_teste, valor_arrecadacao_serie_temporal_lstm_teste])    
    mae_std = mean_absolute_error(std_scaler.inverse_transform(saida_teste), std_scaler.inverse_transform(std_pred))
    print('O MAE para o tributo '+tributo+' usando o "Standard Scaler" foi de '+str(mae_std))
    
    comparativo.loc[tributo, 'StandardScaler'] = mae_std

    # Robust Scaler
    rbt_scaler = RobustScaler()
    valor_treino_rbt = rbt_scaler.fit_transform(df_treino['Valor'].values.reshape(-1, 1))
    valor_teste_rbt = rbt_scaler.transform(df_teste['Valor'].values.reshape(-1, 1))
    
    # A saída (label) é a arrecadação do dia seguinte ao último dia da sequência
    saida_treino = valor_treino_rbt[5:]
    saida_teste = valor_teste_rbt[5:]

    valor_arrecadacao_serie_temporal_lstm_treino = LSTMUtil.cria_intervalos_temporais(valor_treino_rbt)
    valor_arrecadacao_serie_temporal_lstm_teste = LSTMUtil.cria_intervalos_temporais(valor_teste_rbt)

    model = LSTMUnivariada(df_treino)
    checkpoint = ModelCheckpoint('checkpoint_regressor_'+tributo+'_teste_robust_scaler.hdf5', monitor='loss', verbose=2,
                                save_best_only=True, save_weights_only=False,
                                mode='auto', period=1)
    model.compile(optimizer=ko.Adam(lr=0.1), loss='mse')
    model.fit([np_dia_mes_treino, valor_arrecadacao_serie_temporal_lstm_treino], saida_treino, validation_data=([np_dia_mes_teste, valor_arrecadacao_serie_temporal_lstm_teste], saida_teste), 
              epochs=100, batch_size=50, callbacks=[checkpoint])
    
    # Carrega o melhor modelo salvo pelo Checkpoint
    model.load_weights('checkpoint_regressor_'+tributo+'_teste_robust_scaler.hdf5')
    
    rbt_pred = model.predict([np_dia_mes_teste, valor_arrecadacao_serie_temporal_lstm_teste])    
    mae_rbt = mean_absolute_error(rbt_scaler.inverse_transform(saida_teste), rbt_scaler.inverse_transform(rbt_pred))
    print('O MAE para o tributo '+tributo+' usando o "Robust Scaler" foi de '+str(mae_rbt))
    
    comparativo.loc[tributo, 'RobustScaler'] = mae_rbt

    # Power Transformer (yeo-johnson)
    pwr_scaler = PowerTransformer()
    valor_treino_pwr = pwr_scaler.fit_transform(df_treino['Valor'].values.reshape(-1, 1))
    valor_teste_pwr = pwr_scaler.transform(df_teste['Valor'].values.reshape(-1, 1))
    
    # A saída (label) é a arrecadação do dia seguinte ao último dia da sequência
    saida_treino = valor_treino_pwr[5:]
    saida_teste = valor_teste_pwr[5:]

    valor_arrecadacao_serie_temporal_lstm_treino = LSTMUtil.cria_intervalos_temporais(valor_treino_pwr)
    valor_arrecadacao_serie_temporal_lstm_teste = LSTMUtil.cria_intervalos_temporais(valor_teste_pwr)

    model = LSTMUnivariada(df_treino)
    checkpoint = ModelCheckpoint('checkpoint_regressor_'+tributo+'_teste_power_transformer.hdf5', monitor='loss', verbose=2,
                                save_best_only=True, save_weights_only=False,
                                mode='auto', period=1)
    model.compile(optimizer=ko.Adam(lr=0.1), loss='mse')
    model.fit([np_dia_mes_treino, valor_arrecadacao_serie_temporal_lstm_treino], saida_treino, validation_data=([np_dia_mes_teste, valor_arrecadacao_serie_temporal_lstm_teste], saida_teste), 
              epochs=100, batch_size=50, callbacks=[checkpoint])
    
    # Carrega o melhor modelo salvo pelo Checkpoint
    model.load_weights('checkpoint_regressor_'+tributo+'_teste_power_transformer.hdf5')
    
    pwr_pred = model.predict([np_dia_mes_teste, valor_arrecadacao_serie_temporal_lstm_teste])    
    mae_pwr = mean_absolute_error(pwr_scaler.inverse_transform(saida_teste), pwr_scaler.inverse_transform(pwr_pred))
    print('O MAE para o tributo '+tributo+' usando o "Power Transformer" foi de '+str(mae_pwr))
    
    comparativo.loc[tributo, 'PowerTransformer'] = mae_pwr

# Treina a rede neural LSTM com única variável quantitativa utilizando o Power Transformer como scaler, já que foi o de melhor desempenho
for tributo in pd_arrecad_diaria['Tributo'].unique():
    # Utiliza método que extrai o dataset de teste idêntico ao utilizado no Prophet
    df_treino, df_teste = LSTMUtil.gera_teste_identico_prophet(arrecad_diaria[tributo], pd_datas_testes.loc[tributo+' - Prophet - Univariável - Sem Remoção de Outliers', 'Inicio'], pd_datas_testes.loc[tributo+' - Prophet - Univariável - Sem Remoção de Outliers', 'Fim'])   
    
    print('Tributo ' + tributo + ' - Início DF teste : ' + str(
        df_teste.reset_index().loc[0, 'Data']) + ' Fim DF teste : ' + str(
        df_teste.reset_index().loc[len(df_teste) - 1, 'Data']))
    df_treino = LSTMUtil.transforma_dataframe(df_treino, 'Data')
    df_teste = LSTMUtil.transforma_dataframe(df_teste, 'Data')

    # Faz o Label Encoder do dia e mês (apesar de dia e mês serem numéricos, o Label Encoder inicia a contagem em 0 ao invés de 1)
    encoder_dia = LabelEncoder()
    dia_treino_enc = encoder_dia.fit_transform(df_treino['Dia'].values)
    dia_teste_enc = encoder_dia.transform(df_teste['Dia'].values)
    encoder_mes = LabelEncoder()
    mes_treino_enc = encoder_mes.fit_transform(df_treino['Mes'].values)
    mes_teste_enc = encoder_mes.transform(df_teste['Mes'].values)
    encoder_dia_semana = LabelEncoder()
    dia_semana_treino_enc = encoder_dia_semana.fit_transform(df_treino['Dia_Semana'].values)    
    dia_semana_teste_enc = encoder_dia_semana.transform(df_teste['Dia_Semana'].values)

    # Concatena os valores para servirem de input para o modelo
    np_dia_mes_treino = np.concatenate((dia_treino_enc.reshape(-1, 1), mes_treino_enc.reshape(-1, 1), dia_semana_treino_enc.reshape(-1, 1)), axis=1)[5:]
    np_dia_mes_teste = np.concatenate((dia_teste_enc.reshape(-1, 1), mes_teste_enc.reshape(-1, 1), dia_semana_teste_enc.reshape(-1, 1)), axis=1)[5:]

    # Power Transformer (yeo-johnson)
    pwr_scaler = PowerTransformer()
    valor_treino_pwr = pwr_scaler.fit_transform(df_treino['Valor'].values.reshape(-1, 1))
    valor_teste_pwr = pwr_scaler.transform(df_teste['Valor'].values.reshape(-1, 1))

    saida_treino = valor_treino_pwr[5:]
    saida_teste = valor_teste_pwr[5:]

    # Cria intervalos temporais com 3 dimensões para servirem de input à rede LSTM
    valor_arrecadacao_serie_temporal_lstm_treino = LSTMUtil.cria_intervalos_temporais(valor_treino_pwr)
    valor_arrecadacao_serie_temporal_lstm_teste = LSTMUtil.cria_intervalos_temporais(valor_teste_pwr)

    checkpoint = ModelCheckpoint('checkpoint_regressor_'+tributo+'_univariado.hdf5', monitor='loss', verbose=2,
                                save_best_only=True, save_weights_only=False,
                                mode='auto', period=1)
    
    early_stopping = EarlyStopping(monitor='loss', min_delta=0.001, patience=50)
    
    %load_ext tensorboard
    %tensorboard --logdir logs/scalars
    logdir = "logs/scalars/"
    tensorboard_callback = TensorBoard(log_dir=logdir, profile_batch = 100000000) 

    model = LSTMUnivariada(df_treino)
    model.compile(optimizer=ko.Adam(lr=0.1), loss='mse')  
    model.fit([np_dia_mes_treino, valor_arrecadacao_serie_temporal_lstm_treino], saida_treino, validation_data=([np_dia_mes_teste, valor_arrecadacao_serie_temporal_lstm_teste], saida_teste),
              epochs=1000, batch_size=50, callbacks=[checkpoint, tensorboard_callback, early_stopping])
    print(model.summary())
    
    # Carrega o melhor modelo salvo pelo Checkpoint
    model.load_weights('checkpoint_regressor_'+tributo+'_univariado.hdf5')

    # Avalia a predição do test set
    pwr_pred = model.predict([np_dia_mes_teste, valor_arrecadacao_serie_temporal_lstm_teste])
    rmse = mean_squared_error(pwr_scaler.inverse_transform(saida_teste), pwr_scaler.inverse_transform(pwr_pred)) ** (1 / 2)
    mae = mean_absolute_error(pwr_scaler.inverse_transform(saida_teste), pwr_scaler.inverse_transform(pwr_pred))
    
    # Preenche dataframe com os valores para comparação com os resultados do Prophet
    pd_performance.loc[tributo+' - LSTM - Univariável', 'MAE'] = mae
    pd_performance.loc[tributo+' - LSTM - Univariável', 'RMSE'] = rmse
    pd_datas_testes.loc[tributo+' - LSTM - Univariável', 'Inicio'] = df_teste.reset_index().loc[5, 'Data']
    pd_datas_testes.loc[tributo+' - LSTM - Univariável', 'Fim'] = df_teste.reset_index().loc[len(df_teste)-1, 'Data']
    pd_datas_treinos.loc[tributo+' - LSTM - Univariável', 'Inicio'] = df_treino.reset_index().loc[5, 'Data']
    pd_datas_treinos.loc[tributo+' - LSTM - Univariável', 'Fim'] = df_treino.reset_index().loc[len(df_treino)-1, 'Data']

    # Plota as predições em comparação com os valores reais
    fig, (sub1) = plt.subplots(1, 1, sharex=True)
    pred, = plt.plot(df_teste[5:]['Data'], pwr_scaler.inverse_transform(pwr_pred), c='blue', label='Predito')
    real = plt.scatter(df_teste[5:]['Data'], df_teste[5:]['Valor'], s=3, c='orange')
    plt.legend([pred, real],
               ['Predito', 'Real'],
               fontsize=8)
    fig.autofmt_xdate()
    plt.xlabel('Data')
    plt.ylabel('Valor (R$)')
    plt.title(tributo)
    plt.show()
    
# Realiza a comparação com modelos com múltiplas variáveis quantitativas com o Prophet e LSTM, para o ICMS
pd_prophet_icms = ProphetUtil.transforma_dataframe(pd_arrecad_diaria[pd_arrecad_diaria['Tributo']=='ICMS'], ['Data', 'Valor'])

# Baixa os dados do PIB trimestral do Rio Grande do Sul (em valores atualizados) 
pd_pib_rs_trimestral = DownloadDados.download_pib_rs()
pd_pib_br_mensal = DownloadDados.download_pib_br()
pd_pib_br_mensal = CorrigeValores.corrige_inflacao_pib(pd_pib_br_mensal, igp)
pd_emprego_adm_dem_mensal = DownloadDados.download_dados_emprego()

# Adiciona os dados do PIB do RS(trimestre anterior) ao dataframe
pd_prophet_icms = ProphetUtil.adiciona_pib_rs(pd_prophet_icms, pd_pib_rs_trimestral)
# Adiciona os dados do PIB do Brasil(estimativa mês anterior) ao dataframe
pd_prophet_icms = ProphetUtil.adiciona_pib_br(pd_prophet_icms, pd_pib_br_mensal)
# Adiciona os dados de emprego do Brasil(mês anterior) ao dataframe
pd_prophet_icms = ProphetUtil.adiciona_dados_emprego(pd_prophet_icms, pd_emprego_adm_dem_mensal)

# Cria modelo com múltiplas variáveis quantitativas utilizando o Facebook Prophet
prophet = Prophet(daily_seasonality=True)
pd_prophet = pd_prophet_icms
df_treino, df_teste = ProphetUtil.divide_treino_teste(pd_prophet)
df_teste.reset_index(drop=True, inplace=True)
prophet.add_regressor('ADMISSOES_MES_ANTERIOR')
prophet.add_regressor('DEMISSOES_MES_ANTERIOR')
prophet.add_regressor('PIB_BR_MES_ANTERIOR')
prophet.add_regressor('PIB_RS_TRIMESTRE_ANTERIOR')
prophet.fit(df_treino)
predito = prophet.predict(pd.DataFrame(df_teste[['ds', 'ADMISSOES_MES_ANTERIOR', 'DEMISSOES_MES_ANTERIOR', 'PIB_BR_MES_ANTERIOR', 'PIB_RS_TRIMESTRE_ANTERIOR']]))

# Grava os erros
rmse = mean_squared_error(pd.DataFrame(df_teste['y']).values, predito['yhat'].values) ** (1 / 2)
mae = mean_absolute_error(pd.DataFrame(df_teste['y']).values, predito['yhat'].values)

# Plota as predições
fig, (sub1) = plt.subplots(1, 1, sharex=True)
sub1.fill_between(df_teste['ds'], predito['yhat_upper'], predito['yhat_lower'], facecolor='dodgerblue')
pred, = plt.plot(df_teste['ds'], predito['yhat'], c='blue', label='Predito')
pred_sup, = plt.plot(df_teste['ds'], predito['yhat_upper'], c='royalblue')
pred_inf, = plt.plot(df_teste['ds'], predito['yhat_lower'], c='royalblue')
real = plt.scatter(df_teste['ds'], df_teste['y'], s=3, c='orange')
plt.legend([pred, pred_sup, real],
           ['Predito', 'Predito (limites superior e inferior)', 'Real'],
           fontsize=8)
fig.autofmt_xdate()
plt.xlabel('Data')
plt.ylabel('Valor (R$)')
plt.title('ICMS - Prophet - Múltiplas Variáveis Quantitativas')
plt.show()

pd_datas_testes.loc['ICMS - Prophet - Multivariável', 'Inicio'] = df_teste.reset_index().loc[0, 'ds']
pd_datas_testes.loc['ICMS - Prophet - Multivariável', 'Fim'] = df_teste.reset_index().loc[len(df_teste) - 1, 'ds']
pd_datas_treinos.loc['ICMS - Prophet - Multivariável', 'Inicio'] = df_treino.reset_index().loc[0, 'ds']
pd_datas_treinos.loc['ICMS - Prophet - Multivariável', 'Fim'] = df_treino.reset_index().loc[len(df_treino) - 1, 'ds']
pd_performance.loc['ICMS - Prophet - Multivariável', 'MAE'] = mae
pd_performance.loc['ICMS - Prophet - Multivariável', 'RMSE'] = rmse

# Treina a rede neural LSTM com múltiplas variáveis quantitativas utilizando o Power Transformer como scaler, já que foi o de melhor desempenho
from src.ModelosNN import LSTMMultivariada

# Utiliza método que extrai o dataset de teste idêntico ao utilizado no Prophet
df_treino, df_teste = LSTMUtil.gera_teste_identico_prophet_multivariado(pd_prophet_icms, pd_datas_testes.loc['ICMS - Prophet - Multivariável', 'Inicio'], pd_datas_testes.loc['ICMS - Prophet - Multivariável', 'Fim'])   

df_treino = LSTMUtil.transforma_dataframe(df_treino, 'ds')
df_teste = LSTMUtil.transforma_dataframe(df_teste, 'ds')

# Faz o Label Encoder do dia e mês (apesar de dia e mês serem numéricos, o Label Encoder inicia a contagem em 0 ao invés de 1)
encoder_dia = LabelEncoder()
dia_treino_enc = encoder_dia.fit_transform(df_treino['Dia'].values)
dia_teste_enc = encoder_dia.transform(df_teste['Dia'].values)
encoder_mes = LabelEncoder()
mes_treino_enc = encoder_mes.fit_transform(df_treino['Mes'].values)
mes_teste_enc = encoder_mes.transform(df_teste['Mes'].values)
encoder_dia_semana = LabelEncoder()
dia_semana_treino_enc = encoder_dia_semana.fit_transform(df_treino['Dia_Semana'].values)    
dia_semana_teste_enc = encoder_dia_semana.transform(df_teste['Dia_Semana'].values)

# Faz a transformação Power Transformer (yeo-johnson) para dados do PIB trimestral do RS
pwr_scaler_pib_rs = PowerTransformer()
pib_rs_treino_pwr = pwr_scaler_pib_rs.fit_transform(df_treino['PIB_RS_TRIMESTRE_ANTERIOR'].values.reshape(-1, 1))
pib_rs_teste_pwr = pwr_scaler_pib_rs.transform(df_teste['PIB_RS_TRIMESTRE_ANTERIOR'].values.reshape(-1, 1))

# Faz a transformação Power Transformer (yeo-johnson) para dados do PIB mensal do Brasil
pwr_scaler_pib_br = PowerTransformer()
pib_br_treino_pwr = pwr_scaler_pib_br.fit_transform(df_treino['PIB_BR_MES_ANTERIOR'].values.reshape(-1, 1))
pib_br_teste_pwr = pwr_scaler_pib_br.transform(df_teste['PIB_BR_MES_ANTERIOR'].values.reshape(-1, 1))

# Faz a transformação Power Transformer (yeo-johnson) para dados de admissões
pwr_scaler_admissoes = PowerTransformer()
admissoes_treino_pwr =pwr_scaler_admissoes.fit_transform(df_treino['ADMISSOES_MES_ANTERIOR'].values.reshape(-1, 1))
admissoes_teste_pwr = pwr_scaler_admissoes.transform(df_teste['ADMISSOES_MES_ANTERIOR'].values.reshape(-1, 1))

# Faz a transformação Power Transformer (yeo-johnson) para dados de demissõess
pwr_scaler_demissoes = PowerTransformer()
demissoes_treino_pwr =pwr_scaler_demissoes.fit_transform(df_treino['DEMISSOES_MES_ANTERIOR'].values.reshape(-1, 1))
demissoes_teste_pwr = pwr_scaler_demissoes.transform(df_teste['DEMISSOES_MES_ANTERIOR'].values.reshape(-1, 1))

# Concatena os valores para servirem de input para o modelo
np_dia_mes_pib_emprego_treino = np.concatenate((dia_treino_enc.reshape(-1, 1), mes_treino_enc.reshape(-1, 1), dia_semana_treino_enc.reshape(-1, 1), pib_rs_treino_pwr.reshape(-1, 1), pib_br_treino_pwr.reshape(-1, 1), admissoes_treino_pwr.reshape(-1, 1), demissoes_treino_pwr.reshape(-1, 1)), axis=1)[5:]
np_dia_mes_pib_emprego_teste = np.concatenate((dia_teste_enc.reshape(-1, 1), mes_teste_enc.reshape(-1, 1), dia_semana_teste_enc.reshape(-1, 1), pib_rs_teste_pwr.reshape(-1, 1), pib_br_teste_pwr.reshape(-1, 1), admissoes_teste_pwr.reshape(-1, 1), demissoes_teste_pwr.reshape(-1, 1)), axis=1)[5:]

# Faz a transformação Power Transformer (yeo-johnson) para dados da arrecadação
pwr_scaler_arrecad = PowerTransformer()
arrecad_treino_pwr = pwr_scaler_arrecad.fit_transform(df_treino['y'].values.reshape(-1, 1))
arrecad_teste_pwr = pwr_scaler_arrecad.transform(df_teste['y'].values.reshape(-1, 1))

saida_treino = arrecad_treino_pwr[5:]
saida_teste = arrecad_teste_pwr[5:]

# Cria intervalos temporais com 3 dimensões para servirem de input à rede LSTM
valor_arrecadacao_serie_temporal_lstm_treino = LSTMUtil.cria_intervalos_temporais(arrecad_treino_pwr)
valor_arrecadacao_serie_temporal_lstm_teste = LSTMUtil.cria_intervalos_temporais(arrecad_teste_pwr)

checkpoint = ModelCheckpoint('checkpoint_regressor_ICMS_multivariado.hdf5', monitor='loss', verbose=2,
                            save_best_only=True, save_weights_only=False,
                            mode='auto', period=1)

early_stopping = EarlyStopping(monitor='loss', min_delta=0.001, patience=50)

%load_ext tensorboard
%tensorboard --logdir logs/scalars
logdir = "logs/scalars/"
tensorboard_callback = TensorBoard(log_dir=logdir, profile_batch = 100000000)

model = LSTMMultivariada(df_treino)
model.compile(optimizer=ko.Adam(lr=0.1), loss='mse')
model.fit([np_dia_mes_pib_emprego_treino, valor_arrecadacao_serie_temporal_lstm_treino], saida_treino, validation_data=([np_dia_mes_pib_emprego_teste, valor_arrecadacao_serie_temporal_lstm_teste], saida_teste),
          epochs=1000, batch_size=50, callbacks=[checkpoint, tensorboard_callback, early_stopping])

# Carrega o melhor modelo salvo pelo Checkpoint
model.load_weights('checkpoint_regressor_ICMS_multivariado.hdf5')

# Avalia a predição do test set
pwr_pred = model.predict([np_dia_mes_pib_emprego_teste, valor_arrecadacao_serie_temporal_lstm_teste])
rmse = mean_squared_error(pwr_scaler_arrecad.inverse_transform(saida_teste), pwr_scaler_arrecad.inverse_transform(pwr_pred)) ** (1 / 2)
mae = mean_absolute_error(pwr_scaler_arrecad.inverse_transform(saida_teste), pwr_scaler_arrecad.inverse_transform(pwr_pred))

# Preenche dataframe com os valores para comparação com os resultados do Prophet
pd_performance.loc['ICMS - LSTM - Multivariável', 'MAE'] = mae
pd_performance.loc['ICMS - LSTM - Multivariável', 'RMSE'] = rmse
pd_datas_testes.loc['ICMS - LSTM - Multivariável', 'Inicio'] = df_teste.reset_index().loc[5, 'ds']
pd_datas_testes.loc['ICMS - LSTM - Multivariável', 'Fim'] = df_teste.reset_index().loc[len(df_teste)-1, 'ds']
pd_datas_treinos.loc['ICMS - LSTM - Multivariável', 'Inicio'] = df_treino.reset_index().loc[5, 'ds']
pd_datas_treinos.loc['ICMS - LSTM - Multivariável', 'Fim'] = df_treino.reset_index().loc[len(df_treino)-1, 'ds']

# Plota as predições em comparação com os valores reais
fig, (sub1) = plt.subplots(1, 1, sharex=True)
pred, = plt.plot(df_teste[5:]['ds'], pwr_scaler_arrecad.inverse_transform(pwr_pred), c='blue', label='Predito')
real = plt.scatter(df_teste[5:]['ds'], df_teste[5:]['y'], s=3, c='orange')
plt.legend([pred, real],
           ['Predito', 'Real'],
           fontsize=8)
fig.autofmt_xdate()
plt.xlabel('Data')
plt.ylabel('Valor (R$)')
plt.title('ICMS')
plt.show()

''' Para fins de comparação, executa treino e teste nos modelos com única variável quantitativa
nos mesmos períodos com mútliplas variáveis quantitativas.'''

# Modelo do Prophet

prophet = Prophet(daily_seasonality=True)
pd_prophet = pd_prophet_icms
df_treino, df_teste = ProphetUtil.divide_treino_teste(pd_prophet)
df_teste.reset_index(drop=True, inplace=True)
prophet.fit(df_treino)
predito = prophet.predict(pd.DataFrame(df_teste['ds']))

# Grava os erros
rmse = mean_squared_error(pd.DataFrame(df_teste['y']).values, predito['yhat'].values) ** (1 / 2)
mae = mean_absolute_error(pd.DataFrame(df_teste['y']).values, predito['yhat'].values)

# Plota as predições
fig, (sub1) = plt.subplots(1, 1, sharex=True)
sub1.fill_between(df_teste['ds'], predito['yhat_upper'], predito['yhat_lower'], facecolor='dodgerblue')
pred, = plt.plot(df_teste['ds'], predito['yhat'], c='blue', label='Predito')
pred_sup, = plt.plot(df_teste['ds'], predito['yhat_upper'], c='royalblue')
pred_inf, = plt.plot(df_teste['ds'], predito['yhat_lower'], c='royalblue')
real = plt.scatter(df_teste['ds'], df_teste['y'], s=3, c='orange')
plt.legend([pred, pred_sup, real],
           ['Predito', 'Predito (limites superior e inferior)', 'Real'],
           fontsize=8)
fig.autofmt_xdate()
plt.xlabel('Data')
plt.ylabel('Valor (R$)')
plt.title('ICMS - Prophet - Única Variável Quantitativa - Datas Idênticas ao Modelo com Múltiplas Variáveis Quantitativas')
plt.show()

pd_datas_testes.loc['ICMS - Prophet - Univariável (2)', 'Inicio'] = df_teste.reset_index().loc[0, 'ds']
pd_datas_testes.loc['ICMS - Prophet - Univariável (2)', 'Fim'] = df_teste.reset_index().loc[len(df_teste) - 1, 'ds']
pd_datas_treinos.loc['ICMS - Prophet - Univariável (2)', 'Inicio'] = df_treino.reset_index().loc[0, 'ds']
pd_datas_treinos.loc['ICMS - Prophet - Univariável (2)', 'Fim'] = df_treino.reset_index().loc[len(df_treino) - 1, 'ds']
pd_performance.loc['ICMS - Prophet - Univariável (2)', 'MAE'] = mae
pd_performance.loc['ICMS - Prophet - Univariável (2)', 'RMSE'] = rmse

# Modelo LSTM

# Utiliza método que extrai o dataset de teste idêntico ao utilizado no Prophet
df_treino, df_teste = LSTMUtil.gera_teste_identico_prophet(arrecad_diaria['ICMS'], pd_datas_testes.loc['ICMS - Prophet - Univariável (2)', 'Inicio'], pd_datas_testes.loc['ICMS - Prophet - Univariável (2)', 'Fim'])

df_treino = LSTMUtil.transforma_dataframe(df_treino, 'Data')
df_teste = LSTMUtil.transforma_dataframe(df_teste, 'Data')

# Faz o Label Encoder do dia e mês (apesar de dia e mês serem numéricos, o Label Encoder inicia a contagem em 0 ao invés de 1)
encoder_dia = LabelEncoder()
dia_treino_enc = encoder_dia.fit_transform(df_treino['Dia'].values)
dia_teste_enc = encoder_dia.transform(df_teste['Dia'].values)
encoder_mes = LabelEncoder()
mes_treino_enc = encoder_mes.fit_transform(df_treino['Mes'].values)
mes_teste_enc = encoder_mes.transform(df_teste['Mes'].values)
encoder_dia_semana = LabelEncoder()
dia_semana_treino_enc = encoder_dia_semana.fit_transform(df_treino['Dia_Semana'].values)    
dia_semana_teste_enc = encoder_dia_semana.transform(df_teste['Dia_Semana'].values)

# Concatena os valores para servirem de input para o modelo
np_dia_mes_treino = np.concatenate((dia_treino_enc.reshape(-1, 1), mes_treino_enc.reshape(-1, 1), dia_semana_treino_enc.reshape(-1, 1)), axis=1)[5:]
np_dia_mes_teste = np.concatenate((dia_teste_enc.reshape(-1, 1), mes_teste_enc.reshape(-1, 1), dia_semana_teste_enc.reshape(-1, 1)), axis=1)[5:]

# Power Transformer (yeo-johnson)
pwr_scaler = PowerTransformer()
valor_treino_pwr = pwr_scaler.fit_transform(df_treino['Valor'].values.reshape(-1, 1))
valor_teste_pwr = pwr_scaler.transform(df_teste['Valor'].values.reshape(-1, 1))

saida_treino = valor_treino_pwr[5:]
saida_teste = valor_teste_pwr[5:]

# Cria intervalos temporais com 3 dimensões para servirem de input à rede LSTM
valor_arrecadacao_serie_temporal_lstm_treino = LSTMUtil.cria_intervalos_temporais(valor_treino_pwr)
valor_arrecadacao_serie_temporal_lstm_teste = LSTMUtil.cria_intervalos_temporais(valor_teste_pwr)

checkpoint = ModelCheckpoint('checkpoint_regressor_'+tributo+'_univariado(2).hdf5', monitor='loss', verbose=2,
                            save_best_only=True, save_weights_only=False,
                            mode='auto', period=1)

early_stopping = EarlyStopping(monitor='loss', min_delta=0.001, patience=50)

%load_ext tensorboard
%tensorboard --logdir logs/scalars
logdir = "logs/scalars/"
tensorboard_callback = TensorBoard(log_dir=logdir, profile_batch = 100000000) 

model = LSTMUnivariada(df_treino)
model.compile(optimizer=ko.Adam(lr=0.1), loss='mse')
model.fit([np_dia_mes_treino, valor_arrecadacao_serie_temporal_lstm_treino], saida_treino, validation_data=([np_dia_mes_teste, valor_arrecadacao_serie_temporal_lstm_teste], saida_teste),
          epochs=1000, batch_size=50, callbacks=[checkpoint, tensorboard_callback, early_stopping])

# Carrega o melhor modelo salvo pelo Checkpoint
model.load_weights('checkpoint_regressor_'+tributo+'_univariado(2).hdf5')

# Avalia a predição do test set
pwr_pred = model.predict([np_dia_mes_teste, valor_arrecadacao_serie_temporal_lstm_teste])
rmse = mean_squared_error(pwr_scaler.inverse_transform(saida_teste), pwr_scaler.inverse_transform(pwr_pred)) ** (1 / 2)
mae = mean_absolute_error(pwr_scaler.inverse_transform(saida_teste), pwr_scaler.inverse_transform(pwr_pred))

# Preenche dataframe com os valores para comparação com os resultados do Prophet
pd_performance.loc['ICMS - LSTM - Univariável (2)', 'MAE'] = mae
pd_performance.loc['ICMS - LSTM - Univariável (2)', 'RMSE'] = rmse
pd_datas_testes.loc['ICMS - LSTM - Univariável (2)', 'Inicio'] = df_teste.reset_index().loc[5, 'Data']
pd_datas_testes.loc['ICMS - LSTM - Univariável (2)', 'Fim'] = df_teste.reset_index().loc[len(df_teste)-1, 'Data']
pd_datas_treinos.loc['ICMS - LSTM - Univariável (2)', 'Inicio'] = df_treino.reset_index().loc[5, 'Data']
pd_datas_treinos.loc['ICMS - LSTM - Univariável (2)', 'Fim'] = df_treino.reset_index().loc[len(df_treino)-1, 'Data']

# Plota as predições em comparação com os valores reais
fig, (sub1) = plt.subplots(1, 1, sharex=True)
pred, = plt.plot(df_teste[5:]['Data'], pwr_scaler.inverse_transform(pwr_pred), c='blue', label='Predito')
real = plt.scatter(df_teste[5:]['Data'], df_teste[5:]['Valor'], s=3, c='orange')
plt.legend([pred, real],
           ['Predito', 'Real'],
           fontsize=8)
fig.autofmt_xdate()
plt.xlabel('Data')
plt.ylabel('Valor (R$)')
plt.title('ICMS - LSTM - Única Variável Quantitativa - Datas Idênticas ao Modelo com Múltiplas Variáveis Quantitativas')
plt.show()