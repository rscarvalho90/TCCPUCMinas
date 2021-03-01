from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
from pandas import to_datetime
import numpy as np


class ProphetUtil:
    def __init__(self):
        pass

    @staticmethod
    def transforma_dataframe(df, list_colunas):
        """ Dado um dataframe e uma lista com as colunas de data e e valor (nesta ordem), retorna um dataframe
        para que possa ser feito o 'fit' no Prophet. """
        pd_prophet = pd.DataFrame(df).loc[:, list_colunas]
        pd_prophet[list_colunas[0]] = to_datetime(pd_prophet[list_colunas[0]])

        return pd_prophet.rename(columns={list_colunas[0]: 'ds', list_colunas[1]: 'y'})

    @staticmethod
    def divide_treino_teste(df, percentual_treinamento=0.8):
        """ Retorna o dois dataframes com o set de treinamento e o set de testes. """
        df_treino = df[:int(0.8 * len(df))]
        df_teste = df[int(0.8 * len(df)):]

        return df_treino, df_teste


class LSTMUtil:
    def __init__(self):
        pass

    @staticmethod
    def transforma_dataframe(df, nome_coluna_data):
        """ Retorna um dataframe com as colunas necessárias para aplicação à rede neural LSTM. """
        dia = df[nome_coluna_data].dt.day
        mes = df[nome_coluna_data].dt.month

        df = df.drop(nome_coluna_data, axis=1)
        df = df.drop('Tributo', axis=1)
        df.insert(loc=0, column='Mes', value=mes)
        df.insert(loc=0, column='Dia', value=dia)

        return df

    @staticmethod
    def cria_intervalos_temporais(np_array, n_intervalos=5):
        """ Dado um array NumPy com os valores diários, gera sequências temporais com 3 dimensões
        para alimentarem a rede neural LSTM. """

        np_valores = np_array
        np_sequencia = np.empty((0, n_intervalos, 1))

        for i in range(n_intervalos, len(np_valores)):
            # Adiciona os itens que comporão uma sequência
            # Cada item é composto por uma sequência, n_intervalos intervalos de tempo e 1 feature)
            np_item = np.empty((0, n_intervalos, 1))            
            np_item = np.append(np_item, np_valores[(i-n_intervalos):i, 0].reshape(1, n_intervalos, 1), axis=0)
            # Adiciona uma sequência à lista de sequências
            np_sequencia = np.append(np_sequencia, np_item, axis=0)

        return np_sequencia


class ArimaUtil:
    def __init__(self):
        pass

    @staticmethod
    def tunning_parametros(X, valores_p, valores_d, valores_q):
        """ Dado um dataset e três listas de possíveis valores para os parâmetros ARIMA, retorna o conjunto
        de parâmetros com Akaike Information Critera (AIC). """
        X = X.astype('float32')
        menor_aic, melhor_cfg = float("inf"), None
        for p in valores_p:
            for d in valores_d:
                for q in valores_q:
                    order = (p, d, q)
                    try:
                        aic = ArimaUtil.avalia_modelo(X, order)
                        if aic < menor_aic:
                            menor_aic, melhor_cfg = aic, order
                    except:
                        continue
        return melhor_cfg, menor_aic

    @staticmethod
    def avalia_modelo(X, arima_order):
        """ Avalia os parâmetros de um modelo ARIMA, retornando o Akaike Information Critera (AIC). """
        history = [x for x in X]

        model = ARIMA(history, order=arima_order)
        resultado = model.fit()

        return resultado.aic
