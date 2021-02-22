from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
from pandas import to_datetime


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
        df_treino = df[:int(0.8*len(df))]
        df_teste = df[int(0.8*len(df)):]

        return df_treino, df_teste

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
