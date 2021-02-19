from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error


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
