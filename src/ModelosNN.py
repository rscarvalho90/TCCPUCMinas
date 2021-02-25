import tensorflow as tf
from tensorflow.python.keras.layers import Embedding, Flatten, Concatenate, Dense
from tensorflow.python.keras.layers.recurrent import LSTM

class LSTMUnivariada(tf.keras.Model):

    def __init__(self, df):
        super(LSTMUnivariada, self).__init__()
        self.df = df
        self.cria_rede_neural_univariada(df)
        self.valor = Dense(1, activation='linear', name='Valor')

        ''' O primeiro item da lista se refere ao mês e dia, já
        o segundo item se refere à LSTM, com 5 períodos (5 dias anteriores à predição) e uma variável (valor). '''
        self.build([(None, 2), (None, 5, 1)])

    def cria_rede_neural_univariada(self, df):
        """ Cria rede neural "univariada" usando o Keras Functional API, retornando um modelo
        do Keras. """

        dias_distintos = df['Dia'].unique()
        meses_distintos = df['Mes'].unique()

        self.embedding_dia = Embedding(name='dia_embedding', input_length=1,
                                       input_dim=len(dias_distintos),
                                       output_dim=int(round(len(dias_distintos) ** 0.25, 0)))
        self.flatten_dia = Flatten()
        self.embedding_mes = Embedding(name='mes_embedding', input_length=1,
                                       input_dim=len(meses_distintos),
                                       output_dim=int(round(len(meses_distintos) ** 0.25, 0)))
        self.flatten_mes = Flatten()
        self.concatenate_dia_mes = Concatenate(axis=-1, name='dia_mes_concatenate')
        self.dense_dia_mes = Dense(2, activation='relu', name='dia_mes_dense')
        self.lstm_valor = LSTM(1, name='valor_lstm')
        self.dense_valor = Dense(1, activation='relu', name='valor_dense')
        self.concatenate_dia_mes_valor = Concatenate(axis=-1, name='dia_mes_valor_concatenate')
        self.dense_dia_mes_valor = Dense(1, activation='sigmoid', name='dia_mes_valor_dense')

    def call(self, inputs, **kwargs):
        # inputs[0] são os dados de dia e mês
        dia_mes_tensor = tf.convert_to_tensor(inputs[0])

        # inputs[1] são os dados do valor arrecadado
        valor_tensor = tf.convert_to_tensor(inputs[1])

        # dia_mes_tensor[:, 0] são os dados do dia
        # dia_mes_tensor[:, 1] são os dados do mês
        flt_dia = self.flatten_dia(self.embedding_dia(dia_mes_tensor[:, 0]))
        flt_mes = self.flatten_mes(self.embedding_dia(dia_mes_tensor[:, 1]))
        concat_dia_mes = self.concatenate_dia_mes([flt_dia, flt_mes])
        dense_dia_mes = self.dense_dia_mes(concat_dia_mes)
        lstm_valor = self.lstm_valor(valor_tensor)
        dense_valor = self.dense_valor(lstm_valor)
        dia_mes_valor = self.concatenate_dia_mes_valor([dense_dia_mes, dense_valor])

        return self.valor(dia_mes_valor)

    def get_config(self):
        pass
