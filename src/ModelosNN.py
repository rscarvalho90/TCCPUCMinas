import tensorflow as tf
from tensorflow.python.keras.layers import Embedding, Flatten, Concatenate, Dense
from tensorflow.python.keras.layers.recurrent import LSTM


class LSTMUnivariada(tf.keras.Model):

    def __init__(self, df):
        super(LSTMUnivariada, self).__init__()
        self.df = df
        self.cria_rede_neural_univariada(df)
        self.valor = Dense(1, activation='linear', name='Valor')

        ''' O primeiro item da lista se refere ao mês, dia e dia da semana, já
        o segundo item se refere à LSTM, com 5 períodos (5 dias anteriores à predição) e uma variável (valor). '''
        self.build([(None, 3), (None, 5, 1)])

    def cria_rede_neural_univariada(self, df):
        """ Cria rede neural "univariada" usando o Keras Functional API, retornando um modelo
        do Keras. """

        dias_distintos = df['Dia'].unique()
        meses_distintos = df['Mes'].unique()
        dias_semana_distintos = df['Dia_Semana'].unique()

        self.embedding_dia = Embedding(name='dia_embedding', input_length=1,
                                       input_dim=len(dias_distintos),
                                       output_dim=int(round(len(dias_distintos) ** 0.25, 0)))
        self.flatten_dia = Flatten()
        self.embedding_mes = Embedding(name='mes_embedding', input_length=1,
                                       input_dim=len(meses_distintos),
                                       output_dim=int(round(len(meses_distintos) ** 0.25, 0)))
        self.flatten_mes = Flatten()
        self.embedding_dia_semana = Embedding(name='dia_semana_embedding', input_length=1,
                                       input_dim=len(dias_semana_distintos),
                                       output_dim=int(round(len(dias_semana_distintos) ** 0.25, 0)))
        self.flatten_dia_semana = Flatten()
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
        # dia_mes_tensor[:, 2] são os dados do dia da semana
        flt_dia = self.flatten_dia(self.embedding_dia(dia_mes_tensor[:, 0]))
        flt_mes = self.flatten_mes(self.embedding_dia(dia_mes_tensor[:, 1]))
        flt_dia_semana = self.flatten_dia_semana(self.embedding_dia(dia_mes_tensor[:, 2]))
        concat_dia_mes = self.concatenate_dia_mes([flt_dia, flt_mes, flt_dia_semana])
        dense_dia_mes = self.dense_dia_mes(concat_dia_mes)
        lstm_valor = self.lstm_valor(valor_tensor)
        dense_valor = self.dense_valor(lstm_valor)
        dia_mes_valor = self.concatenate_dia_mes_valor([dense_dia_mes, dense_valor])

        return self.valor(dia_mes_valor)

    def get_config(self):
        pass
    

class LSTMMultivariada(tf.keras.Model):

    def __init__(self, df):
        super(LSTMMultivariada, self).__init__()
        self.df = df
        self.cria_rede_neural_multivariada(df)
        self.valor = Dense(1, activation='linear', name='Valor')

        ''' O primeiro item da lista se refere ao mês, dia, dia da semana, PIB do RS no trimestre anterior, 
        PIB do Brasil do mês anterior, admissões no mês anterior e demissões no mês anterior,
        já o segundo item se refere à LSTM, com 5 períodos (5 dias anteriores à predição) e uma variável (valor). '''
        self.build([(None, 7), (None, 5, 1)])

    def cria_rede_neural_multivariada(self, df):
        """ Cria rede neural "multivariada" usando o Keras Functional API, retornando um modelo
        do Keras. """

        dias_distintos = df['Dia'].unique()
        meses_distintos = df['Mes'].unique()
        dias_semana_distintos = df['Dia_Semana'].unique()

        self.embedding_dia = Embedding(name='dia_embedding', input_length=1,
                                       input_dim=len(dias_distintos),
                                       output_dim=int(round(len(dias_distintos) ** 0.25, 0)))
        self.flatten_dia = Flatten()
        self.embedding_mes = Embedding(name='mes_embedding', input_length=1,
                                       input_dim=len(meses_distintos),
                                       output_dim=int(round(len(meses_distintos) ** 0.25, 0)))
        self.flatten_mes = Flatten()
        self.embedding_dia_semana = Embedding(name='dia_semana_embedding', input_length=1,
                                       input_dim=len(dias_semana_distintos),
                                       output_dim=int(round(len(dias_semana_distintos) ** 0.25, 0)))
        self.flatten_dia_semana = Flatten()
        self.flatten_pib_rs = Flatten()
        self.flatten_pib_br = Flatten()
        self.flatten_admissoes = Flatten()
        self.flatten_admissoes = Flatten()
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
        # dia_mes_tensor[:, 2] são os dados do dia da semana
        # dia_mes_tensor[:, 3] são os dados do PIB trimestral do RS no trimestre anterior
        # dia_mes_tensor[:, 4] são os dados do PIB mensal do Brasil no mês anterior
        # dia_mes_tensor[:, 5] são os dados de admissões no mês anterior
        # dia_mes_tensor[:, 6] são os dados de demissões no mês anterior
        flt_dia = self.flatten_dia(self.embedding_dia(dia_mes_tensor[:, 0]))
        flt_mes = self.flatten_mes(self.embedding_dia(dia_mes_tensor[:, 1]))
        flt_dia_semana = self.flatten_dia_semana(self.embedding_dia(dia_mes_tensor[:, 2]))
        flt_pib_rs = self.flatten_pib_rs(dia_mes_tensor[:, 3])
        flt_pib_br =  self.flatten_pib_br(dia_mes_tensor[:, 4])
        flt_admissoes =  self.flatten_admissoes(dia_mes_tensor[:, 5])
        flt_demissoes = self.flatten_admissoes(dia_mes_tensor[:, 6])
        
        concat_dia_mes = self.concatenate_dia_mes([flt_dia, flt_mes, flt_dia_semana, flt_pib_rs, flt_pib_br, flt_admissoes, flt_demissoes])
        dense_dia_mes = self.dense_dia_mes(concat_dia_mes)
        lstm_valor = self.lstm_valor(valor_tensor)
        dense_valor = self.dense_valor(lstm_valor)
        dia_mes_valor = self.concatenate_dia_mes_valor([dense_dia_mes, dense_valor])

        return self.valor(dia_mes_valor)

    def get_config(self):
        pass
