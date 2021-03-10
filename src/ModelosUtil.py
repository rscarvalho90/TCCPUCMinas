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
    
    @staticmethod
    def agrupa_dados_mensais_em_trimestrais(df):
        """ Realiza a soma de cada dia que compõe um determinado trimestre, retornando dados mensais. """
        pd_trimestral = pd.DataFrame(columns=['Trimestre', 'Valor'])
        meses_trimestre = [4, 7, 10, 1]
        soma = 0
        df = df.reset_index(drop=True)
        
        for i in range(0, len(df)):
            mes_atual = df.loc[i, 'Data'].month
            if i!=0:
                mes_anterior = df.loc[i-1, 'Data'].month
            else:
                mes_anterior = mes_atual
             
            # Quando há mudança de trimestre, adiciona o valor da soma trimestral ao dataframe e reseta a soma
            if mes_atual!=mes_anterior and mes_atual in meses_trimestre:
                trimestre = str(mes_anterior)+'/'+str(df.loc[i-1, 'Data'].year)
                pd_trimestral = pd_trimestral.append({'Trimestre': trimestre, 'Valor': soma}, ignore_index=True)
                soma = 0
                
            soma = soma+df.loc[i, 'Valor']
          
        # Aloca a soma restante (trimestre incompleto)
        trimestre = str(mes_anterior)+'/'+str(df.loc[i-1, 'Data'].year)
        pd_trimestral = pd_trimestral.append({'Trimestre': trimestre, 'Valor': soma}, ignore_index=True)
        soma = 0
        
        return pd_trimestral
    
    @staticmethod
    def adiciona_pib(df_tributo, df_pib):
        """ Adiciona as colunas contendo os valores do PIB dos trimestres atual e anterior. """
        for i in range(1, len(df_tributo)):
            trimestre_atual = str(df_tributo.loc[i, 'ds'].month).zfill(2)+'/'+str(df_tributo.loc[i, 'ds'].year)         
            trimestre_anterior = str(df_tributo.loc[i-1, 'ds'].month).zfill(2)+'/'+str(df_tributo.loc[i-1, 'ds'].year)
            
            try:
                pib_trimestre_atual = df_pib[df_pib['Trimestre']==trimestre_atual].PIB.item()
                pib_trimestre_anterior = df_pib[df_pib['Trimestre']==trimestre_anterior].PIB.item()
                
                df_tributo.loc[i, 'pib_trim_atual'] = pib_trimestre_atual
                df_tributo.loc[i, 'pib_trim_ant'] = pib_trimestre_anterior
            except:
                print('Trimeste não encontrado '+trimestre_atual)
          
        df_tributo = df_tributo.dropna()
        
        return df_tributo
    

class LSTMUtil:
    def __init__(self):
        pass

    @staticmethod
    def transforma_dataframe(df, nome_coluna_data):
        """ Retorna um dataframe com as colunas necessárias para aplicação à rede neural LSTM. """
        dia = df[nome_coluna_data].dt.day
        mes = df[nome_coluna_data].dt.month
        ano = df[nome_coluna_data].dt.year

        df = df.drop('Tributo', axis=1)
        df.insert(loc=0, column='Ano', value=ano)
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
    
    @staticmethod
    def gera_teste_identico_prophet(df, data_inicio_teste, data_fim_teste, n_intervalos=5):
        """ Retorna o dois dataframes com o set de treinamento e o set de testes. """
        index_teste = df[df['Data']==data_inicio_teste].index[0]-n_intervalos
        
        df_treino = df[:index_teste]
        df_teste = df[index_teste:]

        return df_treino, df_teste
