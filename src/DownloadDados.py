import requests
import re
import pandas as pd
from lxml import html
from src.Util import DataUtil
import numpy as np


class DownloadDados:
    """ Classe utilizada para baixar os dados remotamente. """

    def __init__(self):
        pass

    @staticmethod
    def download_arrecadacao_sefaz_rs():
        """ Baixa os dados de arrecadação diária da Sefaz RS,
         retornando um Pandas Dataframe."""

        url = 'http://receitadados.fazenda.rs.gov.br/Arquivos/Arrecada%C3%A7%C3%A3o%20Di%C3%A1ria.csv'
        df = pd.read_csv(url, sep=';')
        df['Valor'] = df['Valor'].str.replace(',', '.').astype(float)
        df['Data'] = pd.to_datetime(df['Data'])

        return df

    def download_igp(self):
        """ Baixa os dados do IGP mensal, retornando um Pandas Dataframe."""

        url = 'http://www.ipeadata.gov.br/ExibeSerie.aspx?stub=1&serid37796=37796&serid36482=36482'
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:84.0) Gecko/20100101 Firefox/84.0'}
        pagina_web = requests.get(url, headers=headers, verify=False)
        arvore = html.fromstring(pagina_web.text)
        df = pd.DataFrame(columns=['Mes/Ano', 'Numero Indice'])

        data_util = DataUtil()
        
        linhas_tabela = arvore.xpath('/html/body/form/center[2]/table/tr/td/table[1]/tr')
        
        for i in range(1, len(linhas_tabela)):
            linha = linhas_tabela[i]
            colunas = linha.findall("td")
            ano_mes = str(colunas[0].text_content())
            if colunas[2].text_content() != '\xa0':
                igpm = float(colunas[2].text_content().replace('.', '').replace(',', '.'))
                ano = ano_mes.split('.')[0]
                mes = ano_mes.split('.')[1]
                df.loc[len(df)] = [mes+'/'+ano, igpm]

        return df.iloc[::-1].reset_index(drop=True)
    
    @staticmethod
    def download_pib_rs():
        """ Baixa os dados do PIB trimestral (corrigido pela inflação) do Rio Grande do Sul, retornando um Pandas Dataframe. """
        
        url = 'https://dee.rs.gov.br/upload/arquivos/202012/11100232-serie-historica-pib-terceiro-trimestre.xlsx'
        df = pd.read_excel(url, sheet_name='Série encadeada')
        df = df.loc[6:, ['Unnamed: 0', 'Unnamed: 17']]
        df.columns = ['Trimestre', 'PIB']
        
        trimestre_mes = {'I':'03', 'II':'06', 'III':'09', 'IV':'12'}
        
        df[['Ano', 'Trimestre']] = df['Trimestre'].str.split(pat='.', expand = True)
        
        # Transforma o formato original do trimestre (ano + trimestre em algarismo romano) em formato de mês, correspondendo ao último mês do trimestre
        for x, y in trimestre_mes.items():
            df['Trimestre'] = df['Trimestre'].str.replace(r'^'+x+'$', str(y))
        
        df['Trimestre'] = df['Trimestre']+'/'+df['Ano']
        
        df = df[['Trimestre', 'PIB']].iloc[::-1].reset_index(drop=True)
        
        return df
    
    @staticmethod
    def download_pib_br():
        """ Baixa os dados do PIB mensal (estimado) do Brasil, retornando um Pandas Dataframe. """
        
        url = 'http://www.ipeadata.gov.br/ExibeSerie.aspx?oper=exportCSVBr&serid521274780=521274780&serid521274780=521274780'        
        df = pd.read_csv(url, sep=';')
        df = df.iloc[:, 0:2]
        df.columns = ['Data', 'PIB']
        df['Data'] = df['Data'].astype(str).str.replace('.', '/').replace('0000', '').str.replace('/1$', '/10')
        df['PIB'] = df['PIB'].astype(str).str.replace(',', '.').astype(float)
        
        for i in range(0, len(df)):
            ano = df['Data'].str.split('/')[i][0]
            mes = df['Data'].str.split('/')[i][1]
            df['Data'].loc[i] = mes+'/'+ano
            
        df = df.reset_index(drop=True)
        
        return df
    
    @staticmethod
    def download_dados_emprego():
        """ Baixa os dados de emprego do IPEA Data """
        
        # Baixa os dados de saldo de empregos para preencher campos que vierem sem valores válidos
        url_saldo_caged_antigo = 'http://www.ipeadata.gov.br/ExibeSerie.aspx?oper=exportCSVBr&serid272844966=272844966&serid272844966=272844966'
        df_saldo_caged_antigo = pd.read_csv(url_saldo_caged_antigo, sep=';')
        df_saldo_caged_antigo['Data'] = df_saldo_caged_antigo['Data'].astype(str).str.replace('.', '/').replace('0000', '').str.replace('/1$', '/10')
        df_saldo_caged_antigo = df_saldo_caged_antigo.drop(df_saldo_caged_antigo.columns[2], axis=1)
        df_saldo_caged_antigo.columns = ['Data', 'Saldo']
        
        # Dados antigos do CAGED (Admissões e Demissões)
        url_caged_antigo = 'http://www.ipeadata.gov.br/ExibeSerie.aspx?oper=exportCSVBr&serid231410417=231410417&serid231410418=231410418'
        df_antigo = pd.read_csv(url_caged_antigo, sep=';')
        df_antigo['Data'] = df_antigo['Data'].astype(str).str.replace('.', '/').replace('0000', '').str.replace('/1$', '/10')
        df_antigo = df_antigo.drop(df_antigo.columns[3], axis=1)
        df_antigo.columns = ['Data', 'Admissoes', 'Demissoes']
        datas_nan = df_antigo['Data'][np.isnan(df_antigo['Demissoes'])==True] # Datas cujo valor de demissões é NaN
        saldos_nan = df_saldo_caged_antigo[df_saldo_caged_antigo['Data'].isin(pd.DataFrame(datas_nan)['Data'])] # Saldos em datas cujo valor de demissões é NaN
        df_antigo['Demissoes'][np.isnan(df_antigo['Demissoes'])==True] = df_antigo['Admissoes'][np.isnan(df_antigo['Demissoes'])==True]-saldos_nan['Saldo']        
        df_antigo['Demissoes'] = df_antigo['Demissoes'].astype(int)
        
        # Dados novos do CAGED (Admissões e Demissões)
        url_caged_novo = 'http://www.ipeadata.gov.br/ExibeSerie.aspx?oper=exportCSVBr&serid2096725334=2096725334&serid2096725335=2096725335'
        df_novo = pd.read_csv(url_caged_novo, sep=';')
        df_novo['Data'] = df_novo['Data'].astype(str).str.replace('.', '/').replace('0000', '').str.replace('/1$', '/10')
        df_novo = df_novo.drop(df_novo.columns[3], axis=1)
        df_novo.columns = ['Data', 'Admissoes', 'Demissoes']
        
        df_final = df_novo.iloc[::-1].append(df_antigo.iloc[::-1]).reset_index(drop=True)
        
        for i in range(0, len(df_final)):
            ano = df_final['Data'].str.split('/')[i][0]
            mes = df_final['Data'].str.split('/')[i][1]
            df_final['Data'].loc[i] = mes+'/'+ano
        
        return df_final        
