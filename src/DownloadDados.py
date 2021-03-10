import requests
import pandas as pd
from lxml import html
from src.Util import DataUtil


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
                igpm = float(colunas[2].text_content().replace(',', '.'))
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
        
        df = df[['Trimestre', 'PIB']]
        
        return df
