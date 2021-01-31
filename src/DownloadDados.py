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

        url = 'https://www.portalbrasil.net/igpm/'
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:84.0) Gecko/20100101 Firefox/84.0'}
        pagina_web = requests.get(url, headers=headers, verify=False)
        arvore = html.fromstring(pagina_web.text)
        df = pd.DataFrame(columns=['Mes/Ano', 'Numero Indice'])

        data_util = DataUtil()

        # Faz a leitura da tabela 'Última atualização'
        linhas_tabela = arvore.xpath(
            '/html/body/div[1]/div[3]/main/div/div/div[1]/div[1]/article/div[3]/figure[1]/table[2]/tbody/tr')
        for i in range(0, len(linhas_tabela)):
            row = linhas_tabela[i]
            columns = row.findall("td")
            mes_ano = columns[0].text
            if mes_ano != 'MÊS/ANO':  # Corrige leitura da primeira linha
                mes_ano = columns[0].xpath('strong')[0].text
                numero_indice = float(columns[4].xpath('strong')[0].text.replace('.', '').replace(',', '.'))
                df.loc[len(df)] = [data_util.mesano_str_to_mesano_int(mes_ano), numero_indice]

        # Faz a leitura da tabela principal
        linhas_tabela = arvore.xpath(
            '/html/body/div[1]/div[3]/main/div/div/div[1]/div[1]/article/div[3]/figure[1]/table[3]/tbody/tr')

        for row in linhas_tabela:
            columns = row.findall("td")
            mes_ano = columns[0].text
            if mes_ano == '\n':
                mes_ano = data_util.get_mesano_anterior(df.loc[len(df) - 1, 'Mes/Ano'])
            numero_indice = float(columns[4].text.replace('.', '').replace(',', '.'))
            df.loc[len(df)] = [data_util.mesano_str_to_mesano_int(mes_ano), numero_indice]

        return df
