from datetime import datetime


class CorrigeValores:
    def __init__(self):
        pass

    @staticmethod
    def corrige_inflacao(pd_tributo, pd_inflacao):
        """ Dado um dataframe com os números-índice para correção e outro dataframe com os valores de tributo,
        corrige estes trazendo-os ao valor presente da última data disponível no dataframe de valores de tributo."""

        data_util = DataUtil()

        ultimo_mes_inflacao = pd_inflacao.loc[0, 'Mes/Ano']
        ultimo_numero_indice = pd_inflacao.loc[0, 'Numero Indice']

        for indice, linha in pd_tributo.iterrows():
            mes_atual = linha['Data'].strftime("%m/%Y")

            ''' Nos meses anteriores à última data publicada para a inflação, o valor do tributo será
            multiplicado pela razão entre o útlimo número-índice e o número índice correspondente ao mês seguinte ao do
            tributo. '''
            if datetime.strptime(mes_atual, '%m/%Y').date() < datetime.strptime(ultimo_mes_inflacao, '%m/%Y').date():
                numero_indice_atual = pd_inflacao[pd_inflacao['Mes/Ano'] == mes_atual]['Numero Indice'].item()
            elif datetime.strptime(mes_atual, '%m/%Y').date() >= datetime.strptime(ultimo_mes_inflacao, '%m/%Y').date():
                numero_indice_atual = ultimo_numero_indice

            pd_tributo.loc[indice, 'Valor'] = pd_tributo.loc[indice, 'Valor'] * ultimo_numero_indice / numero_indice_atual

        return pd_tributo
    
    def corrige_inflacao_pib(pd_pib, pd_inflacao):
        """ Dado um dataframe com os números-índice para correção e outro dataframe com os valores do PIB Nominal,
        corrige estes trazendo-os ao valor presente da última data disponível no dataframe de valores de PIB."""

        data_util = DataUtil()

        ultimo_mes_inflacao = pd_inflacao.loc[0, 'Mes/Ano']
        ultimo_numero_indice = pd_inflacao.loc[0, 'Numero Indice']

        for indice, linha in pd_pib.iterrows():
            mes_atual = linha['Data']

            ''' Nos meses anteriores à última data publicada para a inflação, o valor do tributo será
            multiplicado pela razão entre o útlimo número-índice e o número índice correspondente ao mês seguinte ao do
            tributo. '''
            if datetime.strptime(mes_atual, '%m/%Y').date() < datetime.strptime(ultimo_mes_inflacao, '%m/%Y').date():
                numero_indice_atual = pd_inflacao[pd_inflacao['Mes/Ano'] == mes_atual]['Numero Indice'].item()
            elif datetime.strptime(mes_atual, '%m/%Y').date() >= datetime.strptime(ultimo_mes_inflacao, '%m/%Y').date():
                numero_indice_atual = ultimo_numero_indice

            pd_pib.loc[indice, 'PIB'] = pd_pib.loc[indice, 'PIB'] * ultimo_numero_indice / numero_indice_atual

        return pd_pib
    

class DataUtil:
    mes_numero = {'Jan': '01', 'Fev': '02', 'Mar': '03', 'Abr': '04', 'Mai': '05', 'Jun': '06', 'Jul': '07',
                  'Ago': '08', 'Set': '09', 'Out': '10',
                  'Nov': '11', 'Dez': '12'}

    def __init__(self):
        pass

    @staticmethod
    def get_mes(str_data):
        """ Dada uma data no formato yyyy-mm-dd, retorna o mês no formato mm/yyyy. """

        return str_data.strftime('%m/%Y')

    def get_mesano_anterior(self, data_str):
        """ Dada uma data no formato 'mm/yyyy', retorna um string representando o
        mês e ano anterior. """

        mes_int = int(data_str.split('/')[0])

        if mes_int > 0:
            if mes_int - 1 > 0:
                mes_str = [key for key, value in self.mes_numero.items() if (mes_int - 1) == int(value)]
                return mes_str[0] + "/" + str(data_str.split('/')[1])
            if mes_int - 1 <= 0:
                return "Dez/" + str(int(data_str.split('/')[1]) - 1)

    def get_mesano_seguinte(self, data_str):
        """ Dada uma data no formato 'mm/yyyy', retorna um string representando o
        mês e ano seguinte. """

        mes_int = int(data_str.split('/')[0])

        if mes_int > 0:
            if mes_int + 1 <= 12:
                mes_str = [key for key, value in self.mes_numero.items() if (mes_int + 1) == int(value)]
                return mes_str[0] + "/" + str(data_str.split('/')[1])
            if mes_int + 1 > 12:
                return "Jan/" + str(int(data_str.split('/')[1]) + 1)

    def mesano_str_to_mesano_int(self, data_str):
        """ Dada uma data no formato 'MMM/yyyy' (mês com representação literal), retorna um inteiro representando o
        mês e ano. """

        mes_int = [value for key, value in self.mes_numero.items() if data_str.split('/')[0].startswith(key)]

        if len(mes_int) > 0:
            return str(mes_int[0]) + "/" + str(data_str.split('/')[1])

    @staticmethod
    def get_mes(data_str):
        """ Dada uma data no formato 'mm/yyyy', retorna um inteiro representando o mês. """

        mes_str = data_str.split('/')[0]

        return int(mes_str)
