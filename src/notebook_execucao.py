from src.DownloadDados import DownloadDados
from src.Util import CorrigeValores
from src.ModelosUtil import ArimaUtil
import seaborn as sns
import matplotlib.pyplot as plt

dd = DownloadDados()

# Gera um Pandas Dataframe com os dados de arrecadação diária de todos os tributos
pd_arrecad_diaria = DownloadDados.download_arrecadacao_sefaz_rs()

arrecad_diaria = {}

# Gera dataframes de arrecadação diária para cada tributo, sem a correção pela inflação
for tributo in pd_arrecad_diaria['Tributo'].unique():
    arrecad_diaria[tributo] = pd_arrecad_diaria[pd_arrecad_diaria['Tributo'] == tributo].reset_index()
    arrecad_diaria[tributo] = arrecad_diaria[tributo].drop(['index'], axis=1)

# Plota os gráficos das séries temporais dos tributos, sem a correção pela inflação
for tributo in pd_arrecad_diaria['Tributo'].unique():
    sns.lineplot(arrecad_diaria[tributo]['Data'], arrecad_diaria[tributo]['Valor']).set_title(tributo)
    plt.show()

# Corrige os valores pela inflação
igp = dd.download_igp()
pd_arrecad_diaria = CorrigeValores.corrige_inflacao(pd_arrecad_diaria, igp)

# Gera dataframes de arrecadação diária para cada tributo, após a correção pela inflação
for tributo in pd_arrecad_diaria['Tributo'].unique():
    arrecad_diaria[tributo] = pd_arrecad_diaria[pd_arrecad_diaria['Tributo'] == tributo].reset_index()
    arrecad_diaria[tributo] = arrecad_diaria[tributo].drop(['index'], axis=1)

# Plota os gráficos das séries temporais dos tributos, após a correção pela inflação
for tributo in pd_arrecad_diaria['Tributo'].unique():
    sns.lineplot(arrecad_diaria[tributo]['Data'], arrecad_diaria[tributo]['Valor']).set_title(tributo)
    plt.show()

# Cria modelo ARIMA para predição de dados diários
# Plota a autocorrelação dos dataframes
from pandas.plotting import autocorrelation_plot

ax = {}
for tributo in pd_arrecad_diaria['Tributo'].unique():
    ax[tributo] = autocorrelation_plot(arrecad_diaria[tributo]['Valor'], label=tributo, )
    plt.show()

pd.DataFrame(ax['IPVA'].lines[5].get_data()[1]).max()

# Faz o tunning dos parâmetros ARIMA para cada tributo
for tributo in pd_arrecad_diaria['Tributo'].unique():
    valores_p = [3, 4, 5]
    valores_d = [1, 2, 3]
    valores_q = [0, 1, 2]
    melhor_cfg, menor_aic = ArimaUtil.tunning_parametros(arrecad_diaria[tributo]['Valor'].values, valores_p, valores_d,
                                                         valores_q)
    print('A melhor configuração ARIMA para o tributo ' + tributo + ' é ' + str(melhor_cfg) + ' com AIC igual a ' + str(
        menor_aic))
