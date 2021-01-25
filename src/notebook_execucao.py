from src.DownloadDados import DownloadDados
from src.Util import CorrigeValores
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
