from src.DownloadDados import DownloadDados
from src.Util import CorrigeValores
import seaborn as sns
import matplotlib.pyplot as plt

dd = DownloadDados()

# Gera um Pandas Dataframe com os dados de arrecada��o di�ria de todos os tributos
pd_arrecad_diaria = DownloadDados.download_arrecadacao_sefaz_rs()

arrecad_diaria = {}

# Gera dataframes de arrecada��o di�ria para cada tributo, sem a corre��o pela infla��o
for tributo in pd_arrecad_diaria['Tributo'].unique():
    arrecad_diaria[tributo] = pd_arrecad_diaria[pd_arrecad_diaria['Tributo'] == tributo].reset_index()
    arrecad_diaria[tributo] = arrecad_diaria[tributo].drop(['index'], axis=1)

# Plota os gr�ficos das s�ries temporais dos tributos, sem a corre��o pela infla��o
for tributo in pd_arrecad_diaria['Tributo'].unique():
    sns.lineplot(arrecad_diaria[tributo]['Data'], arrecad_diaria[tributo]['Valor']).set_title(tributo)
    plt.show()

# Corrige os valores pela infla��o
igp = dd.download_igp()
pd_arrecad_diaria = CorrigeValores.corrige_inflacao(pd_arrecad_diaria, igp)

# Gera dataframes de arrecada��o di�ria para cada tributo, ap�s a corre��o pela infla��o
for tributo in pd_arrecad_diaria['Tributo'].unique():
    arrecad_diaria[tributo] = pd_arrecad_diaria[pd_arrecad_diaria['Tributo'] == tributo].reset_index()
    arrecad_diaria[tributo] = arrecad_diaria[tributo].drop(['index'], axis=1)

# Plota os gr�ficos das s�ries temporais dos tributos, ap�s a corre��o pela infla��o
for tributo in pd_arrecad_diaria['Tributo'].unique():
    sns.lineplot(arrecad_diaria[tributo]['Data'], arrecad_diaria[tributo]['Valor']).set_title(tributo)
    plt.show()
