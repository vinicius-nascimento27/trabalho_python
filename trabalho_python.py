# ==================================================
# ANÁLISE DE INFRAÇÕES DE VEÍCULOS 2024
# ==================================================

# Importações de bibliotecas
import os                              # ← manipulação de arquivos e diretórios
import pandas as pd                    # ← análise e manipulação de dados
import numpy as np                     # ← operações matemáticas e estatísticas
import matplotlib.pyplot as plt         # ← criação de gráficos
import matplotlib.ticker as mtick       # ← formatação de eixos numéricos em gráficos

# ==================================================
# CONFIGURAÇÃO DO ARQUIVO (já no diretório do Colab)
# ==================================================
csv_file = "infracoes_veiculos_2024.csv"  # ← nome do arquivo CSV de entrada
if not os.path.exists(csv_file):          # ← verifica se o arquivo existe no diretório atual
    raise FileNotFoundError(f"Arquivo '{csv_file}' não encontrado no diretório atual ({os.getcwd()}).")  # ← erro se não existir

# ==================================================
# LEITURA E LIMPEZA
# ==================================================
# Tenta ler o CSV com separador automático e codificação UTF-8
try:
    df = pd.read_csv(csv_file, sep=None, engine="python", encoding="utf-8", on_bad_lines="skip")  # ← leitura com autodetecção
except Exception:
    # Caso dê erro, tenta novamente com codificação Latin-1
    df = pd.read_csv(csv_file, sep=None, engine="python", encoding="latin1", on_bad_lines="skip")

# Remove caracteres ocultos do nome das colunas (como BOM)
df.columns = [c.replace("\ufeff", "").strip() for c in df.columns]  # ← limpeza de cabeçalhos

# Renomeia colunas para nomes padronizados usados no código
df = df.rename(columns={
    "ANOMES": "ANO_MES",
    "DESC_TIPO": "TIPO_VEICULO",
    "AIT": "QTD_INFRACOES",
    "DESCR_MUNIC_INF": "REGIAO"
})

# Converte colunas numéricas para tipo numérico (ignora erros)
df["ANO_MES"] = pd.to_numeric(df.get("ANO_MES"), errors="coerce")       # ← converte ano/mês
df["QTD_INFRACOES"] = pd.to_numeric(df.get("QTD_INFRACOES"), errors="coerce")  # ← converte número de infrações
df = df.dropna(subset=["TIPO_VEICULO", "QTD_INFRACOES", "REGIAO"])     # ← remove linhas com dados ausentes

# ==================================================
# REMOVER TIPOS DE VEÍCULO INDIVIDUAIS
# ==================================================
# Lista de tipos de veículos que serão ignorados na análise
tipos_a_remover = [
    "TRATOR DE RODAS", "MOTOR CASA", "TRICICLO", "CICLOMOTOR",
    "REBOQUE", "CAMINHAO TRATOR", "NAO INFORMADO", "SEMI-REBOQUE"
]
df = df[~df["TIPO_VEICULO"].isin(tipos_a_remover)]  # ← remove os tipos listados

# ==================================================
# FILTRAR REGIÕES COM PELO MENOS 2000 INFRAÇÕES
# ==================================================
total_infracoes_regioes = df.groupby("REGIAO")["QTD_INFRACOES"].sum()  # ← soma total de infrações por região
regioes_com_2000_ou_mais = total_infracoes_regioes[total_infracoes_regioes >= 2000].index  # ← seleciona regiões com ≥2000
df_filtrado = df[df["REGIAO"].isin(regioes_com_2000_ou_mais)].copy()  # ← mantém apenas essas regiões

# ==================================================
# DISTRIBUIÇÃO DE FREQUÊNCIA (PANDAS)
# ==================================================
freq_pd = df.groupby("TIPO_VEICULO")["QTD_INFRACOES"].sum().reset_index().rename(columns={"QTD_INFRACOES":"TOTAL_INFRACOES"})  # ← soma por tipo de veículo
freq_pd = freq_pd.sort_values("TOTAL_INFRACOES", ascending=False)  # ← ordena do maior para o menor
print("\n📊 Distribuição de Frequência por Tipo de Veículo (top rows):")  # ← exibe cabeçalho
print(freq_pd.head(20).to_string(index=False))  # ← mostra os 20 primeiros resultados

# ==================================================
# ESTATÍSTICAS DESCRITIVAS
# ==================================================
media = freq_pd["TOTAL_INFRACOES"].mean()     # ← média
mediana = freq_pd["TOTAL_INFRACOES"].median() # ← mediana
desvio = freq_pd["TOTAL_INFRACOES"].std()     # ← desvio padrão
maximo = freq_pd["TOTAL_INFRACOES"].max()     # ← valor máximo
minimo = freq_pd["TOTAL_INFRACOES"].min()     # ← valor mínimo

# Exibe estatísticas no console
print("\n📈 Estatísticas Descritivas:")
print(f"Média: {media:.2f}")
print(f"Mediana: {mediana:.2f}")
print(f"Desvio padrão: {desvio:.2f}")
print(f"Máximo: {maximo}")
print(f"Mínimo: {minimo}\n")

# ==================================================
# AGRUPAMENTO POR REGIÃO (salvar CSV)
# ==================================================
regioes_agrupadas = (df_filtrado.groupby("REGIAO")["QTD_INFRACOES"].sum().reset_index())  # ← soma infrações por região
regioes_agrupadas = regioes_agrupadas.sort_values("REGIAO")  # ← ordena por nome da região

# ==================================================
# FUNÇÃO: GRÁFICO DE BARRAS (formatado)
# ==================================================
def grafico_barras(df_parte, titulo, cor):
    if df_parte.empty:  # ← evita erro se o dataframe estiver vazio
        print("A parte enviada para grafico_barras está vazia.")
        return
    plt.figure(figsize=(12, 6))  # ← define o tamanho do gráfico
    ax = plt.gca()  # ← obtém os eixos atuais
    barras = ax.bar(df_parte["REGIAO"], df_parte["QTD_INFRACOES"], color=cor)  # ← cria gráfico de barras
    ax.ticklabel_format(style='plain', axis='y')  # ← remove notação científica do eixo Y
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))  # ← formata eixo Y com separadores de milhar
    vmax = df_parte["QTD_INFRACOES"].max()  # ← maior valor (para posicionar texto)
    offset = max(vmax * 0.01, 1)  # ← distância entre o topo da barra e o texto
    for barra in barras:  # ← adiciona rótulos de valores acima das barras
        yval = barra.get_height()
        ax.text(barra.get_x() + barra.get_width()/2, yval + offset, f"{int(yval):,}".replace(",", "."), ha="center", va="bottom", fontsize=9)
    plt.title(titulo)  # ← define título
    plt.xlabel("Regiões (Municípios)")  # ← rótulo eixo X
    plt.ylabel("Total de Infrações")    # ← rótulo eixo Y
    plt.xticks(rotation=45, ha="right")  # ← rotaciona nomes das regiões
    plt.grid(axis="y", linestyle="--", alpha=0.6)  # ← adiciona grade horizontal
    plt.tight_layout()  # ← ajusta layout automaticamente
    plt.show()  # ← exibe o gráfico

# ==================================================
# GRÁFICOS A–J e K–Z — mostrar somente TOP 20 em cada grupo
# ==================================================
regioes_agrupadas["REGIAO"] = regioes_agrupadas["REGIAO"].astype(str)  # ← garante que REGIAO é string
regioes_agrupadas = regioes_agrupadas.rename(columns={"QTD_INFRACOES":"QTD_INFRACOES"})  # ← redundante, apenas mantém nome

# Divide as regiões em dois grupos (A–J e K–Z)
parte1 = regioes_agrupadas[regioes_agrupadas["REGIAO"].str[0].str.upper().between("A", "J")]  # ← grupo A–J
parte2 = regioes_agrupadas[regioes_agrupadas["REGIAO"].str[0].str.upper().between("K", "Z")]  # ← grupo K–Z

# Seleciona top 20 de cada grupo (ordenados)
parte1_top = parte1.sort_values("QTD_INFRACOES", ascending=False).head(20).sort_values("REGIAO").reset_index(drop=True)
parte2_top = parte2.sort_values("QTD_INFRACOES", ascending=False).head(20).sort_values("REGIAO").reset_index(drop=True)

# Gera gráficos de barras para ambos os grupos
grafico_barras(parte1_top, "Top 20 Regiões com Mais Infrações (A–J)", "mediumseagreen")
grafico_barras(parte2_top, "Top 20 Regiões com Mais Infrações (K–Z)", "lightseagreen")

# ==================================================
# REGRESSÃO LINEAR (TENDÊNCIA MENSAL)
# ==================================================
mensal = df.groupby("ANO_MES")["QTD_INFRACOES"].sum().reset_index().sort_values("ANO_MES")  # ← soma mensal de infrações
mensal = mensal.dropna(subset=["ANO_MES"])  # ← remove meses inválidos
mensal["mes_idx"] = range(1, len(mensal) + 1)  # ← cria índice sequencial de meses

# Calcula regressão linear se houver dados suficientes
if len(mensal) >= 2:
    a, b = np.polyfit(mensal["mes_idx"], mensal["QTD_INFRACOES"], 1)  # ← ajusta linha de tendência (y = a*x + b)
    mensal["pred"] = a * mensal["mes_idx"] + b  # ← calcula valores previstos
    print("📉 Regressão Linear (Tendência Mensal):")
    print(f"Inclinação (a): {a:.4f}")
    print(f"Intercepto (b): {b:.2f}\n")
    plt.scatter(mensal["mes_idx"], mensal["QTD_INFRACOES"], label="Observado")  # ← pontos observados
    plt.plot(mensal["mes_idx"], mensal["pred"], label="Tendência", color="red")  # ← linha da tendência
    plt.title("Tendência Mensal de Infrações")  # ← título
    plt.xlabel("Índice de mês (1..n)")          # ← eixo X
    plt.ylabel("Total de Infrações")            # ← eixo Y
    plt.legend()                               # ← legenda
    plt.tight_layout()                         # ← ajuste automático
    plt.show()                                 # ← exibe o gráfico
else:
    print("Dados mensais insuficientes para regressão.")  # ← caso com poucos dados

# ==================================================
# TABELA FORMATADA — TOP 10 regiões por tipo de veículo
# ==================================================
df_tabela = (df_filtrado.groupby(["REGIAO","TIPO_VEICULO"])["QTD_INFRACOES"].sum().unstack(fill_value=0))  # ← tabela pivot com tipos de veículo por região
df_tabela["TOTAL"] = df_tabela.sum(axis=1)  # ← adiciona coluna de total geral
df_tabela = df_tabela.sort_values("TOTAL", ascending=False).head(10)  # ← mantém as 10 regiões com mais infrações

# Renomeia colunas longas para versões curtas
colunas_renomeadas = {
    "CAMINHONETE": "CAMINHON.",
    "CAMIONETA": "CAMION.",
    "MICROONIBUS": "MICROÔN.",
    "MOTOCICLETA": "MOTO",
    "MOTONETA": "MOTON.",
    "UTILITARIO": "UTILIT."
}
df_tabela.rename(columns=colunas_renomeadas, inplace=True)  # ← aplica renomeação
cols = [c for c in df_tabela.columns if c != "TOTAL"] + ["TOTAL"]  # ← move TOTAL para o final
df_tabela = df_tabela[cols]

# Formata valores numéricos com pontos como separador de milhar
df_tabela_fmt = df_tabela.map(lambda x: f"{int(x):,}".replace(",", "."))  # ← substitui applymap() por map() para evitar aviso

# Mostra tabela formatada no notebook
print("\n📋 Top 10 Regiões com Mais Infrações (resumo):")
display(df_tabela_fmt)  # ← exibe tabela formatada no Colab
