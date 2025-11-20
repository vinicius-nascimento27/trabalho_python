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
try:
    df = pd.read_csv(csv_file, sep=None, engine="python", encoding="utf-8", on_bad_lines="skip")
except Exception:
    df = pd.read_csv(csv_file, sep=None, engine="python", encoding="latin1", on_bad_lines="skip")

df.columns = [c.replace("\ufeff", "").strip() for c in df.columns]

df = df.rename(columns={
    "ANOMES": "ANO_MES",
    "DESC_TIPO": "TIPO_VEICULO",
    "AIT": "QTD_INFRACOES",
    "DESCR_MUNIC_INF": "REGIAO"
})

df["ANO_MES"] = pd.to_numeric(df.get("ANO_MES"), errors="coerce")
df["QTD_INFRACOES"] = pd.to_numeric(df.get("QTD_INFRACOES"), errors="coerce")
df = df.dropna(subset=["TIPO_VEICULO", "QTD_INFRACOES", "REGIAO"])

# ==================================================
# REMOVER TIPOS DE VEÍCULO INDIVIDUAIS
# ==================================================
tipos_a_remover = [
    "TRATOR DE RODAS", "MOTOR CASA", "TRICICLO", "CICLOMOTOR",
    "REBOQUE", "CAMINHAO TRATOR", "NAO INFORMADO", "SEMI-REBOQUE"
]
df = df[~df["TIPO_VEICULO"].isin(tipos_a_remover)]

# ==================================================
# FILTRAR REGIÕES COM PELO MENOS 2000 INFRAÇÕES
# ==================================================
total_infracoes_regioes = df.groupby("REGIAO")["QTD_INFRACOES"].sum()
regioes_com_2000_ou_mais = total_infracoes_regioes[total_infracoes_regioes >= 2000].index
df_filtrado = df[df["REGIAO"].isin(regioes_com_2000_ou_mais)].copy()

# ===========================================================
# DISTRIBUIÇÃO DE FREQUÊNCIA (PANDAS)
# ===========================================================

freq_pd = df.groupby("TIPO_VEICULO")["QTD_INFRACOES"] \
            .sum() \
            .reset_index() \
            .rename(columns={"QTD_INFRACOES": "TOTAL_INFRACOES"})

freq_pd = freq_pd.sort_values("TOTAL_INFRACOES", ascending=False)

# ✅ FORMATAÇÃO COM PONTOS COMO SEPARADOR DE MILHAR
freq_pd["TOTAL_INFRACOES"] = freq_pd["TOTAL_INFRACOES"] \
    .apply(lambda x: f"{x:,.0f}".replace(",", "."))

print("\n📊 Distribuição de Frequência por Tipo de Veículo (top rows):")
print(freq_pd.head(20).to_string(index=False))

# ==================================================
# ESTATÍSTICAS DESCRITIVAS
# ==================================================
media = freq_pd["TOTAL_INFRACOES"].str.replace(".", "").astype(int).mean()
mediana = freq_pd["TOTAL_INFRACOES"].str.replace(".", "").astype(int).median()
desvio = freq_pd["TOTAL_INFRACOES"].str.replace(".", "").astype(int).std()
maximo = freq_pd["TOTAL_INFRACOES"].str.replace(".", "").astype(int).max()
minimo = freq_pd["TOTAL_INFRACOES"].str.replace(".", "").astype(int).min()

print("\n📈 Estatísticas Descritivas:")
print(f"Média: {media:.2f}")
print(f"Mediana: {mediana:.2f}")
print(f"Desvio padrão: {desvio:.2f}")
print(f"Máximo: {maximo}")
print(f"Mínimo: {minimo}\n")

# ==================================================
# AGRUPAMENTO POR REGIÃO (salvar CSV)
# ==================================================
regioes_agrupadas = df_filtrado.groupby("REGIAO")["QTD_INFRACOES"].sum().reset_index()
regioes_agrupadas = regioes_agrupadas.sort_values("REGIAO")

# ==================================================
# FUNÇÃO: GRÁFICO DE BARRAS
# ==================================================
def grafico_barras(df_parte, titulo, cor):
    if df_parte.empty:
        print("A parte enviada para grafico_barras está vazia.")
        return

    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    barras = ax.bar(df_parte["REGIAO"], df_parte["QTD_INFRACOES"], color=cor)

    ax.ticklabel_format(style='plain', axis='y')
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))

    vmax = df_parte["QTD_INFRACOES"].max()
    offset = max(vmax * 0.01, 1)

    for barra in barras:
        yval = barra.get_height()
        ax.text(
            barra.get_x() + barra.get_width()/2,
            yval + offset,
            f"{int(yval):,}".replace(",", "."),
            ha="center", va="bottom", fontsize=9
        )

    plt.title(titulo)
    plt.xlabel("Regiões (Municípios)")
    plt.ylabel("Total de Infrações")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

# ==================================================
# GRÁFICOS A–J e K–Z — TOP 20
# ==================================================
regioes_agrupadas["REGIAO"] = regioes_agrupadas["REGIAO"].astype(str)

parte1 = regioes_agrupadas[regioes_agrupadas["REGIAO"].str[0].str.upper().between("A", "J")]
parte2 = regioes_agrupadas[regioes_agrupadas["REGIAO"].str[0].str.upper().between("K", "Z")]

parte1_top = parte1.sort_values("QTD_INFRACOES", ascending=False).head(20).sort_values("REGIAO").reset_index(drop=True)
parte2_top = parte2.sort_values("QTD_INFRACOES", ascending=False).head(20).sort_values("REGIAO").reset_index(drop=True)

grafico_barras(parte1_top, "Top 20 Regiões com Mais Infrações (A–J)", "mediumseagreen")
grafico_barras(parte2_top, "Top 20 Regiões com Mais Infrações (K–Z)", "lightseagreen")

# ==================================================
# REGRESSÃO LINEAR (TENDÊNCIA MENSAL)
# ==================================================
mensal = df.groupby("ANO_MES")["QTD_INFRACOES"].sum().reset_index().sort_values("ANO_MES")
mensal = mensal.dropna(subset=["ANO_MES"])
mensal["mes_idx"] = range(1, len(mensal) + 1)

if len(mensal) >= 2:
    a, b = np.polyfit(mensal["mes_idx"], mensal["QTD_INFRACOES"], 1)
    mensal["pred"] = a * mensal["mes_idx"] + b

    print("📉 Regressão Linear (Tendência Mensal):")
    print(f"Inclinação (a): {a:.4f}")
    print(f"Intercepto (b): {b:.2f}\n")

    plt.scatter(mensal["mes_idx"], mensal["QTD_INFRACOES"], label="Observado")
    plt.plot(mensal["mes_idx"], mensal["pred"], label="Tendência", color="red")
    plt.title("Tendência Mensal de Infrações")
    plt.xlabel("Índice de mês")
    plt.ylabel("Total de Infrações")
    plt.legend()
    plt.tight_layout()
    plt.show()
else:
    print("Dados mensais insuficientes para regressão.")

# ==================================================
# TABELA TOP 10 REGIÕES
# ==================================================
df_tabela = df_filtrado.groupby(["REGIAO","TIPO_VEICULO"])["QTD_INFRACOES"].sum().unstack(fill_value=0)
df_tabela["TOTAL"] = df_tabela.sum(axis=1)
df_tabela = df_tabela.sort_values("TOTAL", ascending=False).head(10)

colunas_renomeadas = {
    "CAMINHONETE": "CAMINHON.",
    "CAMIONETA": "CAMION.",
    "MICROONIBUS": "MICROÔN.",
    "MOTOCICLETA": "MOTO",
    "MOTONETA": "MOTON.",
    "UTILITARIO": "UTILIT."
}
df_tabela.rename(columns=colunas_renomeadas, inplace=True)
cols = [c for c in df_tabela.columns if c != "TOTAL"] + ["TOTAL"]
df_tabela = df_tabela[cols]

df_tabela_fmt = df_tabela.map(lambda x: f"{int(x):,}".replace(",", "."))

print("\n📋 Top 10 Regiões com Mais Infrações (resumo):")
display(df_tabela_fmt)
