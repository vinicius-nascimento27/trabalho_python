# =========================================================
# PIPELINE INICIANTE — PANDAS + NUMPY + MATPLOTLIB + PYSPARK
# Dados: Infrações de Trânsito por Tipo de Veículo (DETRAN-RJ, 2024)
# Etapas: leitura -> limpeza/renomear -> frequência -> regressão (Pandas)
#         + frequência no PySpark
# =========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as spark_sum

# -------- 1) LEITURA (Pandas) --------
# coloque o nome exato do seu arquivo .csv aqui:
df = pd.read_csv("infracoes_veiculos_2024.csv", sep=None, engine="python", encoding="utf-8")

# remover BOM se existir no nome da primeira coluna
df.columns = [c.replace("\ufeff", "") for c in df.columns]

# -------- 2) RENOMEAR COLUNAS (didático para apresentação) --------
df = df.rename(columns={
    "ANOMES": "ANO_MES",                    # ex.: 202401 (jan/2024)
    "DESC_INF": "DESCRICAO_INFRACAO",       # texto da infração (CTB)
    "DESCR_MUNIC_INF": "MUNICIPIO",         # cidade
    "DESC_TIPO": "TIPO_VEICULO",            # tipo do veículo
    "LIT_PTO": "GRAVIDADE",                 # leve/média/grave/gravíssima
    "TIPO": "TIPO_REGISTRO",                # manual, radar etc.
    "AIT": "QTD_INFRACOES"                  # quantidade de autos
})

# -------- 3) LIMPEZA (tipagem + filtro 2024 + remoção de nulos) --------
df["ANO_MES"] = pd.to_numeric(df["ANO_MES"], errors="coerce")
df["QTD_INFRACOES"] = pd.to_numeric(df["QTD_INFRACOES"], errors="coerce")

df_2024 = df[(df["ANO_MES"] >= 202401) & (df["ANO_MES"] <= 202412)]
df_2024 = df_2024.dropna(subset=["TIPO_VEICULO", "QTD_INFRACOES"])

print("\nPrévia (dados limpos 2024):")
print(df_2024.head())

# -------- 4) DISTRIBUIÇÃO DE FREQUÊNCIA (Pandas) --------
freq = (
    df_2024.groupby("TIPO_VEICULO")["QTD_INFRACOES"]
           .sum()
           .sort_values(ascending=False)
)
print("\nDistribuição de Frequência (QTD_INFRACOES por TIPO_VEICULO):")
print(freq)

# gráfico de barras (abre uma janela do matplotlib)
plt.figure(figsize=(10,6))
plt.bar(freq.index, freq.values)
plt.title("Infrações por Tipo de Veículo - 2024")
plt.ylabel("Quantidade de Infrações (AIT)")
plt.xlabel("Tipo de Veículo")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# -------- 5) REGRESSÃO LINEAR SIMPLES (Pandas + NumPy) --------
mensal = (
    df_2024.groupby("ANO_MES")["QTD_INFRACOES"]
           .sum()
           .reset_index()
           .sort_values("ANO_MES")
)
mensal["mes_idx"] = range(1, len(mensal) + 1)

# y ~ a*x + b (ajuste da reta)
a, b = np.polyfit(mensal["mes_idx"], mensal["QTD_INFRACOES"], 1)
mensal["pred"] = a * mensal["mes_idx"] + b

print("\nRegressão Linear Simples (tendência mensal 2024)")
print("Inclinação (a):", a)
print("Intercepto (b):", b)

# (opcional) R² para comentar a qualidade do ajuste
y = mensal["QTD_INFRACOES"].to_numpy()
y_pred = mensal["pred"].to_numpy()
ss_res = ((y - y_pred)**2).sum()
ss_tot = ((y - y.mean())**2).sum()
r2 = 1 - ss_res/ss_tot if ss_tot != 0 else float("nan")
print("R²:", r2)

# gráfico: pontos observados + linha de regressão
plt.figure(figsize=(8,6))
plt.scatter(mensal["mes_idx"], mensal["QTD_INFRACOES"], label="Observado")
plt.plot(mensal["mes_idx"], mensal["pred"], label="Regressão Linear")
plt.title("Tendência Mensal de Infrações - 2024")
plt.xlabel("Mês (jan=1 ... dez=12)")
plt.ylabel("Total de Infrações (AIT)")
plt.legend()
plt.tight_layout()
plt.show()

# -------- 6) PYSPARK — Distribuição de Frequência --------
# (mesma análise no modelo Spark)
spark = SparkSession.builder.appName("Infracoes2024").getOrCreate()

# criar um Spark DataFrame a partir do Pandas já limpo
spark_df = spark.createDataFrame(df_2024[["ANO_MES", "TIPO_VEICULO", "QTD_INFRACOES"]])

# garantir tipos
spark_df = (spark_df
            .withColumn("ANO_MES", col("ANO_MES").cast("int"))
            .withColumn("QTD_INFRACOES", col("QTD_INFRACOES").cast("double")))

# groupBy + sum (igual ao Pandas)
freq_spark = (spark_df.groupBy("TIPO_VEICULO")
              .agg(spark_sum("QTD_INFRACOES").alias("TOTAL_INFRACOES"))
              .orderBy(col("TOTAL_INFRACOES").desc()))

print("\n[PySpark] Distribuição de Frequência (QTD_INFRACOES por TIPO_VEICULO):")
freq_spark.show(truncate=False)
