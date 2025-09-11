import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql.functions import sum as spark_sum

df = pd.read_csv("infracoes_veiculos_2024.csv", sep=None, engine="python", encoding="utf-8")

#Limpeza
df.columns = [c.replace("\ufeff", "") for c in df.columns]

#Renomear
df = df.rename(columns={
    "ANOMES": "ANO_MES",
    "DESC_TIPO": "TIPO_VEICULO",
    "AIT": "QTD_INFRACOES"
})

# converter para numérico
df["ANO_MES"] = pd.to_numeric(df["ANO_MES"], errors="coerce")
df["QTD_INFRACOES"] = pd.to_numeric(df["QTD_INFRACOES"], errors="coerce")

# remover nulos
df = df.dropna(subset=["TIPO_VEICULO", "QTD_INFRACOES"])

#Distribuição de Frequência
spark = SparkSession.builder.appName("Infracoes2024").getOrCreate()

#criar DataFrame
spark_df = spark.createDataFrame(df[["TIPO_VEICULO", "QTD_INFRACOES"]])

#Revisão extra
spark_df = spark_df.na.drop(subset=["TIPO_VEICULO", "QTD_INFRACOES"])

freq_spark = (
    spark_df.groupBy("TIPO_VEICULO")
            .agg(spark_sum("QTD_INFRACOES").alias("TOTAL_INFRACOES"))
            .orderBy("TOTAL_INFRACOES", ascending=False)
)

print("Distribuição de Frequência:")
freq_spark.show(truncate=False)

#Regressão Linear Simples
mensal = df.groupby("ANO_MES")["QTD_INFRACOES"].sum().reset_index().sort_values("ANO_MES")
mensal["mes_idx"] = range(1, len(mensal) + 1)

a, b = np.polyfit(mensal["mes_idx"], mensal["QTD_INFRACOES"], 1)
mensal["pred"] = a * mensal["mes_idx"] + b

print("Regressão Linear (tendência mensal):")
print("Inclinação (a):", a)
print("Intercepto (b):", b, "\n")

#grafico
plt.scatter(mensal["mes_idx"], mensal["QTD_INFRACOES"], label="Observado")
plt.plot(mensal["mes_idx"], mensal["pred"], label="Regressão")
plt.title("Tendência Mensal de Infrações")
plt.xlabel("Mês (jan=1 ... dez=12)")
plt.ylabel("Total de Infrações (AIT)")
plt.legend()
plt.tight_layout()
plt.show()
