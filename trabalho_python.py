import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql.functions import sum as spark_sum

df = pd.read_csv("infracoes_veiculos_2024.csv", sep=None, engine="python", encoding="utf-8")

#LIMPEZA DE DADOS

#Remover espaços em branco
df.columns = [c.replace("\ufeff", "") for c in df.columns]

#Renomear colunas
df = df.rename(columns={
    "ANOMES": "ANO_MES",
    "DESC_TIPO": "TIPO_VEICULO",
    "AIT": "QTD_INFRACOES"
})

# Converter para numérico
df["ANO_MES"] = pd.to_numeric(df["ANO_MES"], errors="coerce")
df["QTD_INFRACOES"] = pd.to_numeric(df["QTD_INFRACOES"], errors="coerce")

# Remover valores nulos
df = df.dropna(subset=["TIPO_VEICULO", "QTD_INFRACOES"])


#DISTRIBUIÇÃO DE FREQUÊNCIA

#Criar sessão spark
spark = SparkSession.builder.appName("Infracoes2024").getOrCreate()

#Criar DataFrame
spark_df = spark.createDataFrame(df[["TIPO_VEICULO", "QTD_INFRACOES"]])

#Agrupar, somar e ordenar
freq_spark = (
    spark_df.groupBy("TIPO_VEICULO") #agrupa todas as linhas pelo tipo de veículo
            .agg(spark_sum("QTD_INFRACOES").alias("TOTAL_INFRACOES")) #soma todas as infrações de cada tipo
            .orderBy("TOTAL_INFRACOES", ascending=False) #ordena do tipo que tem mais infrações para o que tem menos
)

print("Distribuição de Frequência:")
freq_spark.show()

#REGRESSÃO LINEAR
mensal = df.groupby("ANO_MES")["QTD_INFRACOES"].sum().reset_index().sort_values("ANO_MES") #Agrupa por mês e soma as infrações
mensal["mes_idx"] = range(1, len(mensal) + 1) #cria uma numeração sequencial dos meses

a, b = np.polyfit(mensal["mes_idx"], mensal["QTD_INFRACOES"], 1) #Ajusta reta y = a*x + b
mensal["pred"] = a * mensal["mes_idx"] + b #valores previstos pela reta

print("Regressão Linear (tendência mensal):") # mostra os coeficientes da regressão
print("Inclinação (a):", a)
print("Intercepto (b):", b, "\n") #ponto inicial da linha

#grafico
plt.scatter(mensal["mes_idx"], mensal["QTD_INFRACOES"], label="Observado")
plt.plot(mensal["mes_idx"], mensal["pred"], label="Regressão")
plt.title("Tendência Mensal de Infrações")
plt.xlabel("Mês (jan=1 ... dez=12)")
plt.ylabel("Total de Infrações (AIT)")
plt.legend()
plt.tight_layout()
plt.show()
