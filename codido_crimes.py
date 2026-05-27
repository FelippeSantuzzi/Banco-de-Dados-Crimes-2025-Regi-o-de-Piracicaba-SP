# ==========================================
# PROJETO DE MINERAÇÃO DE DADOS
# Criminalidade em cidades de São Paulo
# ==========================================

# ==========================================
# IMPORTAÇÃO DAS BIBLIOTECAS
# ==========================================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# ==========================================
# LEITURA DO ARQUIVO XLSX
# Fonte: SSP-SP / dados reais de criminalidade
# Nome do arquivo: arquivo.xlsx
# ==========================================

df = pd.read_excel('arquivo.xlsx')

# ==========================================
# INFORMAÇÕES DA BASE DE DADOS
# ==========================================

print("=" * 55)
print("INFORMAÇÕES DA BASE DE DADOS")
print("=" * 55)
print(f"Total de registros : {df.shape[0]}")
print(f"Total de atributos : {df.shape[1]}")
print(f"Cidades únicas     : {df['cidade'].nunique()}")
print(f"Tipos de crime     : {df['natureza'].nunique()}")
print(f"\nAtributos:\n{df.dtypes}")
print("\nPrimeiras linhas:")
print(df.head())

# ==========================================
# COLUNAS DOS MESES (1 a 12)
# ==========================================

meses = list(range(1, 13))

# ==========================================
# CRIAR TOTAL ANUAL POR REGISTRO
# ==========================================

df['total_anual'] = df[meses].sum(axis=1)

print("\nTOTAL ANUAL CALCULADO:")
print(df[['cidade', 'natureza', 'total_anual']].head())

# ==========================================
# REMOVER NATUREZAS REDUNDANTES
# Justificativa: o novo arquivo contém colunas
# que são totalizações de outras já presentes
# (ex: total_de_estupro = estupro + estupro_de_vulneravel).
# Incluí-las causaria duplicidade semântica e
# distorceria os clusters e as regras de associação.
# Também removemos contagens de vítimas pois
# repetem informação dos homicídios já presentes.
# ==========================================

naturezas_remover = [
    'numero_de_vitimas_em_homicidio_doloso',
    'numero_de_vitimas_em_homicidio_doloso_por_acidente_de_transito',
    'numero_de_vitimas_em_latrocinio',
    'total_de_estupro',        # soma de estupro + estupro_de_vulneravel
    'total_de_roubos_outros'   # soma dos tipos de roubo
]

df_filtrado = df[~df['natureza'].isin(naturezas_remover)].copy()

print(f"\nNaturezas utilizadas ({df_filtrado['natureza'].nunique()}):")
print(sorted(df_filtrado['natureza'].unique()))

# ==========================================
# PIVOT TABLE: cidades x tipos de crime
# ==========================================

pivot = df_filtrado.pivot_table(
    index='cidade',
    columns='natureza',
    values='total_anual',
    aggfunc='sum',
    fill_value=0
)

print("\nBASE PIVOTADA (cidades x crimes):")
print(f"Dimensão: {pivot.shape}")
print(pivot.head())

# ==========================================
# NORMALIZAÇÃO DOS DADOS
# Técnica: StandardScaler (média=0, desvio=1)
# Justificativa: os tipos de crime têm escalas
# muito diferentes — ex: furto_outros tem centenas
# de ocorrências enquanto latrocinio tem próximo a 0.
# Sem normalização, variáveis de alta escala
# dominariam o algoritmo K-Means.
# ==========================================

scaler = StandardScaler()
dados_normalizados = scaler.fit_transform(pivot)

# ==========================================
# MÉTODO DO COTOVELO
# Justificativa: identificar o k onde a queda
# de inércia começa a desacelerar ("cotovelo"),
# indicando o número ideal de clusters.
# ==========================================

inercias = []
K = range(2, 9)

for k in K:
    modelo = KMeans(n_clusters=k, random_state=42, n_init=10)
    modelo.fit(dados_normalizados)
    inercias.append(modelo.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K, inercias, marker='o', color='steelblue')
plt.title('Método do Cotovelo - Definição do Número de Clusters')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Inércia')
plt.grid(True)
plt.tight_layout()
plt.savefig('grafico_cotovelo.png', dpi=150)
plt.show()

# ==========================================
# SILHOUETTE SCORE PARA MÚLTIPLOS K
# Justificativa: complementa o cotovelo validando
# a coesão interna e separação entre clusters.
# Quanto mais próximo de 1, melhor a separação.
# ==========================================

print("\nSILHOUETTE SCORE POR NÚMERO DE CLUSTERS:")
for k in range(2, 9):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(dados_normalizados)
    score = silhouette_score(dados_normalizados, labels)
    print(f"  k={k} → Silhouette Score: {score:.4f}")

# ==========================================
# K-MEANS FINAL
# Justificativa: k=3 foi escolhido com base no
# cotovelo e no silhouette score. Também é
# interpretável: baixa, média e alta criminalidade.
# ==========================================

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(dados_normalizados)
pivot['cluster'] = clusters

# ==========================================
# SILHOUETTE SCORE FINAL
# ==========================================

silhouette = silhouette_score(dados_normalizados, clusters)
print(f"\nSILHOUETTE SCORE FINAL (k=3): {silhouette:.4f}")
print("Referência: >0.50 = boa separação | 0.25–0.50 = razoável")

# ==========================================
# PCA PARA VISUALIZAÇÃO 2D
# ==========================================

pca = PCA(n_components=2)
componentes = pca.fit_transform(dados_normalizados)

variancia = pca.explained_variance_ratio_
print(f"\nVariância explicada pelo PCA:")
print(f"  PC1: {variancia[0]*100:.1f}%")
print(f"  PC2: {variancia[1]*100:.1f}%")
print(f"  Total: {sum(variancia)*100:.1f}%")

# ==========================================
# GRÁFICO DOS CLUSTERS
# ==========================================

plt.figure(figsize=(12, 8))
sns.scatterplot(
    x=componentes[:, 0],
    y=componentes[:, 1],
    hue=clusters,
    palette='Set2',
    s=120
)

for i, cidade in enumerate(pivot.index):
    plt.text(
        componentes[i, 0] + 0.05,
        componentes[i, 1] + 0.05,
        cidade, fontsize=7
    )

plt.title('Clusters de Criminalidade - Cidades do Interior de SP')
plt.xlabel(f'PCA 1 ({variancia[0]*100:.1f}% variância)')
plt.ylabel(f'PCA 2 ({variancia[1]*100:.1f}% variância)')
plt.legend(title='Cluster')
plt.tight_layout()
plt.savefig('grafico_clusters.png', dpi=150)
plt.show()

# ==========================================
# ANÁLISE DOS CLUSTERS
# ==========================================

print("\nCIDADES POR CLUSTER:")
for c in sorted(pivot['cluster'].unique()):
    cidades = pivot[pivot['cluster'] == c].index.tolist()
    print(f"\n  Cluster {c} ({len(cidades)} cidades):")
    print(f"  {', '.join(cidades)}")

cols_crimes = [c for c in pivot.columns if c != 'cluster']

print("\nMÉDIA DOS CRIMES POR CLUSTER:")
analise_clusters = pivot.groupby('cluster')[cols_crimes].mean().round(2)
print(analise_clusters.T)

# ==========================================
# HEATMAP DE CORRELAÇÃO
# ==========================================

plt.figure(figsize=(14, 10))
sns.heatmap(
    pivot[cols_crimes].corr(),
    cmap='coolwarm',
    annot=False,
    linewidths=0.5
)
plt.title('Correlação entre Tipos de Crimes nas Cidades')
plt.tight_layout()
plt.savefig('heatmap_correlacao.png', dpi=150)
plt.show()

# ==========================================
# PREPARAÇÃO PARA APRIORI
# Objetivo: identificar quais tipos de crime
# tendem a coocorrer na mesma cidade
# ==========================================

df_binario = df_filtrado.copy()

# Binarizar: 1 se houve ao menos 1 ocorrência no ano
# Justificativa: Apriori exige dados binários (transacionais).
# A presença/ausência do crime por cidade é mais
# informativa do que a contagem absoluta aqui.
df_binario['ocorreu'] = np.where(df_binario['total_anual'] > 0, 1, 0)

basket = df_binario.pivot_table(
    index='cidade',
    columns='natureza',
    values='ocorreu',
    aggfunc='max',
    fill_value=0
)

# Converter para booleano (exigido pelo mlxtend)
basket = basket.astype(bool)

print("\nBASE BINÁRIA PARA APRIORI:")
print(f"Dimensão: {basket.shape}")
print(basket.head())

# ==========================================
# ALGORITMO APRIORI
# min_support=0.7 → crime deve aparecer em pelo
# menos 70% das cidades (~36 de 52).
# Justificativa: garante padrões robustos e
# representativos, evitando regras raras ou
# espúrias. Valor testado empiricamente com
# suportes 0.3, 0.5, 0.6, 0.7 e 0.8.
# max_len=3 → limita itemsets a 3 itens para
# gerar regras interpretáveis e em volume razoável.
# ==========================================

frequentes = apriori(
    basket,
    min_support=0.7,
    use_colnames=True,
    max_len=3
)

print(f"\nITEMSETS FREQUENTES ENCONTRADOS: {len(frequentes)}")
print(frequentes.sort_values('support', ascending=False).head(10))

# ==========================================
# REGRAS DE ASSOCIAÇÃO
# min_threshold=0.8 → confiança mínima de 80%
# Justificativa: assegura alta confiabilidade
# preditiva das regras. Regras com lift > 1
# indicam associação real (não aleatória).
# ==========================================

regras = association_rules(
    frequentes,
    metric='confidence',
    min_threshold=0.8,
    num_itemsets=len(frequentes)
)

# Manter apenas regras com lift real (> 1)
regras = regras[regras['lift'] > 1.0]

# Ordenar por Lift
regras = regras.sort_values(by='lift', ascending=False)

print(f"\nREGRAS DE ASSOCIAÇÃO GERADAS: {len(regras)}")
print("\nTOP 15 REGRAS (por Lift):\n")
print(
    regras[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
    .head(15)
    .to_string(index=False)
)

# ==========================================
# EXPORTAR RESULTADOS
# ==========================================

pivot.to_csv('clusters_resultado.csv', encoding='utf-8-sig')
regras.to_csv('regras_associacao.csv', encoding='utf-8-sig')

print("\n" + "=" * 55)
print("ARQUIVOS EXPORTADOS COM SUCESSO!")
print("=" * 55)
print("- clusters_resultado.csv")
print("- regras_associacao.csv")
print("- grafico_cotovelo.png")
print("- grafico_clusters.png")
print("- heatmap_correlacao.png")


# ==========================================
# ANÁLISE DE OUTLIERS — RIO CLARO
# ==========================================
# Durante a visualização dos clusters (PCA),
# Rio Claro apareceu destacado dentro do
# Cluster 1 (alta criminalidade), sugerindo
# que é um outlier mesmo dentro do seu grupo.
#
# Estratégia: calcular o Z-score de cada crime
# para Rio Claro. Z-score mede quantos desvios
# padrão um valor está acima da média geral.
# Z > 2 → outlier moderado
# Z > 3 → outlier severo
# ==========================================

print("\n" + "=" * 55)
print("ANÁLISE DE OUTLIERS — RIO CLARO")
print("=" * 55)

# Z-scores normalizados de todas as cidades
pivot_norm = pd.DataFrame(
    dados_normalizados,
    index=pivot.index,
    columns=cols_crimes
)

# Z-scores específicos do Rio Claro
z_rio = pivot_norm.loc['rio_claro'].sort_values(ascending=False)

print("\nZ-SCORE DO RIO CLARO POR TIPO DE CRIME:")
print("(Valores > 2 indicam outlier em relação às demais cidades)\n")
print(z_rio.round(2).to_string())

# Destacar crimes com Z > 2
outliers_rio = z_rio[z_rio > 2]
print(f"\nCrimes com Z-score > 2 (outliers severos): {len(outliers_rio)}")
for crime, z in outliers_rio.items():
    valor_real = pivot.loc['rio_claro', crime]
    media = pivot[crime].mean()
    print(f"  {crime}:")
    print(f"    Z-score = {z:.2f} | Rio Claro = {int(valor_real)} | Média geral = {media:.1f}")

# ==========================================
# RIO CLARO vs MÉDIA DO SEU CLUSTER
# ==========================================

cluster_rio = pivot.loc['rio_claro', 'cluster']
media_cluster = pivot[pivot['cluster'] == cluster_rio][cols_crimes].mean()
rio_valores = pivot.loc['rio_claro', cols_crimes]

comparacao = pd.DataFrame({
    'Rio Claro': rio_valores,
    'Média Cluster 1': media_cluster,
    'Ratio (RC / Cluster)': (rio_valores / media_cluster).round(2)
}).sort_values('Ratio (RC / Cluster)', ascending=False)

print(f"\nRIO CLARO vs MÉDIA DO CLUSTER {cluster_rio}:")
print(comparacao.round(2).to_string())

# ==========================================
# GRÁFICO — RIO CLARO vs MÉDIA DO CLUSTER
# ==========================================

top_crimes = comparacao.head(8).index.tolist()

x = range(len(top_crimes))
largura = 0.35

fig, ax = plt.subplots(figsize=(14, 6))

barras1 = ax.bar(
    [i - largura/2 for i in x],
    comparacao.loc[top_crimes, 'Rio Claro'],
    largura,
    label='Rio Claro',
    color='tomato'
)
barras2 = ax.bar(
    [i + largura/2 for i in x],
    comparacao.loc[top_crimes, 'Média Cluster 1'],
    largura,
    label='Média Cluster 1',
    color='steelblue'
)

ax.set_xticks(list(x))
ax.set_xticklabels(
    [c.replace('_', '\n') for c in top_crimes],
    fontsize=8
)
ax.set_title('Rio Claro vs Média do Cluster 1 — Top 8 Crimes')
ax.set_ylabel('Total de Ocorrências (ano)')
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('outlier_rio_claro.png', dpi=150)
plt.show()

# ==========================================
# GRÁFICO — Z-SCORES DO RIO CLARO
# ==========================================

plt.figure(figsize=(12, 5))

cores = ['tomato' if z > 2 else 'steelblue' for z in z_rio.values]

plt.barh(
    [c.replace('_', ' ') for c in z_rio.index],
    z_rio.values,
    color=cores
)

plt.axvline(x=2, color='red', linestyle='--', linewidth=1.2, label='Limiar outlier (Z=2)')
plt.axvline(x=0, color='gray', linestyle='-', linewidth=0.8)

plt.title('Z-Score de Rio Claro por Tipo de Crime\n(vermelho = outlier severo, Z > 2)')
plt.xlabel('Z-Score')
plt.legend()
plt.tight_layout()
plt.savefig('zscore_rio_claro.png', dpi=150)
plt.show()

print("\nArquivos gerados:")
print("- outlier_rio_claro.png  → Rio Claro vs Cluster 1")
print("- zscore_rio_claro.png   → Z-scores por crime")