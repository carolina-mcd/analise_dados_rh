# ============================================================
#   PROJETO DE ANÁLISE DE DADOS — RH / PESSOAS
#   Guia completo para iniciantes com Python
# ============================================================
# Execute cada seção no Jupyter Notebook ou VS Code (Jupyter)
# Instale as libs com: pip install pandas matplotlib seaborn
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# Estilo visual
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (10, 5)
plt.rcParams["figure.dpi"] = 120


# ─────────────────────────────────────────────
# SEÇÃO 1 — CARREGANDO E ENTENDENDO OS DADOS
# ─────────────────────────────────────────────
print("=" * 55)
print("  SEÇÃO 1 — CARREGANDO OS DADOS")
print("=" * 55)

df = pd.read_csv("rh_dataset.csv")

print("\n📋 Primeiras 5 linhas do dataset:")
print(df.head())

print(f"\n📐 Dimensões: {df.shape[0]} linhas × {df.shape[1]} colunas")

print("\n🔎 Tipos de dados:")
print(df.dtypes)

print("\n❓ Valores ausentes por coluna:")
print(df.isnull().sum())

# Converter datas para o tipo correto
df["data_admissao"] = pd.to_datetime(df["data_admissao"])
df["data_saida"] = pd.to_datetime(df["data_saida"], errors="coerce")

# Calcular tempo de empresa (em anos)
hoje = pd.Timestamp("2025-12-31")
df["tempo_empresa_anos"] = (
    (df["data_saida"].fillna(hoje) - df["data_admissao"]).dt.days / 365
).round(1)

print("\n✅ Colunas de data convertidas e 'tempo_empresa_anos' criada!")


# ─────────────────────────────────────────────
# SEÇÃO 2 — ANÁLISE EXPLORATÓRIA (EDA)
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("  SEÇÃO 2 — ANÁLISE EXPLORATÓRIA")
print("=" * 55)

print("\n📊 Estatísticas descritivas — colunas numéricas:")
print(df[["salario_mensal", "nota_desempenho", "horas_treinamento_ano",
          "satisfacao_trabalho", "tempo_empresa_anos"]].describe().round(2))

print("\n👥 Distribuição por Departamento:")
print(df["departamento"].value_counts())

print("\n🔄 Taxa de Turnover geral:")
total = len(df)
inativos = df[df["ativo"] == "Não"].shape[0]
turnover = round((inativos / total) * 100, 1)
print(f"   {inativos} desligamentos de {total} funcionários = {turnover}% de turnover")

print("\n📈 Salário médio por Departamento:")
print(
    df.groupby("departamento")["salario_mensal"]
    .mean()
    .sort_values(ascending=False)
    .apply(lambda x: f"R$ {x:,.2f}")
)


# ─────────────────────────────────────────────
# SEÇÃO 3 — VISUALIZAÇÕES
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("  SEÇÃO 3 — CRIANDO GRÁFICOS")
print("=" * 55)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Dashboard de Análise de RH", fontsize=16, fontweight="bold", y=1.01)

# --- Gráfico 1: Headcount por Departamento ---
dept_count = df[df["ativo"] == "Sim"]["departamento"].value_counts()
axes[0, 0].barh(dept_count.index, dept_count.values, color=sns.color_palette("muted", len(dept_count)))
axes[0, 0].set_title("Headcount Ativo por Departamento")
axes[0, 0].set_xlabel("Nº de funcionários")

# --- Gráfico 2: Distribuição de Salários ---
axes[0, 1].hist(df["salario_mensal"], bins=20, color="#4C72B0", edgecolor="white")
axes[0, 1].set_title("Distribuição de Salários")
axes[0, 1].set_xlabel("Salário Mensal (R$)")
axes[0, 1].set_ylabel("Frequência")
axes[0, 1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"R${x/1000:.0f}k"))

# --- Gráfico 3: Turnover por Departamento ---
turnover_dept = (
    df.groupby("departamento")
    .apply(lambda x: (x["ativo"] == "Não").sum() / len(x) * 100)
    .sort_values(ascending=False)
)
axes[0, 2].bar(turnover_dept.index, turnover_dept.values,
               color=sns.color_palette("Reds_r", len(turnover_dept)))
axes[0, 2].set_title("Taxa de Turnover por Departamento (%)")
axes[0, 2].set_ylabel("%")
axes[0, 2].tick_params(axis="x", rotation=30)

# --- Gráfico 4: Satisfação × Nota de Desempenho ---
axes[1, 0].scatter(
    df["satisfacao_trabalho"], df["nota_desempenho"],
    alpha=0.4, c=df["salario_mensal"], cmap="YlOrRd", edgecolors="none"
)
axes[1, 0].set_title("Satisfação × Desempenho")
axes[1, 0].set_xlabel("Satisfação (1-5)")
axes[1, 0].set_ylabel("Nota de Desempenho")

# --- Gráfico 5: Salário médio por Nível ---
sal_nivel = df.groupby("nivel")["salario_mensal"].mean().reindex(
    ["Júnior", "Pleno", "Sênior", "Especialista"])
colors = ["#90CAF9", "#42A5F5", "#1565C0", "#0D47A1"]
axes[1, 1].bar(sal_nivel.index, sal_nivel.values, color=colors)
axes[1, 1].set_title("Salário Médio por Nível")
axes[1, 1].set_ylabel("R$")
axes[1, 1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"R${x/1000:.0f}k"))

# --- Gráfico 6: Distribuição por Gênero ---
genero_count = df[df["ativo"] == "Sim"]["genero"].value_counts()
axes[1, 2].pie(
    genero_count.values,
    labels=genero_count.index,
    autopct="%1.1f%%",
    colors=sns.color_palette("pastel"),
    startangle=140,
)
axes[1, 2].set_title("Diversidade de Gênero (Ativos)")

plt.tight_layout()
plt.savefig("dashboard_rh.png", bbox_inches="tight", dpi=150)
plt.show()
print("\n✅ Dashboard salvo como 'dashboard_rh.png'")


# ─────────────────────────────────────────────
# SEÇÃO 4 — ANÁLISES AVANÇADAS
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("  SEÇÃO 4 — ANÁLISES AVANÇADAS")
print("=" * 55)

# Perfil de quem pede demissão vs. é desligado
print("\n🔍 Perfil médio: Pedido de Demissão vs. Desligamento")
saidas = df[df["ativo"] == "Não"].copy()
perfil = saidas.groupby("motivo_saida")[
    ["salario_mensal", "nota_desempenho", "satisfacao_trabalho", "tempo_empresa_anos"]
].mean().round(2)
print(perfil)

# Correlação entre variáveis numéricas
print("\n📐 Matriz de Correlação (variáveis numéricas):")
colunas_corr = ["salario_mensal", "nota_desempenho", "satisfacao_trabalho",
                "horas_treinamento_ano", "tempo_empresa_anos", "num_promacoes"]
corr_matrix = df[colunas_corr].corr().round(2)
print(corr_matrix)

# Heatmap de correlação
plt.figure(figsize=(9, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0,
            linewidths=0.5, fmt=".2f")
plt.title("Mapa de Correlação — Variáveis de RH", fontweight="bold")
plt.tight_layout()
plt.savefig("correlacao_rh.png", bbox_inches="tight", dpi=150)
plt.show()
print("\n✅ Mapa de correlação salvo como 'correlacao_rh.png'")


# ─────────────────────────────────────────────
# SEÇÃO 5 — PERGUNTAS PARA PRATICAR
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("  SEÇÃO 5 — DESAFIOS PARA VOCÊ RESOLVER!")
print("=" * 55)
desafios = [
    "1. Qual departamento tem a maior média de horas de treinamento?",
    "2. Funcionários com mais promoções têm notas de desempenho maiores?",
    "3. Existe diferença salarial entre gêneros? Calcule a média por gênero.",
    "4. Qual cidade concentra mais funcionários ativos?",
    "5. Filtre apenas funcionários Sêniores e compare o turnover entre departamentos.",
    "6. Crie um gráfico de linha mostrando admissões por ano.",
    "7. Qual é o perfil (nível, depto, satisfação) de quem tem nota > 9.0?",
]
for d in desafios:
    print(f"   {d}")

print("\n🚀 Boa análise! Qualquer dúvida, estude a documentação do Pandas:")
print("   https://pandas.pydata.org/docs/")
