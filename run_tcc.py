# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "yfinance",
#     "numpy",
#     "pandas",
#     "matplotlib",
#     "scipy",
#     "scikit-learn",
# ]
# ///
"""
TCC Lúcio – Simulação de Monte Carlo para Preço do Petróleo Brent
Gera gráficos em qualidade de publicação científica (300 DPI, PDF + PNG).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import yfinance as yf
import scipy.stats as stats
from sklearn.metrics import mean_squared_error

# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURAÇÃO – GRÁFICOS PARA ARTIGO CIENTÍFICO
# ═══════════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "legend.framealpha": 0.0,
    "legend.edgecolor": "none",
    "figure.figsize": (7, 4.5),
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": False,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.spines.left": True,
    "axes.spines.bottom": True,
    "axes.linewidth": 0.5,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
    "savefig.facecolor": "white",
    "lines.linewidth": 1.0,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
})

# Paleta – tons azuis suaves + contraste para zonas de risco
C = {
    "dark": "#1a1a2e",
    "light_blue": "#a8d4e6",   # preenchimento principal (95 %)
    "dark_blue": "#1b3a5c",   # zona de risco (cauda VaR)
    "line": "#0d1b2a",        # contorno da curva
    "blue": "#2980b9",
    "red": "#c0392b",
    "green": "#27ae60",
    "orange": "#e67e22",
    "bar1": "#5b9bd5",
    "bar2": "#ed7d64",
    "gray": "#b0b0b0",
    "sim_gray": "#c0c0c0",
}

OUTPUT_DIR = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "graficos")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUTPUT_DIR, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  -> {name}")


# ═══════════════════════════════════════════════════════════════════════════
#  1. DADOS
# ═══════════════════════════════════════════════════════════════════════════
print("Baixando dados do Brent (BZ=F) …")
raw = yf.download("BZ=F", start="2014-01-01", end="2024-12-30")
data = raw["Close"]
if isinstance(data, pd.DataFrame):
    data = data.squeeze()

log_returns = np.log(data / data.shift(1)).dropna()

# ═══════════════════════════════════════════════════════════════════════════
#  2. PRODUÇÃO / RECEITA PREVISTAS
# ═══════════════════════════════════════════════════════════════════════════
x_years = list(range(2025, 2035))
y_prod = [50.37, 69.715, 121.91, 139.446, 158.775,
          198.195, 180.31, 165.432, 145.27, 141.985]       # milhões bbl
y_receita = [3.526, 4.880, 8.534, 9.761, 11.114,
             13.874, 12.622, 11.580, 10.169, 9.939]        # USD bi

total_prod_uniao = np.sum(y_prod)  # milhões bbl

# ═══════════════════════════════════════════════════════════════════════════
#  3. SIMULAÇÃO DE MONTE CARLO – PREÇO
# ═══════════════════════════════════════════════════════════════════════════
print("Monte Carlo – preço …")
NUM_SIM = 5_000
NUM_DAYS = 252

last_price = float(data.iloc[-1])
mu = float(log_returns.mean())
sigma = float(log_returns.std())

np.random.seed(42)
sim_matrix = np.empty((NUM_DAYS, NUM_SIM))
for i in range(NUM_SIM):
    daily = np.random.normal(mu, sigma, NUM_DAYS)
    sim_matrix[:, i] = last_price * np.exp(np.cumsum(daily))

simulation_df = pd.DataFrame(sim_matrix)

# ── VaR – Preço ────────────────────────────────────────────────────────────
last_row = simulation_df.iloc[-1]
VaR_70 = float(np.percentile(last_row, 30))
VaR_80 = float(np.percentile(last_row, 20))
VaR_95 = float(np.percentile(last_row, 5))
VaR_99 = float(np.percentile(last_row, 1))
mean_last = float(last_row.mean())

print(f"  VaR 70 %: {VaR_70:.2f} | VaR 80 %: {VaR_80:.2f} | "
      f"VaR 95 %: {VaR_95:.2f} | VaR 99 %: {VaR_99:.2f}")
print(f"  Preço médio (dia 252): {mean_last:.2f}")

# ═══════════════════════════════════════════════════════════════════════════
#  4. SIMULAÇÃO DE MONTE CARLO – CARTEIRA DA UNIÃO
# ═══════════════════════════════════════════════════════════════════════════
print("Monte Carlo – carteira União …")
np.random.seed(42)
factor = total_prod_uniao / 1_000  # converter p/ bilhões
sim2 = np.empty((NUM_DAYS, NUM_SIM))
for i in range(NUM_SIM):
    daily = np.random.normal(mu, sigma, NUM_DAYS)
    sim2[:, i] = factor * last_price * np.exp(np.cumsum(daily))

simulation_df_2 = pd.DataFrame(sim2)

last_row_2 = simulation_df_2.iloc[-1]
VaR_80_U = float(np.percentile(last_row_2, 20))
VaR_95_U = float(np.percentile(last_row_2, 5))
VaR_99_U = float(np.percentile(last_row_2, 1))

print(f"  VaR 80 % União: {VaR_80_U:.2f} | "
      f"VaR 95 %: {VaR_95_U:.2f} | VaR 99 %: {VaR_99_U:.2f}")

# ═══════════════════════════════════════════════════════════════════════════
#  4b. SIMULAÇÃO DE MONTE CARLO – CARTEIRA DA UNIÃO (10 ANOS)
# ═══════════════════════════════════════════════════════════════════════════
print("Monte Carlo – carteira União 10 anos …")
NUM_DAYS_10Y = 252 * 10  # 2520 dias úteis
np.random.seed(99)
sim_10y = np.empty((NUM_DAYS_10Y, NUM_SIM))
for i in range(NUM_SIM):
    daily = np.random.normal(mu, sigma, NUM_DAYS_10Y)
    sim_10y[:, i] = factor * last_price * np.exp(np.cumsum(daily))

sim_10y_df = pd.DataFrame(sim_10y)
last_row_10y = sim_10y_df.iloc[-1]

VaR_80_10y = float(np.percentile(last_row_10y, 20))
VaR_95_10y = float(np.percentile(last_row_10y, 5))
VaR_99_10y = float(np.percentile(last_row_10y, 1))
mean_10y = float(last_row_10y.mean())
median_10y = float(np.median(last_row_10y))

print(f"  Média 10 a: {mean_10y:.2f} | Mediana: {median_10y:.2f}")
print(f"  VaR 80 %: {VaR_80_10y:.2f} | VaR 95 %: {VaR_95_10y:.2f} | "
      f"VaR 99 %: {VaR_99_10y:.2f}")

# ═══════════════════════════════════════════════════════════════════════════
#  5. BACKTEST
# ═══════════════════════════════════════════════════════════════════════════
real_tail = data.iloc[-NUM_DAYS:].values.flatten()
sim_mean = simulation_df.mean(axis=1).values[: len(real_tail)]
mse = mean_squared_error(real_tail, sim_mean)
print(f"MSE (backtest): {mse:.4f}")

# ═══════════════════════════════════════════════════════════════════════════
#  6. RECEITA MÉDIA
# ═══════════════════════════════════════════════════════════════════════════
media_preco = float(simulation_df.mean().mean())
receita_10a = sum(v * 1e6 for v in y_prod) * media_preco / 10
print(f"Preço médio simulado: {media_preco:.2f} USD/bbl")
print(f"Receita União (10 a): {receita_10a:,.0f} USD\n")

# ═══════════════════════════════════════════════════════════════════════════
#  GRÁFICOS
# ═══════════════════════════════════════════════════════════════════════════
print("Gerando gráficos …")


# ── Helpers para KDE com preenchimento ──────────────────────────────────────
def kde_curve(values, n_points=500):
    """Retorna (x, y) da KDE gaussiana suavizada."""
    kernel = stats.gaussian_kde(values)
    lo, hi = values.min(), values.max()
    margin = (hi - lo) * 0.08
    x = np.linspace(lo - margin, hi + margin, n_points)
    return x, kernel(x)


def plot_kde_var(ax, values, var_dict, xlabel, title):
    """
    Desenha a KDE preenchida no estilo da imagem de referência:
    - Área clara = zona de confiança
    - Área escura = cauda de risco (abaixo do menor VaR)
    - Linhas verticais + anotações para cada VaR
    """
    x, y = kde_curve(values)

    # Preenchimento claro (toda a curva)
    ax.fill_between(x, y, color=C["light_blue"], alpha=0.65)
    # Contorno
    ax.plot(x, y, color=C["line"], lw=1.4)

    # Cauda escura (abaixo do menor VaR)
    min_var_val = min(var_dict.values())
    ax.fill_between(x, y, where=(x <= min_var_val),
                    color=C["dark_blue"], alpha=0.75)

    # Linhas + anotações
    y_max = y.max()
    positions = [0.92, 0.82, 0.72]  # posições relativas das anotações
    colors_var = [C["blue"], C["red"], C["green"]]
    for i, (label, val) in enumerate(var_dict.items()):
        ax.axvline(val, color=colors_var[i % len(colors_var)],
                   ls="--", lw=1.0, alpha=0.85)
        ax.annotate(f"{label}: {val:.2f}",
                    xy=(val, y_max * positions[i]),
                    xytext=(val + (x.max() - x.min()) * 0.04,
                            y_max * positions[i]),
                    fontsize=8.5, color=colors_var[i % len(colors_var)],
                    arrowprops=dict(arrowstyle="->",
                                    color=colors_var[i % len(colors_var)],
                                    lw=0.8),
                    va="center")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Densidade de probabilidade")
    ax.set_title(title)
    ax.set_ylim(bottom=0)


# ═══════════════════════════════════════════════════════════════════════════

# ── Fig 01 – Preço histórico ────────────────────────────────────────────────
fig, ax = plt.subplots()
ax.plot(data.index, data.values, color=C["line"], linewidth=0.7)
ax.fill_between(data.index, 0, data.values, color=C["light_blue"], alpha=0.3)
ax.set_xlabel("Data")
ax.set_ylabel("Preço (USD/bbl)")
ax.set_title("Preço Histórico do Petróleo Brent (2014–2024)")
save(fig, "01_preco_historico_brent")

# ── Fig 02 – Retornos logarítmicos ─────────────────────────────────────────
fig, ax = plt.subplots()
ax.plot(log_returns.index, log_returns.values,
        color=C["line"], linewidth=0.35, alpha=0.85)
ax.axhline(0, color="black", linewidth=0.5)
ax.set_xlabel("Data")
ax.set_ylabel("Retorno logarítmico")
ax.set_title("Retornos Logarítmicos Diários do Brent (2014–2024)")
save(fig, "02_retornos_logaritmicos")

# ── Fig 03 – Produção anual estimada ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.5, 4.5))
bars = ax.bar(x_years, y_prod, color=C["bar1"], edgecolor="white", width=0.65)
for b, v in zip(bars, y_prod):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5,
            f"{v:.1f}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(x_years)
ax.set_xlabel("Ano")
ax.set_ylabel("Produção (milhões de barris)")
ax.set_title("Produção Anual Estimada da União (2025–2034)")
save(fig, "03_producao_anual_estimada")

# ── Fig 04 – Receita anual estimada ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.5, 4.5))
bars = ax.bar(x_years, y_receita,
              color=C["bar2"], edgecolor="white", width=0.65)
for b, v in zip(bars, y_receita):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.12,
            f"{v:.2f}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(x_years)
ax.set_xlabel("Ano")
ax.set_ylabel("Receita (USD bilhões)")
ax.set_title("Receita Anual Estimada da União (2025–2034)")
save(fig, "04_receita_anual_estimada")

# ── Fig 05 – Monte Carlo: Preço + VaR ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.5, 5))
sample_cols = np.random.choice(NUM_SIM, size=300, replace=False)
for c in sample_cols:
    ax.plot(simulation_df[c].values, color=C["sim_gray"],
            alpha=0.04, linewidth=0.3)
ax.plot(simulation_df.mean(axis=1).values,
        color=C["line"], linewidth=1.6, label="Média")
ax.axhline(VaR_80, color=C["blue"], ls="--",
           lw=1, label=f"VaR 80 %: {VaR_80:.2f}")
ax.axhline(VaR_95, color=C["red"], ls="--",
           lw=1, label=f"VaR 95 %: {VaR_95:.2f}")
ax.axhline(VaR_99, color=C["green"], ls="--",
           lw=1, label=f"VaR 99 %: {VaR_99:.2f}")
ax.set_xlabel("Dias de negociação")
ax.set_ylabel("Preço (USD/bbl)")
ax.set_title("Simulação de Monte Carlo – Preço do Petróleo Brent")
ax.legend(loc="upper left", frameon=True, facecolor="white",
          edgecolor="0.85", framealpha=0.95)
save(fig, "05_monte_carlo_preco")

# ── Fig 06 – Preço médio simulado ──────────────────────────────────────────
fig, ax = plt.subplots()
mean_price = simulation_df.mean(axis=1).values
ax.plot(mean_price, color=C["line"], linewidth=1.2)
ax.fill_between(range(len(mean_price)), mean_price.min() * 0.98,
                mean_price, color=C["light_blue"], alpha=0.3)
ax.set_xlabel("Dias de negociação")
ax.set_ylabel("Preço médio (USD/bbl)")
ax.set_title("Preço Médio Simulado do Petróleo Brent")
save(fig, "06_preco_medio_simulado")

# ── Fig 07 – Monte Carlo: Carteira União + VaR ──────────────────────────────
fig, ax = plt.subplots(figsize=(7.5, 5))
for c in sample_cols:
    ax.plot(simulation_df_2[c].values,
            color=C["sim_gray"], alpha=0.04, linewidth=0.3)
ax.plot(simulation_df_2.mean(axis=1).values,
        color=C["line"], linewidth=1.6, label="Média")
ax.axhline(VaR_80_U, color=C["blue"], ls="--",
           lw=1, label=f"VaR 80 %: {VaR_80_U:.2f}")
ax.axhline(VaR_95_U, color=C["red"], ls="--",
           lw=1, label=f"VaR 95 %: {VaR_95_U:.2f}")
ax.axhline(VaR_99_U, color=C["green"], ls="--",
           lw=1, label=f"VaR 99 %: {VaR_99_U:.2f}")
ax.set_xlabel("Dias de negociação")
ax.set_ylabel("Valor da carteira (USD bilhões)")
ax.set_title("Simulação de Monte Carlo – Carteira da União de Petróleo Brent")
ax.legend(loc="upper left", frameon=True, facecolor="white",
          edgecolor="0.85", framealpha=0.95)
save(fig, "07_monte_carlo_carteira_uniao")

# ── Fig 08 – Valor médio carteira União ─────────────────────────────────────
fig, ax = plt.subplots()
mean_u = simulation_df_2.mean(axis=1).values
ax.plot(mean_u, color=C["line"], linewidth=1.2)
ax.fill_between(range(len(mean_u)), mean_u.min() * 0.98,
                mean_u, color=C["light_blue"], alpha=0.3)
ax.set_xlabel("Dias de negociação")
ax.set_ylabel("Valor médio (USD bilhões)")
ax.set_title("Valor Médio da Carteira da União de Petróleo Brent")
save(fig, "08_valor_medio_carteira_uniao")

# ── Fig 09 – Previsão vs Real ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4.5))
future = pd.bdate_range(start=data.index[-1], periods=NUM_DAYS)
ax.plot(data.index, data.values, color=C["line"], lw=0.65, label="Preço real")
ax.plot(future, simulation_df.mean(axis=1).values,
        color=C["red"], lw=1.2, label="Simulação média")
q05 = simulation_df.quantile(0.05, axis=1).values
q95 = simulation_df.quantile(0.95, axis=1).values
ax.fill_between(future, q05, q95,
                color=C["light_blue"], alpha=0.4, label="IC 90 %")
ax.set_xlabel("Data")
ax.set_ylabel("Preço (USD/bbl)")
ax.set_title("Previsão Monte Carlo do Preço do Petróleo Brent")
ax.legend(frameon=True, facecolor="white", edgecolor="0.85", framealpha=0.95)
save(fig, "09_previsao_monte_carlo")

# ── Fig 10 – CDF + Histograma/KDE – Preço do Brent ────────────────────────────
fig, ax1 = plt.subplots(figsize=(8, 5))

# Eixo principal (esquerda) – histograma + KDE (probabilidade %)
_n10, _bins10, _ = ax1.hist(
    last_row, bins=50,
    weights=np.ones(len(last_row)) / len(last_row) * 100,
    color=C["light_blue"], edgecolor="white", alpha=0.45, zorder=0)
bin_w10 = _bins10[1] - _bins10[0]
x_kde, y_kde = kde_curve(last_row.values)
ax1.plot(x_kde, y_kde * bin_w10 * 100,
         color=C["blue"], lw=1.2, alpha=0.7, zorder=1)
ax1.set_ylabel("Probabilidade (%)")
ax1.set_ylim(bottom=0)

# Eixo secundário (direita) – CDF (% acumulado)
ax2 = ax1.twinx()
sorted_p = np.sort(last_row.values)
cdf = np.arange(1, len(sorted_p) + 1) / len(sorted_p)
ax2.plot(sorted_p, cdf, color=C["line"], lw=1.5, zorder=3)

# Marcadores de 20 em 20 % na curva CDF
for pct in range(20, 100, 20):
    val = float(np.percentile(last_row, pct))
    ax2.plot(val, pct / 100, marker="o", ms=5, color=C["line"], zorder=5)
    ax2.annotate(f"{pct} %",
                 xy=(val, pct / 100),
                 xytext=(5, 4), textcoords="offset points",
                 fontsize=7, color=C["line"])

p4775 = float(np.percentile(last_row, 47.75))
pct_100 = float((last_row <= 100).mean())
pct_acima_100 = (1 - pct_100) * 100

ax2.axvline(p4775, color=C["orange"], ls="--", lw=1.2, zorder=2)
ax2.axvline(100, color=C["green"], ls="--", lw=1.2, zorder=2)

# Área sombreada abaixo de cenário de referência
ax2.fill_between(sorted_p, 0, cdf, where=(sorted_p <= p4775),
                 color=C["orange"], alpha=0.15, zorder=1)

# Área sombreada acima de 100 USD/bbl
ax2.fill_between(sorted_p, cdf, 1.0, where=(sorted_p >= 100),
                 color=C["green"], alpha=0.10, zorder=1)

# Anotação – cenário de referência (~47,75 %)
ax2.annotate(
    f"Cenário de referência: {p4775:.2f} USD/bbl\n"
    f"47,75 % das simulações ficaram\nabaixo deste valor",
    xy=(p4775, 0.4775),
    xytext=(p4775 + 25, 0.18),
    fontsize=8.5, color=C["orange"], fontweight="bold",
    arrowprops=dict(arrowstyle="->", color=C["orange"], lw=1.0),
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C["orange"],
              alpha=0.95, zorder=10),
    va="center", zorder=10)

# Anotação – cenário atual (acima de 100 USD/bbl)
ax2.annotate(
    f"Cenário atual: 100 USD/bbl\n"
    f"{pct_acima_100:.1f} % das simulações ficaram\nacima deste valor",
    xy=(100, pct_100),
    xytext=(130, 0.88),
    fontsize=8.5, color=C["green"], fontweight="bold",
    arrowprops=dict(arrowstyle="->", color=C["green"], lw=1.0),
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C["green"],
              alpha=0.95, zorder=10),
    va="center", zorder=10)

ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
ax2.set_ylabel("Percentual acumulado")
ax1.set_xlabel("Preço simulado (USD/bbl)")
ax1.set_title("Distribuição Cumulativa dos Preços Simulados do Brent")
save(fig, "10_distribuicao_var_preco")

# ── Fig 11 – Histograma + KDE – Carteira União ──────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
bins_11 = np.arange(0, last_row_2.max() + 20, 20)  # intervalos de 20 bi
_n11, _bins11, _ = ax.hist(
    last_row_2, bins=bins_11,
    weights=np.ones(len(last_row_2)) / len(last_row_2) * 100,
    color=C["light_blue"], edgecolor="white", alpha=0.7, zorder=1)
bin_w11 = _bins11[1] - _bins11[0]
x_kde2, y_kde2 = kde_curve(last_row_2.values)
ax.plot(x_kde2, y_kde2 * bin_w11 * 100,
        color=C["line"], lw=1.8, zorder=2)
ax.axvline(VaR_80_U, color=C["blue"], ls="--", lw=1.2,
           label=f"VaR 80 %: {VaR_80_U:.2f}")
ax.axvline(VaR_95_U, color=C["red"], ls="--", lw=1.2,
           label=f"VaR 95 %: {VaR_95_U:.2f}")
ax.axvline(VaR_99_U, color=C["green"], ls="--", lw=1.2,
           label=f"VaR 99 %: {VaR_99_U:.2f}")
ax.set_xlabel("Valor da carteira (USD bilhões)")
ax.set_ylabel("Probabilidade (%)")
ax.set_title("Distribuição da Carteira da União – VaR")
ax.legend(frameon=True, facecolor="white", edgecolor="0.85", framealpha=0.95)
ax.set_ylim(bottom=0)
save(fig, "11_distribuicao_var_carteira_uniao")

# ── Fig 12 – Distribuição Normal teórica dos retornos com VaR ────────────────
fig, ax = plt.subplots(figsize=(8, 5))
x_norm = np.linspace(-4, 4, 500)
y_norm = stats.norm.pdf(x_norm)

# Preenchimento claro (intervalo de confiança 95 %)
ax.fill_between(x_norm, y_norm, color=C["light_blue"], alpha=0.65)
ax.plot(x_norm, y_norm, color=C["line"], lw=1.5)

# Cauda esquerda escura (5 %)
var_z = stats.norm.ppf(0.05)  # ≈ -1.645
ax.fill_between(x_norm, y_norm, where=(x_norm <= var_z),
                color=C["dark_blue"], alpha=0.8)

# Linha vertical VaR
ax.axvline(var_z, color="black", lw=1.2)
ax.annotate(f"VaR 5 % = {var_z:.2f}σ",
            xy=(var_z, stats.norm.pdf(var_z)),
            xytext=(var_z + 0.15, y_norm.max() * 0.85),
            fontsize=9.5, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="black", lw=1))

# Anotações de área
ax.text(-3.0, 0.06,
        "Área escura: 5 % da\nárea total sob a curva\n(zona de risco — VaR)",
        fontsize=8, ha="center", color=C["dark_blue"])
ax.text(1.2, 0.24,
        "Área clara: 95 % da\nárea total sob a curva\n(intervalo de confiança)",
        fontsize=8, ha="center", color="#3b6d8f")

ax.set_xlabel("Desvios-padrão (σ)")
ax.set_ylabel("Densidade de probabilidade")
ax.set_title("Distribuição Normal e Value at Risk (VaR 5 %)")
ax.set_ylim(bottom=0)
save(fig, "12_normal_var_teorico")

# ── Fig 13 – CDF + Histograma/KDE – Preços simulados ────────────────────────
fig, ax1 = plt.subplots(figsize=(8, 5))

# Eixo principal (esquerda) – histograma + KDE (probabilidade %)
_n13, _bins13, _ = ax1.hist(
    last_row, bins=50,
    weights=np.ones(len(last_row)) / len(last_row) * 100,
    color=C["light_blue"], edgecolor="white", alpha=0.45, zorder=0)
bin_w13 = _bins13[1] - _bins13[0]
x_kde13, y_kde13 = kde_curve(last_row.values)
ax1.plot(x_kde13, y_kde13 * bin_w13 * 100,
         color=C["blue"], lw=1.2, alpha=0.7, zorder=1)
ax1.set_ylabel("Probabilidade (%)")
ax1.set_ylim(bottom=0)

# Eixo secundário (direita) – CDF (% acumulado)
ax2 = ax1.twinx()
sorted_p = np.sort(last_row.values)
cdf = np.arange(1, len(sorted_p) + 1) / len(sorted_p)
ax2.plot(sorted_p, cdf, color=C["line"], lw=1.5, zorder=3)

# Marcadores de 20 em 20 % na curva CDF
for pct in range(20, 100, 20):
    val = float(np.percentile(last_row, pct))
    ax2.plot(val, pct / 100, marker="o", ms=5, color=C["line"], zorder=5)
    ax2.annotate(f"{pct} %",
                 xy=(val, pct / 100),
                 xytext=(5, 4), textcoords="offset points",
                 fontsize=7, color=C["line"])

p20 = float(np.percentile(last_row, 20))
p4775 = float(np.percentile(last_row, 47.75))

# Percentual acumulado em 100 USD/bbl
pct_100 = float((last_row <= 100).mean())  # fração abaixo de 100
pct_acima_100 = (1 - pct_100) * 100  # % acima de 100

ax2.axvline(p4775, color=C["orange"], ls="--", lw=1.2, zorder=2)
ax2.axvline(100, color=C["green"], ls="--", lw=1.2, zorder=2)

# Área sombreada abaixo do cenário de referência
ax2.fill_between(sorted_p, 0, cdf, where=(sorted_p <= p4775),
                 color=C["orange"], alpha=0.15, zorder=1)

# Área sombreada acima de 100 USD/bbl
ax2.fill_between(sorted_p, cdf, 1.0, where=(sorted_p >= 100),
                 color=C["green"], alpha=0.10, zorder=1)

# Anotação – cenário de referência (47,75 %)
ax2.annotate(
    f"Cenário de referência: {p4775:.2f} USD/bbl\n"
    f"47,75 % das simulações ficaram\nabaixo deste valor",
    xy=(p4775, 0.4775),
    xytext=(p4775 + 55, 0.18),
    fontsize=8.5, color=C["orange"], fontweight="bold",
    arrowprops=dict(arrowstyle="->", color=C["orange"], lw=1.0),
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C["orange"],
              alpha=0.95, zorder=10),
    va="center", zorder=10)

# Anotação – cenário otimista (acima de 100 USD/bbl)
ax2.annotate(
    f"Cenário atual: 100 USD/bbl\n"
    f"{pct_acima_100:.1f} % das simulações ficaram\nacima deste valor",
    xy=(100, pct_100),
    xytext=(155, 0.88),
    fontsize=8.5, color=C["green"], fontweight="bold",
    arrowprops=dict(arrowstyle="->", color=C["green"], lw=1.0),
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C["green"],
              alpha=0.95, zorder=10),
    va="center", zorder=10)

ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
ax2.set_ylabel("Percentual acumulado")
ax1.set_xlabel("Preço simulado (USD/bbl)")
ax1.set_title("Distribuição Cumulativa dos Preços Simulados")
save(fig, "13_cdf_precos_simulados")

# ── Fig 14 – Histograma + KDE – Carteira União 10 anos ──────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
_n14, _bins14, _ = ax.hist(
    last_row_10y, bins=60,
    weights=np.ones(len(last_row_10y)) / len(last_row_10y) * 100,
    color=C["light_blue"], edgecolor="white", alpha=0.7, zorder=1)
bin_w14 = _bins14[1] - _bins14[0]
x_kde10, y_kde10 = kde_curve(last_row_10y.values)
ax.plot(x_kde10, y_kde10 * bin_w14 * 100,
        color=C["line"], lw=1.8, zorder=2)
ax.axvline(VaR_80_10y, color=C["blue"], ls="--", lw=1.2,
           label=f"VaR 80 %: {VaR_80_10y:.2f}")
ax.axvline(VaR_95_10y, color=C["red"], ls="--", lw=1.2,
           label=f"VaR 95 %: {VaR_95_10y:.2f}")
ax.axvline(VaR_99_10y, color=C["green"], ls="--", lw=1.2,
           label=f"VaR 99 %: {VaR_99_10y:.2f}")
ax.axvline(mean_10y, color=C["orange"], ls="-", lw=1.5,
           label=f"Média: {mean_10y:.2f}")
ax.axvline(median_10y, color=C["dark"], ls=":", lw=1.5,
           label=f"Mediana: {median_10y:.2f}")
ax.set_xlabel("Saldo da carteira da União (USD bilhões)")
ax.set_ylabel("Probabilidade (%)")
ax.set_title("Distribuição do Saldo da Carteira da União em 10 Anos")
ax.legend(frameon=True, facecolor="white", edgecolor="0.85", framealpha=0.95)
ax.set_ylim(bottom=0)
save(fig, "14_carteira_uniao_10_anos")

# ── Fig 15 – CDF + Histograma/KDE – Carteira União (1 ano) ─────────────────────
fig, ax1 = plt.subplots(figsize=(8, 5))

# Eixo principal (esquerda) – histograma + KDE (probabilidade %)
_n15, _bins15, _ = ax1.hist(
    last_row_2, bins=50,
    weights=np.ones(len(last_row_2)) / len(last_row_2) * 100,
    color=C["light_blue"], edgecolor="white", alpha=0.45, zorder=0)
bin_w15 = _bins15[1] - _bins15[0]
x_kde15, y_kde15 = kde_curve(last_row_2.values)
ax1.plot(x_kde15, y_kde15 * bin_w15 * 100,
         color=C["blue"], lw=1.2, alpha=0.7, zorder=1)
ax1.set_ylabel("Probabilidade (%)")
ax1.set_ylim(bottom=0)

# Eixo secundário (direita) – CDF (% acumulado)
ax2 = ax1.twinx()
sorted_c = np.sort(last_row_2.values)
cdf_c = np.arange(1, len(sorted_c) + 1) / len(sorted_c)
ax2.plot(sorted_c, cdf_c, color=C["line"], lw=1.5, zorder=3)

# Marcadores de 20 em 20 % na curva CDF
for pct in range(20, 100, 20):
    val = float(np.percentile(last_row_2, pct))
    ax2.plot(val, pct / 100, marker="o", ms=5, color=C["line"], zorder=5)
    ax2.annotate(f"{pct} %",
                 xy=(val, pct / 100),
                 xytext=(5, 4), textcoords="offset points",
                 fontsize=7, color=C["line"])

# Cenários da carteira
c_ref = float(np.percentile(last_row_2, 47.75))  # ~70 USD/bbl → ~50 bi
pct_c_ref = float((last_row_2 <= c_ref).mean())
c_otim = float(np.percentile(last_row_2, 80))  # cenário otimista (~80 bi)
pct_acima_c_otim = (1 - float((last_row_2 <= c_otim).mean())) * 100

ax2.axvline(c_ref, color=C["orange"], ls="--", lw=1.2, zorder=2)
ax2.axvline(c_otim, color=C["green"], ls="--", lw=1.2, zorder=2)

# Área sombreada abaixo de cenário de referência
ax2.fill_between(sorted_c, 0, cdf_c, where=(sorted_c <= c_ref),
                 color=C["orange"], alpha=0.15, zorder=1)

# Área sombreada acima de cenário otimista
ax2.fill_between(sorted_c, cdf_c, 1.0, where=(sorted_c >= c_otim),
                 color=C["green"], alpha=0.10, zorder=1)

# Anotação – cenário de referência
ax2.annotate(
    f"Cenário de referência: {c_ref:.2f} USD bilhões\n"
    f"{pct_c_ref*100:.1f} % das simulações ficaram\nabaixo deste valor",
    xy=(c_ref, pct_c_ref),
    xytext=(c_ref - 20, 0.25),
    fontsize=8.5, color=C["orange"], fontweight="bold",
    arrowprops=dict(arrowstyle="->", color=C["orange"], lw=1.0),
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C["orange"],
              alpha=0.95, zorder=10),
    va="center", zorder=10)

# Anotação – cenário otimista
ax2.annotate(
    f"Cenário otimista: {c_otim:.2f} USD bilhões\n"
    f"{pct_acima_c_otim:.1f} % das simulações ficaram\nacima deste valor",
    xy=(c_otim, float((last_row_2 <= c_otim).mean())),
    xytext=(c_otim + 25, 0.88),
    fontsize=8.5, color=C["green"], fontweight="bold",
    arrowprops=dict(arrowstyle="->", color=C["green"], lw=1.0),
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C["green"],
              alpha=0.95, zorder=10),
    va="center", zorder=10)

ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
ax2.set_ylabel("Percentual acumulado")
ax1.set_xlabel("Saldo da carteira da União (USD bilhões)")
ax1.set_title("Distribuição Cumulativa da Carteira da União – 1 Ano")
save(fig, "15_cdf_carteira_uniao")

print(f"\nConcluído – {len(os.listdir(OUTPUT_DIR))} arquivos em: {OUTPUT_DIR}")
