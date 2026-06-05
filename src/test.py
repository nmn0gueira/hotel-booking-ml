"""
Teste isolado de generate_ikmeans_bootstrap_stability.
Usa dados sintéticos — não precisa do dataset real nem de labels/ gravados.
"""
import numpy as np
import pandas as pd
import os, sys

# Simula a estrutura de pastas esperada
os.makedirs("figures", exist_ok=True)
os.makedirs("tables", exist_ok=True)

# ── Patch das constantes que a função usa ────────────────────────────────────
import analyze
analyze.MAIN_FS     = "no_value_block"
analyze.MAIN_SCALER = "robust"
analyze.FIGURES_DIR = "figures"
analyze.TABLES_DIR  = "tables"

# ── Substituir preprocess_data por uma versão mock ───────────────────────────
# Em vez de carregar o CSV real, devolve uma matriz aleatória com 500 pontos
import src.preprocessing as _prep
_original_preprocess = _prep.preprocess_data

def _mock_preprocess(df, feature_set="no_value_block", scaler="robust"):
    rng = np.random.default_rng(42)
    # 3 clusters bem separados para o ikmeans conseguir convergir
    X = np.vstack([
        rng.normal([0, 0],  0.3, (170, 8)),
        rng.normal([5, 5],  0.3, (165, 8)),
        rng.normal([0, 10], 0.3, (165, 8)),
    ])
    return X, [f"feat_{i}" for i in range(8)]

_prep.preprocess_data = _mock_preprocess
analyze.preprocess_data = _mock_preprocess  # patch no namespace de analyze também

# ── Raw df mock (só precisa de ter len >= n_pontos) ──────────────────────────
raw_df_mock = pd.DataFrame(np.zeros((500, 2)), columns=["a", "b"])

# ── Correr a função ───────────────────────────────────────────────────────────
print("A correr generate_ikmeans_bootstrap_stability com dados sintéticos...")
analyze.generate_ikmeans_bootstrap_stability(
    raw_df=raw_df_mock,
    main_k=3,
    n_boot=5,       # 5 runs para o teste ser rápido
    frac=0.8,
    seed=0,
)

# ── Verificar outputs ─────────────────────────────────────────────────────────
print("\n=== Verificação dos outputs ===")

for path in [
    "tables/ikmeans_bootstrap_stability.csv",
    "tables/ikmeans_bootstrap_ari.csv",
    "figures/ikmeans_bootstrap_metrics.png",
    "figures/ikmeans_bootstrap_ari.png",
]:
    exists = os.path.exists(path)
    size   = os.path.getsize(path) if exists else 0
    print(f"  {'OK' if exists and size > 0 else 'FAIL'}: {path} ({size} bytes)")

# ── Verificar conteúdo das tabelas ────────────────────────────────────────────
print()
stab = pd.read_csv("tables/ikmeans_bootstrap_stability.csv")
print("ikmeans_bootstrap_stability.csv:")
print(stab.to_string())

print()
ari = pd.read_csv("tables/ikmeans_bootstrap_ari.csv")
print("ikmeans_bootstrap_ari.csv:")
print(ari.to_string())

# ── Verificações lógicas ──────────────────────────────────────────────────────
print()
assert len(stab) == 5, f"Esperava 5 runs, obteve {len(stab)}"
assert "silhouette" in stab.columns, "Coluna silhouette em falta"
assert "ari" in ari.columns, "Coluna ari em falta"
assert len(ari) == 10, f"Esperava 10 pares (C(5,2)), obteve {len(ari)}"  # C(5,2)=10
assert (ari["ari"] >= 0).all(), "ARI negativo encontrado"
print("Todas as verificações passaram.")