"""
test_vectors.py
===============
Zestawy parametrów do analizy landscape NLL dla modelu Agent (KUPL).

Użycie w config.py:
    from test_vectors import GOOD_VECTORS, BAD_VECTORS, ALL_VECTORS

Użycie w run.py (tryb FULL_LANDSCAPE lub DIAGNOSTICS):
    for label, params in ALL_VECTORS.items():
        config.default_test_params = params
        # ... uruchom analizę
"""

# ---------------------------------------------------------------------------
# Parametry: [Loss Aversion, Perception Shape, Learning Rate,
#             Forgetting Fact, Exploration Weight]
# Zakresy:   [(0.01,5.0),    (0.01,0.99),     (0.01,0.99),
#             (0.01,0.5),    (0.01,20.0)]
# ---------------------------------------------------------------------------

GOOD_VECTORS = {

    "ideal": [0.15, 0.75, 0.50, 0.08, 4.0],
    # Loss Aversion    = 0.15  → niskie,    LOWESS ~0.95
    # Perception Shape = 0.75  → wysokie,   zawsze stabilny
    # Learning Rate    = 0.50  → środek,    stabilny
    # Forgetting Fact  = 0.08  → niskie,    dobrze identyfikowalny
    # Exploration Weight = 4.0 → umiarkowane, Forgetting widoczny

    "healthy_participant": [0.30, 0.80, 0.40, 0.15, 6.0],
    # Loss Aversion    = 0.30  → niskie,    dobrze identyfikowalne
    # Perception Shape = 0.80  → wysokie,   zawsze stabilny
    # Learning Rate    = 0.40  → dobry zakres
    # Forgetting Fact  = 0.15  → niskie-średnie, OK
    # Exploration Weight = 6.0 → umiarkowane

    "default": [0.62, 0.66, 0.30, 0.30, 7.5],
    # Obecny default_test_params — punkt odniesienia
}

BAD_VECTORS = {

    "worst_case": [1.8, 0.15, 0.25, 0.38, 1.2],
    # Loss Aversion    = 1.8   → środek zakresu,  LOWESS najniższy
    # Perception Shape = 0.15  → bardzo niskie  → niszczy LA i LR
    # Learning Rate    = 0.25  → przy niskim PS słaby
    # Forgetting Fact  = 0.38  → wysokie + EW niskie = katastrofa
    # Exploration Weight = 1.2 → bardzo niskie  → FF nieidentyfikowalny

    "clinical_profile": [2.5, 0.20, 0.15, 0.40, 0.8],
    # Profil pacjenta z deficytami decyzyjnymi
    # Loss Aversion    = 2.5   → środek, nieprzewidywalny
    # Perception Shape = 0.20  → bardzo niskie
    # Learning Rate    = 0.15  → niskie + niski PS = słaby
    # Forgetting Fact  = 0.40  → prawie maksimum zakresu
    # Exploration Weight = 0.8 → praktycznie zerowe → FF niewidoczny

    "high_all": [4.2, 0.85, 0.85, 0.45, 17.0],
    # Problemy z górnym zakresem
    # Loss Aversion    = 4.2   → bardzo wysokie, LOWESS spada
    # Perception Shape = 0.85  → stabilny (jedyny dobry parametr)
    # Learning Rate    = 0.85  → wysokie, lekki spadek
    # Forgetting Fact  = 0.45  → prawie maksimum
    # Exploration Weight = 17.0 → bardzo wysokie, LOWESS ~0.65
}

# Wszystkie wektory razem — do iteracji
ALL_VECTORS = {**GOOD_VECTORS, **BAD_VECTORS}

# ---------------------------------------------------------------------------
# Opis słowny każdego wektora — do tytułów wykresów
# ---------------------------------------------------------------------------
VECTOR_LABELS = {
    "ideal":               "Idealny — wszystkie parametry w dobrym zakresie",
    "healthy_participant": "Typowy zdrowy uczestnik",
    "default":             "Default — obecny punkt testowy",
    "worst_case":          "Najgorszy przypadek — wszystkie problematyczne kombinacje",
    "clinical_profile":    "Profil kliniczny — pacjent z deficytami decyzyjnymi",
    "high_all":            "Wysokie wartości — problemy górnego zakresu",
}

# ---------------------------------------------------------------------------
# Szybki podgląd przy uruchomieniu jako skrypt
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import numpy as np

    PARAM_NAMES = [
        "Loss Aversion",
        "Perception Shape",
        "Learning Rate",
        "Forgetting Fact",
        "Exploration Weight",
    ]
    BOUNDS = [
        (0.01, 5.0),
        (0.01, 0.99),
        (0.01, 0.99),
        (0.01, 0.5),
        (0.01, 20.0),
    ]

    print("=" * 60)
    print("DOBRE WEKTORY (oczekiwany dobry landscape)")
    print("=" * 60)
    for name, params in GOOD_VECTORS.items():
        print(f"\n[{name}]  {VECTOR_LABELS[name]}")
        for pname, val, (lo, hi) in zip(PARAM_NAMES, params, BOUNDS):
            rel = (val - lo) / (hi - lo) * 100
            print(f"  {pname:<20} = {val:.2f}   ({rel:.0f}% zakresu)")

    print()
    print("=" * 60)
    print("ZŁE WEKTORY (oczekiwany zły landscape)")
    print("=" * 60)
    for name, params in BAD_VECTORS.items():
        print(f"\n[{name}]  {VECTOR_LABELS[name]}")
        for pname, val, (lo, hi) in zip(PARAM_NAMES, params, BOUNDS):
            rel = (val - lo) / (hi - lo) * 100
            print(f"  {pname:<20} = {val:.2f}   ({rel:.0f}% zakresu)")