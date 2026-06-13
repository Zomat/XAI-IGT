# XAI-IGT

Kod źródłowy do pracy magisterskiej:

> **Wyjaśnialność i identyfikowalność modeli procesu uczenia z nagród i kar na przykładzie Iowa Gambling Task**

## Kontekst i cel

Model matematyczny jest **interpretowalny**, gdy jego parametry mają z góry przypisaną semantykę dziedzinową. **Wyjaśnialność** oznacza, że dopasowanie modelu do danych pozwala przypisać obserwowane zachowanie konkretnym mechanizmom decyzyjnym. Wyjaśnienie takie jest wiarygodne tylko przy pełnej **identyfikowalności**: gdy procedura estymacyjna potrafi jednoznacznie odzyskać wartości parametrów z dostępnych danych. Gdy wiele różnych kombinacji parametrów opisuje dane równie dobrze, każda estymata staje się arbitralna, a zbudowane na niej wyjaśnienie – pozorne.

Praca poddaje relację między wyjaśnialnością a identyfikowalnością systematycznej kwantyfikacji w klasie kognitywnych modeli **Iowa Gambling Task (IGT)** — standardowego paradygmatu psychiatrii obliczeniowej, w którym parametry modeli (awersja do strat, tempo uczenia) służą do klinicznej charakterystyki pacjentów.

## Badane modele

| Symbol | Model | Parametry |
|--------|-------|-----------|
| **KPVL** | Autorski model — Kalman-like Prospect Valence Learning; łączy zarządzanie niepewnością inspirowane filtrem Kalmana z asymetryczną funkcją użyteczności teorii perspektywy | λ (awersja do strat), ρ (kształt percepcji), α (tempo uczenia), f (czynnik zapominania), β (eksploracja) |
| **PVL-Delta** | Prospect Valuation Learning z regułą delta (Ahn i wsp., 2008) | A (kształt), w (awersja do strat), a (tempo uczenia), c (spójność) |
| **ORL** | Outcome Representation Learning (Haines i wsp., 2018) | A_rew, A_pun (tempa uczenia), K' (zanik), β_F (częstotliwość), β_P (perseweracja) |

## Metodologia

Analiza obejmuje cztery komplementarne poziomy:

| Poziom | Tryb (`MODE`) | Opis |
|--------|---------------|------|
| Globalne odzyskiwanie parametrów | `RECOVERY`, `DIST_RECOVERY` | MLE na syntetycznych agentach; rozkłady błędów estymatora |
| Diagnostyka stabilności punktowej | `DIAGNOSTICS`, `RELIABILITY_SCAN`, `FULL_LANDSCAPE` | Skan całej przestrzeni parametrów; mapy niezawodności RMSSE; krajobraz NLL |
| Wnioskowanie bayesowskie | `MCMC`, `MCMC_SCAN` | Posterior pojedynczego punktu (corner plot, ESS) oraz przekrojowe mapowanie całej dziedziny (KL, korelacje posterioru) |
| Partycjonowanie przestrzeni | `PSP`, `COMPARISON` | Wiązanie regionów dziedziny z dyskretnymi wzorcami zachowania; porównanie międzymodelowe (cross-fit) |

## Główne wyniki

1. **Zależność od położenia w przestrzeni parametrów.** Jakość estymacji punktowej silnie zależy od regionu przestrzeni: dla modelu KPVL estymacja jest niezawodna w ~¾ przestrzeni, dla modeli referencyjnych – w niespełna 2/5. Rozkłady błędów odbiegają od normalności w przeważającej części dziedziny, co dyskwalifikuje klasyczne przedziały ufności.

2. **Bayesowska korekta obrazu.** Część nieidentyfikowalności jest artefaktem optymalizatora, nie brakiem informacji w danych: istnieją parametry niestabilne przy MLE, a niemal w pełni identyfikowalne bayesowsko.

3. **Dysocjacja identyfikowalności i wzorców zachowania.** Parametr decydujący o klasyfikacji wzorca behawioralnego może być niemal w pełni identyfikowalny, podczas gdy parametry opisujące persewerację wyborów pozostają zawodne niezależnie od wzorca. Klasyfikacja typu pacjenta jest osiągalna; jego ilościowa charakterystyka — nie.

4. **Postulat metodologiczny.** Wiarygodna ocena identyfikowalności wymaga testowania na danych syntetycznych z nieinformacyjnym rozkładem a priori. Silne priors wyprowadzone z danych populacyjnych poprawiają wskaźniki odzyskiwania, ale mogą maskować strukturalne słabości modeli i prowadzić do estymat odwzorowujących normę populacyjną zamiast rzeczywistego profilu pacjenta.

## Struktura projektu

```
src/
├── agents/          # implementacje modeli (KPVL, PVL-Delta, ORL)
├── analysis/        # symulacje, MLE, MCMC, odtwarzanie parametrów, PSP
├── environment/     # środowisko IGT (talie kart)
├── plots/           # generowanie wykresów do pracy
├── config.py        # konfiguracja: model, tryb analizy, hiperparametry
└── run.py           # główny punkt wejścia
regen_*.py           # regeneracja wykresów z istniejących CSV
results/             # wyniki analiz (CSV + PNG)
```

## Uruchomienie

```bash
source igt_env/bin/activate

# Ustaw ACTIVE_AGENT i MODE w src/config.py, następnie:
cd src
python run.py
```

Wyniki trafiają do `results/<model>/<tryb>/`.

## Zależności

Python 3.12 · `numpy` · `scipy` · `pandas` · `matplotlib` · `emcee` (próbkowanie MCMC)
