import numpy as np
from typing import List, NamedTuple, Dict
from scipy.stats import qmc


def get_lhs_samples(n_samples, bounds, seed=None):
    sampler = qmc.LatinHypercube(d=len(bounds), optimization="random-cd", seed=seed)
    raw_sample = sampler.random(n=n_samples)
    lower_bounds = [b[0] for b in bounds]
    upper_bounds = [b[1] for b in bounds]
    return qmc.scale(raw_sample, lower_bounds, upper_bounds)


class CardOutcome(NamedTuple):
    gain: int
    loss: int


Deck = List[CardOutcome]


class Environment:
    def __init__(self, rng=None):
        self.rng = rng if rng is not None else np.random.default_rng()

        self.deck_params = {
            "A": {"fixed_gain": 100, "loss_freq": 0.50, "loss_range": (-350, -150)},
            "B": {"fixed_gain": 100, "loss_freq": 0.10, "loss_value": -1250},
            "C": {"fixed_gain": 50,  "loss_freq": 0.50, "loss_range": (-75, -25)},
            "D": {"fixed_gain": 50,  "loss_freq": 0.10, "loss_value": -250},
        }

        self.pregenerated_decks = self._generate_all_decks(n_cards=1000)

    def _generate_all_decks(self, n_cards: int) -> Dict[str, Deck]:
        decks = {}
        for deck_id, params in self.deck_params.items():
            gains = np.full(n_cards, params["fixed_gain"])
            is_loss = self.rng.random(size=n_cards) < params["loss_freq"]

            if "loss_value" in params:
                loss_magnitudes = np.full(n_cards, params["loss_value"])
            else:
                low, high = params["loss_range"]
                l_min, l_max = sorted((low, high))
                loss_magnitudes = self.rng.integers(l_min, l_max + 1, size=n_cards)

            deck_cards = []
            for g, has_loss, l_val in zip(gains, is_loss, loss_magnitudes):
                final_loss = int(l_val) if has_loss else 0
                deck_cards.append(CardOutcome(gain=int(g), loss=final_loss))

            decks[deck_id] = deck_cards
        return decks

    def get_standard_scheme_cards(self, deck_id: str, start_index: int, count: int) -> Deck:
        source_seq = self.pregenerated_decks.get(deck_id, [])
        seq_len = len(source_seq)
        return [source_seq[(start_index + i) % seq_len] for i in range(count)]

    def create_live_game_decks(self) -> dict:
        return {d: self.get_standard_scheme_cards(d, 0, 200) for d in ["A", "B", "C", "D"]}


class EnvironmentBechara:
    def get_standard_scheme_cards(self, deck_id: str, start_index: int, count: int) -> Deck:
        A_losses = [0] * 40
        for idx, val in [(3,-150),(5,-300),(7,-200),(9,-250),(10,-350),(12,-350),(14,-250),(15,-200),(17,-300),(18,-150),(22,-300),(24,-350),(26,-200),(27,-250),(28,-150),(31,-350),(32,-200),(33,-250),(37,-150),(38,-300)]:
            A_losses[idx-1] = val
        B_losses = [0] * 40
        for idx, val in [(9,-1250),(14,-1250),(21,-1250),(32,-1250)]:
            B_losses[idx-1] = val
        C_losses = [0] * 40
        for idx, val in [(3,-50),(5,-50),(7,-50),(9,-50),(10,-50),(12,-25),(13,-75),(17,-25),(18,-75),(20,-50),(24,-50),(25,-25),(26,-50),(29,-75),(30,-50),(34,-25),(35,-25),(37,-75),(39,-50),(40,-75)]:
            C_losses[idx-1] = val
        D_losses = [0] * 40
        for idx, val in [(10,-250),(20,-250),(29,-250),(35,-250)]:
            D_losses[idx-1] = val

        schemes = {
            "A": [CardOutcome(100, l) for l in A_losses],
            "B": [CardOutcome(100, l) for l in B_losses],
            "C": [CardOutcome(50, l) for l in C_losses],
            "D": [CardOutcome(50, l) for l in D_losses],
        }
        source_seq = schemes.get(deck_id, [])
        seq_len = len(source_seq)
        return [source_seq[(start_index + i) % seq_len] for i in range(count)]

    def create_live_game_decks(self) -> dict:
        return {d: self.get_standard_scheme_cards(d, 0, 200) for d in ["A", "B", "C", "D"]}
