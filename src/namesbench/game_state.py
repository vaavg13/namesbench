from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .image_board import BoardImages


@dataclass
class BoardConfig:
    rows: int
    cols: int
    friendly_fraction: float = 0.5
    seed: Optional[int] = None


@dataclass
class BoardState:
    friendly_positions: Set[int]
    opponent_positions: Set[int]
    index_to_file: Dict[int, Path]
    index_to_team: Dict[int, str]
    remaining_friendly: Set[int] = field(default_factory=set)
    remaining_opponent: Set[int] = field(default_factory=set)
    revealed_cards: Set[int] = field(default_factory=set)

    def __post_init__(self) -> None:
        if not self.remaining_friendly:
            self.remaining_friendly = set(self.friendly_positions)
        if not self.remaining_opponent:
            self.remaining_opponent = set(self.opponent_positions)

    def is_valid_index(self, index: int) -> bool:
        return index in self.index_to_team and index not in self.revealed_cards

    def remove_friendly(self, index: int) -> None:
        self.remaining_friendly.discard(index)
        self.revealed_cards.add(index)

    def remove_opponent(self, index: int) -> None:
        self.remaining_opponent.discard(index)
        self.revealed_cards.add(index)


@dataclass
class RoundResult:
    round_number: int
    clue: str
    count: int
    guesses: List[int]
    intended_targets: List[int]
    correct: List[int]
    wrong: List[int]

    def to_json(self) -> str:
        return json.dumps(
            {
                "round": self.round_number,
                "clue": self.clue,
                "count": self.count,
                "guesses": self.guesses,
                "intended_targets": self.intended_targets,
                "correct": self.correct,
                "wrong": self.wrong,
            }
        )


class GameState:
    def __init__(self, board_state: BoardState) -> None:
        self.board = board_state
        self.round_number = 0
        self.history: List[RoundResult] = []
        self.correct_total = 0
        self.opponent_hits = 0

    def next_round(self) -> int:
        self.round_number += 1
        return self.round_number

    def apply_round(
        self,
        clue: str,
        count: int,
        guesses: Sequence[int],
        intended_targets: Optional[Sequence[int]] = None,
    ) -> RoundResult:
        unique_guesses: List[int] = []
        seen: Set[int] = set()
        for guess in guesses:
            if guess in seen:
                continue
            seen.add(guess)
            unique_guesses.append(guess)

        valid_guesses = unique_guesses[:count]
        correct: List[int] = []
        wrong: List[int] = []

        for guess in valid_guesses:
            if guess in self.board.revealed_cards:
                continue
            if not self.board.is_valid_index(guess):
                self.opponent_hits += 1
                wrong.append(guess)
                continue

            team = self.board.index_to_team.get(guess)
            if team == "friendly":
                if guess in self.board.remaining_friendly:
                    self.board.remove_friendly(guess)
                    self.correct_total += 1
                correct.append(guess)
            else:
                self.opponent_hits += 1
                wrong.append(guess)
                if guess in self.board.remaining_opponent:
                    self.board.remove_opponent(guess)

        intentions = []
        if intended_targets:
            seen_intended: Set[int] = set()
            for index in intended_targets:
                if index in self.board.friendly_positions and index not in seen_intended:
                    intentions.append(index)
                    seen_intended.add(index)

        result = RoundResult(
            round_number=self.round_number,
            clue=clue,
            count=count,
            guesses=list(valid_guesses),
            intended_targets=intentions,
            correct=correct,
            wrong=wrong,
        )
        self.history.append(result)
        return result

    def is_complete(self) -> bool:
        return not self.board.remaining_friendly

    def score(self) -> float:
        if not self.history:
            return 0.0
        rounds = len(self.history)
        return (self.correct_total - self.opponent_hits) / rounds

    @classmethod
    def from_board_images(cls, board_images: "BoardImages") -> "GameState":
        friendly_positions = {idx for idx, team in board_images.index_to_team.items() if team == "friendly"}
        opponent_positions = set(board_images.index_to_team.keys()) - friendly_positions
        board_state = BoardState(
            friendly_positions=friendly_positions,
            opponent_positions=opponent_positions,
            index_to_file=board_images.index_to_file,
            index_to_team=board_images.index_to_team,
            remaining_friendly=set(friendly_positions),
        )
        return cls(board_state)


def generate_positions(total: int, friendly_fraction: float, seed: Optional[int]) -> Tuple[Set[int], Set[int]]:
    rng = random.Random(seed)
    friendly_count = max(1, round(total * friendly_fraction))
    indices = list(range(1, total + 1))
    friendly = set(rng.sample(indices, k=friendly_count))
    opponent = set(indices) - friendly
    return friendly, opponent


