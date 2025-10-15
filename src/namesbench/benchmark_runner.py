from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

from .agents import (
    create_player_runnable,
    create_spymaster_runnable,
    parse_player_response,
    parse_spymaster_response,
)
from .game_state import GameState
from .image_board import BoardImages, build_image_board


def run_game(
    deck_path: Path,
    grid: str,
    model: str,
    seed: Optional[int],
    friendly_fraction: float,
    out_dir: Path,
    debug_images: bool = False,
    progress: bool = True,
) -> Dict[str, object]:
    rows, cols = parse_grid(grid)
    board_images: BoardImages = build_image_board(
        deck_path=deck_path,
        grid_size=(rows, cols),
        output_dir=out_dir,
        friendly_fraction=friendly_fraction,
        seed=seed,
        debug=debug_images,
    )

    game_state = GameState.from_board_images(board_images)

    prompt_context = {}

    spymaster = create_spymaster_runnable(model)
    player = create_player_runnable(model)

    trace: List[Dict[str, object]] = []

    while not game_state.is_complete():
        round_number = game_state.next_round()

        friendly_total = len(game_state.board.friendly_positions)
        opponent_total = len(game_state.board.opponent_positions)

        clue_input = {
            "board_image": str(board_images.composite_path),
            "board_image_data_url": board_images.composite_data_url,
            "round": round_number,
            "remaining": sorted(game_state.board.remaining_friendly),
            "revealed_friendly": sorted(set(game_state.board.friendly_positions) - set(game_state.board.remaining_friendly)),
            "revealed_opponent": sorted(set(game_state.board.opponent_positions) - set(game_state.board.remaining_opponent)),
            **prompt_context,
        }
        spymaster_raw = spymaster.invoke(clue_input)
        spymaster_response = parse_spymaster_response(spymaster_raw)
        if progress:
            print(
                f"[Round {round_number}] Clue → {spymaster_response.clue} - {spymaster_response.count} | Targets: {spymaster_response.targets}"
            )

        count = min(spymaster_response.count, len(game_state.board.remaining_friendly))

        player_input = {
            "board_image": str(board_images.composite_path),
            "board_image_data_url": board_images.composite_data_url,
            "clue": spymaster_response.clue,
            "count": count,
            "revealed_friendly": sorted(set(game_state.board.friendly_positions) - set(game_state.board.remaining_friendly)),
            "revealed_opponent": sorted(set(game_state.board.opponent_positions) - set(game_state.board.remaining_opponent)),
            "friendly_total": friendly_total,
            "opponent_total": opponent_total,
            **prompt_context,
        }
        player_raw = player.invoke(player_input)
        player_response = parse_player_response(player_raw, count)
        if progress:
            print(f"[Round {round_number}] Guesses → {player_response.guesses}")

        round_result = game_state.apply_round(
            clue=spymaster_response.clue,
            count=count,
            guesses=player_response.guesses,
            intended_targets=spymaster_response.targets,
        )
        if progress:
            print(
                f"[Round {round_number}] Result → correct={round_result.correct} wrong={round_result.wrong}"
            )

        trace.append(
            {
                "round": round_number,
                "clue": spymaster_response.clue,
                "count": count,
                "guesses": player_response.guesses,
                "intended_targets": spymaster_response.targets,
                "correct": round_result.correct,
                "wrong": round_result.wrong,
            }
        )

    final_score = game_state.score()

    return {
        "trace": trace,
        "score": final_score,
        "rounds": len(game_state.history),
        "correct_total": game_state.correct_total,
        "opponent_hits": game_state.opponent_hits,
        "composite_image": str(board_images.composite_path),
        "debug_image": str(board_images.debug_path) if board_images.debug_path else None,
    }


def parse_grid(grid: str) -> tuple[int, int]:
    parts = grid.lower().split("x")
    if len(parts) != 2:
        raise ValueError(f"Invalid grid format: {grid}")
    rows, cols = map(int, parts)
    return rows, cols


def save_trace(trace: List[Dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for entry in trace:
            file.write(json.dumps(entry) + "\n")


def append_csv(summary: Dict[str, object], csv_path: Path) -> None:
    write_header = not csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "score",
                "rounds",
                "correct_total",
                "opponent_hits",
                "composite_image",
                "debug_image",
            ],
        )
        if write_header:
            writer.writeheader()
        writer.writerow(summary)


