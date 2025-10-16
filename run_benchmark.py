from __future__ import annotations

import argparse
import hashlib
import statistics
from datetime import datetime
from pathlib import Path

from namesbench.benchmark_runner import append_csv, run_game, save_trace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Dixit benchmark games")
    parser.add_argument("--grid", default="2x4", help="Grid size, e.g., 2x4 or 5x5")
    parser.add_argument("--games", type=int, default=1, help="Number of games to run")
    parser.add_argument("--model", default="gpt-4o", help="Model name for the selected provider")
    parser.add_argument("--provider", default="openai", help="LLM provider, e.g., openai, anthropic, gemini")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--friendly-fraction", type=float, default=0.5, help="Fraction of friendly cards")
    parser.add_argument("--deck", type=Path, default=Path("deck"), help="Path to deck image directory")
    parser.add_argument("--out", type=Path, default=Path("out"), help="Output directory")
    parser.add_argument("--debug-images", action="store_true", help="Output debug board with team colors")
    try:
        # Python 3.9+ supports BooleanOptionalAction
        parser.add_argument("--progress", default=True, action=argparse.BooleanOptionalAction, help="Print per-round progress during games")
    except AttributeError:
        # Fallback for older argparse versions
        parser.add_argument("--progress", dest="progress", action="store_true", help="Print per-round progress during games")
        parser.add_argument("--no-progress", dest="progress", action="store_false", help="Disable per-round progress output")
        parser.set_defaults(progress=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = []
    args.out.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_dir = (args.out / timestamp).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)

    if args.seed is not None:
        base_seed = args.seed
    else:
        grid_hash = hashlib.sha256(args.grid.encode("utf-8")).hexdigest()
        base_seed = int(grid_hash, 16) % (2**32)

    summary_path = run_dir / "summary.csv"

    print(f"Output directory: {run_dir}")

    for game_index in range(args.games):
        game_seed = (base_seed + game_index) % (2**32)
        game_out = run_dir / f"game_{game_index:03d}"
        if args.progress:
            print(f"\n=== Starting game {game_index + 1}/{args.games} → grid={args.grid}, model={args.model} ===")
        result = run_game(
            deck_path=args.deck,
            grid=args.grid,
            model=args.model,
            provider=args.provider,
            seed=game_seed,
            friendly_fraction=args.friendly_fraction,
            out_dir=game_out,
            debug_images=args.debug_images,
            progress=args.progress,
        )

        trace_path = game_out / "trace.jsonl"
        save_trace(result["trace"], trace_path)

        summary = {
            "score": result["score"],
            "rounds": result["rounds"],
            "correct_total": result["correct_total"],
            "opponent_hits": result["opponent_hits"],
            "composite_image": result["composite_image"],
            "debug_image": result["debug_image"],
        }

        append_csv(summary, summary_path)
        summaries.append(summary)

        print(
            f"Game {game_index + 1}/{args.games}: score={summary['score']:.2f}, "
            f"rounds={summary['rounds']}, correct={summary['correct_total']}, "
            f"opponent_hits={summary['opponent_hits']}"
        )

    if summaries:
        scores = [item["score"] for item in summaries]
        rounds = [item["rounds"] for item in summaries]
        print("\nAggregate stats:")
        print(f"Average score: {statistics.mean(scores):.2f}")
        print(f"Average rounds: {statistics.mean(rounds):.2f}")
        print(f"Total opponent hits: {sum(item['opponent_hits'] for item in summaries)}")


if __name__ == "__main__":
    main()


