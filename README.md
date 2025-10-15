# Namesbench Dixit Benchmark

Namesbench evaluates a pair of LangChain agents (spymaster + player) that play a Codenames-style game using Dixit artwork. The board contains only friendly and opponent cards; the objective is to reveal every friendly card in as few rounds as possible while avoiding opponent hits.

## What the project does today

- **Dynamic board generation** (`image_board.py`): Loads PNG/JPG assets from `deck/`, samples a configurable grid, and produces a numbered composite image for the player. Optional debug overlays can highlight team assignments.
- **State tracking** (`game_state.py`): Maintains friendly/opponent pools, prevents already revealed cards from being guessed again, records round outcomes, and computes the benchmark score.
- **LLM agent orchestration** (`agents.py`): Builds prompts for OpenAI vision-capable models via LangChain. The spymaster returns a clue/count pair plus a private list of intended friendly targets; the player responds with indices based on the clue and composite image.
- **Game loop & logging** (`benchmark_runner.py`): Runs each round until all friendly cards are found, clamps guesses to the remaining friendly count, updates state, and saves JSONL traces and CSV summaries (including spymaster intentions for analysis).
- **CLI runner** (`run_benchmark.py`): Command line interface to configure grids, number of games, model, deck folder, output directory, friendly fraction, seed, and debug image generation.

## Running the benchmark

```bash
python run_benchmark.py --grid 2x4 --games 5 --model gpt-4o --deck deck --out out
```

Key options:

- `--grid`: grid dimensions such as `2x4`, `4x4`, `5x5`.
- `--games`: number of independent games to play.
- `--friendly-fraction`: portion of cards marked friendly (defaults to 0.5).
- `--debug-images`: emit team-colored overlays for inspection.
- `--seed`: controls board sampling reproducibility.

Each run writes game traces (`trace.jsonl` per game) and appends summaries to `out/summary.csv`.

## Dependencies

- `Pillow`
- `langchain` and `langchain-openai`
- `python-dotenv`
- `pydantic`
- `numpy` (optional, for image operations)

Ensure `.env` contains `OPENAI_API_KEY`; the CLI loads it automatically via `python-dotenv`.
