from __future__ import annotations

from pathlib import Path

from PIL import Image

from namesbench import benchmark_runner


class DummyRunnable:
    def __init__(self, func):
        self.func = func

    def invoke(self, inputs):
        return self.func(inputs)


def test_run_game_without_llms(monkeypatch, tmp_path):
    deck_dir = tmp_path / "deck"
    deck_dir.mkdir()

    for index in range(8):
        image = Image.new("RGB", (64, 64), (index * 20 % 255, index * 40 % 255, index * 60 % 255))
        image.save(deck_dir / f"card_{index}.png")

    output_dir = tmp_path / "out"

    state = {"remaining": []}

    def spymaster_func(inputs):
        remaining = inputs["remaining"]
        state["remaining"] = list(remaining)
        count = len(remaining) or 1
        targets = ", ".join(str(value) for value in state["remaining"][:count])
        return f"Clue: Test - {count}\nTargets: [{targets}]"

    def player_func(inputs):
        count = inputs["count"]
        guesses = state["remaining"][:count]
        guesses_str = ", ".join(str(value) for value in guesses)
        return f"My guesses: [{guesses_str}]"

    monkeypatch.setattr(
        benchmark_runner,
        "create_spymaster_runnable",
        lambda provider, model: DummyRunnable(spymaster_func),
    )
    monkeypatch.setattr(
        benchmark_runner,
        "create_player_runnable",
        lambda provider, model: DummyRunnable(player_func),
    )

    result = benchmark_runner.run_game(
        deck_path=deck_dir,
        grid="2x4",
        model="stub-model",
        provider="openai",
        seed=42,
        friendly_fraction=0.5,
        out_dir=output_dir,
        debug_images=False,
    )

    assert result["rounds"] == 1
    assert result["correct_total"] > 0
    assert result["opponent_hits"] == 0
    assert Path(result["composite_image"]).exists()
    assert result["debug_image"] is None
    assert result["trace"], "Trace should not be empty"
    assert result["score"] > 0

