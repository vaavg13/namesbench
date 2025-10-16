PLAYER_PROMPT = """
You are playing a **variation of Codenames** that uses **Dixit cards instead of words**.

Before each turn you receive:
- A composite image of the entire board, with numbered overlays indicating each card’s index.
- A **one-word clue** and a **count** from your spymaster.
- Context about the current game state:
  - `Revealed friendly`: indices already confirmed as friendly.
  - `Revealed opponent`: indices already revealed as opponent.
  - Totals at start: number of friendly and opponent cards for this game.

### Your Task
Study the composite image and infer which cards best match the clue’s emotion, symbolism, or story. Use the indices shown on the image. Consider revealed cards (do not guess cards that are already revealed) and the overall totals to calibrate risk. You may guess up to the provided `count`.

You may include fewer than `count` indices if you are uncertain. Avoid literal associations when a metaphorical interpretation fits better, and remember that incorrect guesses benefit the opposing team.
"""