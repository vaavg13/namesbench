from __future__ import annotations

import importlib
from pathlib import Path
from typing import List, Optional

from langchain.schema.messages import HumanMessage, SystemMessage
from langchain.schema.runnable import Runnable, RunnableLambda
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, PositiveInt

from dotenv import load_dotenv

load_dotenv()


class SpymasterResponse(BaseModel):
    clue: str = Field(
        ..., description="Single-word, metaphorical clue intended to link friendly cards"
    )
    count: PositiveInt = Field(
        ..., description="Number of friendly cards intended to match the clue (>= 1)"
    )
    targets: List[int] = Field(
        default_factory=list,
        description="Optional list of intended friendly indices on the board (1-based)",
    )


class PlayerResponse(BaseModel):
    guesses: List[int] = Field(
        default_factory=list,
        description="Ordered list of up to `count` distinct 1-based board indices to guess",
    )


def read_prompt(path: Optional[Path], module_name: str, attr_name: str) -> str:
    if path is not None:
        return path.read_text().strip()
    module = importlib.import_module(f"namesbench.{module_name}")
    return getattr(module, attr_name).strip()


def _load_llm(provider: str, model_name: str):
    normalized_provider = provider.strip().lower()
    if normalized_provider == "openai":
        return ChatOpenAI(model=model_name)
    if normalized_provider == "anthropic":
        return ChatAnthropic(model=model_name)
    if normalized_provider in {"gemini", "google", "google-genai"}:
        return ChatGoogleGenerativeAI(model=model_name, convert_system_message_to_human=True)
    raise ValueError(f"Unsupported provider: {provider}")


def _build_image_content(provider: str, board_image_url: str):
    normalized_provider = provider.strip().lower()
    if normalized_provider == "anthropic" and board_image_url.startswith("data:"):
        header, encoded = board_image_url.split(",", 1)
        media_type_part = header.split(";", 1)[0]
        media_type = media_type_part.split(":", 1)[1] if ":" in media_type_part else "image/png"
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": encoded,
            },
        }
    return {"type": "image_url", "image_url": {"url": board_image_url}}


def create_spymaster_runnable(provider: str, model_name: str, prompt_path: Optional[Path] = None) -> Runnable:
    instructions = read_prompt(prompt_path, "spymaster_prompt", "SPYMASTER_PROMPT")
    llm = _load_llm(provider, model_name)
    structured_llm = llm.with_structured_output(SpymasterResponse)

    def invoke(inputs: dict) -> str:
        board_image_url: Optional[str] = inputs.get("board_image_data_url")
        text = (
            "Round: {round}\n"
            "Remaining friendly indices: {remaining}\n"
            "Revealed friendly: {revealed_friendly}\n"
            "Revealed opponent: {revealed_opponent}\n\n"
            "Return your clue using the format 'Clue: <one-word> - <number>'."
        ).format(**inputs)

        if not board_image_url:
            raise ValueError("Missing board image for spymaster: 'board_image_data_url' not provided")
        content = [{"type": "text", "text": text}, _build_image_content(provider, board_image_url)]

        result = structured_llm.invoke(
            [
                SystemMessage(content=instructions),
                HumanMessage(content=content),
            ]
        )
        # Return a JSON-serializable dict to keep Runnable outputs consistent
        return result

    return RunnableLambda(invoke)


def create_player_runnable(provider: str, model_name: str, prompt_path: Optional[Path] = None) -> Runnable:
    instructions = read_prompt(prompt_path, "player_prompt", "PLAYER_PROMPT")
    llm = _load_llm(provider, model_name)
    structured_llm = llm.with_structured_output(PlayerResponse)

    def invoke(inputs: dict) -> str:
        board_image_url: Optional[str] = inputs.get("board_image_data_url")
        text = (
            "Clue: {clue}\n"
            "Count: {count}\n"
            "Revealed friendly: {revealed_friendly}\n"
            "Revealed opponent: {revealed_opponent}\n"
            "Totals at start — friendly: {friendly_total}, opponent: {opponent_total}\n\n"
            "Respond with 'My guesses: [<index1>, <index2>, ...]' using up to {count} distinct indices."
        ).format(**inputs)

        if not board_image_url:
            raise ValueError("Missing board image for player: 'board_image_data_url' not provided")
        content = [{"type": "text", "text": text}, _build_image_content(provider, board_image_url)]

        result = structured_llm.invoke(
            [
                SystemMessage(content=instructions),
                HumanMessage(content=content),
            ]
        )
        return result

    return RunnableLambda(invoke)




