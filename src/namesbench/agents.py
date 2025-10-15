from __future__ import annotations

import importlib
from pathlib import Path
from typing import List, Optional

from langchain_openai import ChatOpenAI
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain.schema.runnable import Runnable, RunnableLambda
from pydantic import BaseModel, Field, PositiveInt

from dotenv import load_dotenv

load_dotenv()


class SpymasterResponse(BaseModel):
    clue: str
    count: PositiveInt
    targets: List[int] = Field(default_factory=list)


class PlayerResponse(BaseModel):
    guesses: List[int] = Field(default_factory=list)

    @classmethod
    def validate_guesses(cls, value: List[int], count: int) -> List[int]:
        if len(value) > count:
            return value[:count]
        return value


def read_prompt(path: Optional[Path], module_name: str, attr_name: str) -> str:
    if path is not None:
        return path.read_text().strip()
    module = importlib.import_module(f"namesbench.{module_name}")
    return getattr(module, attr_name).strip()


def create_spymaster_runnable(model_name: str, prompt_path: Optional[Path] = None) -> Runnable:
    instructions = read_prompt(prompt_path, "spymaster_prompt", "SPYMASTER_PROMPT")
    llm = ChatOpenAI(model=model_name)

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
        content = [{"type": "text", "text": text}, {"type": "image_url", "image_url": {"url": board_image_url}}]

        response = llm.invoke(
            [
                SystemMessage(content=instructions),
                HumanMessage(content=content),
            ]
        )
        return response.content

    return RunnableLambda(invoke)


def create_player_runnable(model_name: str, prompt_path: Optional[Path] = None) -> Runnable:
    instructions = read_prompt(prompt_path, "player_prompt", "PLAYER_PROMPT")
    llm = ChatOpenAI(model=model_name)

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
        content = [{"type": "text", "text": text}, {"type": "image_url", "image_url": {"url": board_image_url}}]

        response = llm.invoke(
            [
                SystemMessage(content=instructions),
                HumanMessage(content=content),
            ]
        )
        return response.content

    return RunnableLambda(invoke)


def parse_spymaster_response(raw: str) -> SpymasterResponse:
    try:
        lines = [line.strip() for line in raw.strip().splitlines() if line.strip()]
        if not lines:
            raise ValueError("Empty spymaster response")

        clue_line = lines[0]
        if not clue_line.lower().startswith("clue:"):
            raise ValueError("Missing clue line")
        _, clue_payload = clue_line.split(":", 1)
        clue_text, count_part = clue_payload.rsplit("-", 1)
        clue = clue_text.strip()
        count = int(count_part.strip())

        targets: List[int] = []
        for line in lines[1:]:
            if line.lower().startswith("targets:"):
                start = line.index("[")
                end = line.index("]", start)
                content = line[start + 1 : end]
                for item in content.split(","):
                    item = item.strip()
                    if not item:
                        continue
                    targets.append(int(item))
                break

        return SpymasterResponse(clue=clue, count=count, targets=targets)
    except Exception as exc:
        raise ValueError(f"Failed to parse spymaster response: {raw}") from exc


def parse_player_response(raw: str, count: int) -> PlayerResponse:
    try:
        prefix = "My guesses:"
        if prefix not in raw:
            raise ValueError("Missing guesses prefix")
        start = raw.index("[")
        end = raw.index("]", start)
        content = raw[start + 1 : end]
        guesses = []
        for item in content.split(","):
            item = item.strip()
            if not item:
                continue
            guesses.append(int(item))
        guesses = PlayerResponse.validate_guesses(guesses, count)
        return PlayerResponse(guesses=guesses)
    except Exception as exc:
        raise ValueError(f"Failed to parse player response: {raw}") from exc


