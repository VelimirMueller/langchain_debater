from operator import add
from typing import Annotated, Literal, TypedDict


class Turn(TypedDict):
    role: Literal["moderator", "proposer", "critic", "judge"]
    content: str
    round_num: int


class DebateState(TypedDict):
    topic: str
    transcript: Annotated[list[Turn], add]
    focus_question: str
    round_num: int
    max_rounds: int
    verdict: str | None
