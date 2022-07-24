from typing import Optional
from pydantic import BaseModel

from models.mod import Mod

class Score(BaseModel):
    beatmap_md5: str
    beatmap_id: int

    mods: list[Mod]
    mode: int

    acc: float
    n300: int
    n100: int
    n50: int
    ngeki: int
    nkatu: int
    nmiss: int

    combo: int
    score: int
    
    akat_pp: Optional[int] = None
    branch_pp: Optional[int] = None

    @property
    def total_hits(self) -> int:
        return self.n300 + self.n100 + self.n50 + self.nmiss

class ScoresResult(BaseModel):
    scores: list[Score]
