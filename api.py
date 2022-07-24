from fastapi import APIRouter, Query
from constants.mods import Mods
from models.scores import *

import asyncio
import services
import pp

router = APIRouter()

SCORES: dict[int, Score] = {}

def convert_mods(mods: int) -> list[Mod]:
    mod_enum = Mods(mods)
    mod_str = repr(mod_enum)
    
    return [Mod(acronym=mod_str[char : char + 2].upper()) for char in range(0, len(mod_str), 2)]

async def calculate_pp(score: Score) -> None:
    attributes = await pp.get_beatmap_attributes(
        beatmap_id=score.beatmap_id,
        beatmap_md5=score.beatmap_md5,
        mode=score.mode,
        mods=score.mods,
    )
    if not attributes:
        return

    calculator = pp.get_calculator(attributes, score)
    score.branch_pp = int(calculator.calculate_pp())

@router.get("/scores/relax/top", response_model=ScoresResult)
async def top_scores_pp_calc(
    count: int = Query(500, ge=1, le=1000),
    mode: int = Query(0, ge=0, lt=3),
) -> ScoresResult:
    top_scores = await services.database.fetch_all(
        "SELECT scores_relax.*, beatmaps.beatmap_id FROM scores_relax "
        "INNER JOIN beatmaps USING(beatmap_md5) "
        "INNER JOIN users ON scores_relax.userid = users.id "
        "WHERE completed = 3 AND play_mode = :mode AND beatmaps.ranked IN (2, 3) "
        "AND users.privileges & 1 ORDER BY pp DESC LIMIT :count",
        {"mode": mode, "count": count}, 
    )
    if not top_scores:
        return None
    
    scores: list[Score] = []
    for db_score in top_scores:
        if db_score["id"] in SCORES:
            scores.append(SCORES[db_score["id"]])
            continue

        score = Score(
            beatmap_md5=db_score["beatmap_md5"],
            beatmap_id=db_score["beatmap_id"],
            mods=convert_mods(db_score["mods"]),
            mode=db_score["play_mode"],
            acc=db_score["accuracy"],
            n300=db_score["300_count"],
            n100=db_score["100_count"],
            n50=db_score["50_count"],
            ngeki=db_score["gekis_count"],
            nkatu=db_score["katus_count"],
            nmiss=db_score["misses_count"],
            combo=db_score["max_combo"],
            score=db_score["score"],
            akat_pp=db_score["pp"],
        )
        
        scores.append(score)
        SCORES[db_score["id"]] = score

    await asyncio.gather(*[calculate_pp(score) for score in scores])
    return ScoresResult(scores=scores)