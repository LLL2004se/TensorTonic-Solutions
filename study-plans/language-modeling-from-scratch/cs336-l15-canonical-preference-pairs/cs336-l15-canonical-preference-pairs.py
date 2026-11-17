
def canonical_preference_pairs(records: list) -> dict:
    """
    Returns a dict with pairs: a list of prompt_id, winner_id,
    loser_id dictionaries.
    """

    pairs = []

    for record in records:
        prompt_id = record["prompt_id"]

        # Sort candidates by ascending rank
        candidates = sorted(
            record["candidates"],
            key=lambda candidate: candidate["rank"]
        )

        # Generate every winner-loser combination
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                pairs.append({
                    "prompt_id": prompt_id,
                    "winner_id": candidates[i]["id"],
                    "loser_id": candidates[j]["id"]
                })

    return {"pairs": pairs}
