"""
Convert EZStance SubtaskA mixed test set to SemEval format.

Combines noun_phrase (2,663) + claim (5,135) test rows = 7,798 total.
Also writes noun_phrase-only version as ezstance_test_nounphrase.csv.

Output columns match semeval2016_task6_stance.csv:
  id, content, query, stance_label, stance_label_original, source_dataset
  + domain (CE/WE/EdC/EnC/S/R/EP/P) for noun-phrase rows
"""

import pandas as pd
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUBTASK_A = os.path.join(BASE, "ezstance", "subtaskA")

STANCE_LOWER = {"AGAINST": "against", "FAVOR": "favor", "NONE": "none"}

KEYWORD_TO_DOMAIN = {
    "epidemic prevention": "CE", "living with covid": "CE", "herd immunity": "CE",
    "WFH": "CE", "booster": "CE", "vaccine": "CE", "mask mandate": "CE",
    "FDA": "CE", "post-covid": "CE", "Fauci": "CE", "health insurance": "CE",
    "world news": "WE", "Ukraine": "WE", "Russia": "WE", "migrant": "WE",
    "NATO": "WE", "China": "WE", "Mideast": "WE", "Negative population growth": "WE",
    "terrorism": "WE",
    "public education": "EdC", "pop culture": "EdC", "cultural output": "EdC",
    "home schooling": "EdC", "AI assistance writing": "EdC", "arming teachers": "EdC",
    "teacher carry gun": "EdC", "private education": "EdC", "international student": "EdC",
    "prices": "EnC", "gasoline price": "EnC", "online shopping": "EnC",
    "tictok": "EnC", "iphone": "EnC", "reels": "EnC", "Disney": "EnC",
    "medical insurance": "EnC", "ethical consumption": "EnC", "vegetarian": "EnC",
    "world cup": "S", "NBA": "S", "men's football": "S", "women's football": "S",
    "NCAA": "S", "MLB's rule change": "S", "NFL": "S", "WWE": "S",
    "gender equality": "R", "equal rights": "R", "women's rights": "R",
    "LGBTQ": "R", "BLM": "R", "doctors and patients": "R", "racism": "R",
    "asian hate": "R", "gun control": "R",
    "climate change": "EP", "clean energy": "EP", "environmental awareness": "EP",
    "environmental protection agency": "EP", "shut down coal plants": "EP",
    "nuclear energy": "EP", "electric vihicles": "EP",
    "government": "P", "republican": "P", "reform": "P", "leftists": "P",
    "democrat": "P", "democracy": "P", "right wing": "P", "politic": "P",
    "presidential election": "P", "mid-term election": "P",
}


def load_and_convert(path, source_tag, include_domain=False):
    df = pd.read_csv(path)
    out = pd.DataFrame({
        "content":               df["Text"].str.strip(),
        "query":                 df["Target 1"].str.strip(),
        "stance_label":          df["Stance 1"].map(STANCE_LOWER),
        "stance_label_original": df["Stance 1"],
        "source_dataset":        source_tag,
    })
    if include_domain:
        out["domain"] = df["Keywords"].map(KEYWORD_TO_DOMAIN)
    return out


def save(df, out_path):
    df = df.reset_index(drop=True)
    df.insert(0, "id", range(1, len(df) + 1))
    assert df["stance_label"].isna().sum() == 0, "Unmapped stance labels"
    assert df["content"].isna().sum() == 0, "Null content"
    assert df["query"].isna().sum() == 0, "Null query"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows -> {out_path}")
    print(f"  Labels: {df['stance_label_original'].value_counts().to_dict()}")


def main():
    np_df = load_and_convert(
        os.path.join(SUBTASK_A, "noun_phrase", "raw_test_all_onecol.csv"),
        "EZStance-SubtaskA-NounPhrase-test",
        include_domain=True,
    )
    cl_df = load_and_convert(
        os.path.join(SUBTASK_A, "claim", "raw_test_all_onecol.csv"),
        "EZStance-SubtaskA-Claim-test",
        include_domain=True,
    )

    print(f"Noun phrase rows: {len(np_df)}")
    print(f"Claim rows:       {len(cl_df)}")

    # Noun-phrase only — primary test file
    save(np_df, os.path.join(BASE, "data_in", "ezstance_test.csv"))

    # Mixed (noun_phrase + claim) — saved for later
    mixed = pd.concat([np_df, cl_df], ignore_index=True)
    save(mixed, os.path.join(BASE, "data_in", "ezstance_test_mixed.csv"))


if __name__ == "__main__":
    main()
