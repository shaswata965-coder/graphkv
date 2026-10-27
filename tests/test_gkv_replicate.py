"""Paper-table replication: scorer per row, paper lookup, matching released outputs, the report."""

import json
import os

from graphkv_neurips.replicate import paper_value, released_outputs, report, same_text, scorer


def test_scorer_follows_the_printed_numbers():
    assert scorer("morehopqa", "Graph-KV Top-3") == "morehopqa-strict"
    assert scorer("morehopqa", "Graph-KV Full") == "morehopqa"
    assert scorer("2wiki", "Graph-KV Full") == "2wiki"


def test_paper_value():
    assert paper_value("8B-Block", "Graph-KV Top-3", "2wiki", "all") == 75.15
    assert paper_value("8B-Block", "Block-RAG", "hqa", "bridge") == 71.59
    assert paper_value("8B-Block", "Sequential", "nqa", "all") == 63.21
    assert paper_value("8B-Block", "Graph-KV Full", "morehopqa", "hop5") == 47.25
    assert paper_value("8B-SFT", "PCW", "morehopqa", "all") == 11.35  # Tables 5/6
    assert paper_value("8B-Block", "PCW", "2wiki", "all") is None


def _write(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)


def test_released_outputs_are_matched_by_question(tmp_path):
    rows = [{"question": f"q{i}", "generated": f"answer {i}"} for i in range(5)]
    rows[3].pop("generated")
    _write(tmp_path / "results/morehopqa/ldsjmdyTulu3-Block-FT/gapemp_ascent.jsonl", rows[::-1])  # another order
    got = released_outputs("morehopqa", "ldsjmdy/Tulu3-Block-FT", "gapemp", {0: "q0", 3: "q3", 4: "q4", 7: "q7"}, str(tmp_path))
    assert got == {0: "answer 0", 3: None, 4: "answer 4"}
    assert released_outputs("morehopqa", "ldsjmdy/Tulu3-Block-FT", None, {0: "q0"}, str(tmp_path)) == {}


def test_same_text_ignores_white_space_only():
    assert same_text("Paris , France.", "Paris, France.<|eot_id|>")
    assert not same_text("Paris", "Lyon")


def test_report(tmp_path):
    out = tmp_path / "run"
    cfg = dict(model="m", tasks=["2wiki"], methods=["sequential", "block"], released_model="ldsjmdy/Tulu3-Block-FT",
               max_new_tokens=1024, n_total={"2wiki": 12576})
    _write(out / "run_config.json", [cfg])
    qs = [("q0", "comparison", "yes"), ("q1", "inference", "Paris"), ("q2", "inference", "Rome")]
    for label, texts in [("sequential", ["yes", "Paris", "Lyon"]), ("block", ["no", "Paris", "Rome"])]:
        _write(out / "2wiki" / f"{label}.jsonl",
               [dict(index=i, group=g, question=q, answers=[a], generated=t) for i, ((q, g, a), t) in enumerate(zip(qs, texts))])
    data = tmp_path / "data"
    _write(data / "results/2wiki/ldsjmdyTulu3-Block-FT/vanilla_ascent.jsonl",
           [{"question": q, "generated": t} for (q, _, _), t in zip(qs, ["yes", "Paris", "Rome"])])
    _write(data / "results/2wiki/ldsjmdyTulu3-Block-FT/block_ascent.jsonl", [{"question": "another", "generated": "x"}])
    md = report(str(out), str(data))
    assert "| Sequential | ours | 100.0 | 50.0 | 66.7 | 2/3 | 2/3 |" in md
    assert "| | released, same questions | 100.0 | 100.0 | 100.0 | | |" in md
    assert "| | paper, all questions | 84.24 | 63.08 | 77.29 | | |" in md
    assert "| Block-RAG | ours | 0.0 | 100.0 | 66.7 | not released |  |" in md
    assert "| Sequential | 66.7 / 100.0 / 77.29 | 2/3 | 2/3 |" in md
