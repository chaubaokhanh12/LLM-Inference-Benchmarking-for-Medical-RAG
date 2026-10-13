"""Answer quality of a benchmark run, scored by one fixed LLM judge (E3 and Ollama).

    python -m bench.quality_eval results/<run_id> [--n 100]
    python -m bench.quality_eval results/<fp16_run> results/<awq_run>   # + paired comparison

For the first N prompts (by id) that have a reference answer, the run's generated answer
(outputs.jsonl, lowest concurrency, first repeat) is judged on
  faithfulness  share of the answer's claims supported by the context in the prompt
  correctness   1-5 agreement with the reference answer
Judgments are cached per run (quality_judgments.jsonl); quality_summary.json holds the means.
With two runs, prompts whose correctness differs by >= 2 points are counted and listed.
Needs OPENAI_API_KEY; the judge model is --judge-model (default gpt-4.1-mini).
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

JUDGE_SYSTEM = """You grade answers of a medical question-answering system that must answer only from the passages in its prompt.
You get the prompt the system received (passages + question), its answer, and a reference answer.
1. List the factual claims in the answer (ignore citations and advice to see a doctor); for each decide whether the passages support it.
2. "correctness" 1-5 against the reference answer (5 = same facts, 1 = contradicts or misses them).
Return JSON: {"claims": [{"claim": "...", "supported": true|false}], "correctness": <1-5>}"""


def load_run(run_dir: Path) -> tuple[dict, dict[str, str], str]:
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    outputs: dict[str, tuple[int, int, str]] = {}
    for line in (run_dir / "outputs.jsonl").read_text(encoding="utf-8").splitlines():
        o = json.loads(line)
        key = (o["concurrency"], o["repeat"])
        if not o["error"] and (o["prompt_id"] not in outputs or key < outputs[o["prompt_id"]][:2]):
            outputs[o["prompt_id"]] = (*key, o["text"])
    prompts_path = Path(cfg["prompts"])
    if not prompts_path.is_absolute():
        prompts_path = (Path(__file__).resolve().parents[1] / prompts_path).resolve()
    return cfg, {k: v[2] for k, v in outputs.items()}, str(prompts_path)


def judge_run(run_dir: Path, n: int, model: str) -> dict:
    from openai import OpenAI

    cfg, answers, prompts_path = load_run(run_dir)
    prompts = {}
    with open(prompts_path, encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            if p.get("reference_answer") and p["id"] in answers:
                prompts[p["id"]] = p
    ids = sorted(prompts)[:n]
    cache_file = run_dir / "quality_judgments.jsonl"
    cache = {}
    if cache_file.exists():
        for line in cache_file.read_text(encoding="utf-8").splitlines():
            j = json.loads(line)
            cache[(j["prompt_id"], j["judge_model"])] = j
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def one(pid: str) -> dict:
        if (pid, model) in cache:
            return cache[(pid, model)]
        user_msg = next(m["content"] for m in prompts[pid]["messages"] if m["role"] == "user")
        r = client.chat.completions.create(
            model=model, response_format={"type": "json_object"}, max_completion_tokens=1200,
            messages=[{"role": "system", "content": JUDGE_SYSTEM},
                      {"role": "user", "content": f"Prompt:\n{user_msg}\n\nAnswer:\n{answers[pid]}\n\n"
                                                  f"Reference answer: {prompts[pid]['reference_answer']}"}])
        v = json.loads(r.choices[0].message.content or "{}")
        claims = v.get("claims") or []
        return {"prompt_id": pid, "judge_model": model,
                "faithfulness": sum(bool(c.get("supported")) for c in claims) / len(claims) if claims else None,
                "correctness": v.get("correctness")}

    with ThreadPoolExecutor(8) as ex:
        results = list(ex.map(one, ids))
    with open(cache_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    faith = [r["faithfulness"] for r in results if r["faithfulness"] is not None]
    corr = [r["correctness"] for r in results if isinstance(r["correctness"], (int, float))]
    summary = {"config_name": cfg["config_name"], "run": run_dir.name, "judge_model": model, "n": len(results),
               "faithfulness_mean": sum(faith) / len(faith) if faith else None,
               "correctness_mean": sum(corr) / len(corr) if corr else None}
    (run_dir / "quality_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"summary": summary, "by_prompt": {r["prompt_id"]: r for r in results}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", type=Path, nargs="+")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--judge-model", default=os.environ.get("EVAL_LLM_MODEL", "gpt-4.1-mini"))
    args = ap.parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("set OPENAI_API_KEY")
    judged = [judge_run(r, args.n, args.judge_model) for r in args.runs]
    for j in judged:
        print(json.dumps(j["summary"]))
    if len(judged) == 2:
        a, b = (j["by_prompt"] for j in judged)
        diffs = [(pid, a[pid]["correctness"], b[pid]["correctness"]) for pid in sorted(set(a) & set(b))
                 if isinstance(a[pid]["correctness"], (int, float)) and isinstance(b[pid]["correctness"], (int, float))
                 and abs(a[pid]["correctness"] - b[pid]["correctness"]) >= 2]
        print(f"{len(diffs)} prompts differ by >= 2 correctness points: {diffs}")


if __name__ == "__main__":
    main()
