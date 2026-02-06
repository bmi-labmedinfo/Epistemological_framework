from __future__ import annotations
from pathlib import Path
from typing import Iterable, Tuple
from computational_implementation.medical_reasoning_phases.abduction import *
from computational_implementation.medical_reasoning_phases.ranking import *
from computational_implementation.medical_reasoning_phases.deduction import *
from computational_implementation.medical_reasoning_phases.induction import *


class RankedDiagnosticHypothesis(BaseModel):
    diagnosis: str = Field(..., description="The diagnosis.")
    position: Rank = Field(..., description="The ranking position of the hypothesis.")


class RankedHypothesesList(BaseModel):
    hypotheses: List[RankedDiagnosticHypothesis] = Field(
        description="List of diagnostic hypotheses ranked by plausibility (rank 1 = most plausible).")


class HypothesisDeduction(BaseModel):
    diagnosis: str = Field(..., description="Short name of the diagnostic hypothesis or disease entity.")
    predicted_consequences: List[ExpectedFinding] = Field(...,
                                                          description="List of expected clinical consequences (predictions) if this hypothesis is true.")


class DeductionPlan(BaseModel):
    hypotheses: List[HypothesisDeduction] = Field(
        description="Set of hypotheses with their expected consequences, as produced by the deduction (prediction) phase.",
    )


class EvaluatedHypothesis(BaseModel):
    diagnosis: str = Field(..., description="Short name of the diagnostic hypothesis or disease entity.")
    evaluation: Literal["plausible", "refuted"] = Field(...,
                                                        description="Global inductive judgement on the hypothesis after matching expectations with observed findings: 'plausible' or 'refuted'",
                                                        )
    explanation: str = Field(...,
                             description="Short explanation for why the hypothesis is plausible or refuted, based on the match between expected and observed findings.",
                             )
    termination_recommendation: Literal[
        "sufficient_explanation_reached", "continue_testing", "reopen_abduction"] = Field(...,
                                                                                          description=(
                                                                                              "Recommendation about the next step for this hypothesis: "
                                                                                              "'sufficient_explanation_reached' if the diagnostic process could reasonably stop here; "
                                                                                              "'continue_testing' if further tests on this hypothesis are warranted; "
                                                                                              "'reopen_abduction' if this hypothesis is weak and a new abduction step should be considered."
                                                                                          )
                                                                                          )
    evaluated_findings: List[EvaluatedExpectedFinding] = Field(...,
                                                               description="List of expected findings for this hypothesis, each evaluated against the observed data."
                                                               )


class InductionResult(BaseModel):
    hypotheses: List[EvaluatedHypothesis] = Field(
        description="Set of diagnostic hypotheses evaluated by induction after the deduction phase (matching expected vs observed findings and deciding on next steps).",
    )


def _model_validate(model_cls, obj):
    if hasattr(model_cls, "model_validate"):  # pydantic v2
        return model_cls.model_validate(obj)
    return model_cls.parse_obj(obj)           # pydantic v1

def _indent(text: str, n: int = 2) -> str:
    pad = " " * n
    return "\n".join(pad + line if line.strip() else line for line in text.splitlines())

def _status_icon(s: str) -> str:
    return {"confirmed": "✅", "not_observed": "❓", "contradicted": "❌"}.get(s, "•")

def _prio_tag(p: str) -> str:
    return {"high": "🔴 HIGH", "medium": "🟠 MED", "low": "🟢 LOW"}.get(p, p)

def _kind_tag(k: str) -> str:
    return {
        "manifestation": "🩺 manifestation",
        "test_result": "🧪 test_result",
        "clinical_course": "⏳ clinical_course",
    }.get(k, k)

def _normalize_chunk(chunk: Any) -> Tuple[Tuple[Any, ...], Any]:
    """
    Se subgraphs=True, LangGraph può emettere (namespace_tuple, data).
    In caso normale, chunk è solo data.
    """
    if isinstance(chunk, tuple) and len(chunk) == 2 and isinstance(chunk[0], tuple):
        return chunk[0], chunk[1]
    return (), chunk

def _walk(obj: Any) -> Iterable[Any]:
    """Traverse ricorsivo su dict/list per trovare payload annidati."""
    yield obj
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v)

def _extract_abstraction_list(update: Any) -> Optional[List[Dict[str, Any]]]:
    """
    Cerca una lista di dict con chiavi: category, finding, explanation.
    L'update può essere:
      - direttamente la lista
      - oppure un dict che contiene quella lista
    """
    def looks_like_item(x):
        return isinstance(x, dict) and {"category", "finding", "explanation"}.issubset(x.keys())

    if isinstance(update, list) and update and all(looks_like_item(x) for x in update):
        return update

    if isinstance(update, dict):
        for v in update.values():
            if isinstance(v, list) and v and all(looks_like_item(x) for x in v):
                return v

    return None

def _try_parse_anywhere(update: Any, model_cls):
    """
    Prova a validare:
      1) update intero
      2) qualunque sotto-dict annidato in update
    """
    for candidate in _walk(update):
        if isinstance(candidate, dict):
            try:
                return _model_validate(model_cls, candidate)
            except Exception:
                pass
    return None


def fmt_abstraction(update: Any) -> str:
    items = _extract_abstraction_list(update)
    if not items:
        # fallback human: mostra solo chiavi e tipi
        if isinstance(update, dict):
            keys = ", ".join(update.keys())
            return f"🧱 ABSTRACTION — (no standard list found)\nKeys in update: {keys}"
        return "🧱 ABSTRACTION — (no standard list found)"

    out = ["🧱 ABSTRACTION — The features abstracted by the text are:"]
    for i, it in enumerate(items, 1):
        out.append(
            f'* [{i}] The category is "{it.get("category")}" for the finding "{it.get("finding")}" '
            f'and the explanation is "{it.get("explanation")}".'
        )
    return "\n".join(out)

def fmt_abduction(update: Any, label: str) -> str:
    obj = _try_parse_anywhere(update, AbductionHypothesesList)
    if not obj:
        if isinstance(update, dict):
            return f"🧠 {label} — (could not validate AbductionHypothesesList)\nKeys: {', '.join(update.keys())}"
        return f"🧠 {label} — (could not validate AbductionHypothesesList)"

    out = [f"🧠 {label} — Diagnostic hypotheses generated:"]
    for i, h in enumerate(obj.hypotheses, 1):
        out.append(f"\n{i}. {h.diagnosis}")
        out.append(_indent(f"Why: {h.explanation}", 2))
        feats = h.supporting_features or []
        if feats:
            out.append(_indent("Supporting features:", 2))
            for f in feats:
                out.append(_indent(f"- {f}", 4))
    return "\n".join(out)

def _unwrap_to_ranked_list(update: Any) -> Any:
    payload = update

    # unwrap up to a few levels (LangGraph often wraps node outputs)
    for _ in range(5):
        if not isinstance(payload, dict):
            break

        # case: {"ranking": {...}, "stop":..., "iter":...}
        if "ranking" in payload and "hypotheses" not in payload:
            payload = payload["ranking"]
            continue

        # case: {"node_name": {...}}  (single-key wrapper)
        if "hypotheses" not in payload and len(payload) == 1:
            payload = next(iter(payload.values()))
            continue

        break

    return payload

def fmt_ranking(update: Any) -> str:
    payload = _unwrap_to_ranked_list(update)
    #print(payload)
    obj = _try_parse_anywhere(payload, RankedHypothesesList)
    if not obj:
        if isinstance(update, dict):
            return (
                "🏁 RANKING — (could not validate RankedHypothesesList)\n"
                f"Top-level keys: {', '.join(update.keys())}"
            )
        return "🏁 RANKING — (could not validate RankedHypothesesList)"

    out = ["🏁 RANKING — Hypotheses ordered by plausibility:"]
    items = list(obj.hypotheses)
    try:
        items.sort(key=lambda x: x.position.rank)
    except Exception:
        pass

    for rh in items:
        pos = rh.position
        out.append(f"\n#{pos.rank} — {rh.diagnosis}")
        out.append(_indent(f"Parsimony: {pos.parsimony}", 2))
        out.append(_indent(f"Danger: {pos.danger}", 2))
        out.append(_indent(f"Cost: {pos.cost}", 2))
        out.append(_indent(f"Curability: {pos.curability}", 2))

    return "\n".join(out)

def fmt_deduction(update: Any) -> str:
    obj = _try_parse_anywhere(update, DeductionPlan)
    if not obj:
        if isinstance(update, dict):
            return f"🔮 DEDUCTION — (could not validate DeductionPlan)\nKeys: {', '.join(update.keys())}"
        return "🔮 DEDUCTION — (could not validate DeductionPlan)"

    out = ["🔮 DEDUCTION — Expected consequences per hypothesis:"]
    order = {"high": 0, "medium": 1, "low": 2}

    for h in obj.hypotheses:
        out.append(f"\n• {h.diagnosis}")
        preds = sorted(h.predicted_consequences, key=lambda e: (order.get(e.priority, 9), e.kind))
        for ef in preds:
            out.append(_indent(f"- [{_prio_tag(ef.priority)} | {_kind_tag(ef.kind)}] {ef.description}", 2))
    return "\n".join(out)

def fmt_induction(update: Any) -> str:
    obj = _try_parse_anywhere(update, InductionResult)
    if not obj:
        if isinstance(update, dict):
            return f"🧾 INDUCTION — (could not validate InductionResult)\nKeys: {', '.join(update.keys())}"
        return "🧾 INDUCTION — (could not validate InductionResult)"

    out = ["🧾 INDUCTION — Matching expected vs observed:"]
    for h in obj.hypotheses:
        out.append(f"\n• {h.diagnosis}")
        out.append(_indent(f"Evaluation: {h.evaluation}", 2))
        out.append(_indent(f"Termination: {h.termination_recommendation}", 2))
        out.append(_indent(f"Why: {h.explanation}", 2))

        efs = h.evaluated_findings or []
        if efs:
            # mini-summary
            counts = {"confirmed": 0, "not_observed": 0, "contradicted": 0}
            for ef in efs:
                if ef.status in counts:
                    counts[ef.status] += 1
            out.append(_indent(
                f"Findings summary: ✅ {counts['confirmed']} | ❓ {counts['not_observed']} | ❌ {counts['contradicted']}",
                2
            ))
            out.append(_indent("Findings:", 2))
            for ef in efs:
                out.append(_indent(f"- {_status_icon(ef.status)} {ef.description}", 4))
                out.append(_indent(f"↳ {ef.comment}", 6))
    return "\n".join(out)


NODE_TITLES = {
    "abstraction": "Abstraction Phase",
    "abduction_unfocused": "Abduction Phase (Unfocused)",
    "abduction_focused": "Abduction Phase (Focused)",
    "ranking": "Ranking Phase",
    "deduction": "Deduction Phase",
    "induction": "Induction Phase",
}

def _format_node(node_name: str, update: Any) -> str:
    if node_name == "abstraction":
        return fmt_abstraction(update)
    if node_name == "abduction_unfocused":
        return fmt_abduction(update, "ABDUCTION (UNFOCUSED)")
    if node_name == "abduction_focused":
        return fmt_abduction(update, "ABDUCTION (FOCUSED)")
    if node_name == "ranking":
        return fmt_ranking(update)
    if node_name == "deduction":
        return fmt_deduction(update)
    if node_name == "induction":
        return fmt_induction(update)

    # fallback human
    if isinstance(update, dict):
        return f"Update keys: {', '.join(update.keys())}"
    return f"Update type: {type(update).__name__}"


def run_with_human_log(app, inputs: dict, log_path: str = "langgraph_human.log", subgraphs: bool = False):

    p = Path(log_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    final_state = None

    with p.open("a", encoding="utf-8") as f:
        for mode, chunk in app.stream(inputs, {"recursion_limit": 60},stream_mode=["updates", "values"], subgraphs=subgraphs):
            ns, data = _normalize_chunk(chunk)

            if mode == "values":
                final_state = data
                continue

            if mode != "updates":
                continue

            # in updates: data tipicamente è { "node_name": update }
            if not isinstance(data, dict):
                continue

            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ns_str = " / ".join(map(str, ns)) if ns else ""

            for node_name, update in data.items():
                title = NODE_TITLES.get(node_name, node_name)
                header = f"\n----- {title} -----"
                if ns_str:
                    header += f" (ns={ns_str})"
                header += f"\n[{ts}]\n"

                f.write(header)
                f.write(_format_node(node_name, update))
                f.write("\n")
                f.flush()

    return final_state