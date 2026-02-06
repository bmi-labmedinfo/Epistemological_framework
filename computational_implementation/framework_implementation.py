from langgraph.graph import StateGraph, END
from IPython.display import Image, display
from computational_implementation.medical_reasoning_phases.abstraction import *
from computational_implementation.medical_reasoning_phases.abduction import *
from computational_implementation.medical_reasoning_phases.ranking import *
from computational_implementation.medical_reasoning_phases.deduction import *
from computational_implementation.medical_reasoning_phases.induction import *

class EpistState(TypedDict, total=False):
    case_text: str
    model: str
    clust: str
    top_k: int

    abstraction: list
    abduction_unfocused: dict
    abduction_focused: dict
    ranking: dict
    deduction: dict
    induction: dict
    induction_plausible: dict

    reranked: bool

    # loop control
    iter: int

    suffix: str
    stop: bool


#NODES DEFINITION
def n_abstraction(s: EpistState) -> EpistState:
    out = query_abstraction(
        case=s["case_text"],
        model=s["model"],
        iteration=s["iter"])
    return {"abstraction": out}

def route_after_abstraction(s: EpistState) -> str:
    """
    - If abstraction is empty: end
    """
    if s.get("abstraction") is not None:
        return "continue"
    print(">>>Abstraction is empty")
    return "end"

def n_abduction_unfocused(s: EpistState) -> EpistState:
    top_k=s["top_k"]
    out = query_abduction_unfocused(
        case=s["case_text"],
        abstraction_output=s["abstraction"],
        model=s["model"],
        max_unfocused=s["top_k"],
        iteration=s["iter"])
    return {"abduction_unfocused": out, "top_k": top_k}

def n_abduction_focused(s: EpistState) -> EpistState:
    out = query_abduction_focused(
        case=s["case_text"],
        abstraction_output=s.get("abstraction"),
        unfocused_output=s.get("abduction_unfocused"),
        model=s["model"],
        max_focused=s.get("top_k"),
        iteration=s["iter"])
    return {"abduction_focused": out}

def n_ranking(s: EpistState) -> EpistState:
    """
    - First ranking: ranking from abduction_focused
    - After induction: reranking using induction (then end)
    """
    iter_ = s.get("iter", 0)

    # if induction is present, do rerank
    if s.get("induction") is not None and not s.get("reranked", False):
        abduction_like = {"hypotheses": s["induction"]["hypotheses"]}
        out = query_rank_hypotheses(
            case=s["case_text"],
            model=s["model"],
            abduction_output=abduction_like,
            iteration=iter_
        )
        out["hypotheses"].sort(key=lambda h: h["position"]["rank"])

        return {"ranking": out, "reranked": True}

    # first ranking from abduction_focused
    out = query_rank_hypotheses(
        case=s["case_text"],
        model=s["model"],
        abduction_output=s.get("abduction_focused"),
        iteration=iter_
    )
    out["hypotheses"].sort(key=lambda h: h["position"]["rank"])
    return {"ranking": out, "reranked": False}

def n_deduction(s: EpistState) -> EpistState:
    out = query_deduction(
        ranking_output=s["ranking"],
        model=s["model"],
        iteration=s.get("iter", 0)
    )
    return {"deduction": out}

def n_induction(s: EpistState) -> EpistState:
    out_all, out_plausible = query_induction(
        case=s["case_text"],
        model=s["model"],
        deduction_output=s["deduction"],
        iteration=s.get("iter", 0)
    )

    iter_next = s.get("iter", 0) + 1
    return {
        "induction": out_all,
        "induction_plausible": out_plausible,
        "iter": iter_next
    }

def route_after_ranking(s: EpistState) -> str:
    """
    - After first ranking -> continue (deduction/induction)
    - After reranking (ranking post-induction) -> end
    """
    if s.get("reranked", False):
        return "end"
    return "continue"


def graph_definition(configs:Any):
    g = StateGraph(EpistState)

    g.add_node("abstraction", n_abstraction)
    g.add_node("abduction_unfocused", n_abduction_unfocused)
    g.add_node("abduction_focused", n_abduction_focused)
    g.add_node("ranking", n_ranking)
    g.add_node("deduction", n_deduction)
    g.add_node("induction", n_induction)

    g.set_entry_point("abstraction")

    g.add_edge("abstraction", "abduction_unfocused")
    g.add_conditional_edges(
        "abstraction",
        route_after_abstraction,
        {"continue": "abduction_unfocused", "end": END},
    )
    g.add_edge("abduction_unfocused", "abduction_focused")
    g.add_edge("abduction_focused", "ranking")

    # QUI la decisione di uscire dal ciclo
    g.add_conditional_edges(
        "ranking",
        route_after_ranking,
        {"continue": "deduction", "end": END},
    )

    g.add_edge("deduction", "induction")
    g.add_edge("induction", "ranking")

    app = g.compile()
    display(Image(app.get_graph().draw_mermaid_png()))
    return app