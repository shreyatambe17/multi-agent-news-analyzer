import gradio as gr
from langgraph.graph import StateGraph

# ---------------- STATE ----------------
# Explicitly define state keys for LangGraph
class NewsState(dict):
    text: str
    facts: list
    bias_words: list
    score: float
    remark: str
    logs: list


# ---------------- AGENTS ----------------

# Summarizer Agent
def summarizer_agent(text: str) -> str:
    sentences = text.split(".")
    summary = ".".join(sentences[:5])
    return summary


# Fact Extractor Agent
def fact_extractor(state: NewsState) -> NewsState:
    text = state["text"]
    facts = [s.strip() for s in text.split(".") if " is " in s or " was " in s]
    state["facts"] = facts
    state["logs"].append("Fact Extractor Agent executed")
    return state


# Bias Analyzer Agent
def bias_analyzer(state: NewsState) -> NewsState:
    text = state["text"].lower()
    biased_words = ["shocking", "disaster", "failure", "corrupt", "outrage"]
    found_bias = [w for w in biased_words if w in text]
    state["bias_words"] = found_bias
    state["logs"].append("Bias Analyzer Agent executed")
    return state


# Supervisor Agent
def supervisor_agent(state: NewsState) -> NewsState:
    bias_count = len(state["bias_words"])

    if bias_count == 0:
        score = 0.9
        remark = "Mostly Neutral"
    elif bias_count <= 2:
        score = 0.6
        remark = "Moderately Biased"
    else:
        score = 0.3
        remark = "Highly Biased"

    state["score"] = score
    state["remark"] = remark
    state["logs"].append("Supervisor Agent calculated neutrality score")
    return state


# ---------------- LANGGRAPH ----------------
graph = StateGraph(NewsState)

graph.add_node("FactAgent", fact_extractor)
graph.add_node("BiasAgent", bias_analyzer)
graph.add_node("Supervisor", supervisor_agent)

graph.set_entry_point("FactAgent")
graph.add_edge("FactAgent", "BiasAgent")
graph.add_edge("BiasAgent", "Supervisor")

app = graph.compile()


# ---------------- CONTROLLER FUNCTION ----------------
def run_news_analyzer(text: str):

    # Failure case
    if text.strip() == "":
        return "❌ No article provided", "", "", ""

    logs = []

    # 👉 Agentic conditional logic
    if len(text) > 1000:
        text = summarizer_agent(text)
        logs.append("Article length > 1000 → Summarizer Agent executed")

    # IMPORTANT: Initialize full state BEFORE invoking LangGraph
    state = NewsState(
        text=text,
        facts=[],
        bias_words=[],
        score=0.0,
        remark="",
        logs=logs
    )

    result = app.invoke(state)

    return (
        "\n".join(result["facts"]) or "No factual statements found",
        ", ".join(result["bias_words"]) or "No bias indicators found",
        f"{result['score']} ({result['remark']})",
        "\n".join(result["logs"])
    )


# ---------------- GRADIO UI ----------------
ui = gr.Interface(
    fn=run_news_analyzer,
    inputs=gr.Textbox(lines=10, label="Paste News Article"),
    outputs=[
        gr.Textbox(label="Extracted Facts"),
        gr.Textbox(label="Bias Indicators"),
        gr.Textbox(label="Neutrality Score"),
        gr.Textbox(label="Reasoning Log")
    ],
    title="Multi-Agent News Analyzer (Agentic AI)",
    description="Multi-agent system with conditional logic, supervision, and LangGraph"
)

ui.launch()