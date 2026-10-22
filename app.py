import time
from typing import TypedDict

import streamlit as st
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Three-Agent Problem Solver",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Three-Agent Problem Solving System")
st.caption(
    "Built with LangGraph and Groq. Three agents collaborate to analyze, "
    "solve, and review your question."
)

with st.sidebar:
    st.header("About this project")
    st.write("**Framework:** LangGraph")
    st.write("**LLM provider:** Groq")
    st.write("**Model:** `openai/gpt-oss-20b`")
    st.write("**Agents:** Analyzer → Solver → Reviewer")
    st.info(
        "Each agent has a separate role. The workflow passes each agent's "
        "output to the next agent."
    )


# -----------------------------
# Shared LangGraph state
# -----------------------------
class State(TypedDict):
    user_input: str
    analysis: str
    solution: str
    review: str


# -----------------------------
# Create the LLM and graph once
# -----------------------------
@st.cache_resource
def build_graph():
    api_key = st.secrets.get("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing. Add it in Streamlit Community Cloud "
            "under App settings → Secrets."
        )

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,
        api_key=api_key,
    )

    def analyzer_agent(state: State):
        prompt = f"""
You are the Analyzer Agent in a three-agent problem-solving system.
Analyze the question and explain briefly what it is asking.
Do not solve it yet.

Question:
{state['user_input']}
"""
        response = llm.invoke(prompt)
        return {"analysis": response.content}

    def solver_agent(state: State):
        prompt = f"""
You are the Solver Agent.
Answer the user's question clearly and correctly, using the analysis below.

Question:
{state['user_input']}

Analysis from the Analyzer Agent:
{state['analysis']}

Provide a useful solution in simple language.
"""
        response = llm.invoke(prompt)
        return {"solution": response.content}

    def reviewer_agent(state: State):
        prompt = f"""
You are the Reviewer Agent.
Review the proposed answer for correctness, relevance, and clarity.
Correct or improve it if necessary. Return the final answer only.

Original question:
{state['user_input']}

Proposed solution:
{state['solution']}
"""
        response = llm.invoke(prompt)
        return {"review": response.content}

    workflow = StateGraph(State)
    workflow.add_node("analyzer", analyzer_agent)
    workflow.add_node("solver", solver_agent)
    workflow.add_node("reviewer", reviewer_agent)

    workflow.add_edge(START, "analyzer")
    workflow.add_edge("analyzer", "solver")
    workflow.add_edge("solver", "reviewer")
    workflow.add_edge("reviewer", END)

    return workflow.compile()


# -----------------------------
# User input
# -----------------------------
with st.form("question_form"):
    question = st.text_area(
        "Enter your question",
        placeholder="For example: What is machine learning?",
        height=120,
    )
    submitted = st.form_submit_button(
        "Run all three agents",
        type="primary",
        use_container_width=True,
    )

if submitted:
    if not question.strip():
        st.warning("Please enter a question first.")
    else:
        try:
            graph_app = build_graph()

            initial_state: State = {
                "user_input": question.strip(),
                "analysis": "",
                "solution": "",
                "review": "",
            }

            with st.spinner("The Analyzer, Solver, and Reviewer are working..."):
                start_time = time.perf_counter()
                result = graph_app.invoke(initial_state)
                elapsed = time.perf_counter() - start_time

            st.session_state["last_question"] = question.strip()
            st.session_state["last_result"] = result
            st.session_state["last_elapsed"] = elapsed

        except Exception as exc:
            st.error(
                "The workflow could not run. Check that your Groq API key is "
                "configured correctly and that your Groq account can access "
                "the selected model."
            )
            st.exception(exc)


# -----------------------------
# Display the agent outputs
# -----------------------------
if "last_result" in st.session_state:
    result = st.session_state["last_result"]
    elapsed = st.session_state["last_elapsed"]

    st.divider()
    st.subheader("Agent Dashboard")
    st.write("**Question:**", st.session_state["last_question"])

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Agents executed", "3")
    with col2:
        st.metric("Total response time", f"{elapsed:.2f} sec")
    with col3:
        st.metric("Workflow status", "Completed")

    with st.expander("1. Analyzer Agent — Understand the question", expanded=True):
        st.markdown(result.get("analysis", "No analysis was returned."))

    with st.expander("2. Solver Agent — Generate a solution", expanded=True):
        st.markdown(result.get("solution", "No solution was returned."))

    with st.expander("3. Reviewer Agent — Check and improve the solution", expanded=True):
        st.markdown(result.get("review", "No final answer was returned."))

    st.subheader("Final Answer")
    st.markdown(result.get("review", "No final answer was returned."))

    st.caption(
        "Note: response time includes all three LLM calls and may vary with "
        "network conditions, model response time, and API limits."
    )
else:
    st.info(
        "Enter a question above and click **Run all three agents** to see "
        "each agent's output and the final reviewed answer."
    )
