````python
import os
import sys
import io
import traceback
import requests

from typing import TypedDict, List, Optional

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI


# ==========================================
# 1. LLM INITIALIZATION
# ==========================================

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY is not set. "
        "Set it as an environment variable before running the application."
    )

llm_flash = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    google_api_key=api_key
)

llm = llm_flash


# ==========================================
# 2. STATE DEFINITION
# ==========================================

class CrewState(TypedDict):
    messages: List[BaseMessage]
    next_step: Optional[str]
    code: Optional[str]
    report: Optional[str]


# ==========================================
# 3. TOOLS
# ==========================================

@tool
def run_python_code(code: str) -> str:
    """Execute Python code and return the standard output or error trace."""

    if not isinstance(code, str):
        code = str(code)

    clean_code = (
        code
        .replace("```python", "")
        .replace("```", "")
        .strip()
    )

    old_stdout = sys.stdout
    new_stdout = io.StringIO()

    sys.stdout = new_stdout

    try:
        local_scope = {}

        exec(clean_code, {}, local_scope)

        result = new_stdout.getvalue()

    except Exception:
        result = f"Execution Error:\n{traceback.format_exc()}"

    finally:
        sys.stdout = old_stdout

    return (
        result.strip()
        if result.strip()
        else "Success (no terminal output)"
    )


@tool
def generate_test_cases(task_description: str) -> str:
    """Generate specific test scenarios for a given coding task."""

    prompt = (
        "You are a Senior QA Engineer.\n\n"
        f"Generate 3 to 5 highly specific test scenarios "
        f"for the following coding task:\n\n"
        f"{task_description}\n\n"
        "Include both standard cases and edge cases.\n"
        "Return the scenarios as a numbered list."
    )

    response = llm.invoke(prompt)

    return (
        response.content
        if hasattr(response, "content")
        else str(response)
    )


@tool
def search_indian_history(topic: str) -> str:
    """Search Wikipedia for information about Indian history."""

    url = (
        "https://en.wikipedia.org/api/rest_v1/page/summary/"
        + topic.replace(" ", "_")
    )

    try:
        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            data = response.json()

            return data.get(
                "extract",
                "No historical information found."
            )

        return f"Could not find information about {topic}."

    except requests.RequestException as e:
        return f"History search failed: {e}"


# ==========================================
# 4. GRAPH NODES
# ==========================================

def task_input_node(state: CrewState):
    print("\n" + "=" * 50)
    print("--- NEW TASK INITIALIZATION ---")

    user_task = input(
        "Enter the coding task (or type 'exit' to quit): "
    ).strip()

    if user_task.lower() == "exit":
        return {
            "next_step": "exit"
        }

    return {
        "messages": [HumanMessage(content=user_task)],
        "next_step": "developer"
    }


def real_time_developer(state: CrewState):
    print("\n[Developer] Writing dynamic code using LLM...")

    task = state["messages"][-1].content

    dev_prompt = (
        "Write a clean Python script to solve the following coding task.\n\n"
        f"Task:\n{task}\n\n"
        "Only return the Python code. "
        "Do not include explanations or markdown formatting."
    )

    response = llm_flash.invoke(dev_prompt)

    content = response.content

    if isinstance(content, list):

        if content:
            first_item = content[0]

            if isinstance(first_item, dict):
                code_str = first_item.get("text", "")
            else:
                code_str = str(first_item)

        else:
            code_str = ""

    else:
        code_str = str(content)

    print("\n--- GENERATED CODE ---")
    print(code_str)

    return {
        "code": code_str
    }


def real_time_tester(state: CrewState):
    print("\n[Tester] Generating dynamic tests and executing code...")

    task = state["messages"][-1].content

    # Generate test scenarios
    test_cases = generate_test_cases.invoke(task)

    content = test_cases

    if isinstance(content, list):

        if content:
            first_item = content[0]

            if isinstance(first_item, dict):
                cases_str = first_item.get("text", "")
            else:
                cases_str = str(first_item)

        else:
            cases_str = ""

    else:
        cases_str = str(content)

    # Execute generated code
    execution_result = run_python_code.invoke(
        {
            "code": state["code"]
        }
    )

    # Compile report
    report = (
        "### EXECUTION OUTPUT:\n"
        f"{execution_result}\n\n"
        "### TEST SCENARIOS:\n"
        f"{cases_str}"
    )

    return {
        "report": report
    }


def manager_decision_node(state: CrewState):
    print("\n" + "=" * 50)
    print("--- MANAGER DASHBOARD : TEST REPORT ---")

    print(
        state.get(
            "report",
            "No report available."
        )
    )

    print("=" * 50)

    user_input = input(
        "\nCommand (store / another): "
    ).lower().strip()

    if user_input == "store":

        return {
            "next_step": "archiver"
        }

    return {
        "next_step": "task_input"
    }


def archiver_node(state: CrewState):
    print(
        "\n[Archiver] Task stored successfully. "
        "Closing workflow."
    )

    return {
        "next_step": "exit"
    }


# ==========================================
# 5. GRAPH CONSTRUCTION & ROUTING
# ==========================================

rt_workflow = StateGraph(CrewState)


# Nodes
rt_workflow.add_node(
    "task_input",
    task_input_node
)

rt_workflow.add_node(
    "developer",
    real_time_developer
)

rt_workflow.add_node(
    "tester",
    real_time_tester
)

rt_workflow.add_node(
    "manager_decision",
    manager_decision_node
)

rt_workflow.add_node(
    "archiver",
    archiver_node
)


# START → Task Input
rt_workflow.add_edge(
    START,
    "task_input"
)


# Task Input routing
def route_from_input(state: CrewState):

    if state.get("next_step") == "exit":
        return END

    return "developer"


rt_workflow.add_conditional_edges(
    "task_input",
    route_from_input
)


# Developer → Tester
rt_workflow.add_edge(
    "developer",
    "tester"
)


# Tester → Manager
rt_workflow.add_edge(
    "tester",
    "manager_decision"
)


# Manager routing
def route_from_decision(state: CrewState):

    if state.get("next_step") == "archiver":
        return "archiver"

    return "task_input"


rt_workflow.add_conditional_edges(
    "manager_decision",
    route_from_decision
)


# Archiver → END
rt_workflow.add_edge(
    "archiver",
    END
)


# Compile graph
rt_app = rt_workflow.compile()


print(
    "Interactive pipeline compiled and ready for live execution."
)


# ==========================================
# 6. EXECUTION
# ==========================================

if __name__ == "__main__":

    try:

        rt_app.invoke(
            {
                "messages": []
            },
            config={
                "recursion_limit": 50
            }
        )

    except KeyboardInterrupt:

        print(
            "\nStopped by user."
        )

    except Exception as e:

        print(
            f"\nAn error occurred: {e}"
        )
````

### What changed

The important change is that these are **gone**:

```python
import google.generativeai as genai
from google.colab import userdata
```

and this entire Colab-specific section:

```python
api_key = userdata.get("GEMINI_API_KEY")
genai.configure(api_key=api_key)
```

is replaced by:

```python
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is not set...")
```

So the application is no longer dependent on Google Colab. Your LangGraph structure itself remains the same.

### One more thing: `requirements.txt`

Your current requirements contain `fastapi`, `uvicorn`, `langserve`, `sse_starlette`, etc., even though this current `app.py` doesn't import those components.

For **this exact CLI version**, I'd simplify it to:

```text
langchain-core
langchain
langchain-google-genai
langgraph
requests
```

Then locally:

```bash
pip install -r requirements.txt
```

and set your API key.

**Important:** this version fixes the **Colab dependency**, but it does **not yet fix the deeper Tester issue or sandbox the `exec()` execution**. Those are separate changes and should be done deliberately, not smuggled into the same edit like software developers hiding cables behind a desk.
