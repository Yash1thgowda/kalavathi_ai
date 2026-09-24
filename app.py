````python
import os
import io
import traceback
from typing import TypedDict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END


# ============================================================
# 1. FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Kalavathi AI",
    description="AI-powered Developer, Tester and Manager workflow",
    version="1.0.0"
)


# ============================================================
# 2. API KEY / LLM INITIALIZATION
# ============================================================

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError(
        "GEMINI_API_KEY environment variable is not set."
    )


llm_flash = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    google_api_key=api_key
)

llm = llm_flash


# ============================================================
# 3. STATE
# ============================================================

class CrewState(TypedDict, total=False):
    messages: List[BaseMessage]
    next_step: Optional[str]
    code: Optional[str]
    report: Optional[str]


# ============================================================
# 4. REQUEST / RESPONSE MODELS
# ============================================================

class TaskRequest(BaseModel):
    task: str


class TaskResponse(BaseModel):
    task: str
    generated_code: str
    report: str


# ============================================================
# 5. TOOLS
# ============================================================

@tool
def run_python_code(code: str) -> str:
    """
    Execute generated Python code and return its output
    or an error message.
    """

    if not isinstance(code, str):
        code = str(code)

    clean_code = (
        code
        .replace("```python", "")
        .replace("```", "")
        .strip()
    )

    old_stdout = io.StringIO()

    import sys

    previous_stdout = sys.stdout
    sys.stdout = old_stdout

    try:
        local_scope = {}

        exec(
            clean_code,
            {},
            local_scope
        )

        result = old_stdout.getvalue()

        if result.strip():
            return result.strip()

        return "Success (no terminal output)"

    except Exception:
        return (
            "Execution Error:\n"
            + traceback.format_exc()
        )

    finally:
        sys.stdout = previous_stdout


@tool
def generate_test_cases(task_description: str) -> str:
    """
    Generate test scenarios for the coding task.
    """

    prompt = f"""
You are a Senior QA Engineer.

Analyze the following coding task:

{task_description}

Generate 3 to 5 specific test scenarios.

Include:
1. Normal cases
2. Edge cases
3. Boundary cases where applicable

Return only a numbered list of test scenarios.
"""

    response = llm.invoke(prompt)

    content = response.content

    if isinstance(content, list):

        if not content:
            return "No test cases generated."

        first_item = content[0]

        if isinstance(first_item, dict):
            return str(first_item.get("text", ""))

        return str(first_item)

    return str(content)


# ============================================================
# 6. LANGGRAPH NODES
# ============================================================

def developer_node(state: CrewState):
    """
    Developer Agent:
    Generates Python code for the user's task.
    """

    task = state["messages"][-1].content

    prompt = f"""
You are the Developer Agent of Kalavathi AI.

Solve the following programming task:

{task}

Requirements:
- Write clean Python code.
- Make the program executable.
- Handle reasonable edge cases.
- Do not provide explanations.
- Do not use Markdown.
- Return ONLY the Python code.
"""

    response = llm_flash.invoke(prompt)

    content = response.content

    if isinstance(content, list):

        if not content:
            code = ""

        else:
            first_item = content[0]

            if isinstance(first_item, dict):
                code = str(first_item.get("text", ""))

            else:
                code = str(first_item)

    else:
        code = str(content)

    return {
        "code": code
    }


def tester_node(state: CrewState):
    """
    Tester Agent:
    Generates test scenarios and executes the generated code.
    """

    task = state["messages"][-1].content
    code = state.get("code", "")

    test_cases = generate_test_cases.invoke(
        task
    )

    execution_result = run_python_code.invoke(
        {
            "code": code
        }
    )

    report = (
        "### TEST SCENARIOS\n"
        f"{test_cases}\n\n"
        "### CODE EXECUTION RESULT\n"
        f"{execution_result}"
    )

    return {
        "report": report
    }


def manager_node(state: CrewState):
    """
    Manager Agent:
    Reviews the developer and tester output
    and produces the final result.
    """

    task = state["messages"][-1].content
    code = state.get("code", "")
    report = state.get(
        "report",
        "No test report available."
    )

    manager_prompt = f"""
You are the Manager Agent of Kalavathi AI.

Review the work produced by the Developer and Tester.

USER TASK:
{task}

GENERATED CODE:
{code}

TEST REPORT:
{report}

Provide a concise final assessment.

Include:
- Whether the generated code executed successfully
- Important problems found
- Any obvious improvement needed

Do not rewrite the entire code.
"""

    response = llm_flash.invoke(
        manager_prompt
    )

    content = response.content

    if isinstance(content, list):

        if content:

            first_item = content[0]

            if isinstance(first_item, dict):
                manager_result = str(
                    first_item.get("text", "")
                )
            else:
                manager_result = str(first_item)

        else:
            manager_result = "No manager assessment."

    else:
        manager_result = str(content)

    final_report = (
        f"{report}\n\n"
        "### MANAGER ASSESSMENT\n"
        f"{manager_result}"
    )

    return {
        "report": final_report,
        "next_step": "end"
    }


# ============================================================
# 7. LANGGRAPH
# ============================================================

workflow = StateGraph(CrewState)


workflow.add_node(
    "developer",
    developer_node
)

workflow.add_node(
    "tester",
    tester_node
)

workflow.add_node(
    "manager",
    manager_node
)


# START → Developer
workflow.add_edge(
    START,
    "developer"
)


# Developer → Tester
workflow.add_edge(
    "developer",
    "tester"
)


# Tester → Manager
workflow.add_edge(
    "tester",
    "manager"
)


# Manager → END
workflow.add_edge(
    "manager",
    END
)


graph = workflow.compile()


# ============================================================
# 8. FASTAPI ROUTES
# ============================================================

@app.get("/")
def home():
    return {
        "name": "Kalavathi AI",
        "status": "running",
        "workflow": [
            "Developer",
            "Tester",
            "Manager"
        ]
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/run", response_model=TaskResponse)
def run_task(request: TaskRequest):

    task = request.task.strip()

    if not task:
        raise HTTPException(
            status_code=400,
            detail="Task cannot be empty."
        )

    try:

        initial_state: CrewState = {
            "messages": [
                HumanMessage(
                    content=task
                )
            ]
        }

        result = graph.invoke(
            initial_state,
            config={
                "recursion_limit": 20
            }
        )

        return TaskResponse(
            task=task,
            generated_code=result.get(
                "code",
                ""
            ),
            report=result.get(
                "report",
                "No report generated."
            )
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# 9. LOCAL EXECUTION
# ============================================================

if __name__ == "__main__":

    import uvicorn

    port = int(
        os.getenv(
            "PORT",
            "8000"
        )
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
