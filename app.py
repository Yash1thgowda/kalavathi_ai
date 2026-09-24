import os
import io
import traceback
from typing import TypedDict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END


api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable is not set.")


app = FastAPI(
    title="Kalavathi AI",
    description="AI-powered coding development and testing workflow",
    version="1.0.0"
)


llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    google_api_key=api_key,
    temperature=0
)


class CrewState(TypedDict, total=False):
    task: str
    generated_code: str
    test_cases: str
    test_result: str
    report: str


class RunRequest(BaseModel):
    task: str


class RunResponse(BaseModel):
    task: str
    generated_code: str
    report: str


@tool
def run_python_code(code: str) -> str:
    """
    Executes Python code and returns its output.
    """

    output = io.StringIO()

    try:
        namespace = {}

        old_stdout = __import__("sys").stdout
        __import__("sys").stdout = output

        try:
            exec(code, namespace)
        finally:
            __import__("sys").stdout = old_stdout

        result = output.getvalue()

        if not result:
            result = "Code executed successfully with no output."

        return result

    except Exception:
        return traceback.format_exc()


@tool
def generate_test_cases(code: str) -> str:
    """
    Generates test cases for the provided code.
    """

    prompt = f"""
You are a software tester.

Analyze the following Python code and generate useful test cases.

CODE:
{code}

Provide:
1. Normal test cases
2. Edge cases
3. Invalid input cases where applicable
4. Expected output for each case

Keep the response concise and structured.
"""

    response = llm.invoke([HumanMessage(content=prompt)])

    return response.content


def developer_node(state: CrewState) -> CrewState:
    task = state["task"]

    prompt = f"""
You are the Developer agent.

Solve the following programming task.

TASK:
{task}

Requirements:
- Write complete Python code.
- Make the solution executable.
- Do not use Markdown code fences.
- Do not explain the code.
- Return only the Python source code.
"""

    response = llm.invoke([HumanMessage(content=prompt)])

    code = response.content.strip()

    if code.startswith("```"):
        lines = code.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        code = "\n".join(lines).strip()

    return {
        **state,
        "generated_code": code
    }


def tester_node(state: CrewState) -> CrewState:
    code = state["generated_code"]

    try:
        test_cases = generate_test_cases.invoke({
            "code": code
        })

        execution_result = run_python_code.invoke({
            "code": code
        })

        test_result = (
            "GENERATED TEST CASES:\n"
            + test_cases
            + "\n\nINITIAL EXECUTION RESULT:\n"
            + execution_result
        )

    except Exception as e:
        test_result = f"Testing failed:\n{traceback.format_exc()}"

    return {
        **state,
        "test_cases": test_cases if "test_cases" in locals() else "",
        "test_result": test_result
    }


def manager_node(state: CrewState) -> CrewState:
    task = state["task"]
    code = state["generated_code"]
    test_result = state.get("test_result", "")

    prompt = f"""
You are the Manager agent reviewing a software development task.

TASK:
{task}

GENERATED CODE:
{code}

TESTING INFORMATION:
{test_result}

Review the implementation and produce a concise final report.

Include:
- Task summary
- Code status
- Testing status
- Problems found, if any
- Suggested improvements, if any

Do not rewrite the entire code.
"""

    response = llm.invoke([HumanMessage(content=prompt)])

    return {
        **state,
        "report": response.content
    }


workflow = StateGraph(CrewState)

workflow.add_node("developer", developer_node)
workflow.add_node("tester", tester_node)
workflow.add_node("manager", manager_node)

workflow.add_edge(START, "developer")
workflow.add_edge("developer", "tester")
workflow.add_edge("tester", "manager")
workflow.add_edge("manager", END)

graph = workflow.compile()


@app.get("/")
def root():
    return {
        "message": "Kalavathi AI is running",
        "status": "online"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/run", response_model=RunResponse)
def run_task(request: RunRequest):
    task = request.task.strip()

    if not task:
        raise HTTPException(
            status_code=400,
            detail="Task cannot be empty."
        )

    try:
        result = graph.invoke({
            "task": task
        })

        return RunResponse(
            task=task,
            generated_code=result.get("generated_code", ""),
            report=result.get("report", "")
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
