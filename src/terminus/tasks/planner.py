from terminus.agent.factory import SYSTEM_PROMPT
from pydantic import BaseModel
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from terminus.config import CONFIG
from terminus.observability.logging import get_logger


logger = get_logger(__name__)


class TaskType(BaseModel):
    DESIGN = "design"
    IMPLEMENT = "implement"
    TEST = "test"
    REVEIW = "review"
    INTEGRATE = "integrate"
    CONFIGURE = "configure"

class PlannedTask(BaseModel):
    id: str
    title: str
    description: str
    task_type: TaskType
    depends_on: list[str]
    estimated_minutes: int
    output_files: list[str]
    acceptance_criteria: list[str]


class ExecutionPlan(BaseModel):
    project_name: str
    goal_summary: str
    tech_stack : list[str]
    total_estimated_hours: float
    tasks: list[PlannedTask]
    risks: list[str]
    assumptions: list[str]
    

SYSTEM_PROMPT="""

You are an AI Software Architect.

I will give you a **Goal**. Your job is to design a **complete, executable development plan** for a real-world software project, following these strict rules:

Rules:
- 5 to 20 tasks
- Task IDs must be stable snake_case strings (e.g. "task__001")
- depends_on must reference valid task ids in the same plan or be empty array.
- Ordering must form a valide DAG (no cycles): architecture -> schema -> config -> core -> tests -> integrations
- output files must list every file the task will write to disk
- acceptance_criteria must be concreate and verifiable (3-5 items per task)
- task_type must be one of : design, implement, test, reveiw, integrate, configure
"""


def create_plan(goal:str, extra_content:str = "") -> ExecutionPlan:
    """ Call the LLM planner and return a structured ExecutionPlan. """
    provider = CONFIG["llm"]["provider"]
    model = CONFIG["llm"]["model"]
    llm = init_chat_model(f"{provider}_chat_model",model)
    planner_agent = create_agent(
        llm,
        tools=[],
        system_prompt=SYSTEM_PROMPT,
        response_format=ExecutionPlan
    )
    user_message = f"Goal:{goal}"
    if extra_content:
        user_message += f"\nExtra Context:{extra_content}"
    result = planner_agent.invoke({"messages": [{"role":"user","content":user_message}]})
    plan: ExecutionPlan = result["structured_response"]
    logger.info(f"Goal:{goal}")
    logger.info(f"Plan: {plan.model_dump_json(indent=2)}")
    return plan