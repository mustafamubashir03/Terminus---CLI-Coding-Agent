"""The shared message-text helper.

Regression: it must accept a message object, not only a bare content value. The
planner used to unwrap `.content` itself; when that was consolidated into this
helper it passed the whole message and every plan silently failed to parse.
"""

from langchain_core.messages import AIMessage, HumanMessage

from terminus.llm.text import message_text


def test_plain_string_content():
    assert message_text("hello") == "hello"
    assert message_text(AIMessage(content="hello")) == "hello"


def test_none_is_empty():
    assert message_text(None) == ""


def test_text_blocks_are_concatenated():
    content = [{"type": "text", "text": "one"}, {"type": "text", "text": "two"}]
    assert message_text(content) == "one\ntwo"
    assert message_text(AIMessage(content=content)) == "one\ntwo"


def test_non_text_blocks_are_not_dropped():
    content = [{"type": "reasoning", "reasoning": "thinking"}]
    out = message_text(content)
    assert "thinking" in out


def test_block_without_text_is_stringified():
    content = [{"type": "tool_use", "name": "run", "id": "1"}]
    assert "run" in message_text(content)


def test_accepts_a_message_object():
    """The bug this file exists for."""
    msg = AIMessage(content='{"a": 1}')
    assert message_text(msg) == '{"a": 1}'
    # a message whose content is a list of blocks
    msg = AIMessage(content=[{"type": "text", "text": "x"}])
    assert message_text(msg) == "x"


def test_human_messages_work_too():
    assert message_text(HumanMessage(content="hi")) == "hi"


def test_planner_can_parse_a_plan_from_a_message():
    """End-to-end reason the helper exists: planner output -> ExecutionPlan."""
    from terminus.tasks.planner import _extract_json, ExecutionPlan

    plan = {
        "project_name": "P", "goal_summary": "g", "tech_stack": ["python"],
        "total_estimated_hours": 1.0,
        "tasks": [{
            "id": "task__001", "title": "t", "description": "d",
            "task_type": "implement", "depends_on": [], "estimated_minutes": 1,
            "output_files": [], "acceptance_criteria": ["done"],
        }],
        "risks": [], "assumptions": [],
    }
    import json
    msg = AIMessage(content=json.dumps(plan))
    parsed = ExecutionPlan.model_validate(_extract_json(message_text(msg)))
    assert parsed.tasks[0].id == "task__001"
