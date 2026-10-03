"""The tool registry: which tools exist, and what a name resolves to.

Deliberately narrow. These tests assert the contract the rest of Terminus relies
on - discoverability, schemas that come from the tool definitions, name
resolution, sane behaviour for an unknown name - and they assert what the
registry does *not* do, because the failure mode being guarded against is a
second tool-dispatch system creeping in beside LangGraph's.
"""

from __future__ import annotations

import pytest
from langchain.tools import BaseTool

from terminus.agent import factory
from terminus.agents.roles import ROLES
from terminus.tools import registry


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------

def test_every_catalogue_entry_is_a_langchain_tool():
    """The registry holds LangChain tools, not a second representation of them.

    If this ever needs a wrapper type, the wrapper is a new framework layer and
    the answer is to stop wrapping.
    """
    for tool in registry.CATALOGUE:
        assert isinstance(tool, BaseTool), f"{tool!r} is not a BaseTool"
        assert tool.description, f"{tool.name} has no description for the model"


def test_tool_names_are_unique_and_cover_the_catalogue():
    assert len(registry.tool_names()) == len(registry.CATALOGUE)
    assert set(registry.tool_names()) == {t.name for t in registry.CATALOGUE}


def test_catalogue_returns_a_copy_so_a_caller_cannot_register_by_mutation():
    first = registry.catalogue()
    first.pop("read_file")
    assert "read_file" in registry.catalogue()


def test_no_plan_tool_leaks_into_ask_and_vice_versa():
    """The two shell toolsets are deliberately different tools, not one alias."""
    ask = {t.name for t in registry.ask_tools()}
    assert "run_command" in ask
    assert "run_shell_command" not in ask
    assert "run_command_in_directory" not in ask
    # ...and /plan's shell tools are registered under their own names, so
    # resolving "run_command" can never silently hand a worker the /ask shell.
    plan = {t.name for t in registry.plan_tools_for("implement")}
    assert {"run_shell_command", "run_command_in_directory"} <= plan
    assert "run_command" not in plan


# ---------------------------------------------------------------------------
# schemas
# ---------------------------------------------------------------------------

def test_schemas_come_from_the_tool_definitions_not_from_here():
    """Every catalogue tool has a usable, inferred JSON schema.

    Nothing in Terminus writes a schema by hand, so this is really an assertion
    that the registry did not become the place where one grew.
    """
    for tool in registry.CATALOGUE:
        schema = tool.args_schema.model_json_schema()
        assert schema.get("type") == "object", tool.name
        assert isinstance(schema.get("properties"), dict), tool.name
        # the schema must be convertible to what a provider expects
        from langchain_core.utils.function_calling import convert_to_openai_tool

        converted = convert_to_openai_tool(tool)
        assert converted["function"]["name"] == tool.name


def test_ask_tool_schemas_expose_only_their_own_arguments():
    from terminus.tools.filesystem_tools import edit_file, read_file, write_file
    from terminus.tools.shell_tools import run_command

    assert set(read_file.args) == {"file_path"}
    assert set(write_file.args) == {"file_path", "content"}
    assert set(edit_file.args) == {"file_path", "old_text", "new_text"}
    # working_directory exists, and nothing that could widen authority with it
    assert set(run_command.args) == {"command", "working_directory"}


# ---------------------------------------------------------------------------
# resolution
# ---------------------------------------------------------------------------

def test_resolve_returns_the_same_objects_the_catalogue_holds():
    resolved = registry.resolve(("read_file", "grep"))
    assert [t.name for t in resolved] == ["read_file", "grep"]
    by_name = registry.catalogue()
    assert resolved[0] is by_name["read_file"]


def test_resolve_preserves_the_order_it_was_given():
    assert [t.name for t in registry.resolve(("grep", "read_file"))] == [
        "grep",
        "read_file",
    ]


def test_resolve_deduplicates():
    resolved = registry.resolve(("read_file", "read_file", "grep"))
    assert [t.name for t in resolved] == ["read_file", "grep"]


def test_unknown_names_are_skipped_and_reported():
    """A missing tool must not be able to take down an agent."""
    resolved = registry.resolve(("read_file", "no_such_tool", "grep"))
    assert [t.name for t in resolved] == ["read_file", "grep"]
    assert registry.missing_tool_names(["read_file", "nope"]) == ("nope",)


def test_resolve_appends_extra_tools_after_the_named_ones():
    """MCP tools have no catalogue entry; they arrive as extras."""
    from terminus.tools import refusing_tool

    @refusing_tool
    def mcp_probe(path: str) -> str:
        """An MCP-shaped tool."""
        return path

    resolved = registry.resolve(("read_file",), extra=[mcp_probe])
    assert [t.name for t in resolved] == ["read_file", "mcp_probe"]


def test_extra_tools_are_also_deduplicated():
    from terminus.tools.filesystem_tools import read_file

    resolved = registry.resolve(("read_file",), extra=[read_file])
    assert [t.name for t in resolved] == ["read_file"]


# ---------------------------------------------------------------------------
# the toolsets that consume it
# ---------------------------------------------------------------------------

def test_ask_tools_are_the_names_ask_tool_names_declares():
    assert [t.name for t in registry.ask_tools()] == list(registry.ASK_TOOL_NAMES)


def test_factory_ask_tools_come_from_the_registry():
    assert list(factory.ASK_TOOLS) == list(registry.ask_tools())
    catalogue = registry.catalogue()
    resolved = factory.tools_by_name()
    assert set(resolved) == set(registry.ASK_TOOL_NAMES)
    for name, tool in resolved.items():
        assert tool is catalogue[name]


def test_plan_tool_sets_differ_by_task_type():
    design = {t.name for t in registry.plan_tools_for("design")}
    implement = {t.name for t in registry.plan_tools_for("implement")}
    assert "load_skill" in design
    assert "run_shell_command" not in design
    assert "run_shell_command" in implement
    assert "delete_file" in implement, "/plan workers get the destructive fs tools"
    assert "delete_file" not in {t.name for t in registry.ask_tools()}


def test_an_unknown_task_type_falls_back_to_search_only():
    assert [t.name for t in registry.plan_tools_for("nonsense")] == ["search_codebase"]


def test_every_role_tool_name_resolves_in_the_ask_catalogue():
    """A child agent is narrowed by name, so an unresolvable name is dead weight.

    This is the drift the single catalogue exists to prevent: /plan's shell tools
    have different names from /ask's, and role tuples used to name things by hand.
    """
    resolvable = set(registry.tool_names())
    for role in ROLES.values():
        for name in role.tool_names():
            assert name in resolvable, (
                f"role {role.name} names {name!r}, which no catalogue entry has"
            )


# ---------------------------------------------------------------------------
# what the registry must not become
# ---------------------------------------------------------------------------

def test_the_registry_does_not_execute_anything():
    """The registry knows which tools exist. It never runs one.

    Checked against the module's code rather than its text, so the docstring is
    free to explain that it deliberately does not.
    """
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(registry))
    forbidden = {"invoke", "arun", "ainvoke", "run", "subprocess", "Popen"}
    called = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert not (called & forbidden), (
        f"tools/registry.py must not call {called & forbidden}: the registry knows "
        "which tools exist, it does not run them"
    )


def test_langgraph_tool_node_remains_the_only_executor():
    """One execution path: the graph's own ToolNode.

    If a Terminus-side ToolNode ever appears, this fails - which is the point.
    """
    import pathlib

    import terminus

    root = pathlib.Path(terminus.__file__).parent
    offenders = [
        str(path.relative_to(root))
        for path in root.rglob("*.py")
        if "ToolNode(" in path.read_text(encoding="utf-8")
    ]
    assert not offenders, f"a second ToolNode exists: {offenders}"


def test_no_module_reimplements_openai_tool_schema_generation():
    import pathlib

    import terminus

    root = pathlib.Path(terminus.__file__).parent
    offenders = [
        str(path.relative_to(root))
        for path in root.rglob("*.py")
        if "convert_to_openai_tool" in path.read_text(encoding="utf-8")
    ]
    assert not offenders, f"hand-built tool schemas found: {offenders}"


def test_the_catalogue_holds_no_bare_callables():
    """A name->callable table is how a second dispatcher would begin.

    Every entry must be a LangChain tool, so resolving a name cannot produce
    something the graph would have to be taught to execute differently.
    """
    import types

    for tool in registry.CATALOGUE:
        assert not isinstance(tool, types.FunctionType), (
            f"{tool!r} is a plain function, not a tool"
        )
        assert getattr(tool, "run", None) is not None or hasattr(tool, "invoke"), (
            f"{tool!r} is not executable as a LangChain tool"
        )


@pytest.mark.parametrize("name", ["read_file", "write_file", "grep"])
def test_resolution_returns_tools_langgraph_can_bind(name):
    """The resolved object must be bindable - that is the whole contract."""
    from langchain_core.utils.function_calling import convert_to_openai_tool

    tool = registry.resolve([name])[0]
    converted = convert_to_openai_tool(tool)
    assert converted["type"] == "function"
    assert converted["function"]["name"] == name
