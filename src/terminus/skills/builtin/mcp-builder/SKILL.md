---
name: mcp-builder
description: >
  Use when building, reviewing, debugging, or testing a Model Context Protocol server
  or an integration with one: designing the tool surface of an MCP server, wiring tool
  schemas and handlers, connecting to an MCP server from an application, diagnosing a
  server that will not start or whose tools fail, or validating that tools behave
  correctly. Triggers on "build an MCP server", "MCP tool", "Model Context Protocol",
  "expose this as an MCP tool", "my MCP server returns an error", or work on
  mcp.json / MCP client configuration.
  Do NOT use for ordinary REST or GraphQL API design with no MCP involvement, and do
  NOT use for frontend or UI work. Keep the tool surface small; this skill is about
  making a few good tools, not exposing every internal function.
version: 1.0.0
license: Apache-2.0
author: Terminus (adapted from anthropics/skills)
repository: anthropics/skills
tags: [mcp, tools, integration, server, client, schema, testing]
---

# Building MCP Servers and Integrations

MCP is a contract between a model and a capability. The design problem is deciding what
the tool surface should be; the engineering problem is making it reliable.

## Start from the decision, not the function

The most common MCP server wraps internal functions one-to-one and performs badly: a
model has to guess names, invent arguments, and stitch together five calls that should
have been one.

Work from the **decisions a consumer wants to make**:

```
Bad:  list_files(dir), read_file(path), parse_file(path), index_file(path), search(pattern)

Good: search_code(query, scope?)     - find the code that matters
      read_section(path, start, end) - read what search found
```

Rules:

- **Few, high-level tools** beat many primitive ones. Each extra tool is a choice the
  model has to make correctly.
- **Name tools for the job**, not the function: `search_orders` over `query_table`.
- **A tool should be one coherent action.** If its description needs "and also"
  repeatedly, split it.
- **Make the common case one call.** Composing results server-side is cheaper and less
  error-prone than the model stitching them.

## Tool schema design

The schema is documentation the model reads at runtime. Make it good.

```json
{
  "name": "search_orders",
  "description": "Find orders by customer, status, or date range. Returns matching orders with ids, totals, and status. Use for questions about a specific order's state; use list_customers when you do not know the customer.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "customer_email": { "type": "string", "description": "Exact customer email. Optional if status or date_range is given." },
      "status": { "type": "string", "enum": ["pending", "paid", "shipped", "cancelled"], "description": "Filter by status." },
      "date_range": {
        "type": "object",
        "properties": { "from": {"type":"string"}, "to": {"type":"string"} },
        "description": "Inclusive order date range, ISO 8601."
      }
    },
    "required": []
  }
}
```

- The description says what it does, when to use it, **and when to use something else**.
- Every property has a description. Undescribed parameters are guessed.
- Mark optionality honestly. If nothing is required, say so in the description; if
  something is, use `required`.
- Use `enum` for closed sets. It prevents invented values.
- Prefer structured objects over positional strings. `{"from": "...", "to": "..."}`
  survives refactoring; `"a,b"` does not.
- Name parameters the way a user would.

## Server implementation

- **Validate at the boundary.** Reject unknown or malformed arguments with a message
  that tells the caller what to fix. Do not fail deep inside business logic.
- **Return errors the model can act on.** `"No customer with email x"` is actionable;
  a stack trace is not.
- **Do not throw for expected conditions.** A "not found" is a result, not a crash.
- **Make tools idempotent where possible**, or clearly mark the ones that are not.
- **Bound everything**: page size, timeouts, recursion depth, response size. An
  unbounded tool can stall a consumer.
- **Never let a tool write outside its intended scope.** Keep the permission boundary
  the same one the host application uses.

## Connecting an MCP client

- Check the server actually starts and handshakes before debugging tool behaviour.
  Most "MCP is broken" reports are transport or configuration problems.
- Confirm the client can list tools before it can call them; if listing fails, nothing
  else matters.
- Verify the tool name the client sees matches the one you expect. Naming and
  sanitisation differences are a common silent failure.
- Timeouts on startup are often a server that started but did not finish
  initialising.

## Testing

Test tools as a contract, from the consumer's side:

1. **Schema validity** - the schema parses and validates sample input.
2. **Happy path** - a representative call returns the right shape.
3. **Boundary** - empty, very large, and malformed arguments.
4. **Not found** - absent entity returns a clear message, not an exception.
5. **Error propagation** - downstream failure is reported, not swallowed.
6. **Contract** - the response shape matches what the description promised. If the
   description says it returns ids, totals, and status, assert all three.

Test the tool surface from a real MCP client at least once. A handler that works when
called directly can still fail the transport.

## Refining the surface

After real use, look at which tools are never called (candidates for removal or
merging), which are always called in the same sequence (a candidate to combine), and
which frequently fail on arguments (a schema or description problem). That evidence
is the best guide to the next version.

## Reporting

State the tool surface you exposed and why those tools, the transport and runtime
verified, the cases you tested, and anything you deliberately left out.
