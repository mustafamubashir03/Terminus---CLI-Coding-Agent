---
name: database
description: >
  Use when choosing a database, designing or changing a schema, writing or fixing SQL
  queries, planning a migration, adding row level security, modelling multi-tenant data,
  or diagnosing a data problem such as a slow query, a constraint violation, a deadlock,
  or duplicated rows. Triggers on "design the schema", "add a table", "write a migration",
  "this query is slow", "index", "foreign key", "row level security", "RLS", "multi
  tenant", "N+1 query", "duplicate rows", or an ERD.
  Do NOT use for an ORM-only change with no schema or query involved, for caching or
  Redis, for a frontend data-fetching problem, or for choosing a queue.
version: 1.0.0
license: MIT
author: Terminus (adapted from whawkinsiv/solo-founder-skills)
repository: whawkinsiv/solo-founder-skills
tags: [database, sql, schema, migrations, postgres, indexing, rls, performance]
negative_triggers:
  - redis
  - cache invalidation
  - api endpoint
---

# Database Work

Most database bugs are design decisions made early and paid for later. Work the order
below; do not start writing SQL before the shape is settled.

## 1. Choose the access pattern, then the schema

Enumerate the queries the app will actually run, and only then design tables. A
schema designed from an ERD in the abstract is wrong more often than not.

For each table, write down:
- the primary access query, with its filters and sort
- expected row count now, and in two years
- the write rate

Everything else - normalisation, denormalisation, indexes - follows from that list.

## 2. Model multi-tenancy deliberately

If data is shared between tenants, choose one and hold to it:

- **Shared tables, `tenant_id` column.** Simplest; every query must filter on it.
  RLS (below) is the cheapest guard against forgetting.
- **Schema per tenant.** Strong isolation, awkward migrations at scale.
- **Database per tenant.** Strongest isolation, most operational cost.

Mixing modes in one product is where multi-tenant bugs come from. Pick one, write it
down, and make it obvious in the code.

## 3. Constraints belong in the database

Application validation is advisory; a constraint is enforced. Put in the database
what must never be wrong:

- `NOT NULL` on anything required
- `UNIQUE` on natural keys and on every `(tenant_id, external_id)` pair
- foreign keys, with a deliberate `ON DELETE` action - `CASCADE` and `RESTRICT` are
  both valid, but choosing nothing is not
- `CHECK` for values a domain rule constrains
- sensible types: `timestamptz` over `timestamp`, `numeric` over float for money

Money is never a float. Never store a computed total you could derive.

## 4. Index what you query, and nothing else

Every index is a write cost and a storage cost.

```sql
-- Multi-column order matters: this serves tenant + created_at range scans,
-- and created_at alone.
CREATE INDEX CONCAT idx_orders_tenant_created ON orders (tenant_id, created_at DESC);
```

- Put equality columns first, then the range/sort column.
- A composite index serves any leftmost prefix, so `(a, b)` covers `a` too.
- If a query filters on `LOWER(email)`, index the expression, not the column.
- Drop an unused index; find them with `pg_stat_user_indexes`.
- Do not index a low-cardinality boolean alone.

## 5. Write the migration so it is safe to run

A migration runs against live data while the old code is still serving. Therefore:

- **Expand, then contract.** Add the new column, backfill, deploy code that writes
  both and reads the new one, then drop the old. Never in one step.
- Adding a `NOT NULL` column needs a default or a three-step add/backfill/validate.
- Renaming is add + dual-write + backfill + drop, not `ALTER ... RENAME`.
- Set a short `lock_timeout` and retry, so a migration cannot block behind a long
  transaction and take the table down with it.
- Large backfills run in batches, outside the migration, with a way to resume.
- Every migration needs a tested way back, or an explicit decision that there isn't one.

State, when reporting, whether the migration is safe to run with traffic live.

## 6. Row Level Security

RLS moves a tenant guard from "remember to filter" into the database, so a missed
`WHERE` is a bug rather than a breach.

```sql
ALTER TABLE orders ENABLE ROW LEVEL SECURITY;
CREATE POLICY orders_tenant ON orders
  USING (tenant_id = current_setting('app.tenant_id')::uuid);
```

- `USING` covers reads and writes; `WITH CHECK` covers inserts and updates.
- A table with RLS enabled and no policy denies everything - that is the safe
  failure mode, but it is also a common outage.
- The application role must not be the table owner, or it bypasses RLS entirely.
- Write an integration test that proves one tenant cannot read another's rows.

## 7. Diagnose a slow query

Measure, then reason.

```sql
EXPLAIN (ANALYZE, BUFFERS) SELECT ...;
```

Read the plan for: sequential scans on large tables, `Nested Loop` with a large
inner side, `Sort` that spills to disk, and a row estimate far from the actual.

Usual causes, in the order worth checking:
1. A missing or mis-ordered index.
2. An N+1: many small queries from application code. Count the queries first.
3. `SELECT *` over a wide row, fetching columns nobody uses.
4. A function wrapped around an indexed column, which disables the index.
5. Statistics that are stale after a bulk load. `ANALYZE`.
6. Pagination that deepens the offset: `OFFSET 100000` reads and discards 100,000
   rows. Use keyset pagination on the last seen value.

## 8. Common failure modes

- **Duplicate rows** from a retry without a unique constraint or an idempotency key.
  Fix the constraint; catching the error downstream is not a fix.
- **Deadlock** from inconsistent lock ordering. Pick one global order and hold to it.
- **Orphan rows** where a foreign key is absent or was never added.
- **Unbounded query** on a growing table with no index and no `LIMIT`.
- **Silent data loss** from an `UPDATE` with a `WHERE` you did not check first.
  Preview the row count, then update.

## Reporting

State the access pattern you designed for, the constraints you added, the
indexes and the queries they serve, and whether the migration is safe to run
with traffic live. If a data problem is not solved, say which part is unresolved
- do not describe a schema change as "fixed" without showing the query plan.
