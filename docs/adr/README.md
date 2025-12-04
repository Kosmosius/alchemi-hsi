# Architecture Decision Records (ADRs)

An Architecture Decision Record (ADR) captures an important engineering choice along with the context, decision, and consequences. Recording ADRs helps future contributors understand why the architecture, data contracts, tooling stack, or policies evolved in a particular direction.

ADRs in this repo live in `docs/adr/` and are named `NNNN-short-title.md`, where `NNNN` is a zero-padded sequence number (e.g., `0001`, `0002`). Each ADR has a **Status** of **Proposed**, **Accepted**, **Superseded**, or **Rejected** to track its lifecycle.

## When to write an ADR
Write an ADR for any major change to architecture, data contracts, tooling stack, or policy that future contributors need to understand.

## Workflow
1. Author a new ADR using the [template](0000-template.md).
2. Open a pull request that includes the ADR alongside the change it describes.
3. Mark the ADR as **Accepted** once the PR merges. If a new ADR replaces an older one, update the older ADR's status to **Superseded** and link to the replacement.
