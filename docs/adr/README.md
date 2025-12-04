# Architecture Decision Records (ADRs)

Architecture Decision Records capture the context and rationale behind significant technical choices so future contributors understand why the system looks the way it does. Each ADR documents the problem being solved, the selected approach, and the expected consequences.

ADRs are stored in this directory using the naming convention `NNNN-short-title.md`, where `NNNN` is a zero-padded sequence (e.g., `0001`, `0002`). The `Status` field for an ADR must be one of: **Proposed**, **Accepted**, **Superseded**, or **Rejected**.

Write an ADR whenever you make a major change to architecture, data contracts, the tooling stack, or project policies that future contributors need to understand.

## Workflow

1. Author a new ADR file in this directory using the [template](0000-template.md).
2. Open a PR that includes the ADR alongside the change it describes.
3. Once the PR merges, update the ADR status to **Accepted**. If a new ADR replaces a prior decision, mark the old one as **Superseded**.
