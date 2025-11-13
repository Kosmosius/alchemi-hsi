# Triage workflow

This document explains how issues are triaged for the Alchemi HSI project. It covers
label conventions, milestones, automation, and how to seed the shared taxonomy when
setting up a new environment.

## Label conventions

Labels capture area ownership, the hardware sensor involved, the roadmap phase, and
priority. Apply exactly one label from each category when triaging.

| Category  | Label pattern      | Examples                                | Notes |
|-----------|--------------------|------------------------------------------|-------|
| Area      | `area:<name>`      | `area:ingest`, `area:analysis`, `area:platform` | Maps issues to the owning subsystem. |
| Sensor    | `sensor:<name>`    | `sensor:emit`, `sensor:collect`          | Hardware surface touched by the work. |
| Phase     | `phase:<stage>`    | `phase:triage`, `phase:1`, `phase:2`     | Aligns with roadmap phases and milestones. |
| Priority  | `priority:P<level>`| `priority:P0`, `priority:P1`, `priority:P2` | Drives scheduling urgency. |

The [`tools/data/labels.json`](../../tools/data/labels.json) file is the source of truth
for the taxonomy. Use `tools/seed_labels.py` to create or update labels. The script is
idempotent and only updates labels that differ from the specification.

```bash
# Preview changes without applying them
python tools/seed_labels.py --dry-run

# Apply changes (requires gh CLI configured with GH_TOKEN)
GH_TOKEN=<token> python tools/seed_labels.py
```

## Milestones

Roadmap execution uses GitHub milestones named after the delivery phase. The
`Phase 1` milestone is the current active delivery target—ensure new issues are filed
against this milestone so burn-up charts match the project board. Milestone names must
mirror the `phase:*` labels to keep reports aligned with the Phase 1 project.

## Automation

The triage workflow (`.github/workflows/triage.yml`) automatically:

1. Adds every new or edited issue to the organization project at
   `https://github.com/orgs/<owner>/projects/1`.
2. Sets the default project fields so that the item enters the "Triage" status,
   associates to `Phase 1`, and assigns the baseline priority `P2`.
3. Applies the `phase:triage` label when it is missing so the team knows the issue still
   needs categorisation.

Project rules build on these defaults; once the Phase labels or priority change, the
board automation will move cards into the correct swimlanes. When the issue is fully
triaged, replace `phase:triage` with the appropriate `phase:*` label and update
priority as needed.

## Project hygiene checklist

When a new issue arrives:

- [ ] Confirm the issue has been added to the Phase 1 GitHub project.
- [ ] Verify that one label from each category (area, sensor, phase, priority) is set.
- [ ] Ensure the milestone remains `Phase 1` or update it if the work belongs to a later
      phase.
- [ ] If the issue is production impacting, escalate priority (`priority:P0` or
      `priority:P1`) and link the incident.
