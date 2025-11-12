#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 owner/repo [TASKS.yaml]" >&2
  exit 1
fi

REPO="$1"
TASKS="${2:-TASKS.yaml}"

if [[ ! -f "$TASKS" ]]; then
  echo "TASKS file not found: $TASKS" >&2
  exit 1
fi

echo "=== Seeding GitHub from $TASKS into $REPO ==="

# -------------------- LABELS --------------------
echo ">>> Seeding labels..."
yq -r '.labels[]' "$TASKS" | while read -r L; do
  [[ -z "$L" ]] && continue
  COLOR=$(printf '%06x' $((RANDOM % 0xFFFFFF)))
  echo "  label: $L (color #$COLOR)"
  gh label create "$L" --repo "$REPO" --color "$COLOR" --force >/dev/null
done

# -------------------- MILESTONES --------------------
echo ">>> Seeding milestones..."

# Bash associative array: milestone-key -> milestone-number
declare -A M_NUMS

# Each row: KEY \t TITLE \t DUE
while IFS=$'\t' read -r KEY TITLE DUE; do
  [[ -z "$KEY" ]] && continue
  echo "  milestone key=$KEY title=$TITLE due=$DUE"

  # Check if milestone already exists (by title)
  EXIST_NUM=$(
    gh api "repos/$REPO/milestones" --paginate \
      --jq ".[] | select(.title==\"$TITLE\") | .number" 2>/dev/null || true
  )

  if [[ -z "$EXIST_NUM" ]]; then
    # Create new milestone
    if [[ -n "$DUE" && "$DUE" != "null" ]]; then
      DUE_ISO="${DUE}T00:00:00Z"
      NUM=$(
        gh api "repos/$REPO/milestones" --method POST \
          -f "title=$TITLE" \
          -f "due_on=$DUE_ISO" \
          --jq '.number'
      )
    else
      NUM=$(
        gh api "repos/$REPO/milestones" --method POST \
          -f "title=$TITLE" \
          --jq '.number'
      )
    fi
    echo "    created milestone number=$NUM"
  else
    NUM="$EXIST_NUM"
    echo "    reusing existing milestone number=$NUM"
  fi

  M_NUMS["$KEY"]="$NUM"
done < <(yq -r '.milestones[] | [.key, .title, (.due // "")] | @tsv' "$TASKS")

# -------------------- HELPER: create issues from a YAML node --------------------
seed_group() {
  local NODE="$1"   # e.g. .epics[] or .issues[]
  local PREFIX="$2" # usually "" (you already include EPIC — in the title)

  echo ">>> Seeding $NODE ..."

  yq -o=json "$NODE" "$TASKS" | jq -c '.' | while read -r ITEM; do
    local TITLE BODY M_KEY M_NUM

    TITLE=$(jq -r '.title' <<<"$ITEM")
    if [[ -z "$TITLE" || "$TITLE" == "null" ]]; then
      echo "  !! skipping entry with no title"
      continue
    fi

    M_KEY=$(jq -r '.milestone // empty' <<<"$ITEM")
    M_NUM=""
    if [[ -n "$M_KEY" && -n "${M_NUMS[$M_KEY]:-}" ]]; then
      M_NUM="${M_NUMS[$M_KEY]}"
    fi

    BODY=$(jq -r '.body // ""' <<<"$ITEM")

    # Build acceptance + tests sections in Markdown
    local ACC_BODY TEST_BODY

    ACC_BODY=$(
      jq -r '
        .acceptance // [] |
        if length == 0 then "" else
          "## Acceptance criteria\n" + (map("- " + .) | join("\n"))
        end
      ' <<<"$ITEM"
    )

    TEST_BODY=$(
      jq -r '
        .tests // [] |
        if length == 0 then "" else
          "## Tests\n" + (map("- " + .) | join("\n"))
        end
      ' <<<"$ITEM"
    )

    if [[ -n "$ACC_BODY" ]]; then
      BODY+=$'\n\n'"$ACC_BODY"
    fi
    if [[ -n "$TEST_BODY" ]]; then
      BODY+=$'\n\n'"$TEST_BODY"
    fi

    # Labels
    mapfile -t LABELS < <(jq -r '.labels // [] | .[]' <<<"$ITEM")
    local LABEL_ARGS=()
    for lab in "${LABELS[@]}"; do
      [[ -z "$lab" || "$lab" == "null" ]] && continue
      LABEL_ARGS+=(--label "$lab")
    done

    local FULL_TITLE="$TITLE"
    if [[ -n "$PREFIX" ]]; then
      FULL_TITLE="$PREFIX$TITLE"
    fi

    echo "  issue: $FULL_TITLE"
    CMD=(gh issue create --repo "$REPO" --title "$FULL_TITLE" --body "$BODY")
    if [[ -n "$M_NUM" ]]; then
      CMD+=(--milestone "$M_NUM")
    fi
    if ((${#LABEL_ARGS[@]})); then
      CMD+=("${LABEL_ARGS[@]}")
    fi

    "${CMD[@]}"
  done
}

# -------------------- EPICS & ISSUES --------------------
# Your YAML uses `epics:` and `issues:` (not `singleton_issues:`)

seed_group '.epics[]'  ''
seed_group '.issues[]' ''

echo "=== Done. Labels, milestones, epics, and issues seeded into $REPO ==="
