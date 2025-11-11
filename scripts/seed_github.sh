#!/usr/bin/env bash
set -euo pipefail
REPO="${1:-yourname/yourrepo}"
TASKS="${2:-TASKS.yaml}"

# 1) Create labels (idempotent)
yq -r '.labels[]' "$TASKS" | while read -r L; do
  COLOR=$(printf "%06x" $(( RANDOM % 16777215 )))
  gh label create "$L" --repo "$REPO" --color "$COLOR" --force >/dev/null
done

# 2) Create milestones
yq -r '.milestones[] | @base64' "$TASKS" | while read -r row; do
  _jq(){ echo "$row" | base64 --decode | yq -r "$1"; }
  TITLE=$(_jq '.title')
  DUE=$(_jq '.due')
  # github api wants ISO time
  gh api repos/$REPO/milestones -f title="$TITLE" -f due_on="${DUE}T00:00:00Z" >/dev/null || true
done

# 3) Create EPIC issues first (capture numbers)
declare -A EPIC_MAP
yq -r '.epics[] | @base64' "$TASKS" | while read -r row; do
  _jq(){ echo "$row" | base64 --decode | yq -r "$1"; }
  KEY=$(_jq '.key')
  TITLE=$(_jq '.title')
  BODY=$(_jq '.body')
  MILE=$(_jq '.milestone')
  LABELS=$(echo -n "epic"; echo -n ","; echo "$row" | base64 --decode | yq -r '.labels | join(",")')
  NUM=$(gh issue create --repo "$REPO" --title "$TITLE" --body "$BODY" --label "$LABELS" --milestone "$MILE" --json number -q .number)
  EPIC_MAP["$KEY"]=$NUM
  echo "EPIC $KEY -> #$NUM"
done

# 4) Create issues and link parent epic by text reference
yq -r '.issues[] | @base64' "$TASKS" | while read -r row; do
  _jq(){ echo "$row" | base64 --decode | yq -r "$1"; }
  TITLE=$(_jq '.title')
  BODY=$(_jq '.body')
  MILE=$(_jq '.milestone')
  LABELS=$(echo "$row" | base64 --decode | yq -r '.labels | join(",")')
  PARENT=$(_jq '.parent')
  EST=$(_jq '.estimate_h')
  ACC=$(echo "$row" | base64 --decode | yq -r '.acceptance // [] | map("- " + .) | join("\n")')
  TST=$(echo "$row" | base64 --decode | yq -r '.tests // [] | map("- " + .) | join("\n")')

  if [[ "$PARENT" == epic:* ]]; then
    EPIC_KEY="${PARENT#epic:}"
    EPIC_NUM="${EPIC_MAP[$EPIC_KEY]}"
    BODY="${BODY}

**Parent:** #${EPIC_NUM}

**Acceptance**
${ACC}

**Tests**
${TST}

**Estimate(h):** ${EST}"
  else
    BODY="${BODY}

**Acceptance**
${ACC}

**Tests**
${TST}

**Estimate(h):** ${EST}"
  fi

  gh issue create --repo "$REPO" --title "$TITLE" --body "$BODY" --label "$LABELS" --milestone "$MILE" >/dev/null
done

echo "Done seeding $REPO"
