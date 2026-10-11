#!/bin/bash
# Create or update every GitHub label the skills apply, on the current repository.
# --force makes this idempotent: an existing label is updated in place.
# The triage-role strings must match docs/agents/triage-labels.md.
set -euo pipefail

label() { gh label create "$1" --color "$2" --description "$3" --force; }

# Triage roles (state)
label needs-triage    D4C5F9 "Owner needs to evaluate this issue"
label needs-info      FBCA04 "Waiting on reporter for more information"
label ready-for-agent 1D76DB "Fully specified, ready for an AFK agent"
label ready-for-human C5DEF5 "Requires human implementation"
label wontfix         FFFFFF "Will not be actioned"

# Triage roles (category)
label bug             D73A4A "Something isn't working"
label enhancement     A2EEEF "New feature or request"

# Q-n issues (decision and candidates issues; no category role)
label q-n             BFD4F2 "Q-n decision or Q-n candidates issue"

# Out-of-scope record (canonical issues of rejected enhancements)
label out-of-scope    EDEDED "Rejected request: the canonical record of its concept"

# Skill labels (which skill created the issue)
label skill:conversation-to-prd 5319E7 "PRD created by the conversation-to-prd skill"
label skill:prd-to-tickets      0E8A16 "Ticket created by the prd-to-tickets skill"
