#!/bin/bash
# Ensure the conversation-to-prd provenance label exists on the current GitHub repo.
# --force makes this idempotent: it updates the label if it already exists.

gh label create "skill:conversation-to-prd" \
  --color 5319E7 \
  --description "PRD created by the conversation-to-prd skill" \
  --force
