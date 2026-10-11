#!/bin/bash
# Ensure the prd-to-tickets provenance label exists on the current GitHub repo.
# --force makes this idempotent: it updates the label if it already exists.

gh label create "skill:prd-to-tickets" \
  --color 0E8A16 \
  --description "Ticket created by the prd-to-tickets skill" \
  --force
