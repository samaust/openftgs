# Domain Docs

How the skills should consume this repo's domain documentation when exploring the codebase.

## Before exploring, read these

- **`docs/GLOSSARY.md`**: the project's single glossary: every term used in openftgs and its definition, grouped by area.

## File structure

```
/
└── docs/
    ├── GLOSSARY.md
    └── modules/
        ├── cli/
        |    ├── subject1.md
        |    └── subject2.md
        └── config/
            ├── subject1.md
            └── subject2.md
```

## Use the glossary's vocabulary

When your output names a domain concept (in an issue title, a refactor proposal, a hypothesis, a test name), use the term as defined in `docs/GLOSSARY.md`. Don't drift to synonyms the glossary explicitly avoids.

# Create or update the glossary's vocabulary

If the domain concept you need isn't in the glossary then call the skill `/domain-documenting` to add it. It edits on a branch, never as uncommitted changes on `main` (see its "Where the edits go").
