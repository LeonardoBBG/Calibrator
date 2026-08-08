# Schemas

The Pydantic v2 models in `src/sol_fable/models.py` are authoritative. They reject
unknown fields and validate enums, confidence ranges, required fields and stable
structured assessment shapes at every storage boundary. JSON Schema snapshots can
be regenerated with:

```bash
PYTHONPATH=src python -m sol_fable.schema_export
```

