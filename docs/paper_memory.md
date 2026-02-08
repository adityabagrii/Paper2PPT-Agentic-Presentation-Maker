# Persistent Paper Memory (Index + Search)

Store and search lightweight paper summaries locally.

**Index a paper**
```bash
researchos --index-paper -a 2401.12345
```
## Workflow
![Workflow](../images/paper_memory_workflow.png)


**Search the index**
```bash
researchos --search "keyframe selection efficiency"
```

**Where it’s stored**
- Index: `~/.researchos/paper_index.json`
- Per-run entry: `outputs/paper_index_entry.json`

**Stored fields**
- summary
- key claims
- methods
- datasets
- keywords
