# Persistent Paper Memory (Index + Search)

Store and search lightweight paper summaries locally.

**Index a paper**
```bash
paper2ppt --index-paper -a 2401.12345
```

**Search the index**
```bash
paper2ppt --search "keyframe selection efficiency"
```

**Where it’s stored**
- Index: `~/.paper2ppt/paper_index.json`
- Per-run entry: `outputs/paper_index_entry.json`

**Stored fields**
- summary
- key claims
- methods
- datasets
- keywords
