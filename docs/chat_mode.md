# Chat Mode (RAG)

Chat with the model about a paper using stored context and semantic retrieval.

**Command**
```bash
researchos -a 2401.12345 --chat
```
## Workflow
![Workflow](../images/chat_mode_workflow.png)


**What it does**
- Builds and stores `outputs/paper_context.json` (summary + chunks + embeddings)
- Retrieves the most relevant chunks with semantic embeddings
- Answers questions using only the retrieved context

**Output**
- `outputs/chat_history.md`
- `outputs/paper_context.json`

**Notes**
- If embeddings are unavailable, it falls back to keyword retrieval.
