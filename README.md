# GenAI Learning Journey

This repository tracks my learning journey with **Generative AI and LangChain** — every concept I have studied and practiced so far.

---

## What I Have Learned So Far

### 1. Chat Models (LLM Integration)
How to connect and use different AI providers:

| File | What I Learned |
|------|----------------|
| `ChatModels/1_chatmodel_openai.py` | Using OpenAI GPT-4 to answer questions |
| `ChatModels/2_chatmodel_anthropic.py` | Using Anthropic Claude model |
| `ChatModels/3_chatmodel_grok.py` | Using Grok (xAI) model |
| `ChatModels/4_chatmodel_hf_api.py` | Running models via Hugging Face API |
| `LLMs/1_llm_demp.py` | Basic LLM usage demo |

---

### 2. Prompt Templates
How to write structured, reusable instructions for LLMs:

- **PromptTemplate** — create templates with dynamic variables like `{topic}`, `{text}`
- **ChatPromptTemplate** — conversation-style prompts
- Built a chatbot and a prompt generator (`Prompt/` folder)

---

### 3. Output Parsers
Converting raw LLM output into useful, structured formats:

| Parser | What It Does |
|--------|--------------|
| `StrOutputParser` | Returns plain text |
| `JsonOutputParser` | Returns JSON formatted output |
| `StructuredOutputParser` | Output with a defined schema |
| `PydanticOutputParser` | Type-safe output using Pydantic models |

---

### 4. Pydantic
Data validation and structured data modeling:

- Creating custom data models using `BaseModel`
- Adding field-level validation with `Field` (e.g., `gt=18` for age)
- Using `Optional` fields and special types like `EmailStr`
- Integrating Pydantic with LangChain to get structured, validated output from LLMs

---

### 5. Embedding Models
Converting text into numerical vectors so AI can understand meaning and similarity:

| File | What I Learned |
|------|----------------|
| `1_embedding_query_openai.py` | Embedding a single query |
| `2_emeding_openai_doc.py` | Embedding multiple documents |
| `3_embedding_hf_local.py` | Local embeddings using Hugging Face |
| `4_document_similarity.py` | Finding similar documents using **cosine similarity** |

> **Real example:** Given the query `"Tell me about MS Dhoni"`, the system finds the most similar document from a cricket player list using vector math.

---

### 6. Chains (LangChain LCEL)
Connecting multiple steps together using LangChain Expression Language (`|` pipe operator):

| Chain Type | File | What It Does |
|-----------|------|--------------|
| Simple Chain | `simplechain.py` | `prompt | model | parser` — basic pipeline |
| Sequential Chain | `sequentialchain.py` | Runs steps one after another |
| Parallel Chain | `parallel_chain.py` | Runs two tasks simultaneously (OpenAI + Groq) |
| Conditional Chain | `conditional_chain.py` | Routes to different responses based on sentiment |

**Parallel Chain example:**
- Generates notes AND a quiz at the same time using two different models, then merges both into one document.

**Conditional Chain example:**
- Classifies feedback as positive or negative, then generates an appropriate response for each case.

---

### 7. Runnables
Advanced LangChain building blocks for flexible pipelines:

| Runnable | What It Does |
|----------|--------------|
| `RunnableSequence` | Chains steps in order |
| `RunnableParallel` | Runs multiple steps at the same time |
| `RunnablePassthrough` | Passes input through unchanged |
| `RunnableLambda` | Inserts a custom Python function into a chain |
| `RunnableBranch` | Takes different paths based on a condition |

**Smart RunnableBranch example:**
- Generates a report → if the output is over 300 words, it **automatically summarizes** it; otherwise passes it through as-is.

---

## Tech Stack

| Tool / Library | Purpose |
|---------------|---------|
| **Python** | Primary programming language |
| **LangChain** | AI application framework |
| **OpenAI GPT-4** | Main chat and embedding model |
| **Anthropic Claude** | Alternative chat model |
| **Groq (LLaMA 3.3 70B)** | Fast inference model |
| **Hugging Face** | Open-source models and local embeddings |
| **Pydantic** | Data validation and structured output |
| **scikit-learn** | Cosine similarity for document search |
| **python-dotenv** | Managing API keys via `.env` file |

---

## Folder Structure

```
GenAI-learning/
└── langchain-models/
    ├── ChatModels/       # Different AI provider integrations
    ├── LLMs/             # Basic LLM usage
    ├── Prompt/           # Prompt templates and chatbot
    ├── OutPutParser/     # Output parsing techniques
    ├── Pydantic/         # Data validation with Pydantic
    ├── EmbeddedModels/   # Text embeddings and similarity search
    ├── Chain/            # LCEL chains (simple, parallel, conditional)
    ├── Runnables/        # Advanced runnables (branch, parallel, lambda)
    └── Structure/        # TypedDict-based structured output
```

---

## Learning Order (The Path I Followed)

1. **Chat Models** — connecting to AI providers and getting responses
2. **Prompt Templates** — writing reusable, dynamic prompts
3. **Output Parsers** — formatting and structuring LLM responses
4. **Pydantic** — validating and modeling structured data
5. **Embedding Models** — understanding text similarity and vector search
6. **Chains (LCEL)** — building pipelines with `prompt | model | parser`
7. **Runnables** — advanced control flow (branching, parallelism, custom logic)

---

*This learning journey is ongoing — more topics coming soon!*
