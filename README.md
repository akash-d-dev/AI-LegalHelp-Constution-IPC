# 🇮🇳 Indian Legal AI Assistant

A powerful AI-driven agent that answers legal queries related to the **Constitution of India** and the **Indian Penal Code (IPC)** using keyword generation, vector search, and smart tool orchestration.

---

## 🧠 How It Works

1. **User enters a legal query**  
   → e.g., *"When can the Indian government restrict freedom of speech?"*

2. **AI Agent activates tools**:
   - 🔑 `generate_keywords` → Extracts keywords like "freedom", "speech", "restrictions"
   - 📚 `search_db_constitution` → Searches Indian Constitution vector DB
   - ⚖️ `search_db_penal_code` → Searches IPC vector DB
   - 🔄 `enhanced_cross_domain_legal_search` → Performs parallel searches and cross-references between Constitution and IPC
   - 🔄 Synthesizes all retrieved data into a final answer

3. **Returns structured legal answer**  
   With referenced articles and sections.

---

## ⚙️ System Components

- `generate_keywords`: Keyword extractor from user input
- `search_db_constitution`: Vector search on Constitution articles
- `search_db_penal_code`: Vector search on IPC sections
- `enhanced_cross_domain_legal_search`: Advanced cross-document search and fusion
- `ai_agent`: Controls the flow, uses tools, composes the answer

---

## 🔮 Future Add-on: Predict Punishment from Case

A fine-tuned AI model trained on court judgments to:

- Predict punishment based on case description
- Reference relevant IPC sections
- Suggest similar past judgments

**Planned Tool:** `predict_punishment_from_case`

---

## 🚀 Key Features

- **Smart Search**: Multi-stage hybrid search strategy for comprehensive results
- **Cross-Reference**: Automatic linking between Constitution and IPC sections
- **Context-Aware**: Understands legal context and relationships
- **Fast & Accurate**: Optimized vector search with result fusion

---

## 🛠️ Tech Stack

- Python 3.8+
- Milvus Vector Database
- OpenAI GPT Models / Google Gemini Models
- FastAPI Backend

## 📋 Testing & Logs

For test results and detailed logs, check out the generated scripts and logs at:
[Test Scripts & Logs](https://github.com/akash-d-dev/AI-LegalHelp-Constution-IPC/tree/main/backend/agent_system/admin/scripts/generated)

---

## 📚 Data Sources

- Constitution of India
- Indian Penal Code (IPC)
- Legal cross-references database
