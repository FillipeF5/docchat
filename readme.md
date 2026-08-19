# 📄 DocChat – Chatbot RAG Local para Documentos

**DocChat** é uma aplicação Python interativa de CLI que aplica arquitetura **RAG (Retrieval-Augmented Generation)** para permitir consultas semânticas em documentos locais (`.pdf`, `.docx`, `.xlsx`) utilizando modelos de linguagem executados 100% offline via **Ollama**.

A solução garante privacidade total dos dados, eliminando a necessidade de envio de informações confidenciais para APIs de terceiros[cite: 1].

---

## ✨ Funcionalidades e Diferenciais Técnicos

- **Processamento Multi-Formato:** Carregamento automático e extração precisa de dados em arquivos PDF (`pdfplumber`), Word (`docx2txt`) e planilhas Excel (`unstructured`)[cite: 1].
- **Chunking Estratégico:** Divisão de texto via `RecursiveCharacterTextSplitter` ajustada (`chunk_size=800`, `chunk_overlap=250`) para manter o contexto entre parágrafos sem perda de dados técnicos[cite: 1].
- **Embeddings Semânticos:** Uso do modelo multilíngue `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` executado localmente via PyTorch/Hugging Face[cite: 1].
- **Busca Vetorial por Diversidade (MMR):** Armazenamento em **ChromaDB** com recuperação por *Maximal Marginal Relevance* ($k=5$, $fetch\_k=10$), reduzindo redundâncias e trazendo dados mais complementares[cite: 1].
- **Cadeia Modernizada (LCEL):** Pipeline montado inteiramente sobre *LangChain Expression Language* (`RunnablePassthrough`, `ChatPromptTemplate`), garantindo previsibilidade no fluxo de prompt[cite: 1].
- **Interface CLI com Rich & Prompt Toolkit:** Terminal estilizado com histórico de perguntas (`.chat_history`), atalhos de saída e citação transparente das fontes consultadas[cite: 1].

---

## 🚀 Pré-requisitos

- **Python:** 3.10 ou superior
- **Ollama:** Instalado e em execução na máquina (`http://localhost:11434`)[cite: 1]
- **Modelo LLM:** Llama 3 baixado no Ollama (`ollama pull llama3`)[cite: 1]

---

## 🔧 Configuração e Instalação

1. **Clonar o repositório:**
   ```bash
   git clone [https://github.com/FillipeF5/docchat.git](https://github.com/FillipeF5/docchat.git)
   cd docchat