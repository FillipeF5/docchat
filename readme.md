# 📄 DocChat – Chatbot para Documentos com IA Local

**DocChat** é uma aplicação Python que permite conversar com documentos locais (PDF, Word, Excel, etc.) usando modelos de linguagem executados localmente via [Ollama](https://ollama.com). Ideal para buscas semânticas, estudos ou análises documentais, sem depender da nuvem.

## ✨ Funcionalidades

- 📂 Leitura e indexação de documentos (`.pdf`, `.docx`, `.xlsx`, etc.)
- 🧠 Embeddings com `sentence-transformers`
- 🤖 LLM local com `Ollama` (ex: LLaMA3)
- 🔍 Recuperação semântica de trechos relevantes
- 💬 Interface de linha de comando com histórico de chat
- 🖼️ Visualização de fontes com `rich`

---

## 🚀 Requisitos

- Python 3.10 ou superior (não compatível com 3.13 no momento)
- [Ollama](https://ollama.com) instalado e em execução (`ollama serve`)
- Modelos compatíveis carregados no Ollama (ex: `ollama run llama3`)
- Sistema operacional: Windows, Linux ou macOS

---

## 🔧 Instalação

1. **Clone o repositório**
   ```bash
   git clone https://github.com/seu-usuario/docchat.git
   cd docchat


## Uso

**Crie um ambiente virtual**
python -m venv venv
venv\Scripts\activate  # Windows
ou
source venv/bin/activate  # Linux/macOS

**Instale as dependencias**
pip install -r requirements.txt

**Inicie o Ollama**
ollama serve
ollama run llama3

**Run, in bash**
python main.py
