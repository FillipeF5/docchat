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

1. **Crie um ambiente virtual**
    ``` bash 
    python -m venv venv
    venv\Scripts\activate  # Windows

2. **Instale as dependencias**
    ```bash
    pip install -r requirements.txt

3. **Inicie o Ollama**
    ```bash
    ollama serve
    ollama run llama3

4. **Run, in bash**
    ```bash
    python main.py
