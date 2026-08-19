import os
import warnings

# Ocultando avisos de depreciação e warnings das bibliotecas no terminal
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Forçando o Ollama/PyTorch a não tentar usar placas NVIDIA/CUDA no sistema
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OLLAMA_NUM_PARALLEL"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN_WARNING"] = "1"   # Desativa o aviso do HF_TOKEN
os.environ["TOKENIZERS_PARALLELISM"] = "false"              # Evita avisos de concorrência

import sys
import shutil
from pathlib import Path
from typing import List, Tuple, Dict
from rich.console import Console
from rich.panel import Panel

# Importações de carregadores e splitters
from langchain_community.document_loaders import ( 
    PDFPlumberLoader,
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredExcelLoader,
    UnstructuredFileLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Importações de embeddings, LLM e VectorStore (Pacotes dedicados modernos)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM

# Importações do Core para construção de cadeias (LCEL) sem dependência de 'langchain.chains'
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from prompt_toolkit import prompt
from prompt_toolkit.history import FileHistory

DOCS_PATH = "./documents"

class DocumentChatbot:
    def __init__(self):
        self.console = Console()
        try:
            # Configuração de embeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            # Configuração do Ollama com Janela de Contexto Expandida
            self.llm = OllamaLLM(
                model="llama3",
                temperature=0.1,
                num_ctx=4096,
                num_gpu=0,
                top_k=40,
                top_p=0.9,
                repeat_penalty=1.1
            )
            
            # Text splitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,      # menor p/ manter a busca vetorial mais precisa
                chunk_overlap=250,   # ~30% p/ nao perder contexto
                separators=["\n\n", "\n", ". ", " ", ""]  # Preserva parágrafos e frases completas
            )
            
            self.vector_db = None
            self.console.print("[green]✅ Modelos e componentes carregados com sucesso![/]")
        except Exception as e:
            self.console.print(f"[red]❌ Erro na inicialização: {str(e)}[/]")
            raise

    def load_documents(self, folder_path: str) -> None:
        """Carrega e processa os documentos da pasta configurada com suporte a múltiplos formatos"""
        if not Path(folder_path).exists():
            raise ValueError(f"Pasta não encontrada: {folder_path}")

        loaders = {
            '.pdf': PDFPlumberLoader,
            '.docx': Docx2txtLoader,
            '.xlsx': UnstructuredExcelLoader,
        }

        documents = []
        for file_path in Path(folder_path).glob('*'):
            # Ignora arquivos temporários (ex: ~$documento.docx aberto pelo Word)
            if file_path.name.startswith('~$') or file_path.is_dir():
                continue

            ext = file_path.suffix.lower()
            try:
                if ext in loaders:
                    loader = loaders[ext](str(file_path))
                    documents.extend(loader.load())
                else:
                    loader = UnstructuredFileLoader(str(file_path))
                    documents.extend(loader.load())
            except Exception as e:
                self.console.print(f"[red]✗ Erro ao ler {file_path.name}: {str(e)}[/]")

        if not documents:
            raise ValueError("Nenhum documento válido foi encontrado!")

        chunks = self.text_splitter.split_documents(documents)
        
        if os.path.exists("./chroma_db"):
            shutil.rmtree("./chroma_db")
            
        self.vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory="./chroma_db"
        )
        self.console.print("[green]✅ Documentos indexados no ChromaDB com sucesso![/]")

    def format_docs(self, docs) -> str:
        """Função auxiliar para concatenar o conteúdo dos documentos no prompt"""
        return "\n\n".join(doc.page_content for doc in docs)

    def answer_question(self, question: str) -> Tuple[str, List[Dict]]:
        """Gera respostas com base na busca vetorial dos documentos via LCEL"""
        if not self.vector_db:
            return "O banco vetorial não foi inicializado corretamente.", []

        try:
            retriever = self.vector_db.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 5,
                    "fetch_k": 10,
                    "lambda_mult": 0.7
                }
            )
            
            retrieved_docs = retriever.invoke(question)
            
            prompt_template = ChatPromptTemplate.from_template(
                "Você é um assistente especializado em análise rigorosa de documentos.\n"
                "Sua tarefa é responder à pergunta utilizando EXCLUSIVAMENTE os dados presentes no contexto fornecido.\n\n"
                "Diretrizes:\n"
                "1. LEIA TODO O CONTEXTO com atenção antes de responder.\n"
                "2. Identifique termos técnicos, chaves, regras e responsabilidades explicitamente citados no texto.\n"
                "3. Se o texto mencionar termos específicos (ex: chaves primárias, senhas, cargos, setores), INCLUA-OS diretamente na resposta.\n"
                "4. Responda de forma completa, cobrindo todos os pontos solicitados pelo usuário.\n\n"
                "Contexto:\n{context}\n\n"
                "Pergunta: {question}"
            )

            rag_chain = (
                {"context": lambda x: self.format_docs(retrieved_docs), "question": RunnablePassthrough()}
                | prompt_template
                | self.llm
                | StrOutputParser()
            )
            
            response_text = rag_chain.invoke(question)
            
            unique_sources = {}
            for doc in retrieved_docs:
                source = doc.metadata.get('source', 'Fonte desconhecida')
                content = doc.page_content[:250] + ('...' if len(doc.page_content) > 250 else '')
                unique_sources[source] = {"source": source, "content": content}
            
            return response_text, list(unique_sources.values())
            
        except Exception as e:              
            self.console.print(f"[red]Erro ao processar a pergunta: {str(e)}[/]")
            return "Ocorreu um erro interno ao processar sua solicitação.", []

def check_ollama() -> bool:
    """Verifica se o serviço local do Ollama está ativo"""
    import requests
    try:
        response = requests.get('http://localhost:11434', timeout=5)
        return response.status_code == 200
    except Exception:
        return False

def main():
    console = Console()
    console.print(Panel.fit("🔧 Iniciando Document ChatBot", style="bold blue"))
    
    if not check_ollama():
        console.print("[red]❌ O serviço do Ollama não está rodando![/]")
        return

    try:
        bot = DocumentChatbot()
        
        console.print("[bold]📂 Carregando documentos...[/]")
        bot.load_documents(DOCS_PATH)
        console.print("[green]✅ Pronto para responder perguntas![/]\n")
        
        console.print(Panel.fit("💬 Modo Chat (digite 'sair' ou 'exit' para encerrar)", style="bold blue"))
        
        while True:
            try:
                user_input = prompt("❔ Pergunta: ", history=FileHistory(".chat_history"))
                
                if user_input.strip().lower() in ('exit', 'quit', 'sair'):
                    break
                
                if not user_input.strip():
                    continue

                answer, sources = bot.answer_question(user_input)
                
                console.print(Panel.fit(answer, title="💡 Resposta", style="green"))
                
                if sources:
                    sources_text = "\n".join(f"• [bold]{src['source']}[/]" for src in sources)
                    console.print(Panel.fit(sources_text, title="📚 Documentos Consultados", style="blue"))

            except KeyboardInterrupt:
                console.print("\n🛑 Encerrando sessão...")
                break
            except Exception as e:
                console.print(f"[red]Erro durante o chat: {str(e)}[/]")
                continue
                
    except Exception as e:
        console.print(f"[red]Falha crítica na aplicação: {str(e)}[/]")
    finally:
        console.print("\n🔴 Aplicação finalizada.")

if __name__ == "__main__":
    main()