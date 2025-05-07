import os
import sys
import subprocess
import requests
from pathlib import Path
from typing import List, Tuple, Dict
from rich.console import Console
from rich.panel import Panel

# Initialize console
console = Console()

# Install missing dependencies
try:
    from langchain_community.document_loaders import ( # type: ignore
        PyPDFLoader,
        Docx2txtLoader,
        UnstructuredExcelLoader,
        UnstructuredFileLoader
    )
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma # type: ignore
    from langchain.chains import RetrievalQA
    from langchain_community.llms.ollama import Ollama
    from prompt_toolkit import prompt
    from prompt_toolkit.history import FileHistory
except ImportError as e:
    console.print(f"[red]Erro de importação: {str(e)}[/]")
    console.print("[yellow]Instalando dependências necessárias...[/]");
    
    subprocess.run([
    sys.executable, "-m", "pip", "install", 
    "langchain-core",
    "langchain-community",
    "langchain",
    "langchain-huggingface",
    "langchain-ollama",
    "chromadb",
    "sentence-transformers",
    "pydantic==1.10.13",
    "unstructured",
    "rich",
    "prompt_toolkit"
    ], check=True)

    
    console.print("[green]Dependencies installed successfully![/]")
    console.print("[yellow]Please run the script again.[/]")
    sys.exit()


DOCS_PATH = "./documents"

class DocumentChatbot:
    def __init__(self):
        self.console = Console()
        try:
            # Configuração de embeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            # Configuração do Ollama corrigida
            self.llm = Ollama(
                model="llama3",
                temperature=0.1,
                num_ctx=2048,
                top_k=40,
                top_p=0.9,
                repeat_penalty=1.1
            )
            
            # Configure text splitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                is_separator_regex=False
            )
            
            self.vector_db = None
            self.console.print("[green]✅ Models loaded successfully[/]")
        except Exception as e:
            self.console.print(f"[red]❌ Initialization error: {str(e)}[/]")
            raise

    def load_documents(self, folder_path: str) -> None:
        """Load and process documents from specified folder"""
        if not Path(folder_path).exists():
            raise ValueError(f"Folder not found: {folder_path}")

        loaders = {
            '.pdf': PyPDFLoader,
            '.docx': Docx2txtLoader,
            '.xlsx': UnstructuredExcelLoader,
            '.doc': Docx2txtLoader,
            '.xls': UnstructuredExcelLoader
        }

        documents = []
        for file_path in Path(folder_path).glob('*'):
            ext = file_path.suffix.lower()
            try:
                if ext in loaders:
                    loader = loaders[ext](str(file_path))
                    documents.extend(loader.load())
                    self.console.print(f"[green]✓[/] {file_path.name}")
                else:
                    loader = UnstructuredFileLoader(str(file_path))
                    documents.extend(loader.load())
                    self.console.print(f"[yellow]?[/] {file_path.name} (generic format)")
            except Exception as e:
                self.console.print(f"[red]✗[/] {file_path.name} (error: {str(e)})")

        if not documents:
            raise ValueError("No valid documents found!")

        chunks = self.text_splitter.split_documents(documents)
        
        self.vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory="./chroma_db"
        )
        self.console.print("[green]✅ Documents indexed successfully[/]")

    def answer_question(self, question: str) -> Tuple[str, List[Dict]]:
        """Answer questions based on loaded documents"""
        if not self.vector_db:
            return "Documents not loaded properly.", []

        try:
            # Configuração atualizada do retriever
            retriever = self.vector_db.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": 10,
                    "score_threshold": 0.35
                }
            )
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True
            )
            
            result = qa_chain.invoke({"query": f"Responda em português: {question}"})
            
            unique_sources = {}
            for doc in result["source_documents"]:
                source = doc.metadata.get('source', 'Unknown source')
                content = doc.page_content[:250] + ('...' if len(doc.page_content) > 250 else '')
                unique_sources[source] = {"source": source, "content": content}
            
            return result["result"], list(unique_sources.values())
            
        except Exception as e:
            self.console.print(f"[red]Error processing question: {str(e)}[/]")
            return "Sorry, I encountered an error processing your question.", []

def check_ollama() -> bool:
    """Check if Ollama service is running"""
    try:
        response = requests.get('http://localhost:11434', timeout=5)
        return response.status_code == 200
    except:
        return False

def main():
    console.print(Panel.fit("🔧 Starting Document ChatBot", style="bold blue"))
    
    if not check_ollama():
        console.print("[red]❌ Ollama is not running![/]")
        console.print("Please first run in another terminal:")
        console.print("[bold yellow]ollama serve[/]")
        return

    try:
        bot = DocumentChatbot()
        
        console.print("[bold]📂 Loading documents...[/]")
        bot.load_documents(DOCS_PATH)
        console.print("[green]✅ Ready to answer questions![/]\n")
        
        console.print(Panel.fit("💬 Chat Mode (type 'exit' to quit)", style="bold blue"))
        
        while True:
            try:
                user_input = prompt("❔ Question: ", history=FileHistory(".chat_history"))
                
                if user_input.lower() in ('exit', 'quit', 'sair'):
                    break
                    
                answer, sources = bot.answer_question(user_input)
                
                console.print(Panel.fit(answer, title="💡 Answer", style="green"))
                
                if sources:
                    console.print(Panel.fit(
                    "\n".join(
                        f"[bold]{src['source']}[/]"
                        for src in sources
                    ),
                    title="📚 Source Documents",
                    style="blue"
                ))

                
            except KeyboardInterrupt:
                console.print("\n🛑 Shutting down...")
                break
            except Exception as e:
                console.print(f"[red]Chat error: {str(e)}[/]")
                continue
                
    except Exception as e:
        console.print(f"[red]Critical failure: {str(e)}[/]")
    finally:
        console.print("\n🔴 ChatBot shutdown complete")

if __name__ == "__main__":
    main()