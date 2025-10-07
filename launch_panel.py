# launch_panel.py
import os, subprocess, sys, webbrowser

# ajuste aqui se quiser default diferente
API_URL = os.getenv("SAAS_API_URL", "http://127.0.0.1:8000")
API_KEY = os.getenv("SAAS_API_KEY", "minha-chave-forte")

def main():
    env = os.environ.copy()
    env["SAAS_API_URL"] = API_URL
    env["SAAS_API_KEY"] = API_KEY

    # abre o navegador depois que o streamlit subir
    webbrowser.open("http://localhost:8501", new=1)

    # roda o streamlit do seu projeto
    cmd = [sys.executable, "-m", "streamlit", "run", "painel.py", "--server.headless=false"]
    subprocess.Popen(cmd, env=env)

if __name__ == "__main__":
    main()