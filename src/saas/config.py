import os

def get_api_key():
    """
    Retorna a chave da API definida no ambiente
    ou usa o padrão 'minha-chave-forte' se não houver.
    """
    key = os.getenv("SAAS_API_KEY", "minha-chave-forte")
    print(f"[config] SAAS_API_KEY carregada: {key}")
    return key
