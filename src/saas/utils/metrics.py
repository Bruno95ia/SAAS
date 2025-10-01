from prometheus_client import Counter, Histogram, start_http_server
alive_beats = Counter("alive_beats_total", "Sinal de vida do serviço")
def bootstrap_metrics(port=9108):
    start_http_server(port)
