#!/usr/bin/env python3
"""
Gera o template de validação em Excel para o projeto:
- Sheet 1: log_por_video (uma linha por vídeo)
- Sheet 2: exemplos_confusao (frames/snapshots problemáticos)
- Sheet 3: resumo (com fórmulas e gráfico)

Uso:
  source .venv/bin/activate
  python scripts/make_validation_sheet.py --out reports/validacao_quedas.xlsx
"""

import argparse
from pathlib import Path
import pandas as pd
from openpyxl import load_workbook
from openpyxl.chart import BarChart, Reference

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/validacao_quedas.xlsx", help="caminho de saída do .xlsx")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Sheet 1: log por vídeo
    log_cols = [
        "data_teste", "video", "cenario", "iluminacao", "multipessoa(0/1)",
        "qtd_quedas_reais", "quedas_detectadas", "quedas_perdidas", "falsos_positivos",
        "conf_threshold", "observacoes"
    ]
    df_log = pd.DataFrame(columns=log_cols)

    # Sheet 2: Exemplos de confusão
    conf_cols = [
        "video", "frame", "tipo_erro", "confidencia", "arquivo_snapshot", "nota"
    ]
    df_conf = pd.DataFrame(columns=conf_cols)

    # Sheet 3: Resumo
    df_summary = pd.DataFrame([
        ["Total videos", "=COUNTA(log_por_video!B2:B1000)"],
        ["Total quedas reais", "=SUM(log_por_video!F2:F1000)"],
        ["Total quedas detectadas", "=SUM(log_por_video!G2:G1000)"],
        ["Total quedas perdidas", "=SUM(log_por_video!H2:H1000)"],
        ["Total falsos positivos", "=SUM(log_por_video!I2:I1000)"],
        ["Recall (detectadas/reais)", "=IF(B3=0,\"\",B3/B2)"],
        ["FPR aprox. (FP/video)", "=IF(B1=0,\"\",B5/B1)"],
        ["Threshold usado", "preencher"],
        ["Obs gerais", "preencher"],
    ], columns=["metricas", "valor"])

    # Criar Excel inicial
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df_log.to_excel(writer, index=False, sheet_name="log_por_video")
        df_conf.to_excel(writer, index=False, sheet_name="exemplos_confusao")
        df_summary.to_excel(writer, index=False, sheet_name="resumo")

    # Reabrir e adicionar gráfico na aba resumo
    wb = load_workbook(out_path)
    ws = wb["resumo"]

    chart = BarChart()
    chart.title = "Resumo da Validação"
    chart.y_axis.title = "Contagens"
    chart.x_axis.title = "Métricas"

    data = Reference(ws, min_col=2, max_col=2, min_row=2, max_row=6)
    cats = Reference(ws, min_col=1, max_col=1, min_row=2, max_row=6)
    chart.add_data(data, titles_from_data=False)
    chart.set_categories(cats)

    ws.add_chart(chart, "D2")
    wb.save(out_path)

    print(f"Template de validação criado em: {out_path}")

if __name__ == "__main__":
    main()