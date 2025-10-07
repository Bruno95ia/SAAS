# 📓 Dataset Changelog – Detecção de Quedas

Este documento registra todas as alterações significativas no dataset utilizado para treinar o modelo de detecção de quedas.

---

## Formato de versão

* **Major**: Mudanças grandes (ex.: novo esquema de anotação, mudança de classes, reestruturação completa).
* **Minor**: Incrementos de dados (ex.: +50 vídeos, +10k frames, novos cenários).
* **Patch**: Correções pequenas (ex.: ajuste de anotações incorretas, limpeza de duplicados).

Exemplo: `queda_caso-1.2.0`

---

## Histórico

### v1.0.0 – Baseline (2025-09-30)

* 120 vídeos anotados (50 quedas, 70 atividades normais).
* Ambientes: sala, quarto, banheiro.
* FPS: 8; total ~12.000 frames anotados.
* Classes: `Pessoa1`, `Queda`.
* Divisão: 70/20/10 (train/val/test).

### v1.1.0 – Novos negativos difíceis (2025-10-15)

* +60 vídeos (sentar, deitar, pegar objeto).
* Corrigidas 12 anotações inconsistentes.
* Total: ~18.000 frames anotados.

### v1.2.0 – Expansão multipessoa (2025-10-30)

* +40 vídeos com idoso + cuidador em cena.
* Ambientes adicionais: cozinha, corredor.
* Melhor balanceamento entre tipos de quedas.

---

## Próximos passos planejados

* [ ] Adicionar stress tests (quedas parciais fora de quadro, oclusões).
* [ ] Expandir diversidade de perfis (diferentes idades, biotipos, roupas).
* [ ] Documentar métricas de qualidade do dataset (aceite ≥95% revisão dupla).
