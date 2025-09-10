# Disponibilidade de Serviço — Exercício 1.2

Aplicação Flask com upload de CSV, cálculo analítico e simulação estocástica da disponibilidade de um serviço replicado (parâmetros `n`, `k`, `p`), geração de tabelas e gráficos.

## Como executar

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Abra http://localhost:5000 no navegador, envie um CSV no formato:

```csv
n,k,p,rounds
3,1,0.20,20000
3,1,0.50,20000
3,1,0.80,20000
5,3,0.50,20000
5,3,0.70,20000
10,10,0.95,40000
10,10,0.99,40000
```

A aplicação produzirá:
- Tabela com valores analíticos e experimentais lado a lado
- Gráfico de comparação (Analítico vs Experimental)
- Gráfico de erro absoluto
- Curvas analíticas para cada `n` encontrado (para k = 1, n/2, n, variando `p` de 0 a 1)
