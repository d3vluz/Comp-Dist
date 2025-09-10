import os
import math
import uuid
from datetime import datetime
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ALLOWED_EXTENSIONS = {"csv"}

app = Flask(__name__)
app.config["SECRET_KEY"] = "change-me-in-prod"
app.config["UPLOAD_FOLDER"] = os.path.join(app.root_path, "uploads")
app.config["OUTPUT_ROOT"] = os.path.join(app.root_path, "static", "outputs")

def analytic_availability(n: int, k: int, p: float) -> float:
    """
    Calcula a disponibilidade analítica de um sistema composto por n servidores, 
    onde pelo menos k servidores devem estar disponíveis para garantir o funcionamento do serviço.
    A disponibilidade é calculada pela soma das probabilidades de pelo menos k servidores estarem disponíveis,
    considerando que cada servidor está disponível independentemente com probabilidade p.

    Parâmetros:
        n (int): Número total de servidores.
        k (int): Número mínimo de servidores necessários para o serviço estar disponível.
        p (float): Probabilidade de um servidor estar disponível (0 <= p <= 1).

    Retorno:
        float: Disponibilidade analítica do serviço.

    Exceções:
        ValueError: Caso os parâmetros estejam fora dos limites esperados.
    """
    if not (n > 0 and 0 < k <= n and 0.0 <= p <= 1.0):
        raise ValueError("Parâmetros inválidos para n, k, p.")
    s = 0.0
    for i in range(k, n + 1):
        s += math.comb(n, i) * (p ** i) * ((1.0 - p) ** (n - i))
    return s

def simulate_availability(n: int, k: int, p: float, rounds: int = 20000, seed: int = 42) -> float:
    """
    Realiza uma simulação estocástica para estimar a disponibilidade do serviço.
    Em cada rodada, sorteia-se o número de servidores disponíveis segundo uma distribuição binomial,
    e verifica-se se pelo menos k servidores estão disponíveis.

    Parâmetros:
        n (int): Número total de servidores.
        k (int): Número mínimo de servidores necessários.
        p (float): Probabilidade de um servidor estar disponível.
        rounds (int): Número de rodadas de simulação.
        seed (int): Semente para o gerador de números aleatórios.

    Retorno:
        float: Estimativa experimental da disponibilidade do serviço.

    Exceções:
        ValueError: Caso rounds seja menor ou igual a zero.
    """
    if rounds <= 0:
        raise ValueError("rounds deve ser > 0")
    rng = np.random.default_rng(seed)
    successes = rng.binomial(n=n, p=p, size=rounds)
    ok = np.sum(successes >= k)
    return ok / float(rounds)

def allowed_file(filename: str) -> bool:
    """
    Verifica se o arquivo possui uma extensão permitida para upload.

    Parâmetros:
        filename (str): Nome do arquivo.

    Retorno:
        bool: True se o arquivo for permitido, False caso contrário.
    """
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def ensure_run_dir(run_id: str) -> str:
    """
    Garante a existência de um diretório para armazenar os resultados de uma execução identificada por run_id.

    Parâmetros:
        run_id (str): Identificador único da execução.

    Retorno:
        str: Caminho absoluto do diretório criado.
    """
    outdir = os.path.join(app.config["OUTPUT_ROOT"], run_id)
    os.makedirs(outdir, exist_ok=True)
    return outdir

def process_csv(file_path: str, run_id: str) -> Dict[str, Any]:
    """
    Processa um arquivo CSV contendo configurações de disponibilidade, executa os cálculos analíticos e experimentais,
    gera gráficos comparativos e salva os resultados em arquivos.

    Parâmetros:
        file_path (str): Caminho do arquivo CSV de entrada.
        run_id (str): Identificador único da execução.

    Retorno:
        Dict[str, Any]: Dicionário contendo caminhos relativos dos resultados, imagens e tabela HTML para exibição.

    Exceções:
        ValueError: Caso o CSV não contenha as colunas obrigatórias ou haja parâmetros inválidos.
    """
    df = pd.read_csv(file_path)
    df.columns = [c.strip().lower() for c in df.columns]
    required = {"n", "k", "p"}
    if not required.issubset(set(df.columns)):
        raise ValueError("CSV deve conter as colunas: n, k, p. (opcional: rounds)")

    if "rounds" not in df.columns:
        df["rounds"] = 20000  # default

    df["n"] = df["n"].astype(int)
    df["k"] = df["k"].astype(int)
    df["p"] = df["p"].astype(float)
    df["rounds"] = df["rounds"].astype(int)

    # validations
    for idx, row in df.iterrows():
        n, k, p, r = int(row["n"]), int(row["k"]), float(row["p"]), int(row["rounds"])
        if not (n > 0 and 0 < k <= n and 0.0 <= p <= 1.0 and r > 0):
            raise ValueError(f"Parâmetros inválidos na linha {idx+1}: n={n}, k={k}, p={p}, rounds={r}")

    # compute analytic & experimental
    results = []
    for idx, row in df.iterrows():
        n, k, p, r = int(row["n"]), int(row["k"]), float(row["p"]), int(row["rounds"])
        a = analytic_availability(n, k, p)
        e = simulate_availability(n, k, p, rounds=r, seed=42 + idx)  # vary seed per row
        results.append({
            "config": f"n={n}, k={k}, p={p:.2f}",
            "n": n, "k": k, "p": p, "rounds": r,
            "analitico": a,
            "experimental": e,
            "erro_abs": abs(a - e)
        })
    res_df = pd.DataFrame(results)

    outdir = ensure_run_dir(run_id)
    # save results csv
    results_csv_path = os.path.join(outdir, "results.csv")
    res_df.to_csv(results_csv_path, index=False)

    # Charts ----------------------------------------------------------------
    # 1) Comparison chart of analytic vs experimental per row
    plt.figure(figsize=(10, 5))
    x = np.arange(len(res_df))
    width = 0.35
    plt.bar(x - width/2, res_df["analitico"], width, label="Analítico")
    plt.bar(x + width/2, res_df["experimental"], width, label="Experimental")
    plt.xticks(x, res_df["config"], rotation=30, ha="right")
    plt.ylabel("Disponibilidade")
    plt.ylim(0, 1.0)
    plt.title("Disponibilidade: Analítico vs Experimental")
    plt.legend()
    comp_img = os.path.join(outdir, "comparacao.png")
    plt.tight_layout()
    plt.savefig(comp_img, dpi=140)
    plt.close()

    # 2) Error chart
    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(len(res_df)), res_df["erro_abs"], marker="o")
    plt.xticks(np.arange(len(res_df)), res_df["config"], rotation=30, ha="right")
    plt.ylabel("Erro absoluto |A - E|")
    plt.title("Erro absoluto por configuração")
    err_img = os.path.join(outdir, "erro_abs.png")
    plt.tight_layout()
    plt.savefig(err_img, dpi=140)
    plt.close()

    # 3) Curvas analíticas para casos k = 1, n/2, n, para cada n único
    curve_imgs = []
    for n in sorted(res_df["n"].unique()):
        ks = sorted(set([1, int(math.ceil(n/2.0)), n]))
        pgrid = np.linspace(0.0, 1.0, 101)
        plt.figure(figsize=(8, 5))
        for k in ks:
            y = [analytic_availability(n, k, float(p)) for p in pgrid]
            plt.plot(pgrid, y, label=f"k={k}")
        plt.xlabel("p (prob. servidor disponível)")
        plt.ylabel("Disponibilidade do serviço")
        plt.ylim(0, 1.0)
        plt.title(f"Curvas Analíticas — n={n} (k=1, n/2, n)")
        plt.legend()
        fn = os.path.join(outdir, f"curvas_n{n}.png")
        plt.tight_layout()
        plt.savefig(fn, dpi=140)
        plt.close()
        curve_imgs.append(fn)

    # relative paths for templates (under /static)
    def rel(full):
        return os.path.relpath(full, start=os.path.join(app.root_path, "static")).replace("\\", "/")

    images = {
        "comparacao": rel(comp_img),
        "erro": rel(err_img),
        "curvas": [rel(p) for p in curve_imgs]
    }

    return {
        "run_id": run_id,
        "results_csv_rel": rel(results_csv_path),
        "images": images,
        "table": res_df.to_html(classes="table table-striped table-sm", index=False, float_format=lambda v: f"{v:.6f}")
    }

# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def upload():
    """
    Rota principal da aplicação. Permite ao usuário enviar um arquivo CSV contendo as configurações de disponibilidade.
    Após o upload e processamento, redireciona para a página de resultados.

    Métodos HTTP:
        GET: Exibe o formulário de upload.
        POST: Processa o arquivo enviado e executa as análises.
    """
    if request.method == "POST":
        if "file" not in request.files:
            flash("Nenhum arquivo enviado.")
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            flash("Nenhum arquivo selecionado.")
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            upload_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True) 
            file.save(upload_path)

            run_id = datetime.utcnow().strftime("%Y%m%d%H%M%S") + "-" + uuid.uuid4().hex[:8]
            try:
                summary = process_csv(upload_path, run_id)
            except Exception as e:
                flash(f"Erro ao processar o CSV: {e}")
                return redirect(request.url)

            # Pass info via query args to the results route
            return redirect(url_for("results", run_id=summary["run_id"]))
        else:
            flash("Formato inválido. Envie um arquivo .csv")
            return redirect(request.url)
    return render_template("upload.html")

@app.route("/results/<run_id>")
def results(run_id: str):
    """
    Rota responsável por exibir os resultados da análise de disponibilidade para uma execução específica.
    Apresenta tabelas e gráficos gerados a partir do processamento do arquivo CSV enviado pelo usuário.

    Parâmetros:
        run_id (str): Identificador único da execução.

    Retorno:
        Renderização do template de resultados com os dados e imagens correspondentes.
    """
    outdir = ensure_run_dir(run_id)
    results_csv_path = os.path.join(outdir, "results.csv")
    if not os.path.exists(results_csv_path):
        flash("Resultados não encontrados. Tente novamente.")
        return redirect(url_for("upload"))

    df = pd.read_csv(results_csv_path)
    table_html = df.to_html(classes="table table-striped table-sm", index=False, float_format=lambda v: f"{v:.6f}")

    # Caminho relativo para static
    def rel(full):
        return os.path.relpath(full, start=os.path.join(app.root_path, "static")).replace("\\", "/")

    summary = {
        "run_id": run_id,
        "results_csv_rel": rel(results_csv_path),
        "images": {
            "comparacao": rel(os.path.join(outdir, "comparacao.png")),
            "erro": rel(os.path.join(outdir, "erro_abs.png")),
            "curvas": [
                rel(os.path.join(outdir, f))
                for f in sorted(os.listdir(outdir))
                if f.startswith("curvas_n") and f.endswith(".png")
            ]
        },
        "table": table_html
    }
    return render_template("results.html", summary=summary)

if __name__ == "__main__":
    """
    Ponto de entrada da aplicação Flask. Executa o servidor web em modo de desenvolvimento.
    """
    app.run(debug=True, host="0.0.0.0", port=5000)
