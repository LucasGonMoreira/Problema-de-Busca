# -*- coding: utf-8 -*-
"""
Simulação de entregas em grade hexagonal 6x6.
Nomenclatura dos pontos: linha_ponto (ex.: 3_2 = linha 3, ponto 2).
Os caminhos entre pontos são calculados pelo A* implementado neste arquivo.
Um entregador de moto (imagem PNG) percorre cada rota, para em cada casa
(imagem PNG) e registra o combustível gasto. O resultado é animado para as
duas rotas lado a lado. Pensado para rodar no Google Colab.
"""

import heapq
import itertools
import os
import random

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Circle, Rectangle

SEED = 42
LINHAS, PONTOS = 6, 6        # linhas e pontos por linha, ambos de 0 a 5
CD = "0_0"                   # centro de distribuição

# ===== Entregas (formato linha_ponto) =====
ENTREGAS = ["5_0", "5_5", "0_5", "3_5", "4_3", "2_3"]

# ===== Parâmetros da simulação =====
LITROS_POR_UNIDADE = 0.10    # litros gastos por unidade de custo de cada aresta
TANQUE_L = 6.5              # capacidade inicial do tanque, em litros
PASSOS_POR_ARESTA = 2        # quadros da animação em cada aresta
PAUSA_NA_ENTREGA = 2         # quadros parados em cada casa
DPI_GIF = 48                 # resolução do GIF (menor = mais rápido)

# ===== Imagens (PNG) =====
PASTA_IMAGENS = os.path.dirname(os.path.abspath(__file__))  # PNGs junto deste script
ARQ_CASA = os.path.join(PASTA_IMAGENS, "casa-limpa.png")
ARQ_MOTO = os.path.join(PASTA_IMAGENS, "entregador.png")
ZOOM_CASA = 0.13             # tamanho da casa no gráfico (ajuste se necessário)
ZOOM_MOTO = 0.17             # tamanho do entregador no gráfico (ajuste se necessário)


def carregar_imagem(caminho):
    """Lê um PNG como matriz. Avisa com clareza se o arquivo não foi enviado."""
    if not os.path.exists(caminho):
        raise FileNotFoundError(
            f"Imagem não encontrada: {caminho}. "
            "Envie os PNGs para /content antes de rodar o código."
        )
    return plt.imread(caminho)


# ---------------- 1. Grafo ----------------
def criar_grafo(seed=SEED):
    rng = np.random.default_rng(seed)

    # Posição de cada ponto: linha_ponto, com deslocamento nas linhas ímpares
    pos = {}
    for linha in range(LINHAS):
        for ponto in range(PONTOS):
            x = ponto + (0.5 if linha % 2 else 0)
            y = linha * np.sqrt(3) / 2
            pos[f"{linha}_{ponto}"] = (x, y)

    # Cantos da grade: regiões de alto consumo
    xs, ys = zip(*pos.values())
    cantos = [(x, y) for x in (min(xs), max(xs)) for y in (min(ys), max(ys))]

    def dist_canto(p):
        return min(np.hypot(p[0] - cx, p[1] - cy) for cx, cy in cantos)

    G = nx.Graph()
    G.add_nodes_from(pos)
    for a, b in itertools.combinations(pos, 2):
        if np.hypot(*np.subtract(pos[a], pos[b])) < 1.15:  # apenas vizinhos diretos
            penal = 1 + max(0, (1.8 - min(dist_canto(pos[a]), dist_canto(pos[b]))) * 1.5)
            G.add_edge(a, b, weight=round(rng.uniform(1.0, 1.8) * penal, 2))
    return G, pos


# ---------------- 2. A* ----------------
def a_estrela(G, pos, inicio, fim):
    """
    A* explícito: f(n) = g(n) + h(n).
    g(n) = custo acumulado de combustível desde o início.
    h(n) = distância euclidiana até o destino (admissível, pois cada aresta
           tem comprimento ~1 e custo >= 1).
    Retorna (custo, lista de nós do caminho).
    """
    def h(no):
        return float(np.hypot(*np.subtract(pos[no], pos[fim])))

    g = {inicio: 0.0}
    veio_de = {}
    aberta = [(h(inicio), inicio)]
    fechada = set()

    while aberta:
        _, atual = heapq.heappop(aberta)
        if atual in fechada:
            continue
        if atual == fim:
            caminho = [atual]
            while caminho[-1] in veio_de:
                caminho.append(veio_de[caminho[-1]])
            return g[fim], caminho[::-1]
        fechada.add(atual)

        for viz in G.neighbors(atual):
            novo_g = g[atual] + G[atual][viz]["weight"]
            if novo_g < g.get(viz, float("inf")):
                g[viz] = novo_g
                veio_de[viz] = atual
                heapq.heappush(aberta, (novo_g + h(viz), viz))
    return float("inf"), None


def calcular_trechos(G, pos, locais):
    """Usa o A* para calcular custo e caminho entre todos os pares de locais."""
    return {(a, b): a_estrela(G, pos, a, b)
            for a, b in itertools.permutations(locais, 2)}


# ---------------- 3. Custo de uma rota ----------------
def custo_rota(ordem, T):
    seq = [CD, *ordem, CD]
    return sum(T[a, b][0] for a, b in zip(seq, seq[1:]))


# ---------------- 4. Duas formas de definir a ordem ----------------
def vizinho_mais_proximo(T):
    restantes, atual, ordem = list(ENTREGAS), CD, []
    while restantes:
        prox = min(restantes, key=lambda e: T[atual, e][0])
        ordem.append(prox)
        restantes.remove(prox)
        atual = prox
    return ordem


def algoritmo_genetico(T, inicial, seed=SEED, pop_size=100, geracoes=300, mut=0.15):
    rng = random.Random(seed)
    n = len(ENTREGAS)

    def aptidao(o):
        return custo_rota(o, T)

    pop = [list(inicial)] + [rng.sample(ENTREGAS, n) for _ in range(pop_size - 1)]

    for _ in range(geracoes):
        pop.sort(key=aptidao)
        nova = [pop[0][:], pop[1][:]]  # elitismo
        while len(nova) < pop_size:
            p1 = min(rng.sample(pop, 3), key=aptidao)
            p2 = min(rng.sample(pop, 3), key=aptidao)
            # Cruzamento OX
            a, b = sorted(rng.sample(range(n), 2))
            filho = [None] * n
            filho[a:b + 1] = p1[a:b + 1]
            resto = [g for g in p2 if g not in filho]
            for i in range(n):
                if filho[i] is None:
                    filho[i] = resto.pop(0)
            # Mutação por troca
            if rng.random() < mut:
                i, j = rng.sample(range(n), 2)
                filho[i], filho[j] = filho[j], filho[i]
            nova.append(filho)
        pop = nova

    return min(pop, key=aptidao)


# ---------------- 5. Setas de direção ----------------
def desenhar_setas(ax, pos, caminho, cor, deslocamento=0.14, tamanho=0.12):
    """
    Desenha uma seta ao lado de cada segmento do caminho, no sentido do movimento.
    A seta é deslocada perpendicularmente à linha para não cobrir o trajeto.
    """
    for u, v in zip(caminho, caminho[1:]):
        p1, p2 = np.array(pos[u]), np.array(pos[v])
        d = p2 - p1
        comprimento = np.linalg.norm(d)
        if comprimento == 0:
            continue
        d_unit = d / comprimento
        normal = np.array([-d_unit[1], d_unit[0]])  # perpendicular (lado esquerdo)

        meio = (p1 + p2) / 2 + normal * deslocamento
        inicio = meio - d_unit * tamanho
        fim = meio + d_unit * tamanho
        ax.annotate("", xy=fim, xytext=inicio,
                    arrowprops=dict(arrowstyle="-|>", color=cor, lw=1.6,
                                    mutation_scale=14),
                    zorder=5)


# ---------------- 6. Ícones (PNG) ----------------
def desenhar_casa(ax, x, y, imagem):
    """
    Coloca a imagem da casa no ponto. Um círculo branco fica atrás da casa
    e muda para verde quando a encomenda é entregue.
    Retorna o círculo, para que a cor possa ser trocada na animação.
    """
    fundo = Circle((x, y), 0.42, facecolor="#ffffff", edgecolor="#cccccc", zorder=4)
    ax.add_patch(fundo)
    caixa = AnnotationBbox(OffsetImage(imagem, zoom=ZOOM_CASA), (x, y),
                           frameon=False, zorder=5)
    ax.add_artist(caixa)
    return fundo


def criar_moto(ax, imagem):
    """
    Cria UMA vez o entregador do painel. Nos quadros seguintes, a animação
    só altera a posição (moto.xy) e, se preciso, a imagem espelhada.
    """
    caixa = AnnotationBbox(OffsetImage(imagem, zoom=ZOOM_MOTO), (0, 0),
                           frameon=False, zorder=10)
    ax.add_artist(caixa)
    return caixa


# ---------------- 7. Simulação do entregador ----------------
def simular_rota(G, ordem, T, pos):
    """
    Percorre a rota trecho a trecho, aresta a aresta, e registra em cada quadro
    a posição da moto, o combustível acumulado e as entregas já feitas.
    Retorna (quadros, litros_por_entrega, litros_total).
    """
    seq = [CD, *ordem, CD]
    # Cada quadro: (x, y, litros_gastos, entregas_feitas, parou_sem_combustivel)
    quadros = []
    entregues = []
    litros = 0.0
    litros_por_entrega = {}

    x, y = pos[CD]
    quadros.append((x, y, litros, tuple(entregues), False))
    parou = False

    for a, b in zip(seq, seq[1:]):
        if parou:
            break
        caminho = T[a, b][1]
        for u, v in zip(caminho, caminho[1:]):
            p1, p2 = np.array(pos[u]), np.array(pos[v])
            gasto_aresta = G[u][v]["weight"] * LITROS_POR_UNIDADE
            combustivel_restante = TANQUE_L - litros
            if gasto_aresta > combustivel_restante + 1e-9:
                # Para no ponto da aresta em que o tanque chega a zero.
                fracao = max(0.0, combustivel_restante / gasto_aresta)
                for passo in range(1, PASSOS_POR_ARESTA + 1):
                    t = fracao * passo / PASSOS_POR_ARESTA
                    x, y = p1 + (p2 - p1) * t
                    quadros.append((x, y, TANQUE_L, tuple(entregues), True))
                litros = TANQUE_L
                parou = True
                break
            for passo in range(1, PASSOS_POR_ARESTA + 1):
                t = passo / PASSOS_POR_ARESTA
                x, y = p1 + (p2 - p1) * t
                quadros.append((x, y, litros + gasto_aresta * t, tuple(entregues), False))
            litros += gasto_aresta

        if parou:
            break

        if b != CD:                       # chegou a uma casa: entrega e pausa
            entregues.append(b)
            litros_por_entrega[b] = litros
            for _ in range(PAUSA_NA_ENTREGA):
                quadros.append((x, y, litros, tuple(entregues), False))

    return quadros, litros_por_entrega, litros, parou


# ---------------- 8. Painel de cada rota ----------------
def preparar_painel(ax, G, pos, titulo, cor, ordem, T, imagem_casa):
    """Desenha o cenário fixo de um painel: grade, casas, CD e trajeto de referência."""
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#dddddd", width=1.5)

    seq = [CD, *ordem, CD]
    for a, b in zip(seq, seq[1:]):           # trajeto de referência, mais fraco
        caminho = T[a, b][1]
        ax.plot([pos[no][0] for no in caminho], [pos[no][1] for no in caminho],
                color=cor, lw=1.5, alpha=0.25, zorder=3)
        desenhar_setas(ax, pos, caminho, cor)

    # Pontos que não são entrega
    for no, (x, y) in pos.items():
        if no not in ordem and no != CD:
            ax.scatter(x, y, s=120, color="#eeeeee", edgecolors="#bbbbbb", zorder=2)
            ax.text(x, y, no, ha="center", va="center", fontsize=6, color="#888888")

    # Casas de entrega, com a ordem de visita abaixo
    casas = {}
    for k, no in enumerate(ordem, 1):
        x, y = pos[no]
        casas[no] = desenhar_casa(ax, x, y, imagem_casa)
        ax.text(x, y - 0.55, f"{no} ({k}ª)", ha="center", va="center",
                fontsize=7, fontweight="bold", zorder=6)

    # Centro de distribuição
    x, y = pos[CD]
    ax.scatter(x, y, s=500, marker="s", color="#2ca02c", edgecolors="black", zorder=5)
    ax.text(x, y, "CD", ha="center", va="center", fontsize=7,
            color="white", fontweight="bold", zorder=6)

    # Limites fixos: evita que o enquadramento mude a cada quadro
    xs, ys = zip(*pos.values())
    ax.set_xlim(min(xs) - 0.9, max(xs) + 0.9)
    ax.set_ylim(min(ys) - 0.9, max(ys) + 0.9)

    ax.set_title(f"{titulo}\nCusto da rota: {custo_rota(ordem, T):.2f}",
                 fontsize=11, fontweight="bold")
    ax.set_aspect("equal")
    ax.axis("off")
    return casas


# ---------------- 9. Programa principal ----------------
def main():
    imagem_casa = carregar_imagem(ARQ_CASA)
    imagem_moto = carregar_imagem(ARQ_MOTO)
    imagem_moto_esq = np.fliplr(imagem_moto).copy()   # espelhada uma única vez

    G, pos = criar_grafo()
    T = calcular_trechos(G, pos, [CD, *ENTREGAS])

    ordem_viz = vizinho_mais_proximo(T)
    ordem_ag = algoritmo_genetico(T, ordem_viz)

    cenarios = [
        ("Vizinho mais próximo", ordem_viz, "#1f77b4"),
        ("Algoritmo Genético", ordem_ag, "#e31a1c"),
    ]

    # Simula as duas rotas e imprime o consumo
    print("=" * 72)
    print("SIMULAÇÃO DE ENTREGAS: consumo de combustível")
    print(f"Tanque: {TANQUE_L:.1f} L | {LITROS_POR_UNIDADE} L por unidade de custo")
    print("=" * 72)

    simulacoes = []
    totais = {}
    for titulo, ordem, cor in cenarios:
        quadros, por_entrega, litros_total, parou = simular_rota(G, ordem, T, pos)
        simulacoes.append((titulo, ordem, cor, quadros))
        totais[titulo] = litros_total

        print(f"\n{titulo}")
        print(f"  Trilha: {' → '.join([CD, *ordem, CD])}")
        for k, no in enumerate(ordem, 1):
            print(f"  {k}ª entrega em {no:<4} combustível acumulado: {por_entrega[no]:6.2f} L")
        if parou:
            print(f"  PARADA no caminho: tanque vazio após {len(quadros[-1][3])} entrega(s).")
            print(f"  Combustível gasto: {litros_total:.2f} L | restante: 0.00 L")
        else:
            print(f"  Retorno ao CD concluído | gasto: {litros_total:.2f} L | "
                  f"restante: {max(0.0, TANQUE_L - litros_total):.2f} L")

    diferenca = totais["Algoritmo Genético"] - totais["Vizinho mais próximo"]
    print("\n" + "=" * 72)
    print(f"Diferença de consumo (AG − vizinho mais próximo): {diferenca:+.2f} L")
    print("=" * 72)

    # Figura com um painel por rota
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    paineis = []
    proximo_inicio = 0
    for ax, (titulo, ordem, cor, quadros) in zip(axes, simulacoes):
        casas = preparar_painel(ax, G, pos, titulo, cor, ordem, T, imagem_casa)

        # Objetos dinâmicos criados uma única vez por painel
        trilha, = ax.plot([], [], color=cor, lw=3.5, alpha=0.8, zorder=4)
        moto = criar_moto(ax, imagem_moto)
        info = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top",
                       fontsize=9, family="monospace",
                       bbox=dict(boxstyle="round", facecolor="white", edgecolor="#999999"),
                       zorder=20)

        # Medidor de tanque fixo no canto superior direito do painel.
        tanque_fundo = Rectangle((0.878, 0.742), 0.064, 0.18,
                                 transform=ax.transAxes, facecolor="white",
                                 edgecolor="none", zorder=20.5)
        ax.add_patch(tanque_fundo)
        tanque_fill = Rectangle((0.884, 0.748), 0.052, 0.0,
                                transform=ax.transAxes, facecolor="#2ca02c",
                                edgecolor="none", zorder=21)
        ax.add_patch(tanque_fill)
        tanque_contorno = Rectangle((0.878, 0.742), 0.064, 0.18,
                                    transform=ax.transAxes, facecolor="none",
                                    edgecolor="#333333", linewidth=1.5, zorder=22)
        ax.add_patch(tanque_contorno)
        tanque_bocal = Rectangle((0.895, 0.922), 0.03, 0.025,
                                 transform=ax.transAxes, facecolor="#555555",
                                 edgecolor="#333333", linewidth=0.8, zorder=22)
        ax.add_patch(tanque_bocal)
        tanque_titulo = ax.text(0.91, 0.972, "TANQUE", transform=ax.transAxes,
                                ha="center", va="top", fontsize=7,
                                fontweight="bold", zorder=23)
        tanque_valor = ax.text(0.91, 0.716, f"{TANQUE_L:.1f} L",
                               transform=ax.transAxes, ha="center", va="top",
                               fontsize=8, family="monospace", zorder=23)

        paineis.append({
            "ax": ax, "cor": cor, "ordem": ordem, "quadros": quadros,
            "casas": casas, "info": info, "trilha": trilha, "moto": moto,
            "tanque_fill": tanque_fill, "tanque_valor": tanque_valor,
            "xs": [q[0] for q in quadros], "ys": [q[1] for q in quadros],
            "virado_esquerda": False,
            "inicio": proximo_inicio,
        })
        proximo_inicio += len(quadros)

    fig.suptitle(f"Entregas em sequência: Vizinho mais próximo e depois AG (tanque {TANQUE_L:.1f} L)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    total_quadros = sum(len(p["quadros"]) for p in paineis)

    def atualizar(f):
        for p in paineis:
            quadros = p["quadros"]
            aguardando = f < p["inicio"]
            i = 0 if aguardando else min(f - p["inicio"], len(quadros) - 1)
            x, y, litros, entregues, parou = quadros[i]

            # Direção do movimento: mantém a última direção quando a moto está parada
            if i > 0:
                dx = x - quadros[i - 1][0]
                if abs(dx) > 1e-9:
                    p["virado_esquerda"] = dx < 0

            # Trilha já percorrida: só troca os dados da linha existente
            p["trilha"].set_data(p["xs"][:i + 1], p["ys"][:i + 1])

            # Círculo atrás da casa fica verde quando a encomenda já foi entregue
            for no, fundo in p["casas"].items():
                fundo.set_facecolor("#9be59b" if no in entregues else "#ffffff")

            # Entregador: move a caixa existente e troca a imagem se mudou de sentido
            p["moto"].xy = (x, y)
            p["moto"].offsetbox.set_data(imagem_moto_esq if p["virado_esquerda"]
                                         else imagem_moto)

            restante = max(0.0, TANQUE_L - litros)
            fracao_tanque = restante / TANQUE_L if TANQUE_L > 0 else 0.0
            p["tanque_fill"].set_height(0.168 * fracao_tanque)
            p["tanque_fill"].set_facecolor(
                "#2ca02c" if fracao_tanque > 0.5 else
                "#ff9900" if fracao_tanque > 0.15 else "#d62728"
            )
            p["tanque_valor"].set_text(f"{restante:.1f} L")

            terminou = not aguardando and i == len(quadros) - 1
            chegou_cd = terminou and not parou and np.allclose(
                (x, y), pos[CD], atol=1e-9
            )
            if not terminou:
                resultado = "Resultado: em andamento"
            elif chegou_cd:
                resultado = "Resultado: V — chegou ao CD"
            else:
                resultado = "Resultado: X — não chegou ao CD"

            p["info"].set_text(
                f"Combustível gasto: {litros:6.2f} L\n"
                f"Restante no tanque: {restante:6.2f} L\n"
                f"Entregas feitas: {len(entregues)}/{len(p['ordem'])}\n"
                f"Status: {'Aguardando vez' if aguardando else ('PARADO — sem combustível' if parou else 'Em rota')}\n"
                f"{resultado}"
            )

    # Quadro final salvo como imagem estática
    atualizar(total_quadros - 1)
    fig.savefig(os.path.join(PASTA_IMAGENS, "simulacao_final.png"),
                dpi=120, bbox_inches="tight")

    anim = animation.FuncAnimation(fig, atualizar, frames=total_quadros,
                                   interval=70, repeat=False)
    anim.save(os.path.join(PASTA_IMAGENS, "simulacao_entregas.gif"),
              writer="pillow", fps=14, dpi=DPI_GIF)
    plt.show()

    # No Colab, mostra o GIF animado dentro do notebook
    try:
        from IPython.display import Image as ImagemIPython, display
        display(ImagemIPython(filename=os.path.join(
            PASTA_IMAGENS, "simulacao_entregas.gif")))
    except ImportError:
        pass


if __name__ == "__main__":
    main()
