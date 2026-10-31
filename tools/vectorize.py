"""Vetorização offline do canvas de tinta: raster -> polilinhas -> SVG.

Ferramenta offline, sem câmera: recebe o PNG de um canvas acumulado
(850x1200, espaço do papel) e devolve as polilinhas do desenho.

Pipeline:
  1. binarização (a tinta é a classe minoritária do PNG; nos canvas
     do projeto ela é escura sobre fundo claro);
  2. afinamento com cv2.ximgproc.thinning (Zhang-Suen por padrão;
     o OpenCV do laboratório é o 5.0.0 e tem ximgproc);
  3. travessia do esqueleto como grafo: nós são extremidades
     (grau 1), junções (grau >= 3) e pixels isolados (grau 0),
     com pixels de nó vizinhos contraídos num nó só; arestas são os
     caminhos de pixels de grau 2 entre nós; componentes sem nó
     (laços puros) viram polilinhas fechadas;
  4. simplificação de cada polilinha por Douglas-Peucker, com
     epsilon parametrizável em pixels (padrão 1,5 px);
  5. saída SVG com viewBox em milímetros do papel A4.

Escala: a conversão física é compartilhada com ``draw.export`` e usa
uma escala uniforme limitante, para preservar proporções sem exceder
o papel nominal. No A4 padrão, 850 x 1200 px usam 0,247059 mm/px e
produzem uma folha efetiva de 210,00 x 296,47 mm.

Uso:
  python tools/vectorize.py canvas.png [--out saida.svg]
      [--epsilon 1.5] [--thinning zhangsuen|guohall] [--stats]
  python tools/vectorize.py --selfcheck
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from draw import export as stroke_export

SHEET_SCALE = stroke_export.sheet_scale(paper="A4")
PAPER_W_PX = SHEET_SCALE.canvas_width_px
PAPER_H_PX = SHEET_SCALE.canvas_height_px
PAPER_W_MM = SHEET_SCALE.effective_width_mm
PAPER_H_MM = SHEET_SCALE.effective_height_mm
MM_PER_PX = SHEET_SCALE.mm_per_px

THINNING_TYPES = {
    "zhangsuen": cv2.ximgproc.THINNING_ZHANGSUEN,
    "guohall": cv2.ximgproc.THINNING_GUOHALL,
}

_NEIGHBORS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


def load_ink(path):
    """Lê um PNG de canvas e devolve (mascara booleana de tinta, polaridade).

    A tinta é a classe minoritária após limiar no cinza médio: nos
    canvas do projeto ela é escura ("escura") sobre fundo claro.
    """
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"nao foi possivel ler {path}")
    dark = img < 128
    if dark.sum() <= (~dark).sum():
        return dark, "escura"
    return ~dark, "clara"


def thin(ink, thinning="zhangsuen"):
    """Afina a máscara de tinta para um esqueleto de 1 px (booleano)."""
    src = (ink.astype(np.uint8)) * 255
    skel = cv2.ximgproc.thinning(src, None, THINNING_TYPES[thinning])
    return skel > 0


def _degree(skel):
    """Grau de cada pixel do esqueleto (vizinhança de 8)."""
    s = skel.astype(np.int32)
    padded = np.pad(s, 1)
    deg = np.zeros_like(s)
    for dy, dx in _NEIGHBORS:
        deg += padded[1 + dy:1 + dy + s.shape[0], 1 + dx:1 + dx + s.shape[1]]
    return deg * s


def _skeleton_neighbors(flat_idx, skel_flat, width, height):
    """Índices planos dos vizinhos de 8 que também são esqueleto."""
    y, x = divmod(flat_idx, width)
    out = []
    for dy, dx in _NEIGHBORS:
        ny, nx = y + dy, x + dx
        if 0 <= ny < height and 0 <= nx < width:
            nidx = ny * width + nx
            if skel_flat[nidx]:
                out.append(nidx)
    return out


def traverse_graph(skel):
    """Percorre o esqueleto como grafo e devolve as polilinhas brutas.

    Cada polilinha é um dict {"points": array Nx2 (x, y) em px,
    "closed": bool}. Nós do grafo: componentes conexas de pixels
    com grau != 2 (extremidades, junções contraídas e isolados).
    A âncora de um nó é o centróide dos seus pixels.
    """
    height, width = skel.shape
    deg = _degree(skel)
    node_mask = skel & (deg != 2)
    n_nodes, labels = cv2.connectedComponents(
        node_mask.astype(np.uint8), connectivity=8
    )
    labels_flat = labels.ravel()
    skel_flat = skel.ravel()

    anchors = {}
    node_pixels = {}
    for nid in range(1, n_nodes):
        ys, xs = np.nonzero(labels == nid)
        anchors[nid] = (float(xs.mean()), float(ys.mean()))
        node_pixels[nid] = [int(y) * width + int(x) for y, x in zip(ys, xs)]

    polylines = []
    visited = set()  # pares (a, b) de índices planos já percorridos

    def walk_edge(start_pixel, first_step, start_anchor):
        """Caminha de um nó por pixels de grau 2 até outro nó."""
        path = [first_step]
        visited.add((start_pixel, first_step))
        visited.add((first_step, start_pixel))
        prev, cur = start_pixel, first_step
        end_anchor = None
        while labels_flat[cur] == 0:
            nbrs = [
                n for n in _skeleton_neighbors(cur, skel_flat, width, height)
                if n != prev
            ]
            if len(nbrs) != 1:
                # Pixel regular com vizinhança inesperada (bloco 2x2
                # residual): encerra a aresta neste pixel.
                end_anchor = (float(cur % width), float(cur // width))
                break
            nxt = nbrs[0]
            visited.add((cur, nxt))
            visited.add((nxt, cur))
            path.append(nxt)
            prev, cur = cur, nxt
        if end_anchor is None:
            end_anchor = anchors[labels_flat[cur]]
        points = [start_anchor]
        points += [(float(p % width), float(p // width)) for p in path]
        points.append(end_anchor)
        closed = points[0] == points[-1] and len(path) >= 2
        return {"points": np.array(points, dtype=np.float64), "closed": closed}

    # Arestas a partir de cada nó.
    for nid in range(1, n_nodes):
        anchor = anchors[nid]
        has_external = False
        for p in node_pixels[nid]:
            for q in _skeleton_neighbors(p, skel_flat, width, height):
                if labels_flat[q] == nid:
                    continue
                has_external = True
                if (p, q) in visited:
                    continue
                polylines.append(walk_edge(p, q, anchor))
        if not has_external:
            # Componente sem nenhum vizinho fora de si: pixel isolado
            # (grau 0), uma polilinha de um ponto só. A condição é a
            # ausência de vizinhos externos, não "nenhuma aresta saiu
            # daqui": uma aresta pode ter sido emitida a partir do nó
            # da outra ponta, e nesse caso este nó não é isolado.
            polylines.append(
                {"points": np.array([anchor], dtype=np.float64),
                 "closed": False}
            )

    # Componentes sem nó: laços puros (todos os pixels com grau 2) e
    # qualquer resto não visitado.
    covered = np.zeros(skel_flat.shape, dtype=bool)
    for a, b in visited:
        covered[a] = True
        covered[b] = True
    remaining = np.nonzero(skel_flat & ~covered & (labels_flat == 0))[0]
    seen = set()
    for start in remaining:
        start = int(start)
        if start in seen:
            continue
        path = [start]
        seen.add(start)
        prev, cur = None, start
        while True:
            nbrs = [
                n for n in _skeleton_neighbors(cur, skel_flat, width, height)
                if n != prev
            ]
            if not nbrs:
                break
            nxt = nbrs[0]
            if nxt == start and len(path) >= 3:
                break
            if nxt in seen:
                break
            path.append(nxt)
            seen.add(nxt)
            prev, cur = cur, nxt
        points = [(float(p % width), float(p // width)) for p in path]
        closed = len(path) >= 3
        polylines.append(
            {"points": np.array(points, dtype=np.float64), "closed": closed}
        )

    return polylines


def _dp_open(points, epsilon):
    """Douglas-Peucker iterativo sobre uma polilinha aberta."""
    n = len(points)
    if n <= 2:
        return points
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        pa, pb = points[a], points[b]
        ab = pb - pa
        norm = np.hypot(ab[0], ab[1])
        seg = points[a + 1:b]
        if norm == 0:
            dists = np.hypot(seg[:, 0] - pa[0], seg[:, 1] - pa[1])
        else:
            cross = np.abs(
                ab[0] * (pa[1] - seg[:, 1]) - (pa[0] - seg[:, 0]) * ab[1]
            )
            dists = cross / norm
        i = int(np.argmax(dists))
        if dists[i] > epsilon:
            keep[a + 1 + i] = True
            stack.append((a, a + 1 + i))
            stack.append((a + 1 + i, b))
    return points[keep]


def simplify(points, epsilon, closed):
    """Douglas-Peucker; em laço fechado, divide em dois arcos pelas
    duas âncoras mais distantes entre si."""
    if len(points) <= 2 or epsilon <= 0:
        return points
    if not closed:
        return _dp_open(points, epsilon)
    # Âncoras: o ponto mais longe do centróide e o mais longe dele.
    centroid = points.mean(axis=0)
    i_a = int(np.argmax(np.hypot(*(points - centroid).T)))
    i_b = int(np.argmax(np.hypot(*(points - points[i_a]).T)))
    if i_a == i_b:
        return _dp_open(points, epsilon)
    lo, hi = min(i_a, i_b), max(i_a, i_b)
    arc1 = points[lo:hi + 1]
    arc2 = np.vstack([points[hi:], points[:lo + 1]])
    s1 = _dp_open(arc1, epsilon)
    s2 = _dp_open(arc2, epsilon)
    return np.vstack([s1, s2[1:]])


def vectorize_mask(ink, epsilon=1.5, thinning="zhangsuen"):
    """Máscara de tinta -> polilinhas simplificadas (px, espaço do papel)."""
    skel = thin(ink, thinning)
    raw = traverse_graph(skel)
    out = []
    for line in raw:
        pts = simplify(line["points"], epsilon, line["closed"])
        out.append({"points": pts, "closed": line["closed"]})
    return out


def dense_samples(points, closed, step=1.0):
    """Amostra pontos ao longo da polilinha com espaçamento <= step px."""
    if len(points) == 1:
        return points.copy()
    pts = list(points)
    if closed:
        pts = pts + [pts[0]]
    samples = []
    for a, b in zip(pts[:-1], pts[1:]):
        length = float(np.hypot(*(np.array(b) - np.array(a))))
        n = max(1, int(np.ceil(length / step)))
        for i in range(n):
            t = i / n
            samples.append(np.array(a) * (1 - t) + np.array(b) * t)
    samples.append(np.array(pts[-1], dtype=np.float64))
    return np.array(samples)


def distance_stats(ink, polylines):
    """Distâncias polilinha<->tinta nos dois sentidos, em px e mm.

    Sentido 1 (polilinha -> tinta): amostras densas (<= 1 px) das
    polilinhas simplificadas contra o mapa de distância à tinta.
    Sentido 2 (tinta -> polilinha): todos os pixels de tinta contra
    o mapa de distância às polilinhas rasterizadas (linhas de 1 px
    sobre os vértices simplificados). A rasterização quantiza o
    sentido 2 na grade de pixels: é o proxy honesto da saída vetorial.
    """
    dt_to_ink = cv2.distanceTransform(
        (~ink).astype(np.uint8), cv2.DIST_L2, 5
    )
    samples = []
    for line in polylines:
        samples.append(dense_samples(line["points"], line["closed"]))
    samples = np.vstack(samples) if samples else np.zeros((0, 2))
    h, w = ink.shape
    sx = np.clip(np.round(samples[:, 0]).astype(int), 0, w - 1)
    sy = np.clip(np.round(samples[:, 1]).astype(int), 0, h - 1)
    d_fwd = dt_to_ink[sy, sx]

    raster = np.zeros(ink.shape, dtype=np.uint8)
    for line in polylines:
        pts = np.round(line["points"]).astype(np.int32).reshape(-1, 1, 2)
        if len(pts) == 1:
            raster[pts[0, 0, 1], pts[0, 0, 0]] = 1
        else:
            cv2.polylines(
                raster, [pts], bool(line["closed"]), 1, thickness=1,
                lineType=cv2.LINE_8,
            )
    dt_to_poly = cv2.distanceTransform(
        (raster == 0).astype(np.uint8), cv2.DIST_L2, 5
    )
    d_bwd = dt_to_poly[ink]

    def summary(d):
        if len(d) == 0:
            return {"n": 0}
        return {
            "n": int(len(d)),
            "media_px": float(d.mean()),
            "mediana_px": float(np.median(d)),
            "p95_px": float(np.percentile(d, 95)),
            "max_px": float(d.max()),
            "media_mm": float(d.mean() * MM_PER_PX),
        }

    return {
        "polilinha_para_tinta": summary(d_fwd),
        "tinta_para_polilinha": summary(d_bwd),
    }


def write_svg(polylines, out_path, epsilon, thinning):
    """Escreve SVG usando a escala uniforme canônica de ``draw.export``."""
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f"<!-- Draw vectorize: epsilon={epsilon} px, thinning={thinning}, "
        f"paper=A4, scale={MM_PER_PX:.6f} mm/px, "
        f"effective={PAPER_W_MM:.2f}x{PAPER_H_MM:.2f} mm -->",
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{PAPER_W_MM:.2f}mm" height="{PAPER_H_MM:.2f}mm" '
        f'viewBox="0 0 {PAPER_W_MM:.2f} {PAPER_H_MM:.2f}">',
    ]
    for line in polylines:
        pts = line["points"]
        if len(pts) == 1:
            cx, cy = pts[0][0] * MM_PER_PX, pts[0][1] * MM_PER_PX
            lines.append(
                f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="0.30" fill="#000"/>'
            )
            continue
        coords = " ".join(
            f"{x * MM_PER_PX:.2f},{y * MM_PER_PX:.2f}" for x, y in pts
        )
        tag = "polygon" if line["closed"] else "polyline"
        lines.append(
            f'<{tag} points="{coords}" fill="none" stroke="#000" '
            f'stroke-width="0.30"/>'
        )
    lines.append("</svg>")
    Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def describe(ink, polylines, stats):
    """Resumo textual de uma vetorização."""
    n_points = sum(1 for line in polylines if len(line["points"]) == 1)
    n_closed = sum(1 for line in polylines if line["closed"])
    return {
        "pixels_tinta": int(ink.sum()),
        "polilinhas": len(polylines),
        "polilinhas_fechadas": n_closed,
        "polilinhas_de_um_ponto": n_points,
        "saltos_de_caneta": max(0, len(polylines) - 1),
        "distancias": stats,
    }


def _selfcheck():
    """Formas sintéticas conhecidas: retângulo, círculo e X com junção.

    As formas são geradas com traço de 1 px, para a contagem ser
    determinística: o que se confere é a travessia do grafo, a
    simplificação e a distância contra a forma geradora analítica
    (não contra o blob de tinta). Traço de 1 px também é o regime
    dos traços finos do canvas real (1 a 3 px).

    O retângulo é um quadrado rotacionado de 45 graus, desenhado
    com 4 linhas: medido nesta bancada, o retângulo alinhado aos
    eixos com 1 px tem, em conectividade 8, atalhos diagonais nos
    cantos (o pixel vizinho ao canto toca os dois lados), o que
    parte o laço em 8 pedaços; e com traço de 3 px o eixo medial
    cria esporões para os cantos convexos (41 polilinhas nas 3
    formas). Os dois comportamentos são do esqueleto de pixel, não
    da travessia, e o segundo é medido nos canvas reais como
    resíduo. Aqui, com topologia inequívoca, o esperado é exato:
    1 fechada (retângulo) + 1 fechada (círculo) + 4 abertas (os
    braços do X a partir da junção) = 6.
    """
    rect = [(250, 70), (430, 250), (250, 430), (70, 250)]
    img = np.full((PAPER_H_PX, PAPER_W_PX), 255, dtype=np.uint8)
    for a, b in zip(rect, rect[1:] + rect[:1]):
        cv2.line(img, a, b, 0, thickness=1)
    cv2.circle(img, (620, 250), 110, 0, thickness=1)
    cv2.line(img, (150, 700), (450, 1000), 0, thickness=1)
    cv2.line(img, (150, 1000), (450, 700), 0, thickness=1)
    ink = img < 128

    polylines = vectorize_mask(ink, epsilon=1.5)
    n_closed = sum(1 for line in polylines if line["closed"])
    n_open = len(polylines) - n_closed
    print(f"polilinhas: {len(polylines)} (fechadas {n_closed}, "
          f"abertas {n_open}); esperado 6 (2 fechadas, 4 abertas)")
    assert len(polylines) == 6, f"contagem inesperada: {len(polylines)}"
    assert n_closed == 2, f"fechadas inesperadas: {n_closed}"

    def seg_dist(p, a, b):
        a, b = np.array(a, float), np.array(b, float)
        pnt = np.array(p, float)
        ab = b - a
        t = np.clip(np.dot(pnt - a, ab) / np.dot(ab, ab), 0, 1)
        return float(np.hypot(*(pnt - (a + t * ab))))

    def dist_to_shapes(p):
        d_rect = min(
            seg_dist(p, a, b) for a, b in zip(rect, rect[1:] + rect[:1])
        )
        # Círculo: |distância ao centro - raio|.
        d_circ = abs(float(np.hypot(p[0] - 620, p[1] - 250)) - 110.0)
        # X: distância aos dois segmentos.
        d_x = min(seg_dist(p, (150, 700), (450, 1000)),
                  seg_dist(p, (150, 1000), (450, 700)))
        return min(d_rect, d_circ, d_x)

    samples = np.vstack([
        dense_samples(line["points"], line["closed"]) for line in polylines
    ])
    dists = np.array([dist_to_shapes(p) for p in samples])
    print(f"erro contra a forma geradora: n={len(dists)} amostras, "
          f"média {dists.mean():.3f} px ({dists.mean() * MM_PER_PX:.3f} mm), "
          f"p95 {np.percentile(dists, 95):.3f} px, "
          f"máx {dists.max():.3f} px")
    assert dists.mean() <= 1.5, "erro médio acima do esperado"
    stats = distance_stats(ink, polylines)
    print("distâncias raster:", json.dumps(stats, indent=1))
    print("SELFCHECK OK")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Vetoriza um canvas de tinta (PNG) para polilinhas/SVG."
    )
    parser.add_argument("input", nargs="?", help="PNG do canvas de tinta")
    parser.add_argument("--out", default=None, help="SVG de saída")
    parser.add_argument("--epsilon", type=float, default=1.5,
                        help="epsilon do Douglas-Peucker em px (padrao 1,5)")
    parser.add_argument("--thinning", choices=list(THINNING_TYPES),
                        default="zhangsuen",
                        help="algoritmo de afinamento (padrao zhangsuen)")
    parser.add_argument("--stats", action="store_true",
                        help="imprime as distâncias nos dois sentidos")
    parser.add_argument("--selfcheck", action="store_true",
                        help="roda a bancada sintética e sai")
    args = parser.parse_args(argv)

    if args.selfcheck:
        _selfcheck()
        return 0
    if not args.input:
        parser.error("informe o PNG de entrada ou --selfcheck")

    ink, polarity = load_ink(args.input)
    polylines = vectorize_mask(ink, args.epsilon, args.thinning)
    out = args.out or str(Path(args.input).with_suffix(".svg"))
    write_svg(polylines, out, args.epsilon, args.thinning)
    stats = distance_stats(ink, polylines)
    summary = describe(ink, polylines, stats)
    summary["polaridade_tinta"] = polarity
    summary["epsilon_px"] = args.epsilon
    summary["thinning"] = args.thinning
    summary["svg"] = out
    if args.stats:
        print(json.dumps(summary, indent=1, ensure_ascii=False))
    else:
        print(f"{args.input}: {summary['polilinhas']} polilinhas "
              f"({summary['polilinhas_fechadas']} fechadas), "
              f"{summary['saltos_de_caneta']} saltos de caneta, "
              f"dist. média polilinha->tinta "
              f"{stats['polilinha_para_tinta'].get('media_px', float('nan')):.3f} px, "
              f"tinta->polilinha "
              f"{stats['tinta_para_polilinha'].get('media_px', float('nan')):.3f} px "
              f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
