# Treino completo — SegFormer-B0, 3.000 cenas (plano de execução)

Plano preparado em 2026-10-29, branch `feature/main-integracao` (merge
`1dea1c9`, que já contém o pipeline neural). **Nada deste plano foi
executado**: nenhum kernel subido, nenhum treino rodado, nenhum
commit/push feito por este preparo. O treino só deve ser lançado depois
que a GPU for confirmada na interface web do Kaggle (ver "O que falta
do usuário").

## 1. Decisão: pesos de perda

**Pesos recomendados para o treino completo: `[1.0, 1.5, 10.0, 10.0]`**
(classes `[background, paper, hand, ink]`).

### Como foram calculados (só o split de treino, nunca o de teste)

Contagem de pixels nas 2.400 máscaras de treino de
`~/workspace/draw-data/synthetic-v1/train/` (512×512, valores só 0–3,
verificado máscara por máscara), recontada em 2026-10-29:

| Classe | Pixels (train) | Fração |
| --- | --- | --- |
| 0 background | 366.713.811 | 58,288% |
| 1 paper | 245.048.872 | 38,949% |
| 2 hand | 9.488.543 | 1,508% |
| 3 ink | 7.894.374 | 1,255% |
| Total | 629.145.600 | 100% |

Regra do `training/train.py` (`compute_class_weights`, teto 10):
inverso da frequência, normalizado pelo menor peso, com teto 10.
Aplicada a esses números dá exatamente `[1.0, 1.4965, 10.0, 10.0]`
(hand e ink batem no teto; sem teto seriam ~38,6 e ~46,5 na escala
background=1). Arredondado: **`[1.0, 1.5, 10.0, 10.0]`**.

### Por que trocar a config atual

A config atual (`training/config.json`) tem `[1.0, 2.0, 3.0, 10.0]`,
calibrada no dataset de teste de 36 cenas, onde a mão pesava pouco.
No dataset real de 3.000 cenas a mão é só 1,5% dos pixels — quase tão
rara quanto a tinta — e o peso 3,0 a subpondera. O `train.py` **usa
os pesos da config** (`class_weights = tensor(cfg["loss"]...)`);
`compute_class_weights` só imprime os calculados como referência e
não os aplica. Por isso a troca tem de entrar na config, não basta
olhar o log.

### Onde aplicar (passo pendente, não feito neste preparo)

1. `training/config.json` → `loss.cross_entropy_class_weights`:
   `[1.0, 1.5, 10.0, 10.0]`, com a justificativa acima registrada
   (esta seção é o registro).
2. Commit + push na branch de trabalho e **nova versão do dataset
   `guzzfort/draw-training-code`** no Kaggle com o `config.json`
   novo — o kernel lê a config do dataset de código, não do repo.
   Verificar baixando de volta (MD5/diff) antes de lançar o treino.
3. **Nunca recalibrar olhando o split de teste.** As frações de
   val/test foram conferidas só por sanidade (val: 57,7/39,5/1,5/1,3%;
   test: 58,3/38,9/1,6/1,3%) e não entraram no cálculo.

## 2. Inputs do kernel (todos privados, exceto FreiHAND)

| Input | Slug / caminho no kernel |
| --- | --- |
| Dataset sintético, parte 1 (train 0000–0399) | `guzzfort/draw-synthetic-v1-part1` |
| Parte 2 (train 0400–0799) | `guzzfort/draw-synthetic-v1-part2` |
| Parte 3 (train 0800–1199) | `guzzfort/draw-synthetic-v1-part3` |
| Parte 4 (train 1200–1599) | `guzzfort/draw-synthetic-v1-part4` |
| Parte 5 (train 1600–1999) | `guzzfort/draw-synthetic-v1-part5` |
| Parte 6 (train 2000–2399) | `guzzfort/draw-synthetic-v1-part6` |
| Parte 7 (val 0000–0299) | `guzzfort/draw-synthetic-v1-part7` |
| Parte 8 (test 0000–0299) | `guzzfort/draw-synthetic-v1-part8` |
| Código de treino (**versão com a config nova**, ver §1) | `guzzfort/draw-training-code` |
| Mãos reais (só checagem de formato, não treina aqui) | `danieldelro/freihand` |
| Cache de pesos mit-b0, **v2** (com `refs/main` sem newline) | `guzzfort/mit-b0-cache` |

Detalhe das partes e procedimento de junção: ver
`~/workspace/draw-data/synthetic-v1-upload/PARTS.md`. O dataset único
`guzzfort/draw-synthetic-v1` **não deve ser usado**: o upload único de
1,6 GB falhou (resets do proxy) e nunca criou dataset no servidor;
as 8 partes são a fonte canônica, todas com status `ready`.

Cada parte aparece no kernel como
`.../dataset/synthetic/<split>/{images,masks,ink_full}/` — o caminho
base varia entre `/kaggle/input/draw-synthetic-v1-part<N>` e
`/kaggle/input/datasets/guzzfort/draw-synthetic-v1-part<N>`, por isso
a célula de junção abaixo procura nos dois layouts. Só a parte 1 traz
`split_manifest.json` (na raiz de `dataset/synthetic/`).

## 3. Célula de junção (rodar antes da célula de treino)

Copia as 8 partes para um único dataset em `/kaggle/working` (cópia,
não symlink; custa ~1,6 GB de disco do kernel, aceitável) e trava o
treino se as contagens não fecharem. **Importante:** depois da junção,
`DATA` tem de apontar para o diretório reunido — a célula de dataset
do notebook autodetecta o primeiro `dataset/synthetic` que encontrar
nos inputs, o que seria só a parte 1 (400 imagens de treino). A última
linha abaixo corrige isso.

```python
import shutil
from pathlib import Path

out = Path("/kaggle/working/dataset/synthetic")
found = []
for n in range(1, 9):
    cands = [
        Path(f"/kaggle/input/draw-synthetic-v1-part{n}/dataset/synthetic"),
        Path(f"/kaggle/input/datasets/guzzfort/draw-synthetic-v1-part{n}/dataset/synthetic"),
    ]
    src = next((c for c in cands if c.is_dir()), None)
    assert src is not None, f"input ausente para a parte {n}: {cands}"
    found.append(src)
    for f in sorted(src.rglob("*")):
        if f.is_file():
            dst = out / f.relative_to(src)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(f, dst)
print("partes encontradas:", len(found))

# Conferência obrigatória — tem que fechar 2400/300/300.
for split, esperado in (("train", 2400), ("val", 300), ("test", 300)):
    imgs = list((out / split / "images").glob("*.png"))
    masks = list((out / split / "masks").glob("*.png"))
    assert len(imgs) == esperado, f"{split}: {len(imgs)} imagens, esperado {esperado}"
    assert len(masks) == esperado, f"{split}: {len(masks)} máscaras, esperado {esperado}"
    # Amostra de pareamento (nome base igual entre imagem e máscara).
    for img in imgs[:3]:
        assert (out / split / "masks" / img.name).exists(), f"máscara ausente: {img.name}"
    print(split, len(imgs), "imagens +", len(masks), "máscaras OK")
assert (out / "split_manifest.json").exists(), "split_manifest.json ausente (vem da parte 1)"
print("split_manifest.json OK")

# O treino DEVE usar o diretório reunido, não o autodetectado (parte 1).
DATA = out
print("DATA =", DATA)
```

## 4. Célula de treino (cópia de treino completo, não o notebook principal)

Na cópia do notebook para o treino completo, ajustar só a célula de
treino em relação à do teste:

```python
EPOCHS = None  # None = usa a config: 30 épocas, early stopping paciência 5 em val_mean_iou

import subprocess, sys
cmd = [
    sys.executable,
    str(REPO / "training" / "train.py"),
    "--config", str(REPO / "training" / "config.json"),
    "--data-dir", str(DATA),  # DATA = /kaggle/working/dataset/synthetic, da célula de junção
    "--out-dir", "/kaggle/working/out",
    "--batch-size", "4",  # ver §5; a célula atual do notebook não passa batch-size
]
if EPOCHS is not None:
    cmd += ["--epochs", str(EPOCHS)]
print("comando:", " ".join(cmd))
subprocess.run(cmd, check=True)
```

## 5. Hiperparâmetros do treino completo

- Modelo: SegFormer-B0 (`nvidia/mit-b0`), 512×512, normalização
  ImageNet, 4 classes, loss `0.5 × CE com peso + 0.5 × Dice`.
- **Pesos CE: `[1.0, 1.5, 10.0, 10.0]`** (ver §1).
- **Batch 4** (não 8 da config): o código roda em fp32 sem autocast
  (a config promete bf16, mas o código não aplica), e `batch 8` em
  512×512 é o principal risco de OOM numa T4. `gradient_accumulation`
  não está implementado — batch 4 é batch efetivo 4, não 8.
  Não reduzir a resolução: traço fino depende dela.
- Épocas: 30 da config (`EPOCHS = None`), early stopping com
  paciência 5 sobre `val_mean_iou` (pode encurtar o treino).
- Seed 42, AdamW lr 6e-5, cosine com warmup 10%, weight decay 0,01.
- Tempo estimado (faixa, T4): 3.000 cenas em batch 4, da ordem de
  3–4 h se rodar as 30 épocas; early stopping pode encurtar.
  **Em CPU é inviável** — não lançar sem GPU confirmada.

## 6. Checklist final antes de apertar o botão

- [ ] `training/config.json` com pesos `[1.0, 1.5, 10.0, 10.0]`
      commitado e pushado (pendente — ver §1).
- [ ] Nova versão de `guzzfort/draw-training-code` subida e
      verificada por download-back (config nova + `train.py` com
      `dynamo=False` e exportação não-fatal, commit `b370560`).
- [ ] As 8 partes `draw-synthetic-v1-part1..8` anexadas como inputs,
      todas `ready` (conferido em PARTS.md).
- [ ] `danieldelro/freihand` e `guzzfort/mit-b0-cache` v2 anexados.
- [ ] **GPU confirmada na interface web do Kaggle** (ver §7) —
      antes de rodar a célula de treino, conferir no log que
      `torch.cuda.is_available()` é True; se vier CPU, parar.
- [ ] Célula de junção (§3) rodada com 2400/300/300 OK e
      `DATA = /kaggle/working/dataset/synthetic`.
- [ ] Célula de treino (§4) com `EPOCHS = None` e
      `--batch-size 4`.
- [ ] Ao fim: baixar `metrics.json`, `classes.json`,
      `training_config.json`, `split_manifest.json`,
      `draw_segmentation_v1.onnx` e `checkpoints/best.pt`.

## 7. O que falta do usuário

1. **Confirmar a GPU na interface web do Kaggle.** Em todas as
   rodadas até aqui o kernel veio em CPU (`torch 2.11.0+cpu |
   cuda: False`) apesar de `enable_gpu` no metadata e até de
   `--accelerator NvidiaTeslaT4` no push via CLI. A CLI nunca
   entregou GPU neste projeto; a seleção do acelerador (T4/P100)
   precisa ser confirmada na interface web antes do treino completo.
2. **Autorizar o commit/push da troca de pesos** (§1) — este preparo
   não commitou nada, por escopo da tarefa.
3. Decisões que **não** travam este treino, mas ficam para depois:
   validação ONNX vs PyTorch (só acontece se `onnxruntime` estiver
   no kernel; sem ele, `onnx_validation` vem `null` e a entrega não
   pode ser chamada de "validada"); bateria fixa de 15 cenários
   (`tools/seg_metrics.py`, rodada à parte, não pelo notebook);
   holdout real com os frames da sessão
   `captures/training/session_20261029_130013_453419/` (depende da
   decisão Rota A/B e da anotação, ainda pendentes).

## 8. O que verificar no resultado (não prometer antes)

- `metrics.json`: IoU/Dice/precision/recall por classe, ink com
  tolerância 2–3 px, `onnx_validation` (null = validação pulada).
- Comparar com o teste de 2 épocas/36 cenas (mean IoU 0,482, ink
  IoU 0,062 — dataset minúsculo com mãos placeholder, não é
  referência para o modelo final).
- Métricas sintéticas e do holdout real sempre reportadas
  separadamente; queda grande no real é domain gap documentado.
