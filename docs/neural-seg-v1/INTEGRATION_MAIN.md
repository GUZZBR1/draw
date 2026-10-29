# Integração main (38d6293) + pipeline neural — branch feature/main-integracao

Branch criada a partir de `origin/main` (38d6293) com merge da
`feature/neural-seg-v1` (b370560). Merge sem conflitos de conteúdo:
a branch neural só adiciona arquivos (`docs/neural-seg-v1/`,
`notebooks/`, `training/`, `tools/gen_synthetic.py`,
`tools/seg_metrics.py`, `tools/validate_onnx.py`,
`tools/check_freihand.py`, testes novos); a main mudou
`draw/app.py`, `draw/capture.py`, `tests/test_capture.py`,
`README.md` e acrescentou a sessão de câmera
`captures/training/session_20261029_130013_453419/`
(2.518 frames, 640×480, só `frames.csv` de metadados, sem máscaras).

## O que a main trouxe e foi preservado

- `run_check` reescrito: valida a fonte informada por `--source`,
  lê um primeiro frame com prazo e sempre libera a captura.
  Sem câmera, `main.py --check --source 0` sai com exit 1 e mensagem
  clara (verificado nesta branch; não é regressão, é falta de câmera).
- `--capture-all-frames` / `capture_all_frames=True` em `run()`:
  salva cada frame JPEG em `captures/training/session_<data>_<hora>/`
  com `frames.csv` (`frame_index,timestamp_ns,filename,width,height`).
  Foi assim que a sessão acima foi gerada.
- `--neural`, `--list-cameras`, `--check` e as demais flags continuam
  presentes em `main.py --help`.

## Decisão de contrato neural — NÃO ligado ainda, de propósito

- `draw/neural.py` nesta branch é o da main: ONNX antigo, 2 canais
  (`mask_logits` background/paper) + `presence_logit`, limiar 0,5,
  quadrilátero do maior contorno. Não foi alterado.
- O modelo novo do pipeline neural é de 4 canais
  (0=background, 1=paper, 2=hand, 3=ink), saída ONNX `logits`,
  com `paper_presence` DERIVADO da máscara de paper
  (`paper_fraction >= 0,01` e quadrilátero fechado), sem cabeça
  auxiliar — decisão já documentada em `INTEGRATION.md`.
- O modelo novo NÃO foi simplificado para caber no contrato antigo,
  e o contrato antigo NÃO foi trocado ainda, pelos motivos abaixo:
  1. O único modelo treinado até aqui é o de teste de 2 épocas em
     36 cenas com mãos placeholder (mean IoU 0,482 na validação de
     6 imagens; ink IoU 0,062). Não é modelo final.
  2. O ONNX gerado nesse teste (15 MB) ainda não teve validação
     ONNX vs PyTorch (onnxruntime ausente no kernel).
  3. O treino completo (30 épocas, 3.000 cenas com mãos FreiHAND
     reais) ainda não rodou; o upload do dataset grande para o
     Kaggle falhou por resets do proxy e está sendo refeito em partes.
- Ligar o modelo de 4 canais em `draw/neural.py` agora quebraria
  `--neural` para o modelo antigo e entregaria um modelo de teste
  como se fosse final. O passo correto, quando o modelo final
  existir e estiver validado, é o que `INTEGRATION.md` já lista
  para o Codex: ler um output `logits` de 4 canais, softmax+argmax,
  extrair `paper_mask`/`hand_mask`/`ink_mask`, calcular a presença
  derivada e manter o fallback clássico e `DRAW_NEURAL_MODEL`.

## Sessão de câmera da main — uso correto

- Os 2.518 frames são dados reais não anotados, de uma única sessão
  e uma única cena. Não entram no treino de 4 classes sem anotação
  manual (ver `REAL_FRAMES_PLAN.md`): sessão única = fonte única,
  nunca dividir por frame entre treino e teste.
- Uso imediato válido: holdout real (depois de anotar um subconjunto
  estratificado, 60–100 frames) e inspeção visual do detector.
- Sessões futuras não devem ir para o git cru (esta já tem ~66 MB):
  o caminho é dataset privado no Kaggle, com o repo guardando só
  manifesto e metadados.

## Validação nesta branch (antes do commit de merge)

- pytest: 57 passed
- selftest: OK, recall 96,03%
- verification: OK, 108 sondas, pior erro 3,05 px
- `main.py --help`: flags `--neural`, `--check`, `--list-cameras`,
  `--capture-all-frames` presentes
- `main.py --check --source 0`: exit 1, sem câmera (esperado aqui)

## Pendente (fora desta branch, sem merge para main)

- Treino completo no Kaggle (30 épocas, dataset de 3.000 cenas),
  validação ONNX vs PyTorch e bateria de 15 cenários.
- Ligação do modelo final de 4 canais em `draw/neural.py`.
- Anotação do subconjunto da sessão de câmera para holdout real.
- Decisão rota A (holdout) vs rota B (treino) para a sessão real.
