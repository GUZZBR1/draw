# Zona de treinamento do Draw

Branch `training-zone`. Base: `3b95112`. Esta branch guarda o material de treino que saiu da main. A main guarda só o projeto. Main sem treino: `9dd3a5526b5ac1c505b140ae2b07a1018537d06e`.

## O que fica nesta branch

- `captures/`: 15.848 arquivos (15.845 JPG e 3 CSV). Frames de treino e de sessão. Estão na árvore do git, mesmo quando um checkout esparso não os materializa em disco.
- `training/`: `train.py` e `config.json`. Treino SegFormer-B0 em 4 classes.
- `notebooks/`: `train_segformer_kaggle.ipynb` (que já estava na main) e 3 notebooks de treino completo que estavam untracked no working tree do usuário.
- `dataset/`: manifesto e README de exemplos reais, com zero casos reais no manifesto.
- `docs/neural-seg-v1/`: documentação do projeto de segmentação neural v1.
- `tools/` de treino: `gen_synthetic.py`, `evaluate_dataset.py`, `seg_metrics.py`, `validate_onnx.py`, `check_freihand.py`, `label_frames.py` e `eval_real_frames.py`.
- `tests/` de treino: `test_gen_synthetic.py`, `test_evaluate_dataset.py`, `test_seg_metrics.py` e `test_train_pairing.py`.

## O que fica na main

- `draw/` inteiro, incluindo `draw/neural.py` (inferência ONNX em tempo de execução, não treino).
- `main.py`, `requirements.txt`, `template/`, `bridge/`, `scripts/` e `verification/`.
- `tools/measure_paper.py`: medição do detector clássico de papel, sem treino neural.
- Testes do projeto, `tests/selftest.py` e `verification/run_verification.py`.

## Compartilhado

Código compartilhado não é duplicado aqui como cópia nova. Esta branch contém a foto de `3b95112` do código compartilhado, só para as ferramentas de treino rodarem. A versão vigente do código compartilhado é a main. Mudança no código compartilhado entra primeiro na main.

## Origem dos 3 notebooks novos

Os 3 notebooks `train_segformer_full_gpu*.ipynb` foram copiados primeiro para `/tmp` e commitados a partir dessa cópia. As cópias untracked originais permanecem no working tree do usuário, para decisão posterior. `docs/UPSTREAM_PLAN.md` não é treino e não entra nesta branch nem na main. Ele continua só untracked.

## Como rodar

Na raiz desta branch: `python -m pytest tests/ -q`. A suíte completa inclui os testes de treino. Na main, a suíte ajustada exclui esses 4 testes de treino.
