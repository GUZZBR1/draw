# tools/runner: o runner mínimo do Draw

O encanamento pequeno que impede as perdas bobas do ciclo de
experimento. Nasceu das falhas medidas de outubro de 2026, cada uma
com conserto específico:

- **Corrida de ~4 h aprendendo nada** (bug de agenda de LR):
  self-test de 1 batch antes de lançar, conferindo LR efetiva e
  métrica se mexendo, não só "rodou"; e a vigia (`watch.py`) lê as
  métricas incrementais e aponta colapso nas primeiras épocas.
- **Métricas perdidas depois do export**: o kernel escreve as
  métricas antes de conferir o artefato. Regra do script de treino,
  registrada aqui.
- **Download truncado (ONNX de 0 bytes)**: coleta só com resume e
  sha256 conferido (`collect.py`), com manifesto por corrida.
- **Slot de TPU preso por ~11 h**: estado de toda corrida gravado
  em disco a cada leitura (`watch.py`), para qualquer processo
  retomar, conferir e cancelar sem depender de vigia viva.

## Peças

- `recipe.py`: o candidato como dado (kernel, datasets, hardware,
  épocas, aumentos, hiperparâmetros).
- `runner_kaggle.py`: push, status e saída de kernel no Kaggle.
- `cycle.py`: amarra receita → push → vigia → coleta → arquivo,
  com retomada pelo estado em disco.
- `collect.py`: download com resume em laço + hash + manifesto.
- `watch.py`: estado persistido e detecção de colapso.
- `archive.py` + `candidates.jsonl`: o log de candidatos (receita,
  status, métricas, resultados de régua, veredito). É o registro
  por rodada do projeto.
- `tests/`: a suíte do runner (coleta contra servidor que corta a
  conexão, colapso sobre as corridas reais, ciclo com runner falso).

## O que isto deliberadamente não é

- **Não é uma cascata com veto.** Hyperband/ASHA rendem com dezenas
  de configurações; o Draw treina uma ou duas versões por vez. E a
  métrica sintética por época já inverteu o ranking da régua de
  cena nos nossos três candidatos julgados: ela serve para detectar
  corrida morta, nunca para promover ou vetar candidato vivo. Veto
  de verdade só depois do holdout real selado existir.
- **Não é evolução de receitas.** O avaliador do Draw é um treino
  de horas no Kaggle com cota e uma régua de vários portões, não um
  escalar rápido e determinístico. Evolução estilo ShinkaEvolve
  fica para depois do holdout real, se um dia fizer sentido.
- **Não decide troca de modelo.** Quem decide é o Lucas, pelas
  réguas, com o holdout real como portão.

O laboratório de módulos experimentais (mini-régua, varredura de
limiar, provas) mora no repositório draw-bridge. O que se prova lá
e vira necessidade do ciclo desce para cá, pequeno e testado.
