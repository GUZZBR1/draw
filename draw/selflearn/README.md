# draw/selflearn: o loop que aprende sozinho (Loop A, fatia 1)

O Draw em producao e so inferencia com pesos congelados. Este pacote
e o comeco do loop fechado offline que muda isso: o app coleta o que
ve, o monitor aponta onde a rede provavelmente errou, uma fila
ordenada decide o que vale correcao, um teacher offline gera
pseudo-rotulos com veto, e uma rodada de treino no estilo UniMatch
V1 (teacher EMA, weak-to-strong) aprende com rotulados + pseudo.
Aprender acontece entre sessoes, nunca dentro do app em tempo real.

## O loop

```
sessao ao vivo
  -> [0 MONITOR]  monitor.py: entropia/margem ranqueiam suspeita;
     tinta que some entre passadas e sinal de erro; a testemunha
     classica cruza com a rede e separa candidato a FP e a FN
  -> [1 COLETA]   collect.py: ExampleRecorder grava warp + mapa de
     probabilidade + linha de manifesto com sha256, sessao, modelo
     e origem (a linhagem de cada exemplo)
  -> [2 SELECAO]  select.py: score auditavel (desacordo TTA +
     dispersao temporal + impureza de regiao + novidade + estrato
     de risco), teto por sessao e cota aleatoria garantida
  -> [3 TEACHER]  teacher.py: voto temporal (mediana das passadas
     alinhadas) e pseudo-rotulo com veto da testemunha: tinta sem
     escurecimento vai para ignore, nunca para fundo; escurecimento
     que a rede nao viu tambem vai para ignore
  -> [5 TREINO]   tools/selflearn/train_round.py: uma rodada com
     replay integral dos rotulados em todo batch, teacher EMA
     (decay 0.999) rotulando a vista fraca, aluno na vista forte,
     perda supervisionada BCE+Dice + consistencia mascarada pela
     confianca. Exporta ONNX e metrics.json
```

## Modulos

- `monitor.py` funcoes puras de suspeita e divergencia (etapa 0).
- `collect.py` gravador de exemplos no formato de dataset (U1).
- `select.py` fila de active learning com motivos por item (U2).
- `teacher.py` agregacao temporal e pseudo-rotulo com veto (U3).
- `tools/selflearn/train_round.py` a rodada de treino (U5 minimo).
  `--smoke` prova a receita em CPU com dataset sintetico proprio:
  perda supervisionada cai, EMA se move, ONNX exporta.

Testes: `tests/test_selflearn.py`.

## O que NAO esta implementado ainda

- U4: interface de correcao minima (editar a predicao com pincel e
  scribble, orcamento em minutos e gestos).
- U6: portao de promocao automatizado (holdout selado por sessao,
  pior-grupo, stress set historico, pre-registro da rodada).
  Promocao de modelo continua sendo decisao humana.
- PrevMatch-lite (voto de checkpoints de rodadas anteriores).
- Loop B (teacher maior offline com destilacao por canal para o
  aluno) e Loop C (pre-treino mascarado do backbone).
- Integracao com o runner do Kaggle (tools/runner) e a subida
  automatica do dataset versionado.
- Deteccao de mudanca por tile no app (frente de tempo real do
  adendo): o monitor aqui e offline, nao esta ligado ao app.
