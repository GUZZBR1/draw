# Senado: validação dos modelos Kaggle

Data: 2026-10-30
Escopo: SegFormer, `InkStage2` e modelo temporal `draw-ink-pairs-train-v3`.

## Atualização após execução do piloto

Esta atualização sucede a auditoria abaixo, que registrava o estado anterior do checkout. O piloto privado comparou `synthetic_only` com `synthetic_plus_expert19`, mantendo configuração, seed e validação sintética iguais. O baseline obteve `mIoU=0.9177035`; o piloto, `0.9182410`, diferença de `+0.0005375` (aprox. `+0.054` ponto percentual). O IoU de tinta mudou de `0.743343` para `0.743175`, uma queda mínima de `0.000168`. São os mesmos 300 exemplos sintéticos, não uma avaliação de webcam; a diferença é pequena demais para afirmar melhoria prática. Os outputs foram baixados para `captures/triage/kaggle_expert_pilot_output/`.

Os dois gates ONNX passaram: baseline com diferença máxima de logits `5.72e-05`, piloto com `3.81e-05` (tolerância `0.001`); ambos tiveram acordo argmax `1.0` e IoU argmax `1.0` nas quatro classes. A autenticação Kaggle foi restaurada. Não existe holdout real aprovado, portanto não há resultado de generalização para webcam.

Foram instalados `onnx` e `onnxruntime` em um ambiente local isolado e executados `onnx.checker` e inferência CPU com tensores zero nos três artefatos preexistentes. Os grafos abriram, produziram shapes esperados e logits finitos. Para ambos os modelos do novo piloto, os gates Kaggle compararam logits PyTorch–ONNX e passaram. Nenhuma dessas checagens prova qualidade semântica real.

O fluxo de rótulos também foi ampliado: `255` agora é ignorado por loss, Dice e métricas no trainer e em `tools/seg_metrics.py`; as 19 máscaras com consenso especializado triplo foram empacotadas como piloto de treino, sem holdout. As 20 máscaras da sessão antiga continuam reprovadas para avaliação. Veja [SENADO_EXECUCAO_2026-10-30.md](SENADO_EXECUCAO_2026-10-30.md) para os hashes, contagens e limitações atuais.

## Decisão do Senado

Arquitetura aprovada com três trilhas independentes:

1. **Integridade do artefato:** contrato de entrada/saída, inspeção do grafo ONNX, execução CPU e paridade PyTorch–ONNX são gates diferentes. `null`, `skipped`, erro ou apenas exportação significam **não validado**.
2. **Qualidade semântica por frame:** comparar somente modelos que produzam a mesma máscara visível, com os mesmos frames, máscaras revisadas, pré-processamento e métricas. SegFormer e `InkStage2` podem entrar nesta trilha se seus adaptadores e a saída forem equivalentes.
3. **Qualidade temporal:** manter a U-Net pareada numa avaliação própria, pois recebe pares before/after e seu `IoU2` não é comparável ao IoU de tinta do SegFormer.

Cada execução futura deve registrar revisão e estado do código, hashes de configuração, dados/splits, pesos e ONNX, ambiente, seed, contrato do modelo e proveniência dos rótulos. Seleção de checkpoint usa `val`; qualquer relatório final de desempenho precisa de um conjunto congelado que não tenha sido usado para seleção ou ajuste.

## Estado observado na auditoria inicial (histórico)

Os itens nesta seção registram o checkout antes do piloto especializado descrito na atualização acima. Eles explicam por que os gates foram adicionados e não substituem os resultados posteriores.

O estado do checkout foi preservado: `main`, commit `3de3c42`, três commits à frente e 52 atrás de `origin/main`, com alterações e artefatos locais já existentes. Nenhuma sincronização, checkout, instalação de dependência, treino ou kernel Kaggle foi iniciado nesta execução.

### Modelo pareado v3

O log local informa 14.000 pares de treino, 2.000 de avaliação e `IoU2=0,8507` na época 7, com `n=213`. Não explica a unidade de `n`, a definição completa de `IoU2`, nem prova a linhagem e independência dos exemplos avaliados. Portanto, esse número fica classificado como **resultado exploratório reportado pelo log**, sem comparação com SegFormer.

O log registra exportação do ONNX e falha em seguida por `ModuleNotFoundError: onnxruntime`. O checkpoint e o ONNX estão presentes, mas são bytes locais sem proveniência Kaggle reconstruível a partir do log. A inspeção local encontrou strings `pair` e `logits`; `file` não identificou o ONNX como formato validável. Não há `torch`, `onnx` ou `onnxruntime` instalados neste Python 3.12.3. Assim:

- exportação: **registrada pelo log**;
- inspeção estrutural ONNX: **bloqueada**;
- inferência CPU e paridade: **bloqueadas**;
- qualidade temporal reproduzível: **bloqueada**.

### SegFormer e `InkStage2`

Os números fornecidos do relatório Kaggle são contexto, não foram revalidados neste checkout. O relatório não está presente aqui. Os outputs locais disponíveis têm `onnx_validation` ausente/nulo ou falha registrada; isso não passa o gate ONNX. O avaliador [tools/seg_metrics.py](../../tools/seg_metrics.py) rejeita rótulo `255`, enquanto [training/train.py](../../training/train.py) o aceita como pixel ignorado; não usar esse avaliador com máscaras `255` até o contrato ser alinhado. A inferência ONNX do `InkStage2` também não foi demonstrada.

### Dados e alegações

As 719 máscaras reais documentadas pertencem a duas sessões usadas no treino e são pseudo-rótulos, com bordas finas de tinta ainda sem revisão completa. Não há holdout real independente. Elas não sustentam alegação de acurácia real, comparação final ou generalização. Os resultados sintéticos e pseudo-rotulados permanecem exploratórios.

## Identidade dos artefatos locais observados

Hashes SHA-256 calculados nesta auditoria. Eles identificam os arquivos locais atuais; **não demonstram** que sejam os mesmos artefatos usados na execução Kaggle.

| Arquivo | SHA-256 |
|---|---|
| `training/train.py` | `520fe74c039eb8b3600a8af3de4911735427bde77a1111e8080c1819a0e6323e` |
| `training/config.json` | `eda7acfcd13b8b65d9f4073be3449e579624b65bd570ba0bb087f1ab0f17c900` |
| `notebooks/train_segformer_kaggle.ipynb` | `1c24fe606f78e50317123c840950eb84bf73130a9d5c18a7a9b76105ccc0b763` |
| `draw-ink-pairs-train-v3.log` | `fed737d58eecf4c8930ff427b0375822614670c38c2dadad0d07589778bf8cc3` |
| `draw_ink_pairs_v3.onnx` | `1ccc83f22aa9f0c3d68b58b47578fba381c0d3923e7bb9ce1b4ab88cbb9353fe` |
| `pairs_best.pt` | `8f9827eccaecbfa0e7a44b40f843710d99252d0c01059de853e259c2add7daab` |
| Diff não commitado (`git diff`) | `d8ea486b360e1c4561f23d7a781ab585d1e69146f4a3f8dd9616730a4b2b257b` |

## Próximos gates

1. **Preflight de proveniência:** recuperar do Kaggle a revisão, config, versão/hash dos datasets, split e ambiente exatos da corrida; explicar `n=213` e definir `IoU2`.
2. **ONNX por contrato:** instalar runtime apenas no ambiente de avaliação aprovado; validar estrutura, nomes, tipos e shapes; executar várias entradas representativas em CPU e comparar com o modelo PyTorch correspondente. Não adaptar o pair ao validador de quatro classes.
3. **Métricas:** alinhar `ignore_index=255`, suporte, agregação e casos sem tinta antes de usar um avaliador compartilhado; manter métricas estritas e tolerantes separadas.
4. **Benchmark semântico:** congelar uma sessão nova com máscaras humanas revisadas; comparar SegFormer e `InkStage2` nos mesmos frames. Relatar por imagem/sessão, classe, suporte e falsos positivos.
5. **Benchmark temporal:** auditar a origem do split do modelo pareado e definir seu alvo e unidade de amostragem antes de qualquer conclusão.

**Resultado executado nesta etapa:** auditoria do workspace, hashes locais, inventário de dependências e análise do log concluídos. Não foi possível aprovar ONNX ou qualidade dos modelos com as dependências e rótulos disponíveis. O bloqueio para criar ground truth é uma nova sessão e revisão humana das máscaras; não pode ser resolvido por código ou por reinterpretar as métricas atuais.
