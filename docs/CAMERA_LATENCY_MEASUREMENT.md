# Medir latência da câmera

`tools/measure_camera_latency.py` compara o detector clássico com o detector neural ONNX, usando a mesma fonte de vídeo. Cada fase dura 60 segundos por padrão. A ferramenta não grava vídeo nem imagens; grava um CSV por quadro e um resumo JSON.

## Antes de começar

- Feche o Draw e qualquer outro programa que use a câmera.
- Deixe a folha, a câmera e a iluminação paradas até as duas fases terminarem.
- Não use `--capture-all-frames`: esta ferramenta não precisa salvar quadros.
- Os avisos JPEG aparecem no terminal. A ferramenta não conta nem captura os bytes JPEG brutos.

## Executar

Na raiz do repositório, no WSL com a câmera acessível, use o ambiente que tem `onnxruntime`:

```bash
.worktrees/model-kaggle-real-pilot/.venv/bin/python \
  tools/measure_camera_latency.py --source 0 --display
```

O padrão mede 60 segundos sem modelo e depois 60 segundos com o ONNX do piloto, se ele existir em `.worktrees/model-kaggle-real-pilot/models/kaggle-real-pilot-v1/model.onnx`. Se estiver em outro lugar, informe o caminho:

```bash
.worktrees/model-kaggle-real-pilot/.venv/bin/python \
  tools/measure_camera_latency.py \
  --source 0 \
  --neural-model /caminho/para/model.onnx \
  --seconds 60 \
  --display
```

Antes de cada fase, pressione Enter para iniciar. A ferramenta aquece o detector com 10 quadros, fora dos 60 segundos medidos. Para usar exatamente o mesmo modo da câmera do Draw, deixe largura e altura sem informar: o padrão local do Draw é MJPEG 1280×720. Não rode outra aplicação de câmera durante o ensaio.

## Arquivos produzidos

Uma pasta nova em `captures/latency/<data_hora>/` contém:

- `sem_modelo.csv` e `com_modelo.csv`: tempos por quadro;
- `sem_modelo.json` e `com_modelo.json`: percentis, FPS, resolução, backend, falhas de leitura e contagens;
- `comparison.json`: os dois resumos juntos.

Os tempos `read_ms`, `neural_ms`, `classic_ms`, `warp_ms` e `display_ms` ajudam a localizar o custo. O p50 é o valor típico; o p95 mostra quadros lentos. `classic_ms` na fase neural mede o fallback clássico quando o modelo não detecta a folha.

## Limite da medição

O OpenCV entrega quadros já decodificados e não expõe aqui o timestamp original do V4L2. Portanto, `loop_ms` mede o intervalo entre quadros processados, mas não a idade exata do quadro nem toda a latência câmera-tela. Para medir a latência ponta a ponta, filme a cena e a janela do Draw juntas com outro celular em câmera lenta. Para localizar avisos JPEG, será necessário um coletor V4L2 separado que preserve cada buffer MJPEG antes da decodificação.
