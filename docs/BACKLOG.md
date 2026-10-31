# Backlog do Draw

Backlog revisado em 2026-10-31. Os itens abaixo são trabalho pendente ou
registro de entrega; confira o estado do código antes de abrir uma issue.

Prioridade: P0 é o próximo passo do roadmap, P1 é robustez e MVP,
P2 é o futuro.

## Concluído

### Extração e exportação vetorial

`draw/strokes.py` extrai traços vetoriais e exporta SVG/JSON; `draw/export.py`
também cobre SVG em escala física e DXF. A ordem temporal original ainda não
é reconstruída.

### CI e testes automatizados

O workflow está em `.github/workflows/ci.yml`; o tree atual contém `selftest.py`
e testes `pytest`. A data do último run remoto não é registrada neste backlog.

## P0

### 1. Integração do modelo de tinta pareada

Contexto: existe um ONNX experimental em
`models/experimental/ink-pairs-v3/`. Ele precisa de frames alinhados
`before/after` e prevê tinta nova, enquanto o app atual processa uma observação
por frame e mantém um canvas acumulativo. O app agora tem `--pair-ink-model` /
`DRAW_PAIR_INK_MODEL` opt-in e usa a previsão como evidência complementar à
heurística; o runtime do ONNX ainda não foi validado localmente.

Critério de aceite:

- Um adaptador fornece pares de frames retificados com o pré-processamento do
  treino.
- A máscara de tinta nova atualiza o canvas sem apagar traços antigos quando
  não há evento de tinta; a borracha e a mão continuam com comportamento
  definido.
- Teste de contrato usa o ONNX Runtime em mais de uma resolução e compara
  resultado com o checkpoint de treino.
- O modo é opt-in e o app informa claramente quando o modelo não foi validado.

Estado: adaptador e modo opt-in implementados; faltam validação real do ONNX,
teste de webcam, medição de latência e revisão explícita do comportamento de
apagamento/oclusão.

### 2. v0.3: reprodução dos traços no Paint via controle do mouse

Contexto: próximo passo depois do traço vetorial.

Critério de aceite:

- Um modo que reproduz os traços exportados movendo o mouse,
  sem depender de clique manual ponto a ponto.
- Um flag para modo seco, que só lista os movimentos, sem mover o mouse.
- Documentado no README, com aviso de que mover o mouse é uma ação
  visível e reversível apenas desfazendo no app alvo.

## P1

### 3. Robustez da captura quando um marcador some

Contexto: hoje `warp_paper` devolve `None` se menos de 4 marcadores
estiverem visíveis. Uma mão sobre a folha apaga o canvas inteiro.

Critério de aceite:

- Com 3 marcadores, ou com um marcador perdido por poucos frames,
  o sistema mantém a última homografia válida por um tempo curto
  e documentado, em vez de apagar tudo.
- Teste sintético cobre o caso de marcador ausente.
- Nenhuma regressão no selftest.

### 4. Suavização temporal e medição de jitter

Contexto: a homografia é recalculada a cada frame, sem memória.
O canvas pode tremer mesmo com a folha parada.

Critério de aceite:

- Um filtro simples, por exemplo média móvel dos centros dos
  marcadores, com parâmetro documentado.
- Um teste ou script mede o deslocamento frame a frame numa cena
  estática e mostra redução do jitter.
- O filtro pode ser desligado por flag.

### 5. CI e testes automatizados

Contexto: manter o workflow e a suíte alinhados com os contratos atuais.

Critério de aceite:

- Workflow do GitHub Actions instala as dependências, roda o
  selftest e os testes pytest.
- O workflow falha se qualquer um deles falhar.
- README documenta como rodar os testes localmente.

Estado: workflow presente em `.github/workflows/ci.yml`; confirmar o estado
dos runs no provedor antes de considerar o CI operacional.

### 6. Calibração para iluminação real

Contexto: o limiar adaptativo foi validado só na cena sintética.
Mesa real tem sombra, reflexo e papel amarelado.

Critério de aceite:

- Um conjunto de fotos reais, ou um gerador sintético com sombra
  e gradiente de luz, para medir recall e falsos positivos.
- Parâmetros de `extract_ink` documentados, com o trade-off
  entre tinta fraca e ruído.
- Sem regressão no selftest.

### 7. Demo sem webcam

Contexto: quem avalia o projeto nem sempre tem webcam e folha
impressa por perto.

Critério de aceite:

- Um script gera uma sequência animada, por exemplo um traço
  sendo desenhado aos poucos, e roda o pipeline sobre ela.
- O README mostra como rodar a demo e o que esperar.
- A demo não precisa de webcam nem de janela.

## P2

### 8. v1: interpretação de formas, letras e diagramas

Contexto: camada 2 da visão.

Critério de aceite:

- Um primeiro classificador simples, por exemplo círculo, linha
  e retângulo, a partir dos traços vetoriais.
- Teste com formas sintéticas e taxa de acerto documentada.
- Sem acoplamento com um sistema alvo específico.

### 9. v2: integração com um primeiro sistema alvo

Contexto: camada 3 da visão. Escolher um alvo só, antes de tentar
vários.

Critério de aceite:

- Decisão documentada: qual sistema, por que, e qual formato de
  entrada ele aceita.
- Um exemplo de ponta a ponta, do papel ao sistema alvo.
- Sem credenciais ou chaves no repositório.

### 10. Licença e contribuição

Contexto: o repositório ainda não tem arquivo de licença nem guia
de contribuição. Isso trava uso por terceiros e por empresas.

Critério de aceite:

- Decisão do dono do repositório sobre licença e titular do
  copyright, registrada na issue antes de mergear o arquivo.
- `CONTRIBUTING.md` com como rodar, testar e abrir PR.
- Sem segredos, chaves ou dados pessoais no repositório.

## Notas de processo

- Trabalho em branch, PR e revisão. Nada direto na main.
- Cada PR mantém o selftest verde.
- Issue nova só entra no backlog depois de ter critério de aceite.
