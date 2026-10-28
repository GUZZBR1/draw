# Backlog do Draw

Backlog preparado em 2026-10-28, a partir da v0.1. Cada item abaixo é um
rascunho de issue, pronto para abrir no GitHub. Nada aqui foi mergeado:
o fluxo é branch, PR e revisão antes de mergear na main.

Prioridade: P0 é o próximo passo do roadmap, P1 é robustez e MVP,
P2 é o futuro.

## P0

### 1. v0.2: traço vetorial (ordem, caminho e suavização)

Contexto: hoje o canvas é uma máscara de pixels. Para reproduzir no
Paint, interpretar formas e salvar um desenho editável, a tinta precisa
virar traços: lista ordenada de pontos, com suavização.

Critério de aceite:

- `draw/strokes.py` extrai traços de uma máscara de tinta.
- Cada traço tem pontos ordenados, comprimento e um caminho suavizado.
- Exporta SVG e JSON, com largura e altura da folha.
- Teste sintético: cena com círculo, linha e X produz pelo menos
  3 traços, SVG válido e JSON que volta a carregar.
- O pipeline v0.1 continua passando no selftest.

Relacionado: roadmap, v0.2.

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

Contexto: hoje só existe `tests/selftest.py`, sem CI.

Critério de aceite:

- Workflow do GitHub Actions instala as dependências, roda o
  selftest e os testes pytest.
- O workflow falha se qualquer um deles falhar.
- README documenta como rodar os testes localmente.

Estado: feito na branch chore/tudo-pendente, com workflow em
.github/workflows/ci.yml rodando selftest, pytest completo e
verificação geométrica. Falta confirmar o primeiro run verde no
GitHub, que o próprio PR desta branch deve disparar.

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
