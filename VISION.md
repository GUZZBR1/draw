# Draw — Visão

*Documento de visão escrito a partir da ideia original de Lucas Santleme, registrado em 28/10/2026.*

## A ideia em uma frase

O Draw é uma ponte em tempo real entre o desenho no papel e os sistemas
digitais profissionais: a pessoa desenha no papel, do jeito natural, e o
sistema captura, interpreta e reproduz esse desenho em softwares de CAD e
simuladores — devolvendo feedback imediato enquanto a pessoa ainda está
desenhando.

## Por que papel?

Engenheiros, arquitetos e, na prática, a maior parte das pessoas pensa
primeiro no papel. Desenhar no papel não tem fricção: não há menus,
ferramentas, camadas nem mouse. É o pensamento indo direto para o traço.

Os sistemas poderosos — AutoCAD, NVIDIA Isaac Sim / Isaac Lab, simuladores
de física e de materiais — cobram o preço oposto: para usá-los, é preciso
parar de pensar no problema e pensar na ferramenta, desenhando com o mouse,
ponto a ponto.

O Draw elimina essa troca. O papel vira a interface de entrada que esses
sistemas nunca tiveram.

## O loop central

1. **Desenhar no papel** — com lápis, caneta e borracha, como sempre.
2. **Capturar e entender** — uma webcam apontada para a mesa captura cada
   traço. Visão computacional, redes neurais e modelos de visão transformam
   o traço bruto em um modelo estruturado: uma linha deixa de ser "tinta"
   e passa a ser uma parede, uma viga, uma aresta de peça ou a trajetória
   de um braço robótico.
3. **Reproduzir no sistema alvo** — o modelo entra automaticamente no
   AutoCAD, em um simulador de robótica ou em outro sistema profissional,
   como se a pessoa estivesse desenhando lá dentro com o mouse — só que
   está desenhando no papel.
4. **Receber feedback na hora** — a simulação roda sobre o desenho enquanto
   ele é feito: a resistência dos materiais faz sentido? O formato faz
   sentido? Há colisão, sobrecarga, um ponto de estresse? A pessoa vê o
   resultado, apaga com a borracha, corrige no papel — e a correção também
   é capturada e aplicada ao modelo.

## As camadas

A visão é grandiosa, e ela se constrói em camadas, cada uma entregando algo
real:

| Camada | O que faz | Estado |
|---|---|---|
| 1. Captura | Webcam → folha retificada → tinta digital em tempo real (a "mesa digitalizadora sem mesa") | **v0 — em construção** |
| 2. Interpretação | Traço bruto → primitivas e objetos com significado, escala e parâmetros | futura |
| 3. Integração | Modelo → AutoCAD, Isaac Sim/Lab e outros simuladores | futura |
| 4. Feedback | Simulação (materiais, física, colisão) devolvendo resultado em tempo real para quem desenha | futura |

## A primeira barreira

Tudo começa pela camada 1: provar, na mesa real, que uma webcam comum
consegue transformar o que é desenhado numa folha de papel em traço digital
fiel, em tempo real — incluindo apagar com a borracha e continuar
desenhando. Sem captura confiável, não há interpretação; sem interpretação,
não há simulação.

É essa barreira que a v0 ataca.
