# Contribuindo com o Draw

## Fluxo

Trabalho em branch, PR e revisão. Nada direto na main.

```bash
git checkout -b <tipo>/<descricao-curta>
# ... trabalho ...
python tests/selftest.py
python -m pytest tests/ -q
python verification/run_verification.py
git push -u origin <tipo>/<descricao-curta>
```

## Rodando localmente

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install pytest
.venv/bin/python tests/selftest.py
.venv/bin/python -m pytest tests/ -q
.venv/bin/python verification/run_verification.py
```

Sem webcam, o selftest valida o pipeline inteiro com uma cena sintética.

## O que um PR precisa ter

- Selftest verde.
- Verificação geométrica verde quando a mudança toca em
  `draw/paper.py`, `draw/ink.py`, `draw/config.py` ou
  `template/generate_template.py`.
- Teste novo quando o comportamento muda.
- README atualizado quando a interface, as teclas ou o formato de
  saída mudam.
- Sem segredos, chaves ou dados pessoais.
