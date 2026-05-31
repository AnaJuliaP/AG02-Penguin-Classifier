# AG2 - Classificador de Pinguins

Trabalho prático da AG2 de Engenharia de Software - Inatel (1º Sem 2026).

Utilizei o dataset **palmerpenguins** para treinar um modelo de classificação
que identifica a espécie de um pinguim com base em suas medidas físicas.

## Tecnologias

- Python
- Pandas
- scikit-learn

## Como executar

Instale as dependências:

```bash
pip install pandas scikit-learn
```

Execute o script:

```bash
python pinguins.py
```

## Dataset

O dataset contém 333 amostras de pinguins coletadas no Arquipélago Palmer,
com as seguintes colunas: ilha, sexo, comprimento e profundidade do cúlmen,
comprimento da nadadeira, massa corporal e espécie.

## Modelo

Foi utilizada uma **Árvore de Decisão** (Decision Tree) com divisão de
80% treino e 20% teste, atingindo 99% de acurácia.
