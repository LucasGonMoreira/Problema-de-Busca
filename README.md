# Simulação de entregas com combustível

Projeto em Python que compara duas ordens de entrega em uma grade hexagonal:
Vizinho mais próximo e Algoritmo Genético (AG). A animação executa as rotas em
sequência, mostra o nível do tanque e marca o resultado com **V** se o entregador
chegar ao centro de distribuição (CD) ou **X** se ficar sem combustível antes.

## Arquivos

- `busca.py`: simulação e geração dos resultados.
- `casa-limpa.png` e `entregador.png`: imagens usadas na animação.
- `simulacao_entregas.gif`: animação das duas rotas em sequência.
- `simulacao_final.png`: quadro final comparando os resultados.

## Requisitos

Python 3.10 ou superior.

Instale as bibliotecas necessárias:

```bash
python -m pip install -r requirements.txt
```

## Executar

Na pasta do projeto, rode:

```bash
python busca.py
```

O programa grava `simulacao_entregas.gif` e `simulacao_final.png` na própria
pasta do projeto. Os resultados também são exibidos no terminal.

## Configuração

Os parâmetros ficam no início de `busca.py`. `TANQUE_L` define o combustível
inicial em litros; `LITROS_POR_UNIDADE` define o consumo por unidade de custo.
As imagens de entrada devem ficar junto do script.
