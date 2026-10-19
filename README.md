# Processamento de folhas de macieira

Projeto de Processamento Digital de Imagens para classificar quatro condições
visuais de folhas de macieira: saudável, sarna, podridão negra e ferrugem.

O pipeline aplica filtragem bilateral, compara segmentação por Otsu e k-means
com a imagem completa, extrai atributos de forma, cor e textura e avalia SVM e
Random Forest. O bloco de textura combina GLCM e LBP; uma análise complementar
permite compará-los separadamente.

## Instalação

Requer Python 3.10.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Dados

As imagens completas não são versionadas. Para restaurar e verificar o conjunto
registrado no manifesto:

```bash
python scripts/download_data.py
python scripts/download_data.py --verify-only
```

## Execução

Use um diretório novo para cada execução completa:

```bash
python -m scripts.run_experiments prepare --output outputs/reproduction
python -m scripts.report_results calibration --output outputs/reproduction
python -m scripts.run_experiments extract --output outputs/reproduction
python -m scripts.run_experiments select --output outputs/reproduction
python -m scripts.run_experiments evaluate --output outputs/reproduction
python -m scripts.report_results final --output outputs/reproduction
```

O protocolo registra partições e hashes, seleciona configurações apenas no
desenvolvimento e impede sobrescrever silenciosamente uma avaliação iniciada.

Depois da extração, GLCM e LBP podem ser comparados separadamente usando somente
as cinco partes de desenvolvimento:

```bash
python -m scripts.compare_texture_descriptors \
  --experiment outputs/reproduction \
  --output outputs/texture_comparison.csv
```

## Testes

```bash
python -m unittest discover -s tests -v
```

O projeto avalia condições visuais na coleção utilizada. Ele não constitui um
diagnóstico agronômico nem demonstra generalização para fotografias de campo.
