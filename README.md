<<<<<<< HEAD
# Radar Signal Classification Project

## Overview
Este proyecto implementa un clasificador de señales de radar utilizando redes neuronales convolucionales (CNN) para identificar diferentes tipos de modulación y señales.

## Project Structure
```
radar_signal_classification/
│
├── data/                  # Directorio para datasets
├── src/                   # Código fuente principal
│   ├── data_loader.py     # Carga y preprocesa datos
│   ├── model.py           # Arquitectura del modelo
│   ├── train.py           # Entrenamiento y validación cruzada
│   └── utils.py           # Funciones de utilidad
├── results/               # Resultados de entrenamiento
│   ├── models/            # Modelos guardados
│   ├── logs/              # Logs de entrenamiento
│   └── plots/             # Visualizaciones
├── notebooks/             # Cuadernos Jupyter para exploración
├── requirements.txt       # Dependencias del proyecto
└── main.py               # Punto de entrada principal
```

## Instalación

1. Clonar el repositorio
```bash
git clone https://github.com/tu_usuario/radar_signal_classification.git
cd radar_signal_classification
```

2. Crear entorno virtual
```bash
python3 -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. Instalar dependencias
```bash
pip install -r requirements.txt
```

## Uso

### Entrenamiento del Modelo
```bash
python main.py --dataset /ruta/a/tu/dataset.hdf5 --epochs 50 --batch_size 64
```

### Parámetros Opcionales
- `--dataset`: Ruta al archivo HDF5
- `--epochs`: Número de épocas de entrenamiento
- `--batch_size`: Tamaño del lote
- `--cv_splits`: Número de splits para validación cruzada

## Resultados
Los resultados se guardan automáticamente en el directorio `results/`:
- Modelos entrenados en `results/models/`
- Logs de entrenamiento en `results/logs/`
- Gráficos y visualizaciones en `results/plots/`

## Requisitos del Sistema
- Python 3.8+
- GPU recomendada para entrenamiento rápido

## Contribuciones
Las contribuciones son bienvenidas. Por favor, abre un issue o un pull request.

## Licencia
[Especifica tu licencia]
```

## Preparación Final

Para preparar completamente el proyecto:

1. Crea la estructura de directorios
```bash
mkdir -p radar_signal_classification/data
mkdir -p radar_signal_classification/src
mkdir -p radar_signal_classification/results/{models,logs,plots}
mkdir -p radar_signal_classification/notebooks
```

2. Guarda cada archivo en su ubicación correspondiente

3. Copia tu archivo HDF5 a `data/`

## Ejecución

1. Navega al directorio del proyecto
```bash
cd radar_signal_classification
```

2. Ejecuta el script principal
```bash
python main.py
```


=======
# radar_signal_classification
>>>>>>> 926c7a51012297528f9a9be66a696323fbeb628e
