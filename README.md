# P2_AP-IA: Proyecto de Inteligencia Artificial

## Descripción
Este proyecto implementa la solución para la Práctica 2 de la asignatura Aprendizaje de IA, enfocado en el análisis y clasificación de transacciones bancarias mediante técnicas de Machine Learning y PLN.

## Estructura del Proyecto
- `config/`: Archivos de configuración (.env, YAML).
- `data/`: Datos en sus diferentes estados (raw, processed, external).
- `doc/`: Documentación y memoria del proyecto.
- `logs/`: Registros de ejecución.
- `models/`: Modelos entrenados y binarios (incluye el modelo de producción).
- `playground/`: Espacio para pruebas y cuadernos de análisis.
  - `nlp_area_classifier.ipynb`: Clasificación multiclase de **Area** mediante PLN clásico.
  - `nlp_area_transformers.ipynb`: Clasificación multiclase mediante **Transformers**.
  - `nlp_evaluation_comparison.ipynb`: Comparativa avanzada de métricas.
- `src/`: Código fuente modularizado para producción.
  - `data/preprocessing.py`: Limpieza y normalización multilingüe.
  - `models/classifier.py`: Clase `FinancialClassifier` con soporte de aprendizaje incremental.
  - `utils/config.py`: Configuración centralizada de rutas y parámetros.
  - `production_pipeline.py`: Script principal de orquestación del modelo.
- `tests/`: Pruebas unitarias y de integración.

## Instalación

> [!IMPORTANT]
> Se recomienda utilizar **Python 3.11.13**. El proyecto se ha configurado para esta versión debido a la compatibilidad de librerías de ML.

1. Crear un entorno virtual:
   ```bash
   python -m venv .venv
   ```
2. Activar el entorno virtual:
   - Windows: `.venv\Scripts\Activate.ps1`
   - Linux/Mac: `source .venv/bin/activate`
3. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```

## Modelo en Producción
Se ha seleccionado un modelo de **Regresión Logística (SGDClassifier)** con **n-gramas de caracteres (2-5)** por los siguientes motivos:
- **Eficiencia**: Reentrenamiento ultra-rápido (<300ms), ideal para actualizaciones frecuentes (cada 100 entradas).
- **Multilingüe**: El análisis por caracteres captura raíces comunes entre idiomas sin necesidad de modelos pesados.
- **Incremental**: Soporta `partial_fit`, permitiendo actualizar el modelo sin reentrenar desde cero.

## Ejecución

### Entrenar el Modelo de Producción
Para ejecutar el pipeline modularizado y generar el modelo en `models/`:
```bash
# Entrenamiento con datos por defecto:
python main.py --mode train

# Entrenamiento con una base de datos personalizada y nombre modelo:
python main.py --mode train --data ruta/a/tu_base_de_datos.csv --model ruta/a/tu_modelo.joblib
```

### Probar/Evaluar el Modelo
Para evaluar el rendimiento del modelo sobre un conjunto de datos específico:
```bash
python main.py --mode evaluate --data data/processed/db_mod_descript_test.csv
```
> [!NOTE]
> Si el CSV indicado en `--data` contiene una columna **Area**, se mostrará el reporte de clasificación completo. De lo contrario, se imprimirán las predicciones obtenidas.

## Requisitos de Sistema para Transformers
El notebook de Transformers detecta automáticamente si hay una GPU disponible. En caso de usar CPU, el entrenamiento de todos los modelos puede demorar entre 15 y 30 minutos.
