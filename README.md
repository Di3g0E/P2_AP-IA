# P2_AP-IA: Proyecto de Inteligencia Artificial

## Descripción
Este proyecto implementa la solución para la Práctica 2 de la asignatura Aprendizaje de IA, enfocado en el análisis y clasificación de transacciones bancarias mediante técnicas de Machine Learning y PLN.

## Estructura del Proyecto
- `config/`: Archivos de configuración (.env, YAML).
- `data/`: Datos en sus diferentes estados (raw, processed, external).
- `doc/`: Documentación y memoria del proyecto.
- `logs/`: Registros de ejecución.
- `models/`: Modelos entrenados y binarios.
- `playground/`: Espacio para pruebas y cuadernos de análisis.
  - `nlp_expense_classifier.ipynb`: Clasificación binaria de **Type** (Expenses/Income) mediante PLN clásico.
  - `nlp_area_classifier.ipynb`: Clasificación multiclase de **Area** mediante PLN clásico.
  - `nlp_area_transformers.ipynb`: Clasificación multiclase mediante **Transformers** (BETO, RoBERTa, etc.).
- `src/`: Código fuente modularizado.
  - `data/`: Carga y limpieza de datos.
  - `features/`: Ingeniería de variables.
  - `models/`: Definición y entrenamiento de modelos.
  - `evaluation/`: Validación y métricas.
  - `utils/`: Helpers y utilidades.
- `tests/`: Pruebas unitarias y de integración.

## Instalación
1. Crear un entorno virtual:
   ```bash
   python -m venv venv
   ```
2. Activar el entorno virtual:
   - Windows: `.venv\Scripts\Activate.ps1`
   - Linux/Mac: `source venv/bin/activate`
3. Instalar dependencias:
   ```bash
   uv pip install -r P2_AP-IA/requirements.txt --link-mode copy
   ```

## Cuadernos (PLN Avanzado)
En el directorio `playground/` se encuentran tres análisis detallados utilizando Procesamiento de Lenguaje Natural:
1. **Clasificación de Tipo**: Predicción de Gasto vs Ingreso. Todos los modelos clásicos alcanzan el 100% de eficacia.
2. **Clasificación de Área**: Categorización en 9 áreas (Food, Leisure, Invoice, etc.).
3. **Transformers**: Fine-tuning de modelos estado del arte en español (BETO, RoBERTa-BNE) para la clasificación de áreas.

## Ejecución
Para ejecutar el flujo principal del proyecto o los experimentos:
```bash
python main.py
# O para comparar modelos:
python src/experiments/compare_models.py --plot
```

## Requisitos de Sistema para Transformers
El notebook de Transformers detecta automáticamente si hay una GPU disponible. En caso de usar CPU, el entrenamiento de todos los modelos puede demorar entre 15 y 30 minutos.
