# P2_AP-IA: Proyecto de Inteligencia Artificial

## Descripción
Este proyecto implementa la solución para la Práctica 2 de la asignatura Aprendizaje de IA.

## Estructura del Proyecto
- `config/`: Archivos de configuración (.env, YAML).
- `data/`: Datos en sus diferentes estados (raw, processed, external).
- `doc/`: Documentación y memoria del proyecto.
- `logs/`: Registros de ejecución.
- `models/`: Modelos entrenados y binarios.
- `playground/`: Espacio para pruebas rápidas y cuadernos.
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
   - Windows: `.\venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`
3. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```

## Ejecución
Para ejecutar el flujo completo del proyecto:
```bash
python main.py
```
