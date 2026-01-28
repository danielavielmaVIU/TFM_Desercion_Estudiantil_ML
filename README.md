# TFM: Modelos de Detección Temprana de Deserción Estudiantil con Machine Learning

Este repositorio contiene el desarrollo completo del Trabajo Final de Máster (TFM) cuyo objetivo es **diseñar, entrenar y evaluar modelos de aprendizaje automático** capaces de predecir tempranamente la deserción estudiantil en educación superior, integrando prácticas de **MLOps**, trazabilidad experimental y explicabilidad de modelos.

---


## Objetivos de negocio y criterios de éxito

**Objetivos**
General: reducir la deserción estudiantil de manera temprana mediante intervenciones tempranas/oportunas
Específicos:
1. Identificar los factores asociados a la deserción
2. Construir un sistema predictivo que permita detectar a los estudiantes en riesgo
3. Proveer explicaciones claras del uso del modelo
4. Optimizar los recursos institucionales destinados a la retención estudiantil.
**ObjetCriterios de éxito**
1. Identificar al menos el 70% de los estudiantes en riesgo antes de la deserción
2. Modelo interpretable cuyo resultado pueda ser usado por equipos académicos
3. Facilitar intervenciones específicas fundamentadas en la predicción.



##  Objetivo del Proyecto de Mineria

Desarrollar un **sistema predictivo reproducible** que:
1. Anticipe el riesgo de abandono estudiantil en diferentes ventanas temporales (T0, T1, T2).
2. Compare múltiples algoritmos supervisados (Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost, SVM).
3. Genere interpretabilidad mediante valores **SHAP**, permitiendo identificar factores clave asociados al abandono.
4. Mantenga trazabilidad completa de experimentos usando **MLflow**.
5. Gestione datos y artefactos de forma controlada mediante **DVC** (modo local).
6. Siga el marco metodológico **CRISP-DM**, desde la comprensión del negocio hasta la evaluación del modelo.

El propósito final es que el sistema sirva como base para la implementación de un **sistema institucional de alerta temprana**, que guíe intervenciones de retención estudiantil.

---

## Descripción del Proyecto
La deserción estudiantil es un desafío crítico en educación superior. Este proyecto aplica técnicas de analítica avanzada y machine learning para identificar patrones asociados al abandono, explorando:
- variables sociodemográficas  
- factores académicos previos  
- rendimiento durante el primer año
- indicadores administrativos y económicos

El dataset utilizado proviene del **Instituto Politécnico de Portalegre**, disponible públicamente en el UCI Machine Learning Repository, con 4.424 estudiantes y 37 variables entre los años 2008 y 2019.

---
## Metodología: CRISP-DM
El proyecto sigue las seis fases del estándar internacional de minería de datos:

1. **Comprensión del negocio** (en desarrollo)  
2. Comprensión de los datos (en desarrollo)  
3. Preparación de los datos  
4. Modelado  
5. Evaluación  
6. Despliegue (documental)

Cada fase está documentada en el directorio `reports/` y en GitHub Projects.

---

## Tecnologías y Herramientas
### Lenguaje
- Python 3.10

### Librerías principales
- Scikit-learn  
- XGBoost / LightGBM / CatBoost  
- SHAP  
- Pandas / NumPy  
- Matplotlib / Seaborn  

### Herramientas MLOps
- **DVC** (Data Version Control) – manejo de datos en modo local  
- **MLflow** – registro de experimentos y modelos  
- **Git & GitHub** – control de versiones  
- **Entornos virtuales (venv)**

---

## Estructura del Repositorio

TFM_Desercion_Estudiantil_ML/
│
├── data/
│ ├── raw/ # Datos originales (solo .gitkeep + .dvc)
│ └── processed/ # Datos procesados
│
├── src/
│ ├── data/ # Carga y limpieza de datos
│ ├── features/ # Ingeniería de características
│ ├── models/ # Entrenamiento, evaluación y predicción
│ └── utils/ # Configuración, helpers, logging
│
├── notebooks/ # EDA, prototipos y análisis
├── experiments/ # Resultados experimentales (MLflow)
├── reports/ # Documentación del proyecto
│
├── .env # Configuración sensible (no versionado)
├── requirements.txt # Dependencias del proyecto
└── README.md # Documento principal



---

## Estado Actual del Proyecto

- ✔ **Fase 1 – Comprensión del negocio** completada  
- ✔ Dataset integrado y versionado con DVC  
- ✔ Entorno virtual configurado  
- ✔ Estructura MLOps creada  
- ➤ Fase 2 en progreso: descripción, EDA y verificación de la calidad del dato  

---

## Gestión del Proyecto

El avance y planificación del TFM se gestiona mediante **GitHub Projects** (tablero Kanban):
> GitHub. (s.f.). *GitHub Projects* [Herramienta de gestión de proyectos]. https://github.com/features/issues/

---

## Próximos Pasos

- Completar la Fase 2 (EDA + calidad de datos).  
- Diseñar el pipeline de preprocesamiento.  
- Entrenar los modelos iniciales y registrar experimentos.  
- Implementar interpretabilidad SHAP por fase temporal.  
- Evaluar resultados y preparar conclusiones del TFM.

---

## Autor

Proyecto desarrollado por **[Tu nombre]**,  
Máster en Big Data y Ciencia de Datos — Universidad Internacional de Valencia (VIU).
