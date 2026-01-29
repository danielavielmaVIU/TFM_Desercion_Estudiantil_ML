# ANÁLISIS EXPLORATORIO DE DATOS (EDA) INICIAL
## Dataset UCI 697 - Predict Students' Dropout and Academic Success

---

## 1. RESUMEN EJECUTIVO DEL DATASET

| Aspecto | Valor |
|---------|-------|
| **Registros totales** | 4.424 estudiantes |
| **Variables totales** | 37 (36 predictoras + 1 target) |
| **Valores nulos** | 0 (dataset completo) |
| **Período de datos** | 2008-2019 |
| **Institución** | Instituto Politécnico de Portalegre, Portugal |
| **Fuente** | UCI Machine Learning Repository (ID: 697) |

---

## 2. DISTRIBUCIÓN DE LA VARIABLE OBJETIVO (Target)

| Clase | N | Porcentaje | Descripción |
|-------|---|------------|-------------|
| **Graduate** | 2.209 | 49.93% | Completó estudios |
| **Dropout** | 1.421 | 32.12% | Abandonó (desertó) |
| **Enrolled** | 794 | 17.95% | Aún matriculado |
| **TOTAL** | 4.424 | 100.00% | |

### Observaciones:
- El dataset presenta **desbalance de clases**: la clase minoritaria (Enrolled) representa solo el 17.95%
- La tasa de deserción observada es del **32.12%**
- Para clasificación binaria propuesta: **Dropout (32.12%) vs No Dropout (67.88%)**

---

## 3. CLASIFICACIÓN DE VARIABLES POR TIPO

### 3.1 Variables por Tipo de Dato

| Tipo | Cantidad | Variables |
|------|----------|-----------|
| **Categóricas Nominales** | 9 | Marital status, Application mode, Course, Previous qualification, Nacionality, Mother's qualification, Father's qualification, Mother's occupation, Father's occupation |
| **Binarias (0/1)** | 8 | Daytime/evening attendance, Displaced, Educational special needs, Debtor, Tuition fees up to date, Gender, Scholarship holder, International |
| **Numéricas Discretas** | 12 | Application order, Age at enrollment, CU 1st sem (credited, enrolled, evaluations, approved, without evaluations), CU 2nd sem (credited, enrolled, evaluations, approved, without evaluations) |
| **Numéricas Continuas** | 7 | Previous qualification (grade), Admission grade, CU 1st sem (grade), CU 2nd sem (grade), Unemployment rate, Inflation rate, GDP |
| **Target (Categórica)** | 1 | Target |

### 3.2 Variables por Fase Temporal

| Fase | Momento | Variables | N |
|------|---------|-----------|---|
| **T0** | Matrícula | Demográficas + Socioeconómicas + Académicas previas | 21 |
| **T1** | Fin 1er Semestre | + Curricular units 1st sem (6 variables) | ~24 |
| **T2** | Fin 2do Semestre | + Curricular units 2nd sem (6 vars) + Macroeconómicas (3 vars) | 36 |

---

## 4. DICCIONARIO DE VARIABLES

### 4.1 Variables Demográficas

| Variable | Tipo | Valores | Descripción |
|----------|------|---------|-------------|
| **Marital status** | Categórica | 1-6 | Estado civil |
| **Nacionality** | Categórica | 21 códigos | Nacionalidad del estudiante |
| **Gender** | Binaria | 0=Femenino, 1=Masculino | Género |
| **Age at enrollment** | Numérica | 17-70 | Edad al matricularse |
| **International** | Binaria | 0=No, 1=Sí | Estudiante internacional |
| **Displaced** | Binaria | 0=No, 1=Sí | Estudiante desplazado (no local) |
| **Educational special needs** | Binaria | 0=No, 1=Sí | Necesidades educativas especiales |

### 4.2 Variables Socioeconómicas

| Variable | Tipo | Valores | Descripción |
|----------|------|---------|-------------|
| **Mother's qualification** | Categórica | 29 códigos | Nivel educativo de la madre |
| **Father's qualification** | Categórica | 34 códigos | Nivel educativo del padre |
| **Mother's occupation** | Categórica | 32 códigos | Ocupación de la madre |
| **Father's occupation** | Categórica | 46 códigos | Ocupación del padre |
| **Scholarship holder** | Binaria | 0=No, 1=Sí | Tiene beca |
| **Debtor** | Binaria | 0=No, 1=Sí | Es deudor |
| **Tuition fees up to date** | Binaria | 0=No, 1=Sí | Matrícula al día |

### 4.3 Variables Académicas de Ingreso

| Variable | Tipo | Rango | Descripción |
|----------|------|-------|-------------|
| **Application mode** | Categórica | 18 códigos | Modo de ingreso/aplicación |
| **Application order** | Ordinal | 0-9 | Orden de preferencia (0=primera opción) |
| **Course** | Categórica | 17 códigos | Programa académico |
| **Daytime/evening attendance** | Binaria | 0=Nocturno, 1=Diurno | Turno de asistencia |
| **Previous qualification** | Categórica | 17 códigos | Cualificación previa |
| **Previous qualification (grade)** | Numérica | 95-190 | Nota de cualificación previa |
| **Admission grade** | Numérica | 95-190 | Nota de admisión |

### 4.4 Variables de Rendimiento Académico - 1er Semestre

| Variable | Tipo | Rango | Descripción |
|----------|------|-------|-------------|
| **Curricular units 1st sem (credited)** | Numérica | 0-20 | Unidades acreditadas (convalidadas) |
| **Curricular units 1st sem (enrolled)** | Numérica | 0-26 | Unidades matriculadas |
| **Curricular units 1st sem (evaluations)** | Numérica | 0-45 | Número de evaluaciones presentadas |
| **Curricular units 1st sem (approved)** | Numérica | 0-26 | Unidades aprobadas |
| **Curricular units 1st sem (grade)** | Numérica | 0-18.88 | Nota promedio (escala 0-20) |
| **Curricular units 1st sem (without evaluations)** | Numérica | 0-12 | Unidades sin evaluación |

### 4.5 Variables de Rendimiento Académico - 2do Semestre

| Variable | Tipo | Rango | Descripción |
|----------|------|-------|-------------|
| **Curricular units 2nd sem (credited)** | Numérica | 0-19 | Unidades acreditadas |
| **Curricular units 2nd sem (enrolled)** | Numérica | 0-23 | Unidades matriculadas |
| **Curricular units 2nd sem (evaluations)** | Numérica | 0-33 | Número de evaluaciones presentadas |
| **Curricular units 2nd sem (approved)** | Numérica | 0-20 | Unidades aprobadas |
| **Curricular units 2nd sem (grade)** | Numérica | 0-18.57 | Nota promedio |
| **Curricular units 2nd sem (without evaluations)** | Numérica | 0-12 | Unidades sin evaluación |

### 4.6 Variables Macroeconómicas

| Variable | Tipo | Rango | Descripción |
|----------|------|-------|-------------|
| **Unemployment rate** | Numérica | 7.6-16.2% | Tasa de desempleo de Portugal |
| **Inflation rate** | Numérica | -0.8 a 3.7% | Tasa de inflación de Portugal |
| **GDP** | Numérica | -4.06 a 3.51% | Variación del PIB de Portugal |

---



