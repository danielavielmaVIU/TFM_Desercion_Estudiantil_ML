"""
Constantes del dominio - TFM Deserción Estudiantil
===================================================
Este módulo contiene:
- Diccionarios de etiquetas (LABELS)
- Clasificación de variables
- Funciones de utilidad para el dominio
"""

# =============================================================================
# CLASIFICACIÓN DE VARIABLES
# =============================================================================
VARS_BINARIAS = [
    "daytimeevening_attendance",
    "displaced",
    "educational_special_needs",
    "debtor",
    "tuition_fees_up_to_date",
    "gender",
    "scholarship_holder",
    "international"
]

VARS_CATEGORICAS_NOMINALES = [
    "marital_status",
    "application_mode",
    "course",
    "previous_qualification",
    "nacionality",
    "mothers_qualification",
    "fathers_qualification",
    "mothers_occupation",
    "fathers_occupation"
]

VARS_CATEGORICAS_ORDINALES = [
    "application_order"
]

VARS_NUMERICAS = [
    # Continuas
    "previous_qualification_grade",
    "admission_grade",
    "curricular_units_1st_sem_grade",
    "curricular_units_2nd_sem_grade",
    "unemployment_rate",
    "inflation_rate",
    "gdp",
    # Discretas
    "age_at_enrollment",
    "curricular_units_1st_sem_credited",
    "curricular_units_1st_sem_enrolled",
    "curricular_units_1st_sem_evaluations",
    "curricular_units_1st_sem_approved",
    "curricular_units_1st_sem_without_evaluations",
    "curricular_units_2nd_sem_credited",
    "curricular_units_2nd_sem_enrolled",
    "curricular_units_2nd_sem_evaluations",
    "curricular_units_2nd_sem_approved",
    "curricular_units_2nd_sem_without_evaluations"
]

TARGET = ["target"]

TARGET_VALUES = ["Dropout", "Graduate", "Enrolled"]


# =============================================================================
# DICCIONARIOS DE ETIQUETAS (LABELS)
# =============================================================================
LABELS = {
    'marital_status': {
        1: 'Soltero',
        2: 'Casado',
        3: 'Viudo',
        4: 'Divorciado',
        5: 'Unión de hecho',
        6: 'Separado legalmente'
    },
    
    'application_mode': {
        1: '1ra fase - contingente general',
        2: 'Ordenanza nº 612/93',
        5: '1ra fase - contingente especial (Azores)',
        7: 'Titulares otros cursos superiores',
        10: 'Ordenanza nº 854-B/99',
        15: 'Estudiante internacional (bachelor)',
        16: '1ra fase - contingente especial (Madeira)',
        17: '2da fase - contingente general',
        18: '3ra fase - contingente general',
        26: 'Ordenanza nº 533-A/99 ítem b2 (Dif. Plan)',
        27: 'Ordenanza nº 533-A/99 ítem b3 (Otro Inst.)',
        39: 'Mayores de 23 años',
        42: 'Transferencia',
        43: 'Cambio de curso',
        44: 'Titulares técnicos superiores',
        51: 'Cambio institución/curso',
        53: 'Titulares curso técnico superior corto',
        57: 'Cambio institución/curso (Internacional)'
    },
    
    'course': {
        33: 'Tecnologías Producción Biocombustibles',
        171: 'Animación y Diseño Multimedia',
        8014: 'Servicio Social (nocturno)',
        9003: 'Agronomía',
        9070: 'Diseño Comunicación',
        9085: 'Enfermería Veterinaria',
        9119: 'Ingeniería Informática',
        9130: 'Equinicultura',
        9147: 'Gestión',
        9238: 'Servicio Social',
        9254: 'Turismo',
        9500: 'Enfermería',
        9556: 'Higiene Dental',
        9670: 'Gestión Publicidad y Marketing',
        9773: 'Periodismo y Comunicación',
        9853: 'Educación Básica',
        9991: 'Gestión (nocturno)'
    },
    
    'daytimeevening_attendance': {
        0: 'Nocturno',
        1: 'Diurno'
    },
    
    'previous_qualification': {
        1: 'Secundaria',
        2: 'Superior - bachelor',
        3: 'Superior - grado',
        4: 'Superior - máster',
        5: 'Superior - doctorado',
        6: 'Frecuencia ed. superior',
        9: '12º año - no completado',
        10: '11º año - no completado',
        12: 'Otro - 11º año',
        14: '10º año',
        15: '10º año - no completado',
        19: 'Ed. básica 3er ciclo',
        38: 'Ed. básica 2do ciclo',
        39: 'Especialización tecnológica',
        40: 'Superior - grado (1er ciclo)',
        42: 'Técnico superior profesional',
        43: 'Superior - máster (2do ciclo)'
    },
    
    'nacionality': {
        1: 'Portugués',
        2: 'Alemán',
        6: 'Español',
        11: 'Italiano',
        13: 'Neerlandés',
        14: 'Inglés',
        17: 'Lituano',
        21: 'Angoleño',
        22: 'Caboverdiano',
        24: 'Guineano',
        25: 'Mozambiqueño',
        26: 'Santotomense',
        32: 'Turco',
        41: 'Brasileño',
        62: 'Rumano',
        100: 'Moldavo',
        101: 'Mexicano',
        103: 'Ucraniano',
        105: 'Ruso',
        108: 'Cubano',
        109: 'Colombiano'
    },
    
    'mothers_qualification': {
        1: 'Secundaria (12º año)',
        2: 'Superior - Bachelor',
        3: 'Superior - Grado',
        4: 'Superior - Máster',
        5: 'Superior - Doctorado',
        6: 'Frecuencia ed. superior',
        9: '12º año - no completado',
        10: '11º año - no completado',
        11: '7º año (antiguo)',
        12: 'Otro - 11º año',
        14: '10º año',
        18: 'Curso comercio general',
        19: 'Ed. básica 3er ciclo',
        22: 'Curso técnico-profesional',
        26: '7º año escolaridad',
        27: '2do ciclo bachillerato',
        29: '9º año - no completado',
        30: '8º año escolaridad',
        34: 'Desconocido',
        35: 'No sabe leer/escribir',
        36: 'Lee sin 4º año',
        37: 'Ed. básica 1er ciclo',
        38: 'Ed. básica 2do ciclo',
        39: 'Especialización tecnológica',
        40: 'Superior - grado (1er ciclo)',
        41: 'Estudios superiores especializados',
        42: 'Técnico superior profesional',
        43: 'Superior - Máster (2do ciclo)',
        44: 'Superior - Doctorado (3er ciclo)'
    },
    
    'fathers_qualification': {
        1: 'Secundaria (12º año)',
        2: 'Superior - Bachelor',
        3: 'Superior - Grado',
        4: 'Superior - Máster',
        5: 'Superior - Doctorado',
        6: 'Frecuencia ed. superior',
        9: '12º año - no completado',
        10: '11º año - no completado',
        11: '7º año (antiguo)',
        12: 'Otro - 11º año',
        13: '2nd year complementary high school course',
        14: '10º año',
        18: 'Curso comercio general',
        19: 'Ed. básica 3er ciclo',
        20: 'Complementary High School Course',
        22: 'Curso técnico-profesional',
        25: 'Complementary High School Course - not concluded',
        26: '7º año escolaridad',
        27: '2do ciclo bachillerato',
        29: '9º año - no completado',
        30: '8º año escolaridad',
        31: 'General Course of Administration and Commerce',
        33: 'Supplementary Accounting and Administration',
        34: 'Desconocido',
        35: 'No sabe leer/escribir',
        36: 'Lee sin 4º año',
        37: 'Ed. básica 1er ciclo',
        38: 'Ed. básica 2do ciclo',
        39: 'Especialización tecnológica',
        40: 'Superior - grado (1er ciclo)',
        41: 'Estudios superiores especializados',
        42: 'Técnico superior profesional',
        43: 'Superior - Máster (2do ciclo)',
        44: 'Superior - Doctorado (3er ciclo)'
    },
    
    'mothers_occupation': {
        0: 'Estudiante',
        1: 'Directivos/Ejecutivos',
        2: 'Especialistas intelectuales',
        3: 'Técnicos nivel intermedio',
        4: 'Personal administrativo',
        5: 'Servicios/Seguridad/Vendedores',
        6: 'Agricultores/Pesca/Forestal',
        7: 'Trabajadores industria/construcción',
        8: 'Operadores máquinas',
        9: 'Trabajadores no cualificados',
        10: 'Fuerzas armadas',
        90: 'Otra situación',
        99: '(en blanco)',
        122: 'Profesionales de salud',
        123: 'Profesores',
        125: 'Especialistas TIC',
        131: 'Técnicos ciencia/ingeniería',
        132: 'Técnicos/profesionales salud',
        134: 'Técnicos servicios legales/sociales',
        141: 'Oficinistas/secretarios',
        143: 'Operadores datos/contabilidad',
        144: 'Otro personal apoyo administrativo',
        151: 'Trabajadores servicios personales',
        152: 'Vendedores',
        153: 'Trabajadores cuidado personal',
        171: 'Trabajadores construcción cualificados',
        173: 'Trabajadores imprenta/joyería',
        175: 'Trabajadores alimentación/madera',
        191: 'Trabajadores limpieza',
        192: 'Trabajadores no cualif. agricultura',
        193: 'Trabajadores no cualif. industria',
        194: 'Ayudantes preparación comidas'
    },
    
    'fathers_occupation': {
        0: 'Estudiante',
        1: 'Directivos/Ejecutivos',
        2: 'Especialistas intelectuales',
        3: 'Técnicos nivel intermedio',
        4: 'Personal administrativo',
        5: 'Servicios/Seguridad/Vendedores',
        6: 'Agricultores/Pesca/Forestal',
        7: 'Trabajadores industria/construcción',
        8: 'Operadores máquinas',
        9: 'Trabajadores no cualificados',
        10: 'Fuerzas armadas',
        90: 'Otra situación',
        99: '(en blanco)',
        101: 'Oficiales fuerzas armadas',
        102: 'Sargentos fuerzas armadas',
        103: 'Otro personal fuerzas armadas',
        112: 'Directores servicios admin/comerciales',
        114: 'Directores hotel/restauración',
        121: 'Especialistas ciencias físicas/matemáticas',
        122: 'Profesionales de salud',
        123: 'Profesores',
        124: 'Especialistas finanzas/contabilidad',
        131: 'Técnicos ciencia/ingeniería',
        132: 'Técnicos/profesionales salud',
        134: 'Técnicos servicios legales/sociales',
        135: 'Técnicos TIC',
        141: 'Oficinistas/secretarios',
        143: 'Operadores datos/contabilidad',
        144: 'Otro personal apoyo administrativo',
        151: 'Trabajadores servicios personales',
        152: 'Vendedores',
        153: 'Trabajadores cuidado personal',
        154: 'Personal protección/seguridad',
        161: 'Agricultores orientados al mercado',
        163: 'Agricultores/ganaderos subsistencia',
        171: 'Trabajadores construcción cualificados',
        172: 'Trabajadores metalurgia cualificados',
        174: 'Trabajadores electricidad/electrónica',
        175: 'Trabajadores alimentación/madera',
        181: 'Operadores planta fija/máquinas',
        182: 'Trabajadores ensamblaje',
        183: 'Conductores vehículos',
        192: 'Trabajadores no cualif. agricultura',
        193: 'Trabajadores no cualif. industria',
        194: 'Ayudantes preparación comidas',
        195: 'Vendedores ambulantes'
    },
    
    'application_order': {
        0: '1ra opción',
        1: '2da opción',
        2: '3ra opción',
        3: '4ta opción',
        4: '5ta opción',
        5: '6ta opción',
        6: '7ma opción',
        7: '8va opción',
        8: '9na opción',
        9: 'Última opción'
    },
    
    'displaced': {
        0: 'No',
        1: 'Sí'
    },
    
    'educational_special_needs': {
        0: 'No',
        1: 'Sí'
    },
    
    'debtor': {
        0: 'No',
        1: 'Sí'
    },
    
    'tuition_fees_up_to_date': {
        0: 'No',
        1: 'Sí'
    },
    
    'gender': {
        0: 'Femenino',
        1: 'Masculino'
    },
    
    'scholarship_holder': {
        0: 'No',
        1: 'Sí'
    },
    
    'international': {
        0: 'No',
        1: 'Sí'
    }
}

# =============================================================================
# DICCIONARIOS DE ETIQUETAS POST PRE-PROCESAMIENTO(LABELS)
# =============================================================================
LABELS_POSTPROCESAMIENTO = {
      'course': {
        33: 'Tecnologías Producción Biocombustibles',
        171: 'Animación y Diseño Multimedia',
        8014: 'Servicio Social (nocturno)',
        9003: 'Agronomía',
        9070: 'Diseño Comunicación',
        9085: 'Enfermería Veterinaria',
        9119: 'Ingeniería Informática',
        9130: 'Equinicultura',
        9147: 'Gestión',
        9238: 'Servicio Social',
        9254: 'Turismo',
        9500: 'Enfermería',
        9556: 'Higiene Dental',
        9670: 'Gestión Publicidad y Marketing',
        9773: 'Periodismo y Comunicación',
        9853: 'Educación Básica',
        9991: 'Gestión (nocturno)'
    },
    
    'daytimeevening_attendance': {
        0: 'Nocturno',
        1: 'Diurno'
    },
	'application_order': {
        0: '1ra opción',
        1: '2da opción',
        2: '3ra opción',
        3: '4ta opción',
        4: '5ta opción',
        5: '6ta opción',
        6: '7ma opción',
        7: '8va opción',
        8: '9na opción',
        9: 'Última opción'
    },
    
    'displaced': {
        0: 'No',
        1: 'Sí'
    },
    
    'educational_special_needs': {
        0: 'No',
        1: 'Sí'
    },
    
    'debtor': {
        0: 'No',
        1: 'Sí'
    },
    
    'tuition_fees_up_to_date': {
        0: 'No',
        1: 'Sí'
    },
    
    'gender': {
        0: 'Femenino',
        1: 'Masculino'
    },
    
    'scholarship_holder': {
        0: 'No',
        1: 'Sí'
    },
    
    'international': {
        0: 'No',
        1: 'Sí'
    }
}


# =============================================================================
# GUARDAR CLASIFICACIÓN
# =============================================================================
VARIABLE_TYPES = {
    'numeric': VARS_NUMERICAS,
    'binary': VARS_BINARIAS,
    'categorical_nominal': VARS_CATEGORICAS_NOMINALES,
    'ordinal': VARS_CATEGORICAS_ORDINALES,
    'target': TARGET
}



# =============================================================================
# FUNCIONES DE UTILIDAD
# =============================================================================
def get_label(column: str, value: int) -> str:
    """
    Obtiene la etiqueta legible para un valor codificado.
    
    Example:
        >>> get_label('marital_status', 1)
        'Soltero'
    """
    if column in LABELS and value in LABELS[column]:
        return LABELS[column][value]
    return str(value)


def get_all_categorical_columns() -> list:
    """Retorna todas las columnas categóricas."""
    return VARS_BINARIAS + VARS_CATEGORICAS_NOMINALES + VARS_CATEGORICAS_ORDINALES


def get_all_numeric_columns() -> list:
    """Retorna todas las columnas numéricas."""
    return VARS_NUMERICAS