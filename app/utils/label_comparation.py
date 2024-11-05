from typing import List
import unicodedata

def label_comparation(str1: str, str2: str) -> float:
    """
    Calcula el porcentaje de similitud entre dos strings usando distancia de Levenshtein,
    ignorando mayúsculas/minúsculas, tildes y espacios en blanco.

    Args:
        str1: Primera cadena a comparar
        str2: Segunda cadena a comparar

    Returns:
        float: Porcentaje de similitud entre 0 y 100
    """
    def normalize_string(text: str) -> str:
        """
        Normaliza el string: elimina tildes, espacios y convierte a minúsculas
        """
        # Convertir a minúsculas
        text = text.lower()
        # Eliminar espacios en blanco
        text = text.replace(" ", "")
        # Eliminar tildes
        text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')
        return text

    # Normalizar ambos strings
    str1 = normalize_string(str1)
    str2 = normalize_string(str2)

    # Matriz para almacenar los cálculos
    matriz: List[List[int]] = [[0 for x in range(len(str2) + 1)] for x in range(len(str1) + 1)]

    # Inicializar primera fila y columna
    for i in range(len(str1) + 1):
        matriz[i][0] = i
    for j in range(len(str2) + 1):
        matriz[0][j] = j

    # Llenar la matriz
    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            if str1[i - 1] == str2[j - 1]:
                costo: int = 0
            else:
                costo: int = 1
            matriz[i][j] = min(
                matriz[i - 1][j] + 1,  # eliminación
                matriz[i][j - 1] + 1,  # inserción
                matriz[i - 1][j - 1] + costo  # sustitución
            )

    # Calcular la distancia de Levenshtein
    distance: int = matriz[len(str1)][len(str2)]

    # Calcular el porcentaje de similitud
    max_length: int = max(len(str1), len(str2))
    similarity: float = ((max_length - distance) / max_length) * 100

    return round(similarity, 2)

# Ejemplos de uso:
# print(label_comparation("Hola Mundo", "hola mundo"))  # 100.0
# print(label_comparation("México", "mexico"))  # 100.0
# print(label_comparation("Hola  Mundo", "HolaMundo"))  # 100.0