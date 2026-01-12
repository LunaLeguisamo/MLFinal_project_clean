import pandas as pd
import json

"""
Carga los datos y procesa las etiquetas de emociones
mapeándolas a las categorías de Ekman.
"""

def load_split(path):
    """Carga un split de datos desde un archivo TSV."""
    df = pd.read_csv(path, sep="\t", header=None)
    df.columns = ["text", "labels", "id"]
    return df[["text", "labels"]]

"""Cargar splits de datos"""
train = load_split("train.tsv")
dev   = load_split("dev.tsv")
test  = load_split("test.tsv")

"""Mapeo de IDs a emociones GO"""
id2emotion = {
    0: "admiration",
    1: "amusement",
    2: "anger",
    3: "annoyance",
    4: "approval",
    5: "caring",
    6: "confusion",
    7: "curiosity",
    8: "desire",
    9: "disappointment",
    10: "disapproval",
    11: "disgust",
    12: "embarrassment",
    13: "excitement",
    14: "fear",
    15: "gratitude",
    16: "grief",
    17: "joy",
    18: "love",
    19: "nervousness",
    20: "optimism",
    21: "pride",
    22: "realization",
    23: "relief",
    24: "remorse",
    25: "sadness",
    26: "surprise",
    27: "neutral"
}

"""Cargar mapeo de emociones GO a categorías de Ekman"""
with open("ekman_mapping.json", "r") as f:
    ekman_mapping = json.load(f)

"""Crear mapeo inverso de GO a Ekman"""
go2ekman = {}
for ekman, emotions in ekman_mapping.items():
    for e in emotions:
        go2ekman[e] = ekman

"""Procesar datos y mapear etiquetas"""
def parse_label_ids(label_str):
    return [int(x) for x in str(label_str).split(",")]

"""Convertir IDs a emociones GO"""
def ids_to_go_emotions(ids):
    return [id2emotion[i] for i in ids]

"""Mapear emociones GO a categorías de Ekman"""
def map_to_ekman(go_emotions):
    mapped = [go2ekman[e] for e in go_emotions if e in go2ekman]
    if len(mapped) == 0:
        return None
    return mapped[0]  # emoción dominante

"""Procesar un split completo"""
def process_split(df):
    df = df.copy()

    df["label_ids"] = df["labels"].apply(parse_label_ids)
    df["go_emotions"] = df["label_ids"].apply(ids_to_go_emotions)
    df["emotion"] = df["go_emotions"].apply(map_to_ekman)
    df = df.dropna(subset=["emotion"])
    df = df[["text", "emotion"]]
    
    return df

# Procesar splits
train_processed = process_split(train)
dev_processed = process_split(dev)
test_processed = process_split(test)

# Guardar resultados
train_processed.to_csv("train_processed.csv", index=False)
dev_processed.to_csv("dev_processed.csv", index=False)
test_processed.to_csv("test_processed.csv", index=False)

print("Archivos generados:")
print("- train_processed.csv")
print("- dev_processed.csv")
print("- test_processed.csv")

print("\n=== COMPARACIÓN DE TAMAÑOS ===")
print(f"Train original: {len(train)}")
print(f"Train procesado: {len(train_processed)}")

print("\n=== DISTRIBUCIÓN FINAL (TRAIN) ===")
print(train_processed["emotion"].value_counts())
