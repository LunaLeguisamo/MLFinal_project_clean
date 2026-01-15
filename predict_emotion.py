import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
"""Script para predecir la emoción de un texto dado utilizando un modelo preentrenado."""

# Configurar dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar modelo y tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("./emotion_model")
model.to(device)
model.eval()

# Configuración
MODEL_PATH = "./emotion_model" 
id2label = model.config.id2label
MAX_LENGTH = 128

# Función de predicción
def predict_emotion(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=1).item()

    return id2label[predicted_class_id]

# CLI (Command Line Interface)
if __name__ == "__main__":
    print("Emotion Detection (type 'exit' to quit)\n")

    while True:
        text = input("Texto: ")
        if text.lower() in ["exit", "quit"]:
            break

        emotion = predict_emotion(text)
        print(f"Emotion of the day: {emotion}\n")
