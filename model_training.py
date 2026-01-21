import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer, AutoTokenizer
)
from sklearn.metrics import accuracy_score, f1_score
from datasets import load_from_disk

tokenized_train = load_from_disk("data/tokenized_train")
tokenized_dev = load_from_disk("data/tokenized_dev")
tokenized_test = load_from_disk("data/tokenized_test")

"""Entrena un modelo de clasificación de emociones utilizando datos tokenizados.
"""
# Load tokenized 

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="macro")
    }

MODEL_NAME = "distilbert-base-uncased"
NUM_LABELS = 4  # joy, anger, sadness, fear

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=4
)



training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    weight_decay=0.01,
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    report_to="none"  # evita warnings de wandb
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_dev,
    compute_metrics=compute_metrics
)

trainer.train()
test_results = trainer.evaluate(tokenized_test)

print("\n=== RESULTADOS FINALES (TEST) ===")
for k, v in test_results.items():
    print(f"{k}: {v:.4f}")

trainer.save_model("./emotion_model")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model.save_pretrained("./emotion_model")
tokenizer.save_pretrained("./emotion_model")
