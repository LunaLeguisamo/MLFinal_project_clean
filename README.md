## Training the model

This project does not include a pre-trained model.
The emotion classifier must be trained locally.

Steps:

1. Install dependencies
   pip install -r requirements.txt

2. Preprocess and tokenize the data
   python tokenize_data.py

3. Train the model
   python model_training.py

4. Run emotion prediction
   python predict_emotion.py
