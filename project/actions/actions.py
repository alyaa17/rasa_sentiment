# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import numpy as np
import torch
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, GPT2Config


class ActionDefinePhrase(Action):
    def name(self) -> Text:
        return "action_define_phrase"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        def load_model(model_bin_path, model_name="sberbank-ai/rugpt3small_based_on_gpt2"):
            tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            config = GPT2Config.from_pretrained(model_name, num_labels=3)
            model = GPT2ForSequenceClassification(config)
            model.load_state_dict(torch.load(model_bin_path, map_location=torch.device('cpu')))
            return model, tokenizer

        def preprocess_text(text):
            text = text.lower()
            return text

        def predict_sentiment(text, model, tokenizer):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            model.eval()

            processed_text = preprocess_text(text)
            inputs = tokenizer.encode(processed_text, return_tensors="pt", max_length=512, truncation=True,
                                      padding=True)
            inputs = inputs.to(device)

            with torch.no_grad():
                outputs = model(inputs)
                predictions = torch.softmax(outputs.logits, dim=-1)
            return predictions

        def main():
            model_path = '../rugpt3small_based_on_gpt2.bin'
            model, tokenizer = load_model(model_path)

            text = tracker.latest_message.get("text")

            sentiment_labels = ['Neutral', 'Positive', 'Negative']
            predictions = predict_sentiment(text, model, tokenizer)
            predicted_sentiment = sentiment_labels[np.argmax(predictions.cpu().numpy())]

            if predicted_sentiment == 'Positive':
                result = "положительная"
            elif predicted_sentiment == 'Neutral':
                result = "нейтральная"
            else:
                result = "негативная"
            dispatcher.utter_message(text=f"Я распознал фразу: {result}")

        main()

        return []
