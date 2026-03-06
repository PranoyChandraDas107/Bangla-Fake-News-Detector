import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import os

class BanglaFakeNewsPredictor:
    def __init__(self, model_path='models/final_model'):
        print(f"Loading model from: {model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        print(f" Model loaded successfully. Device: {self.device}")
        
    def predict(self, text, return_probability=False):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )
        
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            prediction = torch.argmax(probabilities, dim=-1).item()
        
        if return_probability:
            return prediction, probabilities[0].cpu().numpy()
        return prediction
    
    def predict_batch(self, texts):
        results = []
        for text in texts:
            pred = self.predict(text)
            results.append(pred)
        return np.array(results)
    
    def get_prediction_with_confidence(self, text):
        pred, probs = self.predict(text, return_probability=True)
        
        result = {
            'text': text,
            'prediction': 'Fake News' if pred == 1 else 'Real News',
            'label': pred,
            'confidence': float(probs[pred]),
            'probabilities': {
                'real': float(probs[0]),
                'fake': float(probs[1])
            }
        }
        
        return result
    
    def analyze_news(self, text):
        result = self.get_prediction_with_confidence(text)
        
        confidence = result['confidence']
        if confidence > 0.9:
            confidence_level = "অত্যন্ত উচ্চ"
        elif confidence > 0.7:
            confidence_level = "উচ্চ"
        elif confidence > 0.5:
            confidence_level = "মাঝারি"
        else:
            confidence_level = "নিম্ন"
        
        output = f"""
{'='*60}
 News: {result['text'][:100]}{'...' if len(result['text']) > 100 else ''}

 Analysis Result:
------------------
 Status: {'ভুয়া খবর' if result['label'] == 1 else 'সত্য খবর'}
 Confidence Level: {confidence_level} ({confidence:.2%})

 Detailed Probabilities:
    Real News: {result['probabilities']['real']:.2%}
    Fake News: {result['probabilities']['fake']:.2%}
{'='*60}
        """
        
        return output

def main():
    predictor = BanglaFakeNewsPredictor()
    
    test_news = [
        "বাংলাদেশের প্রধানমন্ত্রী আজ চাঁদে যাবেন",
        "ঢাকায় আজ বৃষ্টি হয়েছে",
        "সকল শিক্ষার্থীকে আগামীকাল বিশ্ববিদ্যালয় ছুটি ঘোষণা",
        "বাংলাদেশ ১৫০ রানে অস্ট্রেলিয়াকে হারিয়েছে",
        "আগামীকাল সারাদেশে ৭২ ঘন্টা কারফিউ জারি"
    ]
    
    print("\n Running test predictions...\n")
    
    for news in test_news:
        result = predictor.analyze_news(news)
        print(result)
        
    print("\n Enter interactive mode (type 'exit' to quit)\n")
    
    while True:
        user_input = input("\nEnter news (type 'exit' to quit): ")
        
        if user_input.lower() == 'exit':
            break
        
        if user_input.strip():
            result = predictor.analyze_news(user_input)
            print(result)

if __name__ == "__main__":
    main()