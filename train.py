import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import os
import json

class BanglaFakeNewsTrainer:
    def __init__(self, model_name="bert-base-multilingual-cased"):
        print(f"Loading model: {model_name}")
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=2,
            ignore_mismatched_sizes=True
        )
        self.model.to(self.device)
        
    def tokenize_data(self, texts):
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
    
    def create_dataset(self, encodings, labels):
        class BanglaDataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels
                
            def __getitem__(self, idx):
                item = {key: val[idx] for key, val in self.encodings.items()}
                item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
                return item
                
            def __len__(self):
                return len(self.labels)
        
        return BanglaDataset(encodings, labels)
    
    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
        acc = accuracy_score(labels, predictions)
        cm = confusion_matrix(labels, predictions)
        
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def train(self, X_train, y_train, X_val, y_val, output_dir='models', num_epochs=5):
        print("\n" + "="*50)
        print("Starting model training...")
        print("="*50)
        
        # Convert to list if numpy array
        if isinstance(X_train, np.ndarray):
            X_train = X_train.tolist()
        if isinstance(X_val, np.ndarray):
            X_val = X_val.tolist()
        
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        
        print("Tokenizing training data...")
        train_encodings = self.tokenize_data(X_train)
        val_encodings = self.tokenize_data(X_val)
        
        train_dataset = self.create_dataset(train_encodings, y_train)
        val_dataset = self.create_dataset(val_encodings, y_val)
        
        training_args = TrainingArguments(
            output_dir=os.path.join(output_dir, 'checkpoints'),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=os.path.join(output_dir, 'logs'),
            logging_steps=50,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_accuracy",
            greater_is_better=True,
            save_total_limit=2,
            report_to="none",
            learning_rate=2e-5,
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
        )
        
        print("\nTraining started... This may take a while...")
        trainer.train()
        
        # Save model
        model_save_path = os.path.join(output_dir, 'final_model')
        os.makedirs(model_save_path, exist_ok=True)
        
        self.model.save_pretrained(model_save_path)
        self.tokenizer.save_pretrained(model_save_path)
        
        print(f"\n Model saved to: {model_save_path}")
        
        return trainer
    
    def evaluate(self, trainer, X_test, y_test):
        print("\n" + "="*50)
        print("Evaluating model...")
        print("="*50)
        
        if isinstance(X_test, np.ndarray):
            X_test = X_test.tolist()
            
        print(f"Test samples: {len(X_test)}")
            
        test_encodings = self.tokenize_data(X_test)
        test_dataset = self.create_dataset(test_encodings, y_test)
        
        results = trainer.evaluate(test_dataset)
        
        print("\n Evaluation results:")
        for key, value in results.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
        
        return results

def main():
    # Load processed data from numpy file
    data_path = 'data/processed_data.npy'
    
    if os.path.exists(data_path):
        print("Loading processed data...")
        processed_data = np.load(data_path, allow_pickle=True).item()
        
        X_train = processed_data['X_train']
        X_val = processed_data['X_val']
        X_test = processed_data['X_test']
        y_train = processed_data['y_train']
        y_val = processed_data['y_val']
        y_test = processed_data['y_test']
        
        print(f"\n Data split:")
        print(f"Train: {len(X_train)} samples")
        print(f"Validation: {len(X_val)} samples")
        print(f"Test: {len(X_test)} samples")
        
        # Check label distribution
        print(f"\nLabel distribution:")
        print(f"Train - Real: {sum(y_train==0)}, Fake: {sum(y_train==1)}")
        print(f"Val - Real: {sum(y_val==0)}, Fake: {sum(y_val==1)}")
        print(f"Test - Real: {sum(y_test==0)}, Fake: {sum(y_test==1)}")
        
    else:
        print(" Processed data not found!")
        print("Please run: python src/preprocess.py first")
        return
    
    # Initialize trainer
    trainer_class = BanglaFakeNewsTrainer()
    
    # Train model
    trainer = trainer_class.train(X_train, y_train, X_val, y_val, num_epochs=5)
    
    # Evaluate model
    results = trainer_class.evaluate(trainer, X_test, y_test)
    
    # Save results
    results_path = 'models/evaluation_results.json'
    os.makedirs('models', exist_ok=True)
    
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    
    print(f"\n Evaluation results saved to: {results_path}")
    print("\n Training completed successfully!")

if __name__ == "__main__":
    main()