import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
import os

class BanglaDataPreprocessor:
    def __init__(self):
        pass
        
    def clean_bangla_text(self, text):
        if pd.isna(text):
            return ""
        
        text = str(text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def load_and_preprocess(self, data_path):
        print(f"Loading data from: {data_path}")
        
        # Read CSV with proper encoding
        df = pd.read_csv(data_path, encoding='utf-8')
        print(f"Total data: {len(df)} entries")
        print(f"Columns: {df.columns.tolist()}")
        
        # Show first few rows
        print("First few rows:")
        print(df.head())
        
        # Column names
        text_column = 'text'
        label_column = 'label'
        
        print(f"Text column: {text_column}")
        print(f"Label column: {label_column}")
        
        # Remove rows with NaN values
        df = df.dropna(subset=[text_column, label_column])
        print(f"After dropping NaN: {len(df)} entries")
        
        # Convert to proper types
        texts = df[text_column].astype(str).tolist()
        labels = df[label_column].astype(int).tolist()
        
        print("Cleaning text...")
        cleaned_texts = [self.clean_bangla_text(text) for text in texts]
        
        # Create new dataframe
        clean_df = pd.DataFrame({
            'cleaned_text': cleaned_texts,
            'label': labels
        })
        
        # Remove empty texts
        clean_df = clean_df[clean_df['cleaned_text'].str.len() > 0]
        
        print(f"Label distribution:")
        print(f"Fake news (1): {sum(clean_df['label']==1)} entries")
        print(f"Real news (0): {sum(clean_df['label']==0)} entries")
        
        return clean_df
    
    def split_data(self, df, test_size=0.2, val_size=0.1):
        print("\nSplitting data...")
        
        if len(df) == 0:
            raise ValueError("No data to split!")
        
        # Convert to lists to avoid numpy issues
        X = df['cleaned_text'].tolist()
        y = df['label'].tolist()
        
        print(f"Total samples: {len(X)}")
        print(f"Unique labels: {set(y)}")
        
        # First split into train and temp
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            test_size=(test_size + val_size),
            random_state=42,
            stratify=y
        )
        
        # Split temp into validation and test
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=0.5,
            random_state=42,
            stratify=y_temp
        )
        
        print(f"Train set: {len(X_train)} entries")
        print(f"Validation set: {len(X_val)} entries")
        print(f"Test set: {len(X_test)} entries")
        
        return X_train, X_val, X_test, y_train, y_val, y_test

def main():
    preprocessor = BanglaDataPreprocessor()
    data_path = 'data/cleanbn_fakenews.csv'
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    try:
        if os.path.exists(data_path):
            print(f"\n📁 Processing file: {data_path}")
            df = preprocessor.load_and_preprocess(data_path)
            
            # Check if we have enough data
            if len(df) < 10:
                print(" Not enough data! Using demo data...")
                raise Exception("Insufficient data")
            
            if len(df[df['label']==0]) == 0 or len(df[df['label']==1]) == 0:
                print(" Only one class found! Using demo data...")
                raise Exception("Single class only")
            
            X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(df)
            
        else:
            print(f" File not found: {data_path}")
            raise Exception("File not found")
            
    except Exception as e:
        print(f" Error: {e}")
        print("Creating demo data...")
        
        # Create demo data
        demo_data = {
            'cleaned_text': [
                'বাংলাদেশের প্রধানমন্ত্রী আজ চাঁদে যাবেন',
                'ঢাকায় আজ বৃষ্টি হয়েছে',
                'সকল শিক্ষার্থীকে আগামীকাল বিশ্ববিদ্যালয় ছুটি ঘোষণা',
                'চট্টগ্রামে নতুন ব্রিজ উদ্বোধন করেছেন প্রধানমন্ত্রী',
                'পৃথিবী আজ ধ্বংস হয়ে যাবে',
                'বাংলাদেশ ক্রিকেট দল আজ ম্যাচ জিতেছে',
                'আগামীকাল সারাদেশে ৭২ ঘন্টা কারফিউ জারি',
                'নতুন বছর ২০২৬ সালে বাংলাদেশ হবে উন্নত দেশ',
                'সকল ব্যাংক আগামীকাল বন্ধ থাকবে',
                'ঈদ উপলক্ষে আগামী সপ্তাহে সরকারি ছুটি'
            ],
            'label': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        }
        df = pd.DataFrame(demo_data)
        
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(df)
    
    # Convert to numpy arrays for saving
    processed_data = {
        'X_train': np.array(X_train, dtype=object),
        'X_val': np.array(X_val, dtype=object),
        'X_test': np.array(X_test, dtype=object),
        'y_train': np.array(y_train), 
        'y_val': np.array(y_val),
        'y_test': np.array(y_test)
    }
    
    # Save as numpy file
    np.save('data/processed_data.npy', processed_data)
    print(f"\n Processed data saved to: data/processed_data.npy")
    
    # Print sample
    print("\n Sample training data:")
    for i in range(min(3, len(X_train))):
        print(f"\nText: {X_train[i][:100]}...")
        print(f"Label: {'Fake' if y_train[i]==1 else 'Real'}")

if __name__ == "__main__":
    main()