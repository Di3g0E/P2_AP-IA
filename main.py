import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from src.utils.config import DATA_PROCESSED_PATH, MODEL_OUTPUT_PATH
from src.data.preprocessing import preprocess_text
from src.models.classifier import FinancialClassifier

def train(data_path, model_path):
    df = pd.read_csv(data_path)
    df['clean_text'] = df['Description'].apply(preprocess_text)
    
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_text'], df['Area'], test_size=0.2, random_state=42, stratify=df['Area']
    )
    
    model = FinancialClassifier()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    print(f"\nSummary Results -> Accuracy: {report['accuracy']:.4f} | Macro F1: {report['macro avg']['f1-score']:.4f} | Weighted F1: {report['weighted avg']['f1-score']:.4f}\n")
    
    model.save(model_path)
    print(f"Model saved to {model_path}\n")

def evaluate(data_path, model_path):
    model = FinancialClassifier.load(model_path)
    df = pd.read_csv(data_path)
    df['clean_text'] = df['Description'].apply(preprocess_text)
    
    y_pred = model.predict(df['clean_text'])
    if 'Area' in df.columns:
        report = classification_report(df['Area'], y_pred, output_dict=True)
        print(f"\nEvaluation Results -> Accuracy: {report['accuracy']:.4f} | Macro F1: {report['macro avg']['f1-score']:.4f} | Weighted F1: {report['weighted avg']['f1-score']:.4f}\n")
    else:
        print(f"\nNo 'Area' column found. Generated {len(y_pred)} predictions.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Financial Transaction Classification Pipeline")
    parser.add_argument("--mode", type=str, choices=["train", "evaluate"], default="train", help="Mode: train or evaluate")
    parser.add_argument("--data", type=str, default=DATA_PROCESSED_PATH, help="Path to the CSV database")
    parser.add_argument("--model", type=str, default=MODEL_OUTPUT_PATH, help="Path to save/load the model")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        train(args.data, args.model)
    elif args.mode == "evaluate":
        evaluate(args.data, args.model)
