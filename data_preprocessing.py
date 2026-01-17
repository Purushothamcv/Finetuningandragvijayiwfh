from datasets import load_dataset
import pandas as pd
def load_quotes_dataset():
    print("Loading dataset from Hugging Face...")
    dataset = load_dataset("Abirate/english_quotes")
    df = pd.DataFrame(dataset['train'])
    print(f"Loaded {len(df)} quotes")
    return df
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = ' '.join(text.split())
    return text
def preprocess_quotes(df):
    print("Preprocessing data...")
    df = df.fillna("")
    df['cleaned_quote'] = df['quote'].apply(clean_text)
    df['cleaned_author'] = df['author'].apply(clean_text)
    df['tags_text'] = df['tags'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')
    df['cleaned_tags'] = df['tags_text'].apply(clean_text)
    df['combined_text'] = (
        df['cleaned_quote'] + ' ' +
        df['cleaned_author'] + ' ' +
        df['cleaned_tags']
    )
    print("Preprocessing complete")
    return df
def main():
    df = load_quotes_dataset()
    df = preprocess_quotes(df)
    df.to_csv('quotes_preprocessed.csv', index=False)
    print("Saved preprocessed data to quotes_preprocessed.csv")
    print("\nSample preprocessed quotes:")
    print(df[['quote', 'author', 'cleaned_quote', 'combined_text']].head(3))
if __name__ == "__main__":
    main()
