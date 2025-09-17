import pandas as pd
from bertopic import BERTopic

def analyze_csv(file_path: str, column_name: str = "text"):
    df = pd.read_csv(file_path)
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in CSV.")
    docs = df[column_name].astype(str).tolist()
    topic_model = BERTopic(verbose=True)
    topics, probs = topic_model.fit_transform(docs)
    return topic_model, topics

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python analyze.py <csv_file> [column_name]")
        sys.exit(1)
    csv_path = sys.argv[1]
    col = sys.argv[2] if len(sys.argv) > 2 else "text"
    model, topics = analyze_csv(csv_path, col)
    print("Top 10 Topics:")
    for i in range(10):
        print(f"Topic {i}:", model.get_topic(i))
