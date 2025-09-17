import pandas as pd
from bertopic import BERTopic

def analyze_csv(file_path: str, column_name: str = "body"):
    df = pd.read_csv(file_path)
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in CSV.")
    docs = df[column_name].astype(str).tolist()
    topic_model = BERTopic(verbose=True)
    topics, probs = topic_model.fit_transform(docs)
    return topic_model, topics

def print_topic_summary(topic_model: BERTopic, topics):
    import collections
    counter = collections.Counter(topics)
    print("\nTopic distribution:\n")
    for topic_id, count in counter.most_common():
        if topic_id == -1:
            label = "Outliers"
        else:
            label = f"Topic {topic_id}"
        print(f"{label}: {count} documents")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python analyze.py <csv_file> [column_name]")
        sys.exit(1)
    csv_path = sys.argv[1]
    col = sys.argv[2] if len(sys.argv) > 2 else "body"
    model, topics = analyze_csv(csv_path, col)
    print_topic_summary(model, topics)

    # Optional: show top words for first 10 topics
    print("\nTop words per topic:\n")
    for i in range(10):
        if i == -1:
            continue
        words = model.get_topic(i)
        print(f"Topic {i}:", [w[0] for w in words[:5]])
