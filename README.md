# Dancing Queen

This repository hosts the **Dancing Queen** project.

## 🚀 Quick Start

1. **Clone the repo**
   ```bash
   git clone https://github.com/JockeOk/DancingQueen.git
   cd DancingQueen
   ```
2. **Create a virtual environment & install dependencies**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # on Windows use `.venv\Scripts\activate`
   pip install -r requirements.txt  # (add bertopic, pandas if you want)
   ```
3. **Run the BERTopic script** against the sample data
   ```bash
   python analyze.py sample_data.csv body
   ```
   The script will:
   * Fit a `BERTopic` model on the `body` column.
   * Print a distribution of topics.
   * Show top words for the first 10 topics.

## 📦 Sample Data

The repository includes **50** sample rows in `sample_data.csv`. Feel free to replace it with your own CSV that contains a text column named `body` (or specify another column when running the script).

## 🧠 About BERTopic

[BERTopic](https://maartengr.github.io/BERTopic/) is an advanced topic modeling library that leverages transformer embeddings and clustering to extract meaningful topics from text.

---

Happy modeling! 🎶