# r/Depression Reddit

**ClusterCast** is a data analysis tool that parses Reddit post data from CSV format, performs sentiment analysis using TextBlob, extracts top keywords, and clusters authors based on sentiment polarity, subjectivity, and word count using K-Means. It includes powerful keyword filtering and interactive Plotly visualizations.

---

## Features

- Load and parse Reddit data from CSV
- Extract posts by specific authors
- Filter posts by keyword(s)
- Extract top keywords used by any author
- Analyze sentiment (polarity, subjectivity, length)
- Vectorize author sentiment and cluster them with K-Means
- Visualize clusters interactively using Plotly

---

## Dependencies

- `csv`
- `collections.Counter`
- `re`
- `textblob`
- `sklearn`
- `numpy`
- `pandas`
- `plotly`

Install required packages with:

```bash
pip install textblob scikit-learn numpy pandas plotly
