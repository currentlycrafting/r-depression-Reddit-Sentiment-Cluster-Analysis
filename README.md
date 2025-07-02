# r/Depression Reddit

**r/Depression Reddit Data Analysis** tool that parses Reddit post data from CSV format, performs sentiment analysis using TextBlob, extracts top keywords, and clusters authors based on sentiment polarity, subjectivity, and word count using K-Means. It includes powerful keyword filtering and interactive Plotly visualizations.

--

<img width="1123" alt="Screenshot 2025-07-01 at 10 38 58 PM" src="https://github.com/user-attachments/assets/7e2d81b4-e1c6-4aa3-bc05-c7b56c2ba6d1" />


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
