
# Ultra Simple ArXiv Chatbot

A lightweight and easy-to-use chatbot for exploring ArXiv scientific papers.
No external NLP dependencies — uses simple TF-IDF vectorization to perform searches, summarize abstracts, and answer research questions.

---

## Overview

This Streamlit-based chatbot allows users to:

* Ask questions related to scientific papers in selected categories.
* Search for ArXiv papers based on keywords.
* Get simplified summaries of abstracts.
* Explore basic dataset statistics through visualizations.

It is perfect for researchers, students, or anyone who wants to interactively explore ArXiv papers without setting up complex NLP pipelines.

---

## Features

* Simple TF-IDF based search and ranking of papers.
* Summarizes abstracts without external NLP libraries.
* Interactive chat interface for answering questions.
* Displays dataset statistics with graphs (e.g., publication trends).
* Sample questions to help users get started.

---

## Prerequisites

* Python 3.8 or higher
* Install dependencies:

```bash
pip install streamlit pandas numpy scikit-learn plotly
```

---

## How to Run

1. Place your ArXiv metadata JSON file locally (download from [arXiv OAI snapshot](https://www.kaggle.com/datasets/Cornell-University/arxiv)).

2. Run the chatbot:

```bash
streamlit run app.py
```

3. Open your browser and visit `http://localhost:8501`.

---

## Configuration

* **ArXiv JSON File Path**:
  Specify the full path to the ArXiv dataset in the sidebar.

* **Max Papers to Load**:
  Set a reasonable limit (e.g., 1500 for testing).

* **Category Selection**:
  Available categories:

  * Computer Science (`cs`)
  * Mathematics (`math`)
  * Physics (`physics`)
  * Statistics (`stat`)
  * Quantitative Biology (`q-bio`)
  * Economics (`econ`)
  * Electrical Engineering (`eess`)

---

## Application Workflow

1. **Initialize Chatbot**:
   Load and preprocess selected number of papers from the dataset in the chosen category.

2. **Chat Interface**:

   * Ask general research questions.
   * Sample questions provided for inspiration.

3. **Paper Search**:

   * Search by keywords to find relevant papers.
   * Display top matching results with metadata and abstracts.

4. **Dataset Statistics**:

   * Visualize paper counts by year.
   * See top categories.
   * Sample table of papers for quick browsing.

---

## Example Usage

* Ask: “What is machine learning?”
* Result: The chatbot retrieves top relevant papers and summarizes abstracts to answer the query.

---

## Tips

* Start small (e.g., 1500 papers) for fast initialization.
* Computer Science (`cs`) category has good coverage for experimentation.
* Use specific keywords for better search results.
* Clear browser cache if the interface behaves unexpectedly.

---

## Future Improvements

* Add advanced NLP-based summarization.
* Persistent user sessions with saved queries.
* Better error handling and input validation.
* More visualization options for dataset insights.

---


