
# Jesse Livermore GenAI Mentor Bot  
## Retrieval-Augmented Chatbot + Breakout Strategy Backtester

This project implements a **GenAI-based chatbot** trained on curated Q&A data about legendary trader **Jesse Livermore**, combined with a **stock backtesting engine** that simulates his breakout-and-trend-following strategy.

The entire system is built using:

- **RAG (Retrieval Augmented Generation)** using FAISS, MiniLM, and FLAN-T5  
- **A Livermore Breakout Strategy Backtest Engine** implemented in Python  
- **Streamlit Web UI** with two interactive tabs:
  -  **Chat with Jesse (RAG chatbot)**
  -  **Backtest Strategy (Hybrid UI with editable parameters)**  

This project is the final submission for the **Generative AI Course Project**.

---

#  Features

### **1. Part I — RAG Chatbot**
- Takes user questions related to:
  - Personal life
  - Trading strategy
  - Timing
  - Risk management
  - Adaptability
  - Psychology  
- Retrieves relevant Q&A from your **624-row dataset**
- Uses **MiniLM embeddings** + **FAISS vector search**
- Generates answers using **FLAN-T5**, rewritten in Jesse Livermore's tone
- Displays:
  - Chat history  
  - Retrieved sources  
  - Clear persona-based responses  

---

### **2. Part II — Backtest Strategy Engine**
Implements Jesse Livermore's breakout approach:

- Buy when:
  - Price breaks above N-day high, and  
  - Price > SMA_short and Price > SMA_long  
- Sell when:
  - Price drops below N-day low  

UI includes **Hybrid Controls**:

| Parameter | Default | Editable? |
|----------|---------|-----------|
| Breakout Window | 20 | ✔ |
| SMA Short | 50 | ✔ |
| SMA Long | 200 | ✔ |

Displays:

- Summary metrics  
- Buy & Hold vs Strategy cumulative returns  
- Recent signals  
- Full dataset preview  

---

#  Project Architecture

The system consists of three major components:

## 1. Retrieval-Augmented Generation (RAG) Layer
- **Embedding Model:** `sentence-transformers/paraphrase-MiniLM-L6-v2`
- **Vector Store:** FAISS (Facebook AI Similarity Search)
- **Generator Model:** `google/flan-t5-large`
- **Custom Prompting:** Generates responses in the style/persona of Jesse Livermore
- **Pipeline Flow:**
  1. Convert all Q&A pairs into embeddings  
  2. FAISS retrieves the top-K most relevant entries  
  3. Retrieved Q&A snippets are placed into a prompt  
  4. FLAN-T5 generates the final answer  

---

## 2. Trading Strategy Backtest Layer
Implements a simplified version of Jesse Livermore’s breakout methodology.

### Indicators Used
- **SMA (Simple Moving Average)**  
- **N-Day High & N-Day Low Breakout Levels**

### Trade Logic
- **Buy When:**
  - Close breaks above *N-day High*, **AND**
  - Price > SMA_short, **AND**
  - Price > SMA_long  

- **Sell When:**
  - Close breaks below *N-day Low*

### Outputs:
- Daily buy/sell **position signals**
- Buy & Hold vs Strategy **daily returns**
- **Cumulative aggregated returns**
- Display of **last 5 rows** and **full dataset**

---

## 3. Streamlit UI Layer
A clean, two-tab interface:

### **Tab 1 — Chat with Jesse**
- Chat interface using `st.chat_message()` and `st.chat_input()`
- Displays conversation history
- Shows sources retrieved by FAISS
- Generates responses via FLAN-T5

### **Tab 2 — Backtest Strategy**
Includes:
- Inputs for:
  - Ticker  
  - Date range  
  - Breakout window  
  - SMA short/long  
- Strategy results:
  - Metrics summary  
  - Line chart (Buy & Hold vs Strategy)  
  - Data table and DF expander  

## Directory Structure
rag-investment-chatbot/
│
├── app.py # Streamlit UI (two-tab interface)
├── requirements.txt
├── README.md
│
├── data/
│ └── Team Livermore.xlsx # 624-row curated Q&A dataset
│
├── src/
│ ├── init.py
│ ├── data_prep.py # Loads Excel sheets into unified DataFrame
│ ├── rag_index.py # Embedding + FAISS retrieval index
│ ├── rag_chatbot.py # RAG pipeline (retrieval + FLAN-T5)
│ └── strategy.py # Livermore breakout strategy implementation
│
└── .venv/ # Python virtual environment

---

# Tech Stack
- **Python 3.12**
- **Streamlit** — web UI framework
- **Transformers + FLAN-T5** — LLM generation
- **Sentence-Transformers (MiniLM)** — semantic embeddings
- **FAISS** — vector search index
- **Pandas / NumPy** — financial data processing
- **yfinance** — stock price retrieval


## Installation & Setup

Follow these steps to run the project from scratch.

### Clone the repository**

- git clone git@github.com:chalapati-rao-neu/rag-investment-chatbot.git
- cd rag-investment-chatbot
- python3 -m venv .venv
- source .venv/bin/activate   # macOS/Linux
- .\.venv\Scripts\activate #Windows
- pip install -r requirements.txt
- python -m streamlit run app.py


# App Usage Guide

### ** Tab 1 — Chat with Jesse**
Ask anything about:
- Psychology  
- Timing  
- Strategy development  
- Risk management  
- Life philosophy  

Bot will:
- Retrieve relevant Q&A rows
- Display sources used
- Generate a persona-based response

---

### ** Tab 2 — Backtest Strategy**
Enter:
- Ticker (e.g., AAPL, TSLA, MSFT)
- Date range
- Breakout window (default = 20)
- SMA short (default = 50)
- SMA long (default = 200)

Click **Run Backtest** to see:
- Summary metrics
- Cumulative returns chart
- Last 5 rows
- Full dataset in expander

## Dataset Description

The dataset contains **624 manually curated Q&A pairs** across 6 labels:

| Label                | Count |
|----------------------|-------|
| Personal Life        | ~112 |
| Strategy Development | ~97  |
| Timing               | ~122 |
| Risk Management      | ~105 |
| Adaptability         | ~98  |
| Psychology           | ~90  |

Excel structure:
- **Questions**
- **Answers**
- **Label**


