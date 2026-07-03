# BBC News Text Mining and Processing

🌐 **[English](#english) | [Italiano](#italiano)**

Text mining pipeline for the classification and clustering of BBC News articles / Pipeline di text mining per la classificazione e il clustering di articoli BBC News.

**Authors / Autori**
- Alberto Cera (925269)
- Davide Fabio Loreti (865309)
- Carlo Pegoraro (865329)

Project for the **Data Science** course, **University of Milano-Bicocca** / Progetto per il corso di **Data Science**, **Università degli Studi di Milano-Bicocca**.

---

# English

Text mining pipeline for the classification and clustering of BBC News articles, developed as a project for the **Data Science** course at the **University of Milano-Bicocca**.

The pipeline covers the full workflow from raw text to model evaluation: exploratory data analysis, preprocessing, text representation (TF-IDF and contextual embeddings), supervised classification, and unsupervised clustering.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Pipeline](#pipeline)
- [Notebooks](#notebooks)
- [Results](#results)
  - [Classification](#classification)
  - [Clustering](#clustering)
- [Key Takeaways](#key-takeaways)
- [Requirements](#requirements)
- [How to Run](#how-to-run)
- [Notes](#notes)

## Project Overview

The goal of the project is to build a complete, reproducible text mining pipeline and to compare how different **text representations** affect performance on two very different tasks:

- **Supervised classification** of news articles into topical categories.
- **Unsupervised clustering** of the same articles, without using label information.

Two representation families are compared throughout: **sparse, frequency-based** representations (Bag of Words / TF-IDF) and **dense, contextual embeddings** (Sentence Transformers / BERT-style).

## Dataset

- **Source:** [BBC Full Text Document Classification: Kaggle](https://www.kaggle.com/datasets/shivamkushwaha/bbc-full-text-document-classification)
- **Size:** 2,232 articles
- **Categories (5):** Business, Entertainment, Politics, Sport, Tech
- **Class balance:** reasonably balanced across categories, suitable for both classification and clustering

> Download the dataset from Kaggle and place it under `data/raw/bbc-fulltext/` before running the notebooks.

## Project Structure

```
Text_Mining_Project/
│
├── data/
│   ├── raw/
│   │   └── bbc-fulltext/        # Original BBC News dataset
│   └── processed/               # Processed datasets, vectorizers, embeddings
│
├── notebooks/
│   ├── 01_data_loading_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_text_representation.ipynb
│   ├── 04_text_classification.ipynb
│   └── 05_text_clustering.ipynb
│
├── requirements.txt
└── README.md
```

Each notebook consumes the outputs produced by the previous one, so they must be run in order.

## Pipeline

1. **Data loading & EDA**: load the raw dataset, inspect class distribution and text length.
2. **Preprocessing**: two dedicated pipelines depending on the downstream representation:
   - **Strong preprocessing** (for BoW/TF-IDF): lowercasing, punctuation/number removal, tokenization, stopword removal, lemmatization (spaCy).
   - **Light preprocessing** (for contextual embeddings): newline removal and whitespace cleaning only --> no lowercasing, stopword removal, or lemmatization, to preserve the semantic/contextual information that pretrained language models rely on.
3. **Text representation**:
   - **TF-IDF**: sparse representation, shape `(2232, 6985)`, computed both on the train/test split (classification) and on the full corpus (clustering).
   - **Sentence embeddings**: dense representation from `all-MiniLM-L6-v2` (Sentence Transformers), shape `(2232, 384)`.
4. **Classification**: Logistic Regression, SVM (RBF kernel), and Random Forest, each trained on both TF-IDF and embeddings.
5. **Clustering**: KMeans and Agglomerative Clustering on both representations, with the number of clusters selected via silhouette analysis (k = 2…20).

## Notebooks

### `01_data_loading_eda.ipynb`
Loads the BBC News dataset from the raw folder, performs exploratory data analysis, examines class distribution and basic text statistics, and saves the initial dataframe to the processed folder.

### `02_preprocessing.ipynb`

Loads `df_bbc.pkl` (produced by notebook 01) and builds **two parallel cleaned versions** of the text, stored as new columns, since TF-IDF/BoW and BERT-style embeddings need very different treatment.

**Libraries:** `re`, `string`, `nltk` (stopwords), `spaCy` (`en_core_web_sm`, for lemmatization).

- **`preprocess_for_bow(text)`** — strong normalization for BoW/TF-IDF, column `clean_text_bow`:
  1. remove newlines
  2. lowercase
  3. remove punctuation (regex on `string.punctuation`)
  4. remove digits
  5. collapse extra whitespace
  6. run the text through spaCy and keep only **alphabetic, non-stopword tokens**, replacing each with its **lemma** (`token.lemma_`)
  7. join the resulting tokens back into a string

- **`preprocess_for_bert(text)`**: light/minimal normalization for embeddings, column `clean_text_bert`:
  1. remove newlines
  2. collapse extra whitespace
  3. **no** lowercasing, lemmatization, or stopword removal — the rationale (documented in the notebook) is that aggressive normalization would alter the input distribution the pretrained transformer was trained on, potentially degrading embedding quality.

Both functions are applied to the whole dataframe with `progress_apply` (via `tqdm`). The notebook then:
- computes `bow_length` and `bert_length` (word counts after cleaning) for each document;
- plots boxplots of these lengths by category, to compare how much each pipeline shrinks the text;
- saves the resulting dataframe (`text`, `clean_text_bow`, `clean_text_bert`, `label`, length columns) to **`data/processed/df_bbc_preprocessed.pkl`**.

### `03_text_representation.ipynb`

Loads `df_bbc_preprocessed.pkl` and builds the numerical representations used by the classification and clustering notebooks.

**1. Train/test split for classification**
- `train_test_split` on `clean_text_bow`, `test_size=0.2`, `stratify=y`, `random_state=42` → 1,785 train / 447 test documents, preserving class proportions.

**2. TF-IDF for classification**
- `TfidfVectorizer(max_df=0.8, min_df=5, ngram_range=(1,1))` fitted **only on the training set** and applied to train/test (`X_train_tfidf`, `X_test_tfidf`), to avoid data leakage.
- `max_df=0.8` discards overly frequent (uninformative) terms, `min_df=5` discards very rare terms/typos.

**3. TF-IDF on the full corpus (for clustering)**
- A second `TfidfVectorizer` with the same parameters is fitted on **all 2,232 documents** (`clean_text_bow`), producing `X_all_tfidf` with shape `(2232, 6985)` — used later for unsupervised clustering, where no train/test split makes sense.

**4. Sentence embeddings**
- Model: `all-MiniLM-L6-v2` from `sentence-transformers`.
- Applied to the **lightly-cleaned** `clean_text_bert` column, batch size 32.
- Output: `embeddings_all`, a dense NumPy array of shape `(2232, 384)`.

**5. Saved artifacts** (all under `data/processed/`):

| File | Content |
|---|---|
| `df_bbc_preprocessed_split.pkl` | dataframe with added `is_train` / `is_test` boolean columns |
| `tfidf_vectorizer_clf.joblib` | TF-IDF vectorizer fitted on the training split |
| `tfidf_vectorizer_all.joblib` | TF-IDF vectorizer fitted on the full corpus |
| `X_train_tfidf.joblib` / `X_test_tfidf.joblib` | TF-IDF matrices for classification |
| `X_all_tfidf.joblib` | TF-IDF matrix on the full corpus, for clustering |
| `embeddings_all.npy` | Sentence-Transformer embeddings for the full corpus |

These artifacts are then loaded directly by `04_text_classification.ipynb` (TF-IDF train/test + embeddings) and `05_text_clustering.ipynb` (TF-IDF all + embeddings), so this notebook must be run before either of them.

### `04_text_classification.ipynb`
Trains and evaluates Logistic Regression, SVM, and Random Forest on both TF-IDF and embeddings, using an 80/20 stratified train/test split (`random_state = 42`). Reports Accuracy, Macro F1-score, and confusion matrices for all six configurations.

### `05_text_clustering.ipynb`
Performs unsupervised clustering with KMeans and Agglomerative Clustering on both representations. Selects the optimal number of clusters via silhouette analysis, evaluates results with internal (silhouette) and external (ARI, NMI, V-measure, purity, mapped Macro F1) metrics, and produces PCA visualizations and cluster interpretability analyses (top TF-IDF terms per cluster).

## Results

### Classification

Best configuration: **TF-IDF + SVM**, with TF-IDF outperforming BERT-style embeddings across every classifier tested.

| Embedding | Model | Accuracy | Macro F1 |
|---|---|---|---|
| TF-IDF | SVM | **0.9799** | **0.9799** |
| TF-IDF | Random Forest | 0.9799 | 0.9793 |
| TF-IDF | Logistic Regression | 0.9776 | 0.9774 |
| BERT | SVM | 0.9732 | 0.9727 |
| BERT | Random Forest | 0.9709 | 0.9704 |
| BERT | Logistic Regression | 0.9441 | 0.9435 |

Most misclassifications occur between **Business** and **Politics**, which share economic/political vocabulary. **Sport** is the easiest category to classify.

### Clustering

Best configuration: **Sentence embeddings + KMeans (k = 5)**, which aligns with the true number of BBC categories and captures high-level semantic structure better than TF-IDF.

| Model | k | Silhouette | ARI | NMI | Purity | Macro F1 (mapped) |
|---|---|---|---|---|---|---|
| Embeddings + KMeans | 5 | 0.1258 | 0.9035 | 0.8744 | 0.9592 | 0.9582 |
| Embeddings + Agglomerative (cosine, avg) | 5 | 0.0798 | 0.4995 | 0.6035 | 0.6048 | 0.4763 |
| TF-IDF (SVD200) + Agglomerative (ward) | 16 | 0.0581 | 0.3742 | 0.5670 | 0.8513 | 0.8461 |
| TF-IDF + KMeans | 16 | 0.0403 | 0.2756 | 0.5355 | 0.7939 | 0.8000 |

TF-IDF clustering favors a much larger number of clusters (k = 16), reflecting finer-grained lexical variability rather than the true topical structure.

## Key Takeaways

- Text representation choice has a major impact on both tasks, and **no single representation is universally optimal**.
- **Sparse (TF-IDF) representations excel in supervised classification**, thanks to the distinctive vocabulary of each news category.
- **Dense embeddings excel in unsupervised clustering**, better capturing high-level semantic similarity between documents.
- The best representation is task-dependent, a trade-off between interpretability (TF-IDF) and semantic richness (embeddings).

## Requirements

```
numpy
pandas
scikit-learn
matplotlib
seaborn
joblib
sentence-transformers
tqdm
umap-learn      # optional, for visualization
spacy           # for lemmatization
```

Install with:

```bash
pip install -r requirements.txt
```

The notebooks are designed to run both locally and on Google Colab.

## How to Run

1. Save the project folder to your Google Drive (if running on Colab) and make sure the project root matches the paths used in the notebooks, or clone the repo locally.
2. Install the dependencies listed in `requirements.txt`.
3. Run the notebooks **in order**, since each one depends on the outputs of the previous step:

```
01_data_loading_eda.ipynb
02_preprocessing.ipynb
03_text_representation.ipynb
04_text_classification.ipynb
05_text_clustering.ipynb
```

## Notes

- All file paths are handled programmatically for portability across local and Colab environments.
- Intermediate results (cleaned text, vectorizers, embeddings, model outputs) are stored in `data/processed/`.
- The project is fully reproducible by following the execution order above with the fixed random seed (`random_state = 42`).

[⬆️ Back to top / Torna su](#bbc-news-text-mining--classification--clustering)

---

# Italiano

Pipeline di text mining per la classificazione e il clustering di articoli BBC News, sviluppata come progetto per il corso di **Data Science** all'**Università degli Studi di Milano-Bicocca**.

La pipeline copre l'intero workflow, dal testo grezzo alla valutazione dei modelli: analisi esplorativa dei dati, preprocessing, rappresentazione del testo (TF-IDF ed embedding contestuali), classificazione supervisionata e clustering non supervisionato.

## Indice

- [Panoramica del Progetto](#panoramica-del-progetto)
- [Dataset](#dataset-1)
- [Struttura del Progetto](#struttura-del-progetto)
- [Pipeline](#pipeline-1)
- [Notebook](#notebook)
- [Risultati](#risultati)
  - [Classificazione](#classificazione)
  - [Clustering](#clustering-1)
- [Conclusioni Principali](#conclusioni-principali)
- [Requisiti](#requisiti)
- [Come Eseguire il Progetto](#come-eseguire-il-progetto)
- [Note](#note)

## Panoramica del Progetto

L'obiettivo del progetto è costruire una pipeline di text mining completa e riproducibile, per confrontare come diverse **rappresentazioni testuali** influenzano le prestazioni su due task molto diversi tra loro:

- **Classificazione supervisionata** degli articoli in categorie tematiche.
- **Clustering non supervisionato** degli stessi articoli, senza usare l'informazione sulle etichette.

Vengono confrontate due famiglie di rappresentazioni: quelle **sparse, basate sulla frequenza** (Bag of Words / TF-IDF) e quelle **dense e contestuali** (Sentence Transformers / stile BERT).

## Dataset

- **Fonte:** [BBC Full Text Document Classification: Kaggle](https://www.kaggle.com/datasets/shivamkushwaha/bbc-full-text-document-classification)
- **Dimensione:** 2.232 articoli
- **Categorie (5):** Business, Entertainment, Politics, Sport, Tech
- **Bilanciamento delle classi:** ragionevolmente bilanciato tra le categorie, adatto sia alla classificazione che al clustering

> Scarica il dataset da Kaggle e posizionalo in `data/raw/bbc-fulltext/` prima di eseguire i notebook.

## Struttura del Progetto

```
Text_Mining_Project/
│
├── data/
│   ├── raw/
│   │   └── bbc-fulltext/        # Dataset BBC News originale
│   └── processed/               # Dataset processati, vectorizer, embedding
│
├── notebooks/
│   ├── 01_data_loading_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_text_representation.ipynb
│   ├── 04_text_classification.ipynb
│   └── 05_text_clustering.ipynb
│
├── requirements.txt
└── README.md
```

Ogni notebook utilizza gli output prodotti dal precedente, quindi vanno eseguiti in ordine.

## Pipeline

1. **Caricamento dati & EDA**: caricamento del dataset grezzo, analisi della distribuzione delle classi e della lunghezza dei testi.
2. **Preprocessing**: due pipeline dedicate a seconda della rappresentazione a valle:
   - **Preprocessing forte** (per BoW/TF-IDF): lowercasing, rimozione di punteggiatura/numeri, tokenizzazione, rimozione stopword, lemmatizzazione (spaCy).
   - **Preprocessing leggero** (per gli embedding contestuali): solo rimozione newline e pulizia degli spazi, niente lowercasing, rimozione stopword o lemmatizzazione, per preservare l'informazione semantica/contestuale su cui si basano i modelli linguistici pre-addestrati.
3. **Rappresentazione del testo**:
   - **TF-IDF**: rappresentazione sparsa, shape `(2232, 6985)`, calcolata sia sullo split train/test (classificazione) sia sull'intero corpus (clustering).
   - **Sentence embedding**: rappresentazione densa da `all-MiniLM-L6-v2` (Sentence Transformers), shape `(2232, 384)`.
4. **Classificazione**: Logistic Regression, SVM (kernel RBF) e Random Forest, ciascuno addestrato sia su TF-IDF che su embedding.
5. **Clustering**: KMeans e Agglomerative Clustering su entrambe le rappresentazioni, con il numero di cluster selezionato tramite analisi della silhouette (k = 2…20).

## Notebook

### `01_data_loading_eda.ipynb`
Carica il dataset BBC News dalla cartella raw, esegue l'analisi esplorativa dei dati (EDA), esamina la distribuzione delle classi e le statistiche di base del testo, e salva il dataframe iniziale nella cartella processed.

### `02_preprocessing.ipynb`

Carica `df_bbc.pkl` (prodotto dal notebook 01) e costruisce **due versioni parallele del testo pulito**, salvate come nuove colonne, poiché TF-IDF/BoW e gli embedding stile BERT richiedono trattamenti molto diversi.

**Librerie usate:** `re`, `string`, `nltk` (stopword), `spaCy` (`en_core_web_sm`, per la lemmatizzazione).

- **`preprocess_for_bow(text)`** — normalizzazione forte per BoW/TF-IDF, colonna `clean_text_bow`:
  1. rimozione degli a-capo
  2. lowercase
  3. rimozione della punteggiatura (regex su `string.punctuation`)
  4. rimozione dei numeri
  5. pulizia degli spazi multipli
  6. passaggio del testo in spaCy, mantenendo solo i **token alfabetici e non-stopword**, sostituiti con il rispettivo **lemma** (`token.lemma_`)
  7. ricomposizione dei token in un'unica stringa

- **`preprocess_for_bert(text)`** — normalizzazione leggera/minimale per gli embedding, colonna `clean_text_bert`:
  1. rimozione degli a-capo
  2. pulizia degli spazi multipli
  3. **nessuna** lowercase, lemmatizzazione o rimozione stopword — la motivazione (documentata nel notebook) è che una normalizzazione aggressiva altererebbe la distribuzione di input su cui il transformer pre-addestrato è stato allenato, con un possibile peggioramento della qualità degli embedding.

Entrambe le funzioni vengono applicate all'intero dataframe con `progress_apply` (tramite `tqdm`). Il notebook poi:
- calcola `bow_length` e `bert_length` (conteggio parole dopo la pulizia) per ogni documento;
- traccia dei boxplot di queste lunghezze per categoria, per confrontare quanto ciascuna pipeline riduce il testo;
- salva il dataframe risultante (`text`, `clean_text_bow`, `clean_text_bert`, `label`, colonne di lunghezza) in **`data/processed/df_bbc_preprocessed.pkl`**.

### `03_text_representation.ipynb`

Carica `df_bbc_preprocessed.pkl` e costruisce le rappresentazioni numeriche usate dai notebook di classificazione e clustering.

**1. Split train/test per la classificazione**
- `train_test_split` su `clean_text_bow`, `test_size=0.2`, `stratify=y`, `random_state=42` → 1.785 documenti di train / 447 di test, mantenendo le proporzioni tra classi.

**2. TF-IDF per la classificazione**
- `TfidfVectorizer(max_df=0.8, min_df=5, ngram_range=(1,1))` addestrato **solo sul train set** e poi applicato a train/test (`X_train_tfidf`, `X_test_tfidf`), per evitare data leakage.
- `max_df=0.8` scarta i termini troppo frequenti (poco informativi), `min_df=5` scarta i termini troppo rari/refusi.

**3. TF-IDF sull'intero corpus (per il clustering)**
- Un secondo `TfidfVectorizer` con gli stessi parametri viene addestrato su **tutti i 2.232 documenti** (`clean_text_bow`), producendo `X_all_tfidf` con shape `(2232, 6985)` — usato successivamente per il clustering non supervisionato, dove non ha senso uno split train/test.

**4. Sentence embedding**
- Modello: `all-MiniLM-L6-v2` da `sentence-transformers`.
- Applicato alla colonna **leggermente pulita** `clean_text_bert`, batch size 32.
- Output: `embeddings_all`, un array NumPy denso di shape `(2232, 384)`.

**5. Artefatti salvati** (tutti in `data/processed/`):

| File | Contenuto |
|---|---|
| `df_bbc_preprocessed_split.pkl` | dataframe con le colonne booleane aggiuntive `is_train` / `is_test` |
| `tfidf_vectorizer_clf.joblib` | vectorizer TF-IDF addestrato sullo split di train |
| `tfidf_vectorizer_all.joblib` | vectorizer TF-IDF addestrato sull'intero corpus |
| `X_train_tfidf.joblib` / `X_test_tfidf.joblib` | matrici TF-IDF per la classificazione |
| `X_all_tfidf.joblib` | matrice TF-IDF sull'intero corpus, per il clustering |
| `embeddings_all.npy` | embedding Sentence-Transformer per l'intero corpus |

Questi artefatti vengono poi caricati direttamente da `04_text_classification.ipynb` (TF-IDF train/test + embedding) e `05_text_clustering.ipynb` (TF-IDF all + embedding): questo notebook va quindi eseguito prima di entrambi.

### `04_text_classification.ipynb`
Addestra e valuta Logistic Regression, SVM e Random Forest sia su TF-IDF che su embedding, usando uno split train/test stratificato 80/20 (`random_state = 42`). Riporta Accuracy, Macro F1-score e confusion matrix per tutte e sei le configurazioni.

### `05_text_clustering.ipynb`
Esegue il clustering non supervisionato con KMeans e Agglomerative Clustering su entrambe le rappresentazioni. Seleziona il numero ottimale di cluster tramite analisi della silhouette, valuta i risultati con metriche interne (silhouette) ed esterne (ARI, NMI, V-measure, purity, Macro F1 mappato), e produce visualizzazioni PCA e analisi di interpretabilità dei cluster (termini TF-IDF principali per ciascun cluster).

## Risultati

### Classificazione

Configurazione migliore: **TF-IDF + SVM**, con TF-IDF che supera gli embedding stile BERT in tutti i classificatori testati.

| Embedding | Modello | Accuracy | Macro F1 |
|---|---|---|---|
| TF-IDF | SVM | **0.9799** | **0.9799** |
| TF-IDF | Random Forest | 0.9799 | 0.9793 |
| TF-IDF | Logistic Regression | 0.9776 | 0.9774 |
| BERT | SVM | 0.9732 | 0.9727 |
| BERT | Random Forest | 0.9709 | 0.9704 |
| BERT | Logistic Regression | 0.9441 | 0.9435 |

La maggior parte degli errori di classificazione avviene tra **Business** e **Politics**, categorie che condividono vocabolario economico/politico. **Sport** è la categoria più facile da classificare.

### Clustering

Configurazione migliore: **Sentence embedding + KMeans (k = 5)**, che si allinea al numero reale di categorie BBC e cattura la struttura semantica di alto livello meglio del TF-IDF.

| Modello | k | Silhouette | ARI | NMI | Purity | Macro F1 (mappato) |
|---|---|---|---|---|---|---|
| Embeddings + KMeans | 5 | 0.1258 | 0.9035 | 0.8744 | 0.9592 | 0.9582 |
| Embeddings + Agglomerative (cosine, avg) | 5 | 0.0798 | 0.4995 | 0.6035 | 0.6048 | 0.4763 |
| TF-IDF (SVD200) + Agglomerative (ward) | 16 | 0.0581 | 0.3742 | 0.5670 | 0.8513 | 0.8461 |
| TF-IDF + KMeans | 16 | 0.0403 | 0.2756 | 0.5355 | 0.7939 | 0.8000 |

Il clustering su TF-IDF predilige un numero di cluster molto più alto (k = 16), riflettendo una variabilità lessicale più fine piuttosto che la reale struttura tematica.

## Conclusioni Principali

- La scelta della rappresentazione testuale ha un impatto rilevante su entrambi i task, e **nessuna rappresentazione è universalmente ottimale**.
- **Le rappresentazioni sparse (TF-IDF) eccellono nella classificazione supervisionata**, grazie al vocabolario distintivo di ciascuna categoria di notizie.
- **Gli embedding densi eccellono nel clustering non supervisionato**, catturando meglio la similarità semantica di alto livello tra i documenti.
- La rappresentazione migliore dipende dal task — un compromesso tra interpretabilità (TF-IDF) e ricchezza semantica (embedding).

## Requisiti

```
numpy
pandas
scikit-learn
matplotlib
seaborn
joblib
sentence-transformers
tqdm
umap-learn      # opzionale, per la visualizzazione
spacy           # per la lemmatizzazione
```

Installazione:

```bash
pip install -r requirements.txt
```

I notebook sono progettati per essere eseguiti sia in locale che su Google Colab.

## Come Eseguire il Progetto

1. Salva la cartella del progetto su Google Drive (se usi Colab) e assicurati che la root del progetto corrisponda ai path usati nei notebook, oppure clona la repo in locale.
2. Installa le dipendenze elencate in `requirements.txt`.
3. Esegui i notebook **in ordine**, poiché ciascuno dipende dagli output del passo precedente:

```
01_data_loading_eda.ipynb
02_preprocessing.ipynb
03_text_representation.ipynb
04_text_classification.ipynb
05_text_clustering.ipynb
```

## Note

- Tutti i percorsi dei file sono gestiti programmaticamente per garantire portabilità tra ambiente locale e Colab.
- I risultati intermedi (testo pulito, vectorizer, embedding, output dei modelli) sono salvati in `data/processed/`.
- Il progetto è completamente riproducibile seguendo l'ordine di esecuzione sopra indicato, con seed fisso (`random_state = 42`).

[⬆️ Torna su / Back to top](#bbc-news-text-mining--classification--clustering)
