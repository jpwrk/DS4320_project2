# DS 4320 Project 2: Detecting AI Generated Amazon Product Reviews

Executive Summary:

Name:

NetID:

DOI:

Press Release: https://myuva-my.sharepoint.com/:t:/g/personal/cqb3tc_virginia_edu/IQDlY9Na6WWwSrwoMR1eTHWuAWzJHlphDbIilZiiC27-g7c?e=f4Re2W

Pipeline:

License:

## Problem Definition

General Problem: Detecting AI-generated text

Specific Problem: Detecting AI-generated fake product reviews on e-commerce platforms to help consumers and platforms identify inauthentic content and make more trustworthy purchasing decisions.

Online product reviews are one of the most influential factors in purchasing decisions. However, the rise of large language models has made it easy to generate hundreds of convincing fake reviews at scale to manipulate consumers. This can cause consumers to waste money on faulty products, and therefore trust in online platforms erodes. Building a system that can reliably detect AI-generated reviews is therefore both a consumer protection issue and a platform integrity issue.

The general problem of detecting AI-generated text is broad and applies to many domains including academic essays, news articles, and social media posts. We refined our focus to product reviews specifically because they represent a high-stakes, high-volume, and commercially motivated use case where fake content is already known to be widespread and have a large impact. Product reviews also have a natural document structure as each review typically contains metadata like rating, date, product category, and review text, which makes it an ideal fit for the document model in MongoDB!

AI is Flooding Shopping Sites with Fake Reviews! {LINK}

## Domain Exposition


| Term | Definition |
|------|------------|
| Large Language Model (LLM) | An AI system trained on large amounts of text data that can generate human-like text (e.g. ChatGPT, Claude) |
| AI-generated text | Text produced by an LLM rather than a human author |
| Fake review | A product review that is inauthentic, either AI-generated, paid for, or otherwise deceptive |
| Sentiment analysis | NLP technique for identifying the emotional tone of a piece of text (positive, negative, neutral) |
| Perplexity | A measure of how "surprising" text is to a language model — AI text tends to have low perplexity |
| Burstiness | The variation in sentence length and structure — human writing tends to be more "bursty" than AI writing |
| Review bombing | Coordinated posting of fake reviews to artificially inflate or tank a product's rating |
| Verified purchase | A platform label indicating the reviewer actually bought the product — a signal of authenticity |
| TF-IDF | Term Frequency-Inverse Document Frequency — a metric used to evaluate how important a word is in a document |
| Ground truth | A labeled dataset where the correct answer (human vs AI) is already known, used to train and evaluate models |
| Precision | The proportion of flagged reviews that are actually fake — measures false positive rate |
| Recall | The proportion of actual fake reviews that were successfully detected — measures false negative rate |
| F1 Score | A combined metric balancing precision and recall, commonly used to evaluate classification models |
| NLP | Natural Language Processing — the field of AI concerned with understanding and generating human language |
| Feature extraction | The process of pulling meaningful signals (e.g. word patterns, punctuation use) from raw text for model input |

Online shopping has become one of the primary ways people decide what to buy, and product reviews are a huge part of that. The problem is that AI tools like ChatGPT have made it increasingly easy to generate fake reviews, and businesses have been caught doing this to boost profit. This project sits at the crossroads of AI, e-commerce, and fraud detection. Solving this matters for everyday shoppers who just want to know if a product is actually good, for honest sellers who are being undercut by competitors cheating the system, and for platforms like Amazon that have a responsibility to keep their marketplaces trustworthy.

Background Readings: https://myuva-my.sharepoint.com/:f:/g/personal/cqb3tc_virginia_edu/IgB1OF9D4x85TKbckAlYNOahARWHV9UlHz1elna-uluEN4g?e=cqZ7rU


| # | Title | Brief Description | Link |
|---|-------|-------------------|------|
| 1 | AI Generated Text Detection | Compares three models (Logistic Regression, BiLSTM, DistilBERT) for detecting AI vs human text using topic-based splitting to prevent data leakage. DistilBERT achieved the best ROC-AUC of 0.96. | [Link](https://myuva-my.sharepoint.com/:b:/g/personal/cqb3tc_virginia_edu/IQAkBuyZasSMTLWCtRiL1fkOAScDqEm-HXmwHyV3i3gPuFg?e=CLDRmf) |
| 2 | Detecting AI-Generated Text with Pre-Trained Models using Linguistic Features | Uses linguistic features like perplexity, burstiness, and readability scores combined with transformer models on the HC3 dataset. RoBERTa achieved 99.73% accuracy. | [Link](https://myuva-my.sharepoint.com/:b:/g/personal/cqb3tc_virginia_edu/IQBMeZXkMkvDTpkXqRAIJ7YbAUu-SMuHNENSQNrSfcc1cp0?e=gBDPts) |
| 3 | Detecting AI Generated Text Using Neural Networks | A graduate thesis and systematic literature review of 50 papers on synthetic text detection, plus an experiment on mutation-based adversarial attacks that can fool detectors. | [Link](https://myuva-my.sharepoint.com/:b:/g/personal/cqb3tc_virginia_edu/IQDH0fhQUNLqTJg-QWf_uJ2ZARvazZebdz3UGSKIuoJbXog?e=PwWXDp) |
| 4 | Detecting AI-Generated Texts in Cross-Domains | Proposes RoBERTa-Ranker, a fine-tunable model that detects AI text across different domains with minimal labeled data, outperforming GPTZero and DetectGPT. | [Link](https://myuva-my.sharepoint.com/:b:/g/personal/cqb3tc_virginia_edu/IQAmrAlJ6CTVSbbdIt46Qu2yATDMQK05sHSPO8SZTReZF3A?e=paTVOK) |
| 5 | A Practical Synthesis of Detecting AI-Generated Textual, Visual, and Audio Content | Broad survey covering detection methods across text, image, and audio modalities, including watermarking, ensemble approaches, and real-world case studies. | [Link](https://myuva-my.sharepoint.com/:b:/g/personal/cqb3tc_virginia_edu/IQCL2eWlfczLRb0iNysdnVlPAQJtFFnuYaIHwqaJXqbonBc?e=r39v6K) |


## Data Creation

The dataset used in this project is the "Fake Reviews Dataset" published on Kaggle by the user mexwell, sourced from a 2022 research study by Sasikanth et al. The dataset was derived from Amazon Review Data (2018), a large publicly available collection of real product reviews scraped from Amazon across multiple categories. The fake (computer-generated) reviews were synthetically generated to mimic authentic reviews, then labeled accordingly. The dataset contains 40,000 total reviews — 20,000 labeled as OR (Original/human-written) and 20,000 labeled as CG (Computer-Generated/fake), making it a balanced binary classification dataset. It covers the top 10 Amazon product categories to ensure broad representativeness across consumer goods.
The data was downloaded directly from Kaggle at https://www.kaggle.com/datasets/mexwell/fake-reviews-dataset and loaded into MongoDB Atlas as a document database for this project. Each row in the original CSV file was converted into a MongoDB document using Python and pymongo, with each document representing a single product review. No additional scraping, cleaning beyond null removal, or augmentation was performed on the data. The dataset was already sanitized for public research use by its original creators, making it appropriate for academic analysis without additional ethical review.


| File | Description | Link |
|------|-------------|------|
| load_data.ipynb | Loads the CSV dataset, converts rows to MongoDB documents, and inserts them into Atlas | [Link](https://github.com/jpwrk/DS4320_project2/blob/main/load_data.ipynb) |

The choice to use the mexwell Kaggle dataset was made because of its size (40,000 reviews), the balanced labeling, the multi-category coverage, and its status as a sanitized public research dataset. This reduces domain-specific overfitting and ethical risk. The most significant source of uncertainty in this project is the unknown generation method used to produce the fake reviews. Since the exact LLM and parameters are undocumented, it is unclear how well the fake reviews in this dataset represent the kinds of AI-generated content being produced today, which could cause the model to underperform on reviews generated by newer systems.

Several sources of bias exist in this dataset. The human-written reviews were sourced exclusively from Amazon, meaning they reflect the writing style, vocabulary, and cultural norms of Amazon's predominantly English-speaking, Western user base, which limits generalizability to other platforms or languages. The computer-generated fake reviews were also produced using a single generation method (that is also not fully documented by the dataset's creators), meaning the fake reviews may not represent the full diversity of modern AI writing styles from newer models like GPT-4 or Claude. Additionally, the dataset is artificially balanced at 50/50 human vs. fake, which does not reflect real-world conditions where fake reviews are actually a much smaller proportion of total reviews. Finally, the dataset only covers the top 10 Amazon product categories, which may not capture writing patterns in niche or specialized product markets.

The artificial 50/50 class balance should be clearly acknowledged when interpreting model accuracy as real-world performance will likely be lower since the model has never been tested on a realistic imbalanced distribution. To partially account for this, evaluation metrics like precision and recall should be prioritized over raw accuracy, as they reveal how the model performs on each class independently. The limitation to Amazon reviews can be mitigated by framing findings narrowly by stating that conclusions apply specifically to e-commerce product review text.


## Metadata

Guidelines:
- Every document must contain all four fields — category, rating, label, and text
- label must be exactly "OR" (Original/human) or "CG" (Computer-Generated/fake) — no other values are valid
- rating must be an integer between 1 and 5 inclusive — decimal ratings are not valid
- text must be a non-empty string — blank review text should be dropped during loading
- category must be one of the 10 Amazon product categories present in the dataset
- No nested documents or arrays are used — all fields are flat key-value pairs


| Property | Value |
|----------|-------|
| Database Name | fake_reviews_db |
| Collection Name | reviews |
| Total Documents | 40,000 |
| Human-Written Reviews (OR) | 20,000 (50%) |
| AI-Generated Reviews (CG) | 20,000 (50%) |
| Number of Features | 4 (excluding _id) |
| Product Categories | 10 |
| Rating Range | 1 – 5 |
| Language | English only |
| Source | Amazon Review Data (2018) via Kaggle |
| Missing Values | None (dropped during loading) |


| Feature | Data Type | Description | Example |
|---------|-----------|-------------|---------|
| _id | ObjectId | Auto-generated unique identifier assigned by MongoDB to each document | ObjectId('64a1f3b2c9e4a12d3f8b4567') |
| category | String | The Amazon product category the review belongs to | "Home_and_Kitchen" |
| rating | Integer | Star rating given by the reviewer on a scale of 1 to 5 | 5 |
| label | String | Classification of the review — OR for human-written, CG for computer-generated | "CG" |
| text | String | The full text body of the product review | "This blender works great and is very easy to clean!" |


| Metric | Value |
|--------|-------|
| Feature | rating |
| Data Type | Integer |
| Range | 1 - 5 |
| Mean (full dataset) | 4.1 |
| Std Deviation | 1.1 |
| Most Common Value | 5 (heavily skewed toward 5-star ratings) |
| Distribution Shape | Left-skewed — majority of reviews are 4 or 5 stars |
| Uncertainty Source 1 | Rating reflects reviewer perception, not objective product quality |
| Uncertainty Source 2 | AI-generated fake reviews tend to cluster at 5 stars, introducing label-correlated bias |
| Uncertainty Source 3 | Amazon's review system allows unverified purchases to leave ratings, reducing authenticity signal |
| Uncertainty Source 4 | Integer rounding — no decimal precision, so fine-grained sentiment is lost |
| Mitigation | Treat rating as a categorical feature (1-5 buckets) rather than continuous to reduce assumptions about its distribution |

