# 💄 Dupe Finder: same formula, smaller price

Dupe Finder is an AI application that finds cheaper beauty products whose formula is closest
to a product you love. Pick a product (for example Tatcha The Water Cream, $70) and the app
ranks cheaper products from other brands by how similar their ingredient lists are, explains
which key ingredients they share, and shows how much you save, including price per ounce.

It also works the other way round: paste any ingredient list (a pharmacy brand, a product
seen on TikTok) and the app finds the Sephora products with the closest formulas.

---

## 1. Quick start (no installation, no coding)

**Option A: Google Colab (recommended)**

1. Open `Dupe_Finder.ipynb` in Google Colab: go to [colab.research.google.com](https://colab.research.google.com),
   choose **Upload** (or **GitHub** and paste this repository's link), and select the notebook.
2. Click **Runtime → Run all** (in French: *Exécution → Tout exécuter*).
3. If Colab warns that the notebook was not authored by Google, click **Run anyway**.
4. Wait 1 to 3 minutes. A link ending in `.gradio.live` appears at the bottom: click it.

The notebook installs the libraries, downloads the dataset from Kaggle, and launches the app.
If the Kaggle download fails, it asks you to upload `product_info.csv` (included in this repository).

**Option B: on your own computer**

```bash
pip install -r requirements.txt
python app.py
```
Then open http://127.0.0.1:7860 in your browser. `product_info.csv` must be in the same folder.
---

## 2. User guide

### Tab "Find a dupe"
1. Type a product you love in **Product you love**: a name ("Drunk Elephant Protini"), or a
   description ("vitamin C serum for dark spots"). Press **Search**, or click one of the
   examples under **Try one**.
2. Check that **Choose the exact product** shows the right product; open the list to pick another.
3. Optional: open **Filters** to include more expensive products, other product types or the
   same brand, and to choose how many dupes to show.
4. Click **Find dupes**.

Each result shows:
- **Formula match (%)**: how close the ingredient list is to the original (see section 3).
- A verdict: *Near-twin* (60%+), *Very close* (40–60%), *Similar* (25–40%) or *Loose match*.
- The price, the saving in dollars and percent, and the price per ounce of both products.
- How many ingredients they share, including how many of the original's first 10 ingredients
  (the most concentrated ones).
- Pink tags for shared active ingredients (niacinamide, retinol, hyaluronic acid...).
- The most important shared ingredients. Click the product name to look it up on Sephora.

### Tab "Paste an ingredient list"
1. Paste an ingredient list, with ingredients separated by commas (an example is pre-filled).
2. Optional: set a maximum price and a product type.
3. Click **Find closest formulas**.

### Tab "How it works"
A short explanation of the method, for curious users.

### Good to know
- A high match means similar ingredients in a similar order, not an identical product.
  Concentrations, textures and fragrance are not public, so treat the results as a shortlist to test.
- Prices come from the Sephora US website (data collected in 2023) and may have changed.
- Products without an ingredient list, minis, sets and tools are left out (6,010 products remain).

---

## 3. How it works

| Step | What happens | AI / data-science concept |
|---|---|---|
| Clean | Ingredient labels are split and standardised ("Aqua/Water/Eau" = "Water", "(Rice)" common names removed, "may contain" shades dropped) | Text preprocessing, normalisation |
| Weigh | Each ingredient gets a weight = rarity across all products × position in the list (labels list ingredients by decreasing concentration) × 1.5 for known active ingredients | TF-IDF feature engineering with domain knowledge |
| Compare | Each product becomes a vector; products are compared with cosine similarity | Vector space model, content-based recommender system |
| Search | Products are described by brand, name, category and Sephora highlights, encoded with the Hugging Face model `all-MiniLM-L6-v2`, so users can search in their own words; this is combined with fuzzy name matching | Sentence embeddings (transformer NLP), semantic search |
| Explain | Shared top ingredients, shared actives and price per ounce are shown for each result | Explainable recommendations |

If the Hugging Face model cannot be downloaded (no internet), the app automatically falls back
to keyword search; the dupe matching itself does not need the model.

**Beyond the course content:** the recommender uses domain-specific weighting (ingredient
order as a proxy for concentration) on top of TF-IDF, combines a pretrained transformer model
for semantic search with classic text matching, and explains every recommendation.

---

## 4. Files

| File | Content |
|---|---|
| `Dupe_Finder.ipynb` | One-click Google Colab notebook (installs, downloads data, launches the app) |
| `app.py` | Web interface (Gradio) |
| `dupe_engine.py` | Matching engine: cleaning, weighting, similarity, search |
| `requirements.txt` | Python libraries needed |
| `product_info.csv` | Sephora product data (from Kaggle, see below) |
| `PROMPTS.md` | Prompts used with AI tools during the project |

---

## 5. Data, models and packages used (citations)

- **Data:** Nady Inky (2023). *Sephora Products and Skincare Reviews* [dataset]. Kaggle.
  https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews
  (file `product_info.csv`: 8,494 products scraped from sephora.com in March 2023).
- **Model:** Reimers, N. & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese
  BERT-Networks*. Model `sentence-transformers/all-MiniLM-L6-v2`, Hugging Face.
  https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- **Packages:** Gradio (Abid et al., 2019, https://gradio.app), sentence-transformers,
  scikit-learn (Pedregosa et al., 2011), pandas, NumPy, SciPy, kagglehub.
- **AI assistant:** Claude (Anthropic) was used to brainstorm the idea and write the code;
  see `PROMPTS.md`.
