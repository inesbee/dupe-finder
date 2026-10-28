"""
Dupe Finder - web app (Gradio)
==============================
Run:  python app.py        (then open the link that appears)
In Google Colab the notebook launches it with a public link.
"""

import html
import os
import urllib.parse

import gradio as gr

from dupe_engine import DupeFinder

CSV_PATH = os.environ.get("DUPE_CSV", "product_info.csv")
SHARE = os.environ.get("DUPE_SHARE", "0") == "1"

print("Loading products and building the formula index...")
finder = DupeFinder(CSV_PATH)
df = finder.df
print(f"Ready: {len(df):,} products, search mode = {finder.search_mode}")

CATEGORIES = ["All categories"] + sorted(df["category"].dropna().unique().tolist())

# ---------------------------------------------------------------------------
# Look & feel
# ---------------------------------------------------------------------------
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Bodoni+Moda:opsz,wght@6..96,500;6..96,700&family=Instrument+Sans:wght@400;500;600&display=swap');
:root{
  --df-bg:#FBF7F9; --df-ink:#2B1B2E; --df-muted:#76657A; --df-line:#E9DDE4;
  --df-lip:#B3124A; --df-blush:#F6D3DF; --df-card:#FFFFFF; --df-good:#1F7A55;
}
.gradio-container{background:var(--df-bg)!important;font-family:'Instrument Sans',system-ui,sans-serif!important;color:var(--df-ink);max-width:1060px!important}
#df-head{padding:28px 4px 8px}
#df-head h1{font-family:'Bodoni Moda',serif;font-weight:700;font-size:clamp(40px,7vw,68px);line-height:1;margin:0;letter-spacing:-.01em;color:var(--df-ink)}
#df-head p{font-size:17px;color:var(--df-muted);max-width:60ch;margin:12px 0 0;line-height:1.5}
.df-target,.df-card,.df-empty,.df-note{color:var(--df-ink)!important}
.df-target *,.df-card *{color:inherit}
.df-target{border-left:4px solid var(--df-ink);padding:10px 16px;margin:6px 0 18px;background:var(--df-card)}
.df-target .t-name{font-family:'Bodoni Moda',serif;font-size:22px;font-weight:500}
.df-target .t-meta{color:var(--df-muted);font-size:14px;margin-top:2px}
.df-card{display:grid;grid-template-columns:118px 1fr;gap:18px;background:var(--df-card);border:1px solid var(--df-line);border-radius:14px;padding:18px 20px;margin-bottom:12px}
.df-score{text-align:center;border-right:1px solid var(--df-line);padding-right:14px}
.df-score .n{font-family:'Bodoni Moda',serif;font-size:44px;line-height:1;color:var(--df-lip)!important;font-weight:700}
.df-score .l{font-size:12px;color:var(--df-muted)!important;margin-top:6px;line-height:1.3}
.df-name{font-size:18px;font-weight:600;line-height:1.3}
.df-name a{color:inherit;text-decoration:none;border-bottom:1px solid var(--df-line)}
.df-name a:hover{border-color:var(--df-lip)}
.df-brand{color:var(--df-muted)!important;font-size:14px}
.df-target .t-meta,.df-inci,.df-note{color:var(--df-muted)!important}
.df-price .save{color:var(--df-good)!important}
.df-inci mark{color:var(--df-ink)!important}
.df-price{display:flex;flex-wrap:wrap;gap:14px;align-items:baseline;margin:8px 0 10px;font-size:15px}
.df-price .p{font-weight:600;font-size:20px}
.df-price .save{color:var(--df-good);font-weight:600}
.df-verdict{font-size:14px;margin-bottom:8px}
.df-chips{display:flex;flex-wrap:wrap;gap:6px;margin:6px 0}
.df-chip{background:var(--df-blush);color:var(--df-ink);border-radius:999px;padding:3px 10px;font-size:13px}
.df-inci{font-size:13px;color:var(--df-muted);line-height:1.55;margin-top:6px}
.df-inci mark{background:transparent;color:var(--df-ink);font-weight:600;border-bottom:2px solid var(--df-blush)}
.df-empty{padding:28px;text-align:center;color:var(--df-muted);border:1px dashed var(--df-line);border-radius:14px}
.df-note{font-size:13px;color:var(--df-muted);margin-top:14px;line-height:1.5}
button.primary{background:var(--df-lip)!important;border-color:var(--df-lip)!important}
@media (max-width:640px){.df-card{grid-template-columns:1fr}.df-score{border-right:0;border-bottom:1px solid var(--df-line);padding:0 0 10px;text-align:left;display:flex;gap:12px;align-items:baseline}}
"""

THEME = gr.themes.Soft(primary_hue="pink", neutral_hue="stone",
                       font=[gr.themes.GoogleFont("Instrument Sans"), "system-ui", "sans-serif"])


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------
def esc(x) -> str:
    return html.escape(str(x))


def money(x) -> str:
    return f"${x:,.2f}".replace(".00", "")


def sephora_link(row) -> str:
    q = urllib.parse.quote_plus(f"{row['brand_name']} {row['product_name']}")
    return f"https://www.sephora.com/search?keyword={q}"


def target_html(row) -> str:
    size = f" · {esc(row['size'])}" if isinstance(row["size"], str) else ""
    return (f"<div class='df-target'><div class='t-name'>{esc(row['product_name'])}</div>"
            f"<div class='t-meta'>{esc(row['brand_name'])} · {money(row['price_usd'])}{size} · "
            f"{esc(row['category'])} · {len(row['ingredient_list'])} ingredients</div></div>")


def card_html(m, target_price=None, target_ppo=None) -> str:
    r = m.row
    price_line = f"<span class='p'>{money(r['price_usd'])}</span>"
    if isinstance(r["size"], str):
        price_line += f"<span>{esc(r['size'])}</span>"
    if target_price:
        diff = target_price - r["price_usd"]
        if diff > 0:
            price_line += f"<span class='save'>{money(diff)} cheaper ({diff / target_price:.0%} less)</span>"
    if target_ppo and r["price_per_oz"] == r["price_per_oz"]:   # not NaN
        price_line += f"<span>{money(r['price_per_oz'])}/oz vs {money(target_ppo)}/oz</span>"
    chips = "".join(f"<span class='df-chip'>{esc(s)}</span>" for s in m.shared_stars)
    chips = f"<div class='df-chips'>{chips}</div>" if chips else ""
    shared = ", ".join(f"<mark>{esc(x)}</mark>" for x in m.shared_top)
    return f"""
<div class='df-card'>
  <div class='df-score'><div class='n'>{m.score:.0f}%</div><div class='l'>formula match</div></div>
  <div>
    <div class='df-name'><a href='{sephora_link(r)}' target='_blank' rel='noopener'>{esc(r['product_name'])}</a></div>
    <div class='df-brand'>{esc(r['brand_name'])}</div>
    <div class='df-price'>{price_line}</div>
    <div class='df-verdict'><b>{m.verdict}.</b> Shares {m.n_shared} ingredients, including
      {m.top10_shared} of the original's first 10 (the most concentrated ones).</div>
    {chips}
    <div class='df-inci'>Most important shared ingredients: {shared or 'none'}</div>
  </div>
</div>"""


NOTE = ("<div class='df-note'>The match score compares ingredient lists, weighting rare and "
        "highly concentrated ingredients more. It does not know exact percentages, textures or "
        "fragrances, so read it as a shortlist to test, not a guarantee. Prices are from the "
        "Sephora US catalogue (2023).</div>")


def empty(msg: str) -> str:
    return f"<div class='df-empty'>{esc(msg)}</div>"


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
def option_label(i: int) -> str:
    r = df.loc[i]
    return f"{r['brand_name']} - {r['product_name']} ({money(r['price_usd'])})"


def search(query):
    ids = finder.search_products(query, k=10)
    if not ids:
        return gr.update(choices=[], value=None), empty("Type a product, a brand or what it does.")
    choices = [(option_label(i), int(i)) for i in ids]
    return (gr.update(choices=choices, value=int(ids[0])),
            empty("Check the product picked below, then select Find dupes."))


def find(idx, cheaper, same_cat, other_brands, k):
    if idx is None:
        return empty("Search for a product first, then pick it from the list.")
    idx = int(idx)
    target = df.loc[idx]
    matches = finder.find_dupes(idx, k=int(k), cheaper_only=cheaper,
                                same_category=same_cat, other_brands=other_brands)
    head = target_html(target)
    if not matches:
        return head + empty("No cheaper match in this category. Try unticking a filter.")
    ppo = target["price_per_oz"] if target["price_per_oz"] == target["price_per_oz"] else None
    cards = "".join(card_html(m, target["price_usd"], ppo) for m in matches)
    return head + cards + NOTE


def from_ingredients(text, max_price, category, k):
    if not text or text.count(",") < 3:
        return empty("Paste a full ingredient list, separated by commas (at least 4 ingredients).")
    matches, ings = finder.find_from_ingredients(text, k=int(k),
                                                 max_price=max_price or None, category=category)
    known = sum(1 for x in ings if x in finder.vocab)
    head = (f"<div class='df-target'><div class='t-name'>Your ingredient list</div>"
            f"<div class='t-meta'>{len(ings)} ingredients read, {known} found in the catalogue</div></div>")
    if not matches:
        return head + empty("Nothing matches these filters. Raise the price limit or pick All categories.")
    return head + "".join(card_html(m) for m in matches) + NOTE


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
EXAMPLE_INCI = ("Water, Glycerin, Niacinamide, Butylene Glycol, Pentylene Glycol, Sodium Hyaluronate, "
                "Panthenol, Allantoin, Ceramide NP, Carbomer, Sodium Hydroxide, Phenoxyethanol, "
                "Ethylhexylglycerin")

with gr.Blocks(title="Dupe Finder") as demo:
    gr.HTML("<div id='df-head'><h1>Dupe Finder</h1>"
            "<p>Same formula, smaller price. Pick a beauty product you love and see which cheaper "
            f"products share its ingredients. {len(df):,} Sephora products compared by AI.</p></div>")

    with gr.Tab("Find a dupe"):
        with gr.Row():
            query = gr.Textbox(label="Product you love",
                               placeholder="e.g. Tatcha Water Cream, or 'vitamin C serum for dark spots'",
                               scale=4)
            search_btn = gr.Button("Search", scale=1)
        pick = gr.Dropdown(label="Choose the exact product", choices=[], interactive=True)
        with gr.Accordion("Filters", open=False):
            with gr.Row():
                cheaper = gr.Checkbox(value=True, label="Only cheaper products")
                same_cat = gr.Checkbox(value=True, label="Same type of product")
                other_brands = gr.Checkbox(value=True, label="Other brands only")
                k = gr.Slider(3, 10, value=5, step=1, label="Number of dupes")
        go = gr.Button("Find dupes", variant="primary")
        out = gr.HTML(empty("Start by typing a product you love."))
        gr.Examples([["Tatcha The Water Cream"], ["Drunk Elephant Protini moisturizer"],
                     ["Charlotte Tilbury Magic Cream Moisturizer"], ["La Mer Moisturizing Soft Cream"],
                     ["Glow Recipe Watermelon Glow AHA"]], inputs=[query], label="Try one",
                    fn=search, outputs=[pick, out], run_on_click=True, cache_examples=False)

        search_btn.click(search, query, [pick, out])
        query.submit(search, query, [pick, out])
        go.click(find, [pick, cheaper, same_cat, other_brands, k], out)

    with gr.Tab("Paste an ingredient list"):
        gr.Markdown("Found a product somewhere else (a pharmacy brand, a gift, a TikTok find)? "
                    "Paste its ingredient list from the box or the brand's website to see which "
                    "Sephora products have the closest formula.")
        inci = gr.Textbox(label="Ingredient list", lines=5, value=EXAMPLE_INCI)
        with gr.Row():
            max_price = gr.Number(label="Maximum price ($, optional)", value=None)
            cat = gr.Dropdown(CATEGORIES, value="All categories", label="Type of product")
            k2 = gr.Slider(3, 10, value=5, step=1, label="Number of results")
        go2 = gr.Button("Find closest formulas", variant="primary")
        out2 = gr.HTML()
        go2.click(from_ingredients, [inci, max_price, cat, k2], out2)

    with gr.Tab("How it works"):
        gr.Markdown(f"""
**1. Read the label.** Every product's ingredient list is cleaned so that the same ingredient
written differently ("Aqua/Water/Eau" and "Water") counts once.

**2. Weigh what matters.** Ingredients are listed from most to least concentrated, so the first
ones weigh more. Rare ingredients (niacinamide, peptides, retinol) weigh more than ingredients
found in almost everything (water, glycerin): this is the TF-IDF idea from text mining. Known
active ingredients get a small extra boost.

**3. Compare formulas.** Each product becomes a vector of weighted ingredients, and products are
compared with cosine similarity, giving the % formula match.

**4. Understand the request.** The search box uses a Hugging Face sentence-embedding model
(all-MiniLM-L6-v2), so you can describe a product in your own words. Current search mode:
*{finder.search_mode}*.

Data: *Sephora Products and Skincare Reviews* (Kaggle, Nady Inky, 2023), {len(df):,} products
with complete ingredient lists after cleaning.
""")

# Always open in light mode, even when the computer is in dark mode
FORCE_LIGHT = """() => {
  const url = new URL(window.location);
  if (url.searchParams.get('__theme') !== 'light') {
    url.searchParams.set('__theme', 'light');
    window.location.href = url.href;
  }
}"""

if __name__ == "__main__":
    demo.launch(share=SHARE, css=CSS, theme=THEME, js=FORCE_LIGHT)
