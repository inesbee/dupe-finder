# Prompts used with AI tools

The assignment asks us to report where and how we used AI tools. We used **Claude (Anthropic)**
for brainstorming, coding and documentation. Prompts are reported as written (some were in French).

> Team: add below any prompts you used with ChatGPT, Gemini, Copilot, Canva AI, etc.
> for the slides, script, voiceover or video editing.

## 1. Choosing the idea (Claude)

| Prompt | What Claude produced | How we used it |
|---|---|---|
| *[uploaded the assignment PDF]* "I have this assignment to do (knowing we know nothing about coding)" | Explanation of a no-code-friendly approach (Python + Gradio + Hugging Face, run in Google Colab) and first app ideas | Chose the technical approach |
| "maybe do something related to my field of work" / "I also have a passion for podcasts" | Ideas linked to digital law and to podcasts | Shortlist |
| "the members are [Marketing Management & Digital, beauty chair; Strategy & Management of International Business]" / "boo boring" | Ideas fitting a marketing / beauty / business team: **Dupe Finder**, Pod Me, Lost in Translation | Proposed Dupe Finder to the group, which agreed |
| "should I ask the groupchat if they have any ideas..." / "please offer another text i can send" | Draft messages to coordinate the team and split tasks | Team coordination |

## 2. Building the app (Claude)

| Prompt | What Claude produced |
|---|---|
| "i never used claude code, how does it go?" | Plan: Claude writes and tests the Python code; we run it in Google Colab |
| *[uploaded product_info.csv from Kaggle]* | Data exploration (8,494 products, ingredient format, categories, sizes), then the code: `dupe_engine.py` (ingredient cleaning, TF-IDF × position weighting, cosine similarity, Hugging Face semantic search), `app.py` (Gradio interface), the Colab notebook, this README and user guide |

Claude also tested the app on well-known products (Tatcha Water Cream, Drunk Elephant Protini,
La Mer Moisturizing Soft Cream, Charlotte Tilbury Magic Cream, Olaplex No. 3) and adjusted the
method after looking at the results (fixing how ingredient names with "/" were cleaned, giving
active ingredients a small extra weight, removing duplicate results, improving name search).

## 3. Demo video

| Tool | Prompt | Use |
|---|---|---|
| *(to fill in)* | | |
