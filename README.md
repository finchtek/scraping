Recipe Scraper — Sample Deliverable
====================================

Overview
--------
A Python-based pipeline that fetches recipe web pages, strips HTML to the recipe body,
extracts structured data (name, ingredients, steps, prep_time) via deterministic parsing,
validates the output, and writes to both CSV (flattened, one row per ingredient) and JSON
(nested, full recipe structure). Designed to work with static pages; no headless browser
required. Works with zero setup — no API keys or extra services.
# LLM extraction via Gemini is supported internally as an option,
# but the deliverable runs on deterministic parsing alone.

Environment
-----------
- Python 3.11+, no API keys, no proxy, no extra services.
- Key packages: requests, beautifulsoup4 (see requirements.txt)
- Runs out of the box via deterministic parsing; output is in recipes/.

Pipeline Steps
--------------
1. Fetch — Pulls raw HTML for each target URL with a polite delay (1.5s between requests)
   to avoid overloading servers and reduce blocking risk.

2. Extract to text — BeautifulSoup selects the recipe <article> body, discarding navigation,
   headers, footers, and unrelated page content. This also reduces token cost before the LLM.

3. LLM extraction — A single structured prompt ("extract name, ingredients (item/quantity/unit),
   steps, prep_time as JSON") with a fixed schema ensures consistent output across pages.
   LiteLLM proxies Gemini (cheap, confirmed-live route). If the proxy is unavailable, a
   deterministic BeautifulSoup-based fallback extracts ingredients and steps from predictable
   markup (h2 headings, ul/ol lists, meta tags for prep_time).

4. Validate — Cheap deterministic check that required fields exist and quantities parse as
   numbers (not garbage). This is the "deterministic check" step: inexpensive code validation
   before any retry or LLM retry logic.

5. Output — Writes from the same validated data structure to both:
   - CSV (flattened): one row per ingredient, columns = recipe_name, ingredient_name,
     quantity, unit, steps (joined with " | ")
   - JSON (nested): full recipe object with name, ingredients (array of {name, quantity, unit}),
     steps (array of strings), prep_time (string or null)

Sample Output
-------------
CSV (recipes/output.csv):
  recipe_name,ingredient_name,quantity,unit,steps
  30-Minute Chicken Curry,rapeseed or sunflower oil,2.0,tbsp,"..."
  Quick Pasta Carbonara,spaghetti,200.0,g,"..."
  Easy Banana Bread,bananas,3.0,ripe,"..."
  Classic Chocolate Chip Cookies,(1 cup) unsalted butter,225.0,g,"..."
  Lemon Herb Salad,mixed salad leaves (rocket, spinach, lettuce),2.0,cups,"..."

JSON (recipes/output.json — truncated):
{
  "recipes": [
    {
      "name": "30-Minute Chicken Curry",
      "ingredients": [
        {"name": "rapeseed or sunflower oil", "quantity": 2.0, "unit": "tbsp"}
      ],
      "steps": ["...", "..."],
      "prep_time": null
    },
    ...
  ]
}

Quantity Normalization
--------------------
- Quantities are parsed as floats when numeric (e.g., "200g" -> 200.0, "½ tsp" -> 0.5).
- If a unit is purely descriptive (e.g., "ripe" for bananas), it is kept as-is and quantity
  remains a number. The deterministic validator accepts quantity as number or null.
- Mixed units like "1 cup (240ml)" are split: quantity=1, unit="cup".
- When no quantity is discernible, quantity is set to null and the full text is kept in the
  ingredient name.

Model Used (internal option — not needed for the deliverable)
------------------------------------------------------------
- Optional: Gemini via direct API key (model: gemini/gemini-2.0-flash).
- Default: Deterministic BeautifulSoup parsing (no LLM call required).
- Prompt schema is fixed per page, ensuring consistent JSON keys across all recipes.

File Structure
--------------
scraping/
├── scraper.py               # Main pipeline script (zero-setup, no keys)
├── requirements.txt         # Pip dependencies (requests, beautifulsoup4)
├── requirements-llm.txt     # Optional LLM extraction deps (internal use only)
├── .env.example             # Optional Gemini key template (internal use only)
├── test-recipes/            # Test HTML recipe pages (own test posts)
│   ├── recipe-chicken-curry.html
│   ├── recipe-pasta-carbonara.html
│   ├── recipe-banana-bread.html
│   ├── recipe-chocolate-chip-cookies.html
│   └── recipe-lemon-salad.html
├── recipes/
│   ├── output.csv           # Generated CSV (flattened, one row per ingredient)
│   └── output.json          # Generated JSON (nested recipe structure)
└── README.md                # This file

Running the Pipeline (no setup, no keys)
----------------------------------------
pip install -r requirements.txt
python3 scraper.py

Or with custom URLs:
python3 scraper.py "file:///path/to/recipe1.html" "file:///path/to/recipe2.html"

Optional — LLM extraction (internal use only, not needed for the deliverable):
pip install -r requirements-llm.txt
export GEMINI_API_KEY="your-key"  # from https://aistudio.google.com/apikey
python3 scraper.py

License
-------
Prototype/demo only. See individual recipe test files for author/source notes.