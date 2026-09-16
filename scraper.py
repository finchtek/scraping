#!/usr/bin/env python3
"""
Recipe Scraper Pipeline
-----------------------
- Fetches raw HTML from recipe URLs with politeness delays
- Strips HTML to the relevant recipe content block via BeautifulSoup
- Extracts structured data (name, ingredients, steps, prep_time) via LLM
- Validates required fields and quantity parsing
- Writes output to both CSV (flattened, one row per ingredient) and JSON (nested)
"""

import json
import csv
import time
import re
import os
import sys
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any

from bs4 import BeautifulSoup, Tag

import requests

# --- LiteLLM / LLM extraction ---
try:
    import litellm
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# LiteLLM configuration — direct Gemini key, no proxy needed.
# Quick install: export GEMINI_API_KEY="your-key" (or GOOGLE_API_KEY)
# Get a key at https://aistudio.google.com/apikey
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
LITELLM_MODEL = os.getenv("LITELLM_MODEL", "gemini/gemini-2.0-flash")
# Optional: only set this if you really run a LiteLLM proxy. Otherwise leave unset.
LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL", "")

# Politeness: seconds between requests
REQUEST_DELAY = 1.5  # seconds

# Target recipe URLs (5–10 real URLs; skip sites with explicit no-scrape clauses)
# Local test HTML files — own test posts (portable: resolved relative to this file)
def _local_recipe(name: str) -> str:
    from pathlib import Path
    return Path(__file__).parent.joinpath("test-recipes", name).as_uri()


RECIPE_URLS = [
    _local_recipe("recipe-chicken-curry.html"),
    _local_recipe("recipe-pasta-carbonara.html"),
    _local_recipe("recipe-banana-bread.html"),
    _local_recipe("recipe-chocolate-chip-cookies.html"),
    _local_recipe("recipe-lemon-salad.html"),
]

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Ingredient:
    name: str
    quantity: Optional[float] = None
    unit: Optional[str] = None
    # raw_text: str = ""  # keep the original text if needed


@dataclass
class RecipeResult:
    name: str
    ingredients: List[Ingredient]
    steps: List[str]
    prep_time: Optional[str] = None
    # raw_soup_text: str = ""  # optional: keep a snippet


# ---------------------------------------------------------------------------
# 1. Fetch — pull raw HTML with politeness delay
# ---------------------------------------------------------------------------

def fetch_html(url: str, timeout: int = 20) -> str:
    """Fetch raw HTML for a single URL with a small delay (politeness).
    
    Supports both http/https URLs and local file paths (file://).
    """
    global last_request_time
    # Handle local file paths
    if url.startswith("file://"):
        path = url[7:]  # strip file:// prefix
        try:
            with open(path, "r", encoding="utf-8") as f:
                html = f.read()
            last_request_time = time.time()
            return html
        except FileNotFoundError:
            print(f"[WARN] Local file not found: {path}", file=sys.stderr)
            return ""
    
    now = time.time()
    elapsed = now - last_request_time
    if elapsed < REQUEST_DELAY:
        time.sleep(REQUEST_DELAY - elapsed)
    try:
        resp = requests.get(url, timeout=timeout, headers={
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) "
                "AppleWebKit/537.36 "
                "(KHTML, like Gecko) "
                "Chrome/125.0.0.0 Safari/537.36"
            )
        })
        resp.raise_for_status()
        last_request_time = time.time()
        return resp.text
    except requests.RequestException as e:
        print(f"[WARN] Failed to fetch {url}: {e}", file=sys.stderr)
        return ""


last_request_time = 0.0

# ---------------------------------------------------------------------------
# 2. Extract to text — strip HTML down to the relevant content block
# ---------------------------------------------------------------------------

def extract_recipe_body(html: str, url: str) -> Optional[str]:
    """Use BeautifulSoup to select the recipe body, not the whole page.

    Returns the prettified HTML of the recipe container, or None if not found.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Heuristic selectors — try common recipe container IDs/classes in order
    selectors = [
        "article",
        'div[role="main"]',
        'div.recipe',
        'div[class*="recipe"]',
        'main',
        '#recipe',
        '.recipe-content',
        '.entry-content',
    ]

    container = None
    for sel in selectors:
        container = soup.select_one(sel)
        if isinstance(container, Tag) and container.get_text(strip=True):
            break

    if container is None:
        # fallback: just return the <body> text stripped of scripts/styles
        for script in soup(["script", "style", "nav", "header", "footer"]):
            script.decompose()
        container = soup.body

    if container is None:
        print(f"[WARN] No recipe body found at {url}", file=sys.stderr)
        return None

    # Return prettified HTML of the container so the LLM gets structured markup
    return container.prettify()


# ---------------------------------------------------------------------------
# 3. LLM extraction — structured prompt for consistent output
# ---------------------------------------------------------------------------

# Use double braces to escape JSON-like blocks from .format() interpretation
LLM_PROMPT_TEMPLATE = """You are a recipe data extraction assistant. Extract the following fields from the recipe HTML below.

Output ONLY a valid JSON object with these exact keys. Do not include any surrounding text, markdown fences, or prose.

{{
  "name": "Recipe title",
  "ingredients": [
    {{
      "name": "Ingredient name",
      "quantity": 4,        // number or null if not present
      "unit": "cups"       // string or null if not present
    }}
  ],
  "steps": [
    "Step 1 description",
    "Step 2 description",
    ...
  ],
  "prep_time": "30 minutes"   // string or null if not present
}}

If any field cannot be determined, set it to null (for prep_time) or an empty list (for ingredients/steps).

Recipe HTML:
{html_block}
"""

def extract_with_llm(html_block: str) -> Optional[Dict[str, Any]]:
    """Send the HTML block to Gemini (direct key) and parse the JSON response.

    Needs GEMINI_API_KEY or GOOGLE_API_KEY set. Returns None on failure
    so the pipeline falls back to deterministic parsing.
    """
    if not LITELLM_AVAILABLE:
        print("[WARN] litellm not installed — skipping LLM extraction (pip install -r requirements.txt)", file=sys.stderr)
        return None

    if not GEMINI_API_KEY and not LITELLM_BASE_URL:
        # No key, no proxy — skip quietly to deterministic fallback
        print("[INFO] No GEMINI_API_KEY set — using deterministic fallback", file=sys.stderr)
        return None

    prompt = LLM_PROMPT_TEMPLATE.format(html_block=html_block)

    try:
        kwargs: Dict[str, Any] = dict(
            model=LITELLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1000,
        )
        if GEMINI_API_KEY:
            kwargs["api_key"] = GEMINI_API_KEY
        if LITELLM_BASE_URL:
            # Only used if someone explicitly runs a proxy
            kwargs["api_base"] = LITELLM_BASE_URL

        response = litellm.completion(**kwargs)  # type: ignore[has-type]

        # litellm returns a completion object; grab the text content safely
        choices = getattr(response, "choices", None)
        if choices and len(choices) > 0:
            msg = getattr(choices[0], "message", None)
            if msg:
                content = getattr(msg, "content", None)
            else:
                content = None
        else:
            content = None

        if not content:
            print("[WARN] LLM returned no content", file=sys.stderr)
            return None

        # Strip potential markdown fences if the model ignores the "no fences" instruction
        if content.startswith("```"):
            content = content.strip("`")
            lines = content.splitlines()
            content = "\n".join(lines[1:]) if len(lines) > 1 else "```"

        data = json.loads(content)
        return data
    except Exception as e:
        print(f"[WARN] LLM extraction failed: {e}", file=sys.stderr)
        # Fall back to deterministic soup-based extraction below
        return None


# ---------------------------------------------------------------------------
# 4. Deterministic validation — cheap check before any retry logic
# ---------------------------------------------------------------------------

def validate_recipe(data: Dict[str, Any]) -> bool:
    """Quick deterministic check that required fields exist and quantities are numbers."""

    # name must be a non-empty string
    if not data.get("name") or not isinstance(data["name"], str):
        print(f"[VALIDATE] Missing or invalid name: {data.get('name')}", file=sys.stderr)
        return False

    # ingredients must be a list
    ingredients = data.get("ingredients")
    if not isinstance(ingredients, list):
        print("[VALIDATE] ingredients is not a list", file=sys.stderr)
        return False

    # each ingredient should have name (str) and quantity (number or null)
    for i, ing in enumerate(ingredients):
        if not isinstance(ing, dict):
            print(f"[VALIDATE] ingredient {i} is not a dict", file=sys.stderr)
            return False
        if not isinstance(ing.get("name"), str) or not ing["name"].strip():
            print(f"[VALIDATE] ingredient {i} missing name", file=sys.stderr)
            return False
        qty = ing.get("quantity")
        # quantity, if present, must be a number (int/float) or null
        if qty is not None and not isinstance(qty, (int, float)):
            print(f"[VALIDATE] ingredient {i} quantity is not a number: {qty}", file=sys.stderr)
            return False

    # steps, if present, should be a list of strings
    steps = data.get("steps")
    if steps is not None and not isinstance(steps, list):
        print("[VALIDATE] steps is not a list", file=sys.stderr)
        return False

    # prep_time, if present, should be a string or null
    pt = data.get("prep_time")
    if pt is not None and not isinstance(pt, str):
        print("[VALIDATE] prep_time is not a string or null", file=sys.stderr)
        return False

    return True


# ---------------------------------------------------------------------------
# 5. Post-processing: normalize the raw LLM output into RecipeResult
# ---------------------------------------------------------------------------

def normalize_ingredients(raw_ingredients: List[Dict[str, Any]]) -> List[Ingredient]:
    """Convert raw ingredient dicts from LLM into Ingredient dataclasses."""
    normalized = []
    for ing in raw_ingredients:
        name = ing.get("name", "").strip()
        qty = ing.get("quantity")
        unit = ing.get("unit")
        # Attempt to parse quantity as float if it's a string that looks numeric
        if isinstance(qty, str):
            try:
                qty = float(qty)
            except ValueError:
                qty = None
        normalized.append(Ingredient(name=name, quantity=qty, unit=unit))
    return normalized


def normalize_steps(raw_steps: List[str]) -> List[str]:
    """Clean up step strings."""
    cleaned = []
    for s in raw_steps:
        s = s.strip()
        if s:
            cleaned.append(s)
    return cleaned


# ---------------------------------------------------------------------------
# 6. Output — CSV (flattened, one row per ingredient) and JSON (nested)
# ---------------------------------------------------------------------------

def write_csv(result: RecipeResult, filepath: str):
    """Write flattened CSV: one row per ingredient.

    Columns: recipe_name, ingredient_name, quantity, unit, steps (joined)
    """
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["recipe_name", "ingredient_name", "quantity", "unit", "steps"])
        steps_joined = " | ".join(result.steps) if result.steps else ""
        for ing in result.ingredients:
            writer.writerow([
                result.name,
                ing.name,
                ing.quantity if ing.quantity is not None else "",
                ing.unit if ing.unit is not None else "",
                steps_joined,
            ])


def write_json(result: RecipeResult, filepath: str):
    """Write nested JSON with the full validated structure."""
    data = {
        "name": result.name,
        "ingredients": [{"name": ing.name, "quantity": ing.quantity, "unit": ing.unit} for ing in result.ingredients],
        "steps": result.steps,
        "prep_time": result.prep_time,
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(urls: List[str], out_csv: str, out_json: str):
    """Run the full scrape-extract-validate-output pipeline."""
    results: List[RecipeResult] = []

    for url in urls:
        print(f"\n=== Fetching: {url} ===")
        html = fetch_html(url)
        if not html:
            print(f"[SKIP] No HTML fetched for {url}", file=sys.stderr)
            continue

        print(f"=== Extracting body for: {url} ===")
        body_html = extract_recipe_body(html, url)
        if body_html is None:
            continue

        # Try LLM extraction first
        raw_data = extract_with_llm(body_html)

        # Fallback: if LLM failed, try deterministic soup-based extraction
        if raw_data is None or not validate_recipe(raw_data):
            print("[INFO] LLM extraction failed or invalid — attempting deterministic fallback", file=sys.stderr)
            raw_data = deterministic_extract(body_html)

        if raw_data is None or not validate_recipe(raw_data):
            print(f"[SKIP] Validation failed for {url}, skipping", file=sys.stderr)
            continue

        # Normalize
        ing_norm = normalize_ingredients(raw_data.get("ingredients", []))
        steps_norm = normalize_steps(raw_data.get("steps", []))

        prep_time = raw_data.get("prep_time")
        name = raw_data.get("name", url.split("/")[-2].replace("-", " ").title())

        result = RecipeResult(
            name=name,
            ingredients=ing_norm,
            steps=steps_norm,
            prep_time=prep_time,
        )
        results.append(result)

        # Be polite between URLs
        time.sleep(REQUEST_DELAY)

    # Aggregate all ingredients across results for CSV (flattened)
    # Each recipe gets its own CSV row block
    # Write individual CSVs per result, or a combined one
    # Here we'll write one combined CSV and one JSON per result

    # Combined CSV: one row per ingredient across all recipes
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["recipe_name", "ingredient_name", "quantity", "unit", "steps"])
        for res in results:
            steps_joined = " | ".join(res.steps) if res.steps else ""
            for ing in res.ingredients:
                writer.writerow([
                    res.name,
                    ing.name,
                    ing.quantity if ing.quantity is not None else "",
                    ing.unit if ing.unit is not None else "",
                    steps_joined,
                ])

    # Combined JSON: nest by recipe name
    combined_json = {
        "recipes": [
            {
                "name": res.name,
                "ingredients": [{"name": ing.name, "quantity": ing.quantity, "unit": ing.unit} for ing in res.ingredients],
                "steps": res.steps,
                "prep_time": res.prep_time,
            }
            for res in results
        ]
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(combined_json, f, indent=2, ensure_ascii=False)

    print(f"\n=== Pipeline complete ===")
    print(f"Wrote {len(results)} recipes to {out_csv} and {out_json}")
    for res in results:
        print(f"  - {res.name}: {len(res.ingredients)} ingredients, {len(res.steps)} steps")


FRACTION_MAP = {
    "½": 0.5, "⅓": 1 / 3, "⅔": 2 / 3, "¼": 0.25, "¾": 0.75,
    "⅛": 0.125, "⅜": 0.375, "⅝": 0.625, "⅞": 0.875,
}

KNOWN_UNITS = {
    "g", "kg", "mg", "ml", "l", "litre", "liter",
    "tsp", "tbsp", "teaspoon", "teaspoons", "tablespoon", "tablespoons",
    "cup", "cups", "oz", "fl", "lb", "lbs", "pint", "quart",
    "pinch", "dash", "bunch", "handful", "can", "cans", "tin", "tins",
    "slice", "slices", "piece", "pieces", "stick", "sticks",
    "sprig", "sprigs", "stalk", "stalks", "scoop", "scoops",
    "clove", "cloves",
}

_QTY_RE = re.compile(
    r"^(?P<int>\d+(?:\.\d+)?)?\s*(?P<frac>[½⅓⅔¼¾⅛⅜⅝⅞])?"
    r"\s*(?P<rest>.*)$"
)


def _split_quantity(text: str):
    """Split '2 tbsp oil' / '½ tsp turmeric' / '350g passata' into (qty, unit, name).

    Returns (None, None, text) when no leading quantity is found.
    """
    m = _QTY_RE.match(text.strip())
    if not m or (not m.group("int") and not m.group("frac")):
        return None, None, text
    qty = float(m.group("int")) if m.group("int") else 0.0
    if m.group("frac"):
        qty += FRACTION_MAP[m.group("frac")]
    rest = m.group("rest").strip()
    if not rest:
        return None, None, text
    parts = rest.split(None, 1)
    if len(parts) == 2:
        unit, name = parts
        # Only accept the second token as a unit if it looks like one;
        # otherwise keep it in the name ("4 garlic cloves" -> unit None).
        if unit.lower().rstrip(".,") not in KNOWN_UNITS:
            return qty, None, rest
        return qty, unit, name.strip()
    # e.g. "200g" with no name left — keep whole text, quantity known but
    # unit/name ambiguous, so leave unparsed rather than guessing wrong
    return None, None, text


def deterministic_extract(body_html: str) -> Optional[Dict[str, Any]]:
    """Deterministic fallback: extract recipe data using BeautifulSoup alone.

    This avoids needing an LLM call and produces a JSON-compatible dict.
    It works well on static recipe pages with predictable markup.
    """
    soup = BeautifulSoup(body_html, "html.parser")

    # Try to find the title
    title = None
    for sel in ["h1", "h2", ".recipe-title", ".entry-title", "title"]:
        found = soup.select_one(sel)
        if found and found.get_text(strip=True):
            title = found.get_text(strip=True)
            break

    # Try to find ingredient list — scoped selectors first so we don't
    # accidentally grab steps (which live in section.method > ol).
    ingredients = []
    ing_sels = [
        ".ingredients li",
        ".recipe-ingredients li",
        "section.ingredients li",
        "span[itemprop='recipeIngredient']",
        "ul li",  # generic fallback
    ]
    for sel in ing_sels:
        found = soup.select(sel)
        if not found:
            continue
        for li in found:
            text = li.get_text(separator=" ", strip=True)
            if not text:
                continue
            if text.endswith(":"):
                continue  # subsection header like "For the chicken:"
            qty, unit, name = _split_quantity(text)
            ingredients.append({"name": name, "quantity": qty, "unit": unit})
        if ingredients:
            break  # scoped selector hit — stop trying broader ones

    # If no ingredients found via selectors, try getting all text and guessing
    if not ingredients:
        # Last resort: return whatever text we have
        text = soup.get_text(separator="\n", strip=True)
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        return {"name": title or "Unknown", "ingredients": [{"name": l, "quantity": None, "unit": None} for l in lines[:10]], "steps": [], "prep_time": None}

    # Try to find steps
    steps = []
    step_sels = [
        "ol li", ".instructions li", ".recipe-instructions li",
        "div[role='listitem']"
    ]
    for sel in step_sels:
        found = soup.select(sel)
        if found:
            steps = [li.get_text(strip=True) for li in found if li.get_text(strip=True)]
            break

    return {"name": title or "Unknown", "ingredients": ingredients, "steps": steps, "prep_time": None}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Strip argparse args; if remaining args look like URLs, use them; otherwise defaults
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    urls = argv if argv else RECIPE_URLS

    out_csv = os.path.join(os.path.dirname(__file__), "recipes", "output.csv")
    out_json = os.path.join(os.path.dirname(__file__), "recipes", "output.json")

    # Ensure output dir exists
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    run_pipeline(urls, out_csv, out_json)