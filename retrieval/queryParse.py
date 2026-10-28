"""Parse a shopper query into the attribute values it constrains.

The output vocabulary is exactly the one in data/metadata.csv, so downstream
code can index attribute tables with it. Aliases also cover the words a shopper
uses when the catalogue word is different ("sneakers" -> Sports Shoes).

Two details matter for scoring:

* A longer phrase masks the shorter keys it contains: "sports sandals" is a
  Sports Sandals query, not a Sports Shoes query plus a Sandals query.
* The first value of an alias list is the word's literal meaning; the rest is
  the colour/type family it is often confused with. Family colours carry a
  weight below 1 in `weights`, so "grey" nudges Silver instead of treating it
  as a full match; type aliases are true synonyms and keep full weight.
"""
from __future__ import annotations

import re

# Colour aliases often name a *different* catalogue value: a silver shoe is
# Silver, not Grey, so family colours get partial credit. Type aliases are real
# synonyms ("sneakers" = Sports Shoes or Casual Shoes), so they stay at 1.0.
COLOR_FAMILY_WEIGHT = 0.5
TYPE_FAMILY_WEIGHT = 1.0
MASK_SHORT_MATCHES = True

COLOR_ALIASES: dict[str, list[str]] = {
    "black": ["Black", "Charcoal"],
    "white": ["White", "Off White"],
    "off white": ["Off White", "White"],
    "cream": ["Cream", "Off White", "Beige"],
    "grey": ["Grey", "Charcoal", "Silver"],
    "gray": ["Grey", "Charcoal", "Silver"],
    "charcoal": ["Charcoal", "Grey"],
    "silver": ["Silver", "Metallic"],
    "metallic": ["Metallic", "Silver", "Gold"],
    "blue": ["Blue", "Navy Blue", "Turquoise Blue"],
    "navy": ["Navy Blue", "Blue"],
    "turquoise": ["Turquoise Blue", "Teal"],
    "red": ["Red", "Maroon", "Burgundy", "Rust"],
    "maroon": ["Maroon", "Burgundy", "Red"],
    "burgundy": ["Burgundy", "Maroon", "Red"],
    "brown": ["Brown", "Tan", "Coffee Brown", "Mushroom Brown", "Taupe"],
    "tan": ["Tan", "Brown", "Beige"],
    "beige": ["Beige", "Cream", "Tan", "Khaki", "Off White", "Mushroom Brown"],
    "khaki": ["Khaki", "Beige", "Tan", "Cream"],
    "olive": ["Olive", "Khaki"],
    "green": ["Green", "Olive", "Sea Green", "Fluorescent Green"],
    "sea green": ["Sea Green", "Teal"],
    "teal": ["Teal", "Turquoise Blue", "Sea Green"],
    "gold": ["Gold", "Metallic"],
    "bronze": ["Bronze", "Copper", "Brown"],
    "copper": ["Copper", "Bronze"],
    "orange": ["Orange", "Rust", "Peach"],
    "yellow": ["Yellow", "Mustard"],
    "mustard": ["Mustard", "Yellow", "Khaki"],
    "pink": ["Pink", "Peach", "Magenta"],
    "purple": ["Purple", "Lavender", "Magenta"],
    "lavender": ["Lavender", "Purple"],
    "magenta": ["Magenta", "Pink", "Purple"],
    "peach": ["Peach", "Pink", "Nude"],
    "nude": ["Nude", "Beige", "Peach", "Tan"],
    "multi": ["Multi"],
}

TYPE_ALIASES: dict[str, list[str]] = {
    "casual shoes": ["Casual Shoes"],
    "casual": ["Casual Shoes"],
    "sneakers": ["Sports Shoes", "Casual Shoes"],
    "sneaker": ["Sports Shoes", "Casual Shoes"],
    "trainers": ["Sports Shoes"],
    "sports shoes": ["Sports Shoes"],
    "sports": ["Sports Shoes"],
    "running": ["Sports Shoes"],
    "athletic": ["Sports Shoes"],
    "formal shoes": ["Formal Shoes"],
    "formal": ["Formal Shoes"],
    "oxfords": ["Formal Shoes"],
    "derby": ["Formal Shoes"],
    "loafers": ["Formal Shoes", "Casual Shoes"],
    "heels": ["Heels"],
    "heel": ["Heels"],
    "pumps": ["Heels"],
    "stilettos": ["Heels"],
    "sports sandals": ["Sports Sandals"],
    "sports sandal": ["Sports Sandals"],
    "sport sandals": ["Sports Sandals"],
    "sandals": ["Sandals", "Sports Sandals"],
    "sandal": ["Sandals", "Sports Sandals"],
    "flip flops": ["Flip Flops"],
    "flip-flops": ["Flip Flops"],
    "flipflops": ["Flip Flops"],
    "slippers": ["Flip Flops"],
    "thongs": ["Flip Flops"],
    "flats": ["Flats"],
    "ballerinas": ["Flats"],
}

GENDER_ALIASES: dict[str, str] = {
    "men": "Men", "man": "Men", "mens": "Men", "male": "Men",
    "women": "Women", "woman": "Women", "womens": "Women",
    "female": "Women", "ladies": "Women",
}


def _matches(text: str, aliases: dict) -> list[tuple[int, int, list[str]]]:
    """Every whole-word/phrase alias hit, with its character span."""
    found = []
    for key, values in aliases.items():
        for m in re.finditer(rf"\b{re.escape(key)}\b", text):
            found.append((m.start(), m.end(), values))
    return found


def _slot(text: str, aliases: dict, family_weight: float) -> tuple[list[str], dict]:
    """Values the slot constrains, plus a weight per value.

    Longer matches win: any alias sitting inside an accepted phrase is dropped,
    so "sports sandals" does not also fire "sports" -> Sports Shoes.
    """
    kept: list[tuple[int, int, list[str]]] = []
    for start, end, values in sorted(_matches(text, aliases),
                                     key=lambda m: (-(m[1] - m[0]), m[0])):
        inside = any(s <= start and end <= e and (e - s) > (end - start)
                     for s, e, _ in kept)
        if MASK_SHORT_MATCHES and inside:
            continue
        kept.append((start, end, values))

    weights: dict[str, float] = {}
    for _s, _e, values in kept:
        for i, value in enumerate(values):
            weight = 1.0 if i == 0 else family_weight
            weights[value] = max(weights.get(value, 0.0), weight)
    ordered = sorted(weights, key=lambda v: (-weights[v], -len(v), v))
    return ordered, {v: weights[v] for v in ordered}


def parse_query(text: str) -> dict:
    """Return {"color", "type", "gender", "weights"} for the query.

    `color` / `type` list the metadata values named (empty = unconstrained);
    `weights` maps every listed value to 1.0 (the word's own value) or
    < 1 (a family value it is often confused with). `gender` is None when the
    query names neither gender or names both.
    """
    low = " ".join(text.lower().split())
    genders = sorted({GENDER_ALIASES[k] for k in GENDER_ALIASES
                      if re.search(rf"\b{k}\b", low)})
    color, color_w = _slot(low, COLOR_ALIASES, COLOR_FAMILY_WEIGHT)
    type_, type_w = _slot(low, TYPE_ALIASES, TYPE_FAMILY_WEIGHT)
    return {
        "color": color,
        "type": type_,
        "gender": genders[0] if len(genders) == 1 else None,
        "weights": {"color": color_w, "type": type_w},
    }
