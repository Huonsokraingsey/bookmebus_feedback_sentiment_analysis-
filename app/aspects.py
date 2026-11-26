"""Keyword-based aspect groups for the dashboard (Stage 2).

IMPORTANT: this is simple KEYWORD MATCHING, not a trained aspect-based sentiment model.
A review "mentions" an aspect if it contains one of the aspect's keywords. The sentiment shown for an aspect is the
sentiment of the whole review (not of the sentence about that aspect).

Edit the lists below freely (one easy-to-edit dictionary). Khmer lists checked and extended by a native
Khmer speaker of the team (KEYWORD_REVIEW.xlsx, 2026-11-25). Khmer has no spaces between words, so Khmer keywords are
matched anywhere in the text; English keywords are matched as whole words (so "ac" does not match "back").
"""
import re

ASPECTS = {
    'Seat & comfort': {
        "km": ["កៅអី", "កន្លែងអង្គុយ", "ចង្អៀត", "ផាសុកភាព", "ផាសុខភាព", "គ្រែ", "ដាក់ជើង", "កៅអ៊ី", "កៅអីរឹង", "ពិបាកអង្គុយ", "ពុលឡាន", "រលាក់", "វាំងនន"],
        "en": ["seat", "seats", "comfortable", "uncomfortable", "comfort", "cramped", "legroom", "leg room", "bed", "beds"],
    },
    'Driver & driving': {
        "km": ["អ្នកបើកបរ", "តៃកុង", "អ្នកបើក", "បើកបរ", "បើកលឿន", "តាកុង", "តៃកុន", "អ្នកបើកឡាន", "អ្នកបើករថយន្ត", "កញ្ឆក់", "កន្ត្រាក់", "ជាន់ហ្វ្រាំង", "ចាប់ប្រាំង", "ជែង"],
        "en": ["driver", "drivers", "driving", "drove", "drive"],
    },
    'Punctuality & waiting': {
        "km": ["យឺត", "ទាន់ពេល", "ត្រូវម៉ោង", "ទៀងទាត់", "ពន្យារ", "ពន្យា", "រង់ចាំ", "យឺតយ៉ាវ", "ពន្យាពេល", "ទៀងម៉ោង", "ខកពេល", "ឈប់ច្រើន", "ចេញយឺត", "មកដល់យឺត"],
        "en": ["late", "delay", "delayed", "on time", "punctual", "wait", "waited", "waiting"],
    },
    'Price & money': {
        "km": ["តម្លៃ", "ថ្លៃ", "ថោក", "លុយ", "ប្រាក់", "បង់លុយ", "ធូរថ្លៃ", "សងលុយ", "ចំណាយ"],
        "en": ["price", "prices", "expensive", "cheap", "money", "cost", "value", "refund"],
    },
    'Staff & service': {
        "km": ["បុគ្គលិក", "សេវាកម្ម", "សេវា", "រួសរាយ", "ឈ្លើយ", "រាក់ទាក់", "រស់រាយ", "សុជីវធម៌", "គួរសម", "ទំនួលខុសត្រូវ", "សម្ដីអាក្រក់"],
        "en": ["staff", "service", "friendly", "rude", "helpful"],
    },
    'Cleanliness & toilets': {
        "km": ["ស្អាត", "កខ្វក់", "អនាម័យ", "ក្លិន", "បន្ទប់ទឹក", "បង្គន់", "សំរាម", "សម្អាត", "សំអាត", "ធូលី", "ក្លិនផ្សែង"],
        "en": ["clean", "dirty", "smell", "smelly", "toilet", "toilets", "hygiene"],
    },
    'Air conditioning & temperature': {
        "km": ["ម៉ាស៊ីនត្រជាក់", "ត្រជាក់", "ក្តៅ", "ក្ដៅ", "ម៉ាសុីនត្រជាក់", "មាសុីនត្រជាក់", "កង្ហា", "ក្តៅស្អុះ", "ម៉ាស៊ីនកម្តៅ"],
        "en": ["air con", "aircon", "air-con", "ac", "a/c", "air conditioning", "hot", "cold"],
    },
    'Booking, tickets & app': {
        "km": ["កក់", "សំបុត្រ", "អនឡាញ", "កម្មវិធី", "គេហទំព័រ", "អែប", "លេខកៅអី", "កក់សំបុត្រ", "ទិញសំបុត្រ", "ដូរកៅអី"],
        "en": ["book", "booking", "booked", "ticket", "tickets", "app", "website", "online", "bookmebus"],
    },
    'Safety': {
        "km": ["គ្រោះថ្នាក់", "សុវត្ថិភាព", "ខ្សែក្រវាត់", "សុវត្តិភាព", "សុវតិ្ថភាព", "ប្រុងប្រយ័ត្ន", "ហួសល្បឿន", "ខ្សែក្រវ៉ាត់", "បំពានច្បាប់", "ង៉ក់"],
        "en": ["safe", "safety", "unsafe", "dangerous", "danger", "seatbelt", "seat belt", "accident"],
    },
}


def compile_patterns(aspects=ASPECTS):
    """One regular expression per aspect: Khmer keywords anywhere, English keywords as whole words."""
    patterns = {}
    for aspect, words in aspects.items():
        parts = [re.escape(word) for word in words.get("km", [])]
        parts += [r"(?<![a-z])" + re.escape(word) + r"(?![a-z])" for word in words.get("en", [])]
        patterns[aspect] = re.compile("|".join(parts), re.IGNORECASE)
    return patterns


PATTERNS = compile_patterns()


def find_aspects(text):
    """List of the aspects whose keywords appear in the text."""
    return [aspect for aspect, pattern in PATTERNS.items() if pattern.search(str(text))]


def aspect_table(texts, sentiments):
    """One row per aspect: how many reviews mention it and the sentiment split of those reviews."""
    import pandas as pd
    rows = []
    found = [find_aspects(text) for text in texts]
    for aspect in ASPECTS:
        hits = [s for aspects, s in zip(found, sentiments) if aspect in aspects]
        n = len(hits)
        rows.append({
            "aspect": aspect, "reviews": n,
            "% positive": round(100 * hits.count("positive") / n, 1) if n else 0.0,
            "% uncertain": round(100 * hits.count("uncertain") / n, 1) if n else 0.0,
            "% negative": round(100 * hits.count("negative") / n, 1) if n else 0.0,
        })
    return pd.DataFrame(rows).sort_values("reviews", ascending=False).reset_index(drop=True)
