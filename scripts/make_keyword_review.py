"""Builds KEYWORD_REVIEW.xlsx: a spreadsheet for a native Khmer speaker to check the web app's word lists
(aspect keywords in app/aspects.py and hidden words in app/stopwords_km.txt) without touching any code.

Run:  python scripts/make_keyword_review.py
"""
import os
import sys
from collections import Counter

import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "app"))
import aspects  # noqa: E402
import indicators  # noqa: E402

OUT = os.path.join(ROOT, "KEYWORD_REVIEW.xlsx")

KEYWORD_MEANING = {
    "កៅអី": "seat", "កន្លែងអង្គុយ": "seating place", "ចង្អៀត": "cramped / narrow", "ផាសុកភាព": "comfort",
    "ផាសុខភាព": "comfort (other spelling)", "គ្រែ": "bed", "ដាក់ជើង": "put the legs (legroom)",
    "អ្នកបើកបរ": "driver", "តៃកុង": "driver (informal)", "អ្នកបើក": "driver (short)", "បើកបរ": "to drive",
    "បើកលឿន": "drive fast", "យឺត": "late / slow", "ទាន់ពេល": "on time", "ត្រូវម៉ោង": "on time",
    "ទៀងទាត់": "punctual / regular", "ពន្យារ": "delay", "ពន្យា": "delay (other spelling)", "រង់ចាំ": "to wait",
    "តម្លៃ": "price", "ថ្លៃ": "expensive / fee", "ថោក": "cheap", "លុយ": "money", "ប្រាក់": "money",
    "បុគ្គលិក": "staff", "សេវាកម្ម": "service", "សេវា": "service (short)", "រួសរាយ": "friendly", "ឈ្លើយ": "rude",
    "ស្អាត": "clean", "កខ្វក់": "dirty", "អនាម័យ": "hygiene", "ក្លិន": "smell", "បន្ទប់ទឹក": "bathroom / toilet",
    "បង្គន់": "toilet", "ម៉ាស៊ីនត្រជាក់": "air conditioner", "ត្រជាក់": "cold", "ក្តៅ": "hot",
    "ក្ដៅ": "hot (other spelling)", "កក់": "to book / reserve", "សំបុត្រ": "ticket", "អនឡាញ": "online",
    "អ៊ីនធឺណិត": "internet", "កម្មវិធី": "app / program", "គេហទំព័រ": "website", "គ្រោះថ្នាក់": "danger / accident",
    "សុវត្ថិភាព": "safety", "ខ្សែក្រវាត់": "seat belt",
}
HIDDEN_MEANING = {
    "ខ្ញុំ": "I", "យើង": "we", "គាត់": "he / she", "គេ": "they / someone", "អ្នក": "you", "វា": "it",
    "ពួកគេ": "they", "បង": "older sibling (form of address)", "ពូ": "uncle (form of address)", "នេះ": "this",
    "នោះ": "that", "ហ្នឹង": "this / that (informal)", "នឹង": "will / with", "ណា": "which / where", "អ្វី": "what",
    "ដែរ": "also", "ផង": "too / please", "ទៀត": "more / again", "ហើយ": "and / already", "ក៏": "also / so",
    "តែ": "but / only", "គឺ": "is (that is)", "ជា": "is / be", "ថា": "that (say that)", "ដែល": "which / that",
    "បាន": "did / can (past)", "មាន": "have / there is", "ត្រូវ": "must / correct", "ការ": "(makes a noun: -ing, -tion)",
    "ៗ": "(repetition mark)", "និង": "and", "ឬ": "or", "ប៉ុន្តែ": "but", "បើ": "if", "ទោះ": "although",
    "ព្រោះ": "because", "ដោយ": "by / with", "ជាមួយ": "with", "សម្រាប់": "for", "ចំពោះ": "towards / about",
    "ឱ្យ": "to / let", "អោយ": "to / let (other spelling)", "ឲ្យ": "to / let (other spelling)", "ពី": "from",
    "ទៅ": "go / to", "មក": "come / to here", "នៅ": "at / in / stay", "ក្នុង": "in", "លើ": "on", "ដល់": "until / reach",
    "របស់": "of (possessive)", "ខាង": "side", "តាម": "along / by", "ពេល": "when / time", "ណាស់": "very",
    "ខ្លាំង": "strongly / very", "ច្រើន": "many / much", "តិច": "few / little", "បន្តិច": "a little", "ខ្លះ": "some",
    "ទាំង": "all / both", "មួយ": "one", "ពីរ": "two", "គ្នា": "each other", "រួច": "already / finished",
}

INSTRUCTIONS = [
    ["How to check the word lists of the web app (about 20–30 minutes)"],
    [""],
    ["The web app uses two Khmer word lists. A native Khmer speaker should check them. You do NOT need to edit any code."],
    [""],
    ["Sheet 'Aspect keywords'  – the keywords that decide which topic a review is about (seat, driver, price, …)."],
    ["   → In column 'keep?' write 'no' for a keyword that is wrong or misleading. Add a comment if needed."],
    ["Sheet 'Add keywords'      – write NEW keywords that customers often use for a topic (one per row)."],
    ["   → Choose the aspect from the list (or write a new aspect name), write the Khmer word, and its meaning."],
    ["   → Think of informal words, other spellings and common short forms (e.g. តៃកុង for driver)."],
    ["Sheet 'Hidden words'      – small function words (I, this, and, very, …) that are HIDDEN from the word charts."],
    ["   → Write 'no' in 'keep hidden?' if a word carries real meaning and should be shown in the charts."],
    ["Sheet 'Frequent words'    – the 150 most frequent words in our reviews that are currently SHOWN in the charts."],
    ["   → 'hide?': write 'yes' for function words that should be hidden (like the ones in 'Hidden words')."],
    ["   → 'add to aspect': choose an aspect if the word should be a keyword of that topic."],
    [""],
    ["Note: Khmer keywords are matched anywhere inside a review (Khmer has no spaces), so a very short keyword can match"],
    ["      inside other words. Prefer complete words."],
    ["Note: hiding a word only changes the charts. The sentiment model still reads every word."],
    [""],
    ["When you are done: save the file (keep the .xlsx format) and send it back. The team / Claude updates the app."],
]


def frequent_shown_words(n=150):
    """The n words shown in the charts that appear in the most reviews (all processed splits)."""
    hidden = indicators.hidden_words()
    counts = Counter()
    for split in ["train", "val", "test_translated", "test_native"]:
        data = pd.read_csv(os.path.join(ROOT, "data", "processed", f"{split}.csv"))
        for tokens in data["tokens"].astype(str).str.split():
            counts.update(indicators.review_words(tokens, hidden))
    rows = []
    for word, count in counts.most_common(n):
        rows.append({"word": word, "reviews": count, "already in aspect": ", ".join(aspects.find_aspects(word)),
                     "hide? (yes/no)": "", "add to aspect": "", "comment": ""})
    return pd.DataFrame(rows)


def main():
    aspect_rows = []
    for aspect, words in aspects.ASPECTS.items():
        for word in words["km"]:
            aspect_rows.append({"aspect": aspect, "language": "Khmer", "keyword": word,
                                "meaning": KEYWORD_MEANING.get(word, ""), "keep? (yes/no)": "yes", "comment": ""})
        for word in words["en"]:
            aspect_rows.append({"aspect": aspect, "language": "English", "keyword": word, "meaning": word,
                                "keep? (yes/no)": "yes", "comment": ""})
    add_rows = pd.DataFrame({"aspect": [""] * 40, "new Khmer keyword": [""] * 40,
                             "meaning (English)": [""] * 40, "comment": [""] * 40})
    hidden = sorted(indicators.load_khmer_stopwords(), key=lambda w: list(HIDDEN_MEANING).index(w)
                    if w in HIDDEN_MEANING else 999)
    hidden_rows = pd.DataFrame({"word": hidden, "meaning": [HIDDEN_MEANING.get(w, "") for w in hidden],
                                "keep hidden? (yes/no)": ["yes"] * len(hidden), "comment": [""] * len(hidden)})

    with pd.ExcelWriter(OUT, engine="openpyxl") as writer:
        pd.DataFrame(INSTRUCTIONS).to_excel(writer, sheet_name="Read me", index=False, header=False)
        pd.DataFrame(aspect_rows).to_excel(writer, sheet_name="Aspect keywords", index=False)
        add_rows.to_excel(writer, sheet_name="Add keywords", index=False)
        hidden_rows.to_excel(writer, sheet_name="Hidden words", index=False)
        frequent_shown_words().to_excel(writer, sheet_name="Frequent words", index=False)

    # formatting: wide columns, bold header, frozen header row, yes/no and aspect drop-down lists
    wb = load_workbook(OUT)
    aspect_list = '"' + ",".join(a.replace(",", "") for a in aspects.ASPECTS) + '"'
    for ws in wb.worksheets:
        for column in ws.columns:
            width = max(len(str(cell.value or "")) for cell in column)
            ws.column_dimensions[column[0].column_letter].width = min(max(14, width + 4), 60)
        for row in ws.iter_rows():
            for cell in row:
                cell.font = Font(size=12, bold=(cell.row == 1))
                cell.alignment = Alignment(vertical="center")
        if ws.title == "Read me":
            ws.column_dimensions["A"].width = 120
            ws["A1"].font = Font(size=14, bold=True)
            continue
        ws.freeze_panes = "A2"
        for cell in ws[1]:
            cell.fill = PatternFill("solid", fgColor="DDEBF7")
        headers = [c.value for c in ws[1]]
        for name in headers:
            col = ws.cell(row=1, column=headers.index(name) + 1).column_letter
            if "(yes/no)" in str(name):
                dv = DataValidation(type="list", formula1='"yes,no"', allow_blank=True)
            elif name in ("aspect", "add to aspect") and ws.title != "Aspect keywords":
                dv = DataValidation(type="list", formula1=aspect_list, allow_blank=True, showErrorMessage=False)
            else:
                continue
            ws.add_data_validation(dv)
            dv.add(f"{col}2:{col}{max(ws.max_row, 41)}")
    wb.save(OUT)
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
