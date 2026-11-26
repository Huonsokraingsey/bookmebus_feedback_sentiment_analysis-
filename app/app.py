"""Feedback Sentiment Dashboard for businesses (Part C add-on).

Run with:  streamlit run app/app.py

Two tabs:
  * "Analyse a file"   – upload a CSV/Excel file of feedback and get a sentiment dashboard
  * "Analyse one review" – type one review and see what both models predict

Page layout only: models are loaded in sentiment.py, word statistics are computed in indicators.py, aspect keywords
are in aspects.py, files are read in file_io.py. The app only USES the models trained in the notebooks; it never
retrains anything, and every evaluation number it shows is read from results/metrics/.
"""
import html
import io
import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP_DIR)
import aspects  # noqa: E402
import file_io  # noqa: E402
import indicators  # noqa: E402
import sentiment  # noqa: E402

COLORS = {"positive": "#2e7d32", "uncertain": "#9e9e9e", "negative": "#c62828"}
EMOJI = {"positive": "😊", "uncertain": "❔", "negative": "😞"}
KHMER_FONT = "Noto Sans Khmer, Khmer OS, Khmer UI, Leelawadee UI, sans-serif"
DEMO_FILE = os.path.join(sentiment.ROOT, "data", "demo", "sample_feedback.csv")
BATCH = {"tfidf_logreg": 500, "xlmr": 32}
MODEL_NAMES = {"tfidf_logreg": "Fast – TF-IDF + Logistic Regression", "xlmr": "Accurate – XLM-R"}
EXAMPLES = {   # the same 5 examples as in notebooks 02–04
    "សេវាកម្មល្អណាស់ អ្នកបើកបររួសរាយរាក់ទាក់": "very good service, friendly driver",
    "អ្នកបើកបរមិនសូវគោរពអ្នកដំណើរទេ": "the driver doesn't really respect passengers (negation)",
    "ឡានចេញយឺតពីរម៉ោង ហើយម៉ាស៊ីនត្រជាក់មិនត្រជាក់": "the bus left two hours late and the AC was not cold",
    "សូមអរគុណច្រើន ទោះបីឡានយឺតពីរម៉ោងក៏ដោយ": "thank you very much, even though the bus was two hours late (indirect complaint)",
    "មិនអាក្រក់ទេ ធ្វើដំណើរបានស្រួល": "not bad, a comfortable trip",
}

st.set_page_config(page_title="Feedback Sentiment Dashboard", page_icon="💬", layout="wide")


# ================================================================ cached work
@st.cache_resource(show_spinner="Loading the model…")
def get_model(model_key):
    return sentiment.load_tfidf() if model_key == "tfidf_logreg" else sentiment.load_xlmr()


@st.cache_data(show_spinner=False)
def get_tokens(texts):
    return sentiment.tokenize(texts)


@st.cache_data(show_spinner=False)
def predict_batch(model_key, texts, token_lists):
    """Probabilities (and, for TF-IDF, the top words) for one batch of reviews. Cached per batch."""
    bundle = get_model(model_key)
    if model_key == "tfidf_logreg":
        return sentiment.predict_tfidf(bundle, texts, token_lists)
    return sentiment.predict_xlmr(bundle, list(texts)), [[] for _ in texts]


@st.cache_data(show_spinner=False)
def get_metrics():
    return sentiment.load_metrics()


def style(fig, height=380):
    fig.update_layout(font_family=KHMER_FONT, height=height, margin=dict(l=10, r=10, t=40, b=10),
                      legend_title_text="")
    fig.update_yaxes(tickfont_size=15)      # Khmer words need a bigger font to be readable
    return fig


def stacked_share_chart(table, label_column, height):
    """Horizontal bars split into % positive / uncertain / negative, with the number of reviews at the end."""
    long = table.melt(id_vars=[label_column, "reviews"], value_vars=["% positive", "% uncertain", "% negative"],
                      var_name="sentiment", value_name="share")
    long["sentiment"] = long["sentiment"].str.replace("% ", "")
    fig = px.bar(long, x="share", y=label_column, color="sentiment", orientation="h", color_discrete_map=COLORS,
                 hover_data={"reviews": True}, category_orders={"sentiment": indicators.SENTIMENTS},
                 labels={"share": "% of the reviews", label_column: ""})
    fig.update_layout(barmode="stack")
    for _, r in table.iterrows():
        fig.add_annotation(x=101, y=r[label_column], text=f"{r['reviews']}", showarrow=False, xanchor="left", font_size=11)
    fig.update_xaxes(range=[0, 110])
    return style(fig, height)


def highlight(text, word):
    """HTML of a review with every occurrence of `word` highlighted."""
    safe = html.escape(str(text))
    if word:
        safe = safe.replace(html.escape(word), f"<mark>{html.escape(word)}</mark>")
    return safe.replace(chr(10), "<br>")       # keep the line breaks of the review


def badge(label):
    return (f"<span style='background:{COLORS[label]};color:white;padding:1px 8px;border-radius:10px;"
            f"font-size:0.8em'>{label}</span>")


metrics = get_metrics()


def model_label(key):
    name = MODEL_NAMES[key]
    if key in metrics:
        m = metrics[key]["test_native"]
        note = " (provisional)" if metrics[key].get("test_native_label_source") == "silver" else ""
        name += f"  ·  macro-F1 {m['macro_f1']:.2f} on real Khmer reviews{note}"
    return name


def available_models():
    return ["tfidf_logreg"] + (["xlmr"] if sentiment.xlmr_available() else [])


# ================================================================ sidebar: upload and setup (file analysis)
with st.sidebar:
    st.header("📄 File analysis")
    st.subheader("1. Upload feedback")
    uploaded = st.file_uploader("CSV or Excel file (max 10 MB, 5,000 rows)", type=["csv", "xlsx", "xls"],
                                max_upload_size=10)
    if st.button("Try sample data", width="stretch"):
        st.session_state["use_demo"] = True
    if uploaded is not None:
        st.session_state["use_demo"] = False

    table, source_name, file_error = None, None, None
    try:
        if uploaded is not None:
            table, source_name = file_io.read_feedback_file(uploaded.name, uploaded.getvalue()), uploaded.name
        elif st.session_state.get("use_demo"):
            with open(DEMO_FILE, "rb") as f:
                table, source_name = file_io.read_feedback_file("sample_feedback.csv", f.read()), "sample_feedback.csv"
    except file_io.FileProblem as problem:
        file_error = str(problem)
    except FileNotFoundError:
        file_error = "The sample file data/demo/sample_feedback.csv is missing: run `python scripts/make_demo_data.py`."
    if file_error:
        st.error(file_error)
    elif table is not None:
        st.success(f"**{source_name}** loaded ({len(table):,} rows). The results are in the **📄 Analyse a file** tab.")

    text_column, date_column, category_column, model_key = None, "(none)", "(none)", "tfidf_logreg"
    if table is not None:
        st.subheader("2. Columns")
        columns = list(table.columns)
        text_column = st.selectbox("Column with the feedback text", columns, index=file_io.guess_text_column(table))
        others = ["(none)"] + [c for c in columns if c != text_column]
        date_column = st.selectbox("Date column (optional, for the trend chart)", others,
                                   index=others.index(file_io.guess_date_column(table, text_column)))
        category_column = st.selectbox("Category column (optional, e.g. branch or route)",
                                       [c for c in others if c != date_column or c == "(none)"])

        st.subheader("3. Model")
        model_key = st.radio("Sentiment model", available_models(), format_func=model_label)
        if not sentiment.xlmr_available():
            st.caption("The accurate XLM-R model is not available: run `notebooks/04_transformer_xlmr.ipynb` "
                       "to create `models/xlmr_best/`.")

    st.divider()
    st.caption("🔒 Your file and text stay in memory for this session only. Nothing is saved to disk.")

# ================================================================ header
st.title("💬 Feedback Sentiment Dashboard")
st.write("See at a glance what customers are happy or unhappy about, from Khmer customer feedback.")

with st.expander("ℹ️ About this model"):
    rows = []
    for key in ["tfidf_logreg", "xlmr"]:
        if key in metrics:
            m = metrics[key]
            rows.append({"model": MODEL_NAMES[key],
                         "macro-F1, translated Khmer reviews": m["test_translated"]["macro_f1"],
                         "macro-F1, real Khmer reviews": m["test_native"]["macro_f1"],
                         "labels of the real-Khmer test": m["test_native_label_source"]})
    label_source = next((m["test_native_label_source"] for m in metrics.values()), "unknown")
    st.markdown(f"""
- **Training data:** about 2,900 BookMeBus customer reviews (bus, van and ferry trips in Cambodia), translated into Khmer.
- **Labels:** each review was labeled positive or negative from its text by an AI (Claude): **silver labels**.
  The real-Khmer test reviews currently use **{label_source}** labels{' (results are provisional until human gold labels exist)' if label_source == 'silver' else ''}.
- **Two classes:** the model predicts *positive* or *negative*. When it is less than 60% sure, the review is shown as **uncertain**.
- **Scores on the test sets** (macro-F1 = average F1 of both classes; 1.0 = perfect), read from `results/metrics/`:
""")
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.markdown("""
- **Known limits:** trained on bus/ferry reviews, so other businesses may be less accurate; polite or indirect complaints
  ("thank you, even though the bus was late") and sarcasm are hard; English or other non-Khmer text is less reliable.
""")

if not sentiment.tfidf_available():
    st.error("**The trained models are missing** (the `models/` folder is not shared, it is too big). Create them first:\n\n"
             "1. run `notebooks/01_data_preparation.ipynb` and `notebooks/02_statistical_ml_tfidf_logreg.ipynb` "
             "→ creates `models/tfidf_logreg.joblib` (Fast model, needed)\n"
             "2. optional: run `notebooks/04_transformer_xlmr.ipynb` (GPU) → creates `models/xlmr_best/` (Accurate model)\n\n"
             "(or run `notebooks/00_colab_run_all.ipynb`, which runs all notebooks), then reload this page.")
    st.stop()

tab_file, tab_text = st.tabs(["📄 Analyse a file", "✍️ Analyse one review"])


# ================================================================ tab 2: one review
def render_single_review():
    st.write("Type or paste **one** customer review (Khmer works best) and see what both models predict.")

    def use_example():
        choice = st.session_state["example_choice"]
        if choice in EXAMPLES:
            st.session_state["single_text"] = choice

    st.selectbox("…or pick an example", ["(choose an example)"] + list(EXAMPLES), key="example_choice",
                 on_change=use_example, format_func=lambda t: t if t not in EXAMPLES else f"{t}  –  {EXAMPLES[t]}")
    text = st.text_area("Review", key="single_text", height=110,
                        placeholder="e.g. អ្នកបើកបរមិនសូវគោរពអ្នកដំណើរទេ").strip()
    if not text:
        st.info("Type a review above (then press Ctrl+Enter or click outside the box).")
        return
    if not file_io.has_khmer(text):
        st.warning("This text has no Khmer letters. The models were trained on Khmer, so the result is less reliable.")

    tokens = sentiment.tokenize([text])[0]
    results = st.columns(len(available_models()))
    for column, key in zip(results, available_models()):
        with column:
            with st.container(border=True):
                st.markdown(f"**{MODEL_NAMES[key]}**")
                if key == "tfidf_logreg":
                    probs, _ = sentiment.predict_tfidf(get_model(key), (text,), [tokens])
                else:
                    with st.spinner("XLM-R is reading the review…"):
                        probs = sentiment.predict_xlmr(get_model(key), [text])
                label, confidence = sentiment.to_sentiment(probs)
                label, confidence = label[0], confidence[0]
                st.markdown(f"<div style='font-size:1.8em;color:{COLORS[label]}'>{EMOJI[label]} {label.upper()}</div>",
                            unsafe_allow_html=True)
                st.caption(f"confidence {confidence:.0%}" + (" → less than 60% sure, read it yourself"
                                                              if label == "uncertain" else ""))
                fig = go.Figure(go.Bar(x=[probs[0, 1], probs[0, 0]], y=["positive", "negative"], orientation="h",
                                       marker_color=[COLORS["positive"], COLORS["negative"]],
                                       text=[f"{probs[0, 1]:.1%}", f"{probs[0, 0]:.1%}"], textposition="auto"))
                fig.update_xaxes(range=[0, 1], tickformat=".0%")
                st.plotly_chart(style(fig, 160), width="stretch", key=f"single_bar_{key}")

    # Explanation from the TF-IDF model: which words pushed the decision
    contributions = sentiment.explain_tfidf(get_model("tfidf_logreg"), tokens)
    st.markdown("##### Which words pushed the TF-IDF model?")
    st.caption("Each word is colored by its weight in the TF-IDF model × how important it is in this review: "
               "**green** pushes towards positive, **red** towards negative, no color = (almost) no effect. "
               "XLM-R reads the whole sentence in context, so it cannot be explained word by word like this.")
    strongest = max([abs(v) for v in contributions.values()] + [1e-9])
    spans = []
    for token in tokens:
        value = contributions.get(token, 0.0)
        alpha = min(1.0, abs(value) / strongest) * 0.85
        color = "46,125,50" if value > 0 else "198,40,40"
        spans.append(f"<span style='background:rgba({color},{alpha:.2f});padding:2px 4px;margin:2px;"
                     f"border-radius:4px;display:inline-block' title='{value:+.3f}'>{html.escape(token)}</span>")
    st.markdown(f"<div style='font-family:{KHMER_FONT};font-size:1.3em;line-height:2.2'>{''.join(spans)}</div>",
                unsafe_allow_html=True)
    if contributions:
        table = pd.DataFrame(sorted(contributions.items(), key=lambda x: -abs(x[1]))[:8],
                             columns=["word or word pair", "push"])
        table["towards"] = np.where(table["push"] > 0, "positive", "negative")
        st.dataframe(table.round(3), hide_index=True, width="content")
    with st.expander("How the review was cut into words (khmer-nltk)"):
        st.write(" · ".join(tokens) if tokens else "(no words)")


with tab_text:
    render_single_review()


# ================================================================ tab 1: file dashboard
def render_file_dashboard():
    if table is None:
        st.info("👈 Upload a CSV or Excel file with one review per row in the sidebar, or click **Try sample data**.")
        return

    # ---------------------------------------------------------- prepare and predict
    texts_all = table[text_column].astype(str).str.strip()
    is_empty = texts_all == ""
    data = table[~is_empty].copy()
    texts = tuple(texts_all[~is_empty])
    if len(texts) == 0:
        st.error(f"The column **{text_column}** is empty. Choose the column that contains the feedback text.")
        return

    token_lists = get_tokens(texts)
    if model_key == "xlmr" and get_model("xlmr")["device"] == "cpu":
        st.warning("XLM-R is running on the CPU (no GPU found): this can take a few minutes for large files.")

    probs_parts, top_parts = [], []
    size = BATCH[model_key]
    progress = st.progress(0.0, text="Analysing reviews…") if model_key == "xlmr" else None
    for start in range(0, len(texts), size):
        probs, tops = predict_batch(model_key, texts[start:start + size],
                                    tuple(map(tuple, token_lists[start:start + size])))
        probs_parts.append(probs)
        top_parts.extend(tops)
        if progress:
            progress.progress(min(1.0, (start + size) / len(texts)), text="Analysing reviews…")
    if progress:
        progress.empty()
    probs = np.vstack(probs_parts)

    labels, confidence = sentiment.to_sentiment(probs)
    data["sentiment"] = labels
    data["confidence"] = confidence.round(3)
    data["prob_negative"] = probs[:, 0].round(4)
    data["prob_positive"] = probs[:, 1].round(4)
    data["language"] = ["Khmer" if file_io.has_khmer(t) else "not Khmer" for t in texts]
    data["_tokens"] = list(token_lists)
    data["_text"] = list(texts)
    data["top words"] = [", ".join(words) for words in top_parts] if model_key == "tfidf_logreg" else ""

    st.caption(f"File: **{source_name}** · {len(table):,} rows · model: {MODEL_NAMES[model_key]}"
               + (f" · {int(is_empty.sum())} empty rows skipped" if is_empty.any() else ""))
    not_khmer = int((data["language"] == "not Khmer").sum())
    if not_khmer:
        st.warning(f"{not_khmer} reviews contain no Khmer letters. The model was trained on Khmer text, so its "
                   "predictions for these reviews are less reliable (see the *language* column).")

    view = data
    if category_column != "(none)":
        categories = sorted(data[category_column].astype(str).unique())
        chosen = st.multiselect(f"Show only these {category_column} values", categories, default=categories)
        view = data[data[category_column].astype(str).isin(chosen)]
        if len(view) == 0:
            st.warning("No reviews for this selection.")
            return

    # ---------------------------------------------------------- 1. overview
    st.subheader("Overview")
    share = view["sentiment"].value_counts(normalize=True)
    low_conf = int((view["confidence"] < sentiment.UNCERTAIN_BELOW).sum())
    cols = st.columns(5)
    cols[0].metric("Reviews analysed", f"{len(view):,}", border=True)
    cols[1].metric("😊 Positive", f"{share.get('positive', 0):.0%}", border=True)
    cols[2].metric("😞 Negative", f"{share.get('negative', 0):.0%}", border=True)
    cols[3].metric("❔ Uncertain", f"{share.get('uncertain', 0):.0%}", border=True,
                   help="The model is less than 60% sure: read these reviews yourself.")
    cols[4].metric("Low confidence", f"{low_conf:,}", border=True,
                   help="Number of predictions with a top probability below 0.6")

    # ---------------------------------------------------------- 2. distribution (+ per category)
    st.subheader("Sentiment distribution")
    counts = view["sentiment"].value_counts().reindex(indicators.SENTIMENTS).fillna(0)
    fig = go.Figure(go.Pie(labels=counts.index, values=counts.values, hole=0.55, sort=False,
                           marker_colors=[COLORS[s] for s in counts.index], textinfo="percent+label"))
    if category_column == "(none)":
        st.plotly_chart(style(fig, 340), width="stretch")
    else:
        left, right = st.columns([2, 3])
        left.plotly_chart(style(fig, 340), width="stretch")
        per_category = (view.groupby(view[category_column].astype(str))["sentiment"].value_counts(normalize=True)
                        .unstack(fill_value=0).reindex(columns=indicators.SENTIMENTS, fill_value=0) * 100).round(1)
        per_category.columns = [f"% {c}" for c in per_category.columns]
        per_category["reviews"] = view[category_column].astype(str).value_counts()
        per_category = per_category.reset_index().rename(columns={category_column: "category"})
        top = per_category.sort_values("reviews", ascending=False).head(15).sort_values("% negative")
        with right:
            st.markdown(f"**Sentiment per {category_column}** (most negative at the top, max. 15)")
            st.plotly_chart(stacked_share_chart(top, "category", 340), width="stretch")

    # ---------------------------------------------------------- 3. trend over time
    if date_column != "(none)":
        st.subheader("Trend over time")
        dates = pd.to_datetime(view[date_column], errors="coerce", format="mixed")
        if dates.notna().sum() == 0:
            st.warning(f"The column **{date_column}** contains no dates that could be read.")
        else:
            if dates.isna().any():
                st.caption(f"{int(dates.isna().sum())} rows without a readable date are left out of this chart.")
            span_days = (dates.max() - dates.min()).days
            period = st.radio("Group by", ["Week", "Month"], index=0 if span_days <= 120 else 1, horizontal=True)
            frame = pd.DataFrame({"date": dates, "sentiment": view["sentiment"]}).dropna()
            frame["period"] = frame["date"].dt.to_period("W" if period == "Week" else "M").dt.start_time
            trend = frame.groupby(["period", "sentiment"]).size().unstack(fill_value=0) \
                .reindex(columns=indicators.SENTIMENTS, fill_value=0).reset_index()
            long = trend.melt(id_vars="period", var_name="sentiment", value_name="reviews")
            fig = px.area(long, x="period", y="reviews", color="sentiment", color_discrete_map=COLORS,
                          category_orders={"sentiment": indicators.SENTIMENTS},
                          labels={"period": period, "reviews": "Number of reviews"})
            st.plotly_chart(style(fig, 360), width="stretch")
            trend["% negative"] = (trend["negative"] / trend[indicators.SENTIMENTS].sum(axis=1) * 100).round(1)
            fig = px.line(trend, x="period", y="% negative", markers=True, color_discrete_sequence=[COLORS["negative"]],
                          labels={"period": period}, title=f"Share of negative reviews per {period.lower()}")
            fig.update_yaxes(range=[0, 100])
            st.plotly_chart(style(fig, 260), width="stretch")

    # ---------------------------------------------------------- 4. most mentioned words
    hidden = indicators.hidden_words()
    st.subheader("Most mentioned words – are they good or bad?")
    st.caption("The 20 words mentioned in the most reviews. Each bar shows the share of positive, uncertain and negative "
               "reviews that mention the word. Common function words (and, the, ខ្ញុំ, នេះ…) are hidden here; the model "
               "still reads them.")
    all_words = indicators.word_table(view["_tokens"], view["sentiment"], hidden, min_reviews=1)
    if len(all_words) == 0:
        st.info("Not enough words to show.")
    else:
        st.plotly_chart(stacked_share_chart(all_words.head(20).iloc[::-1], "word", 560), width="stretch")

    # ---------------------------------------------------------- 5. indicator words
    st.subheader("Strongest indicator words")
    min_reviews = st.slider("Only words that appear in at least … reviews", 2, 30, 3)
    words = indicators.word_table(view["_tokens"], view["sentiment"], hidden, min_reviews=min_reviews)
    st.caption("A **high negative score** means the word appears much more often in negative reviews than in positive "
               "ones (and the other way round for positive). Score = smoothed log-odds ratio.")
    if len(words) == 0:
        st.info("No word appears often enough: lower the slider.")
    else:
        left, right = st.columns(2)
        negative_words = words.sort_values("indicator score", ascending=False).head(15)
        negative_words = negative_words[negative_words["indicator score"] > 0].iloc[::-1]
        positive_words = words.sort_values("indicator score").head(15)
        positive_words = positive_words[positive_words["indicator score"] < 0].assign(
            **{"positive score": lambda d: -d["indicator score"]}).iloc[::-1]
        with left:
            fig = px.bar(negative_words, x="indicator score", y="word", orientation="h",
                         hover_data=["reviews", "% negative"], color_discrete_sequence=[COLORS["negative"]],
                         title="Linked to NEGATIVE feedback", labels={"indicator score": "negative score", "word": ""})
            st.plotly_chart(style(fig, 460), width="stretch")
        with right:
            fig = px.bar(positive_words, x="positive score", y="word", orientation="h",
                         hover_data=["reviews", "% positive"], color_discrete_sequence=[COLORS["positive"]],
                         title="Linked to POSITIVE feedback", labels={"word": ""})
            st.plotly_chart(style(fig, 460), width="stretch")
        with st.expander("All words (sortable table)"):
            search = st.text_input("Search a word", key="word_search")
            shown = words[words["word"].str.contains(search, regex=False)] if search else words
            st.dataframe(shown, hide_index=True, width="stretch",
                         column_config={"indicator score": st.column_config.NumberColumn(
                             help="> 0: more typical of negative reviews · < 0: more typical of positive reviews")})

    # ---------------------------------------------------------- 6. word explorer
    st.subheader("Word explorer")
    st.caption("Pick a word (or type any word) to see how customers talk about it.")
    left, right = st.columns(2)
    options = all_words["word"].head(300).tolist() if len(all_words) else []
    picked = left.selectbox("Pick a frequent word", options,
                            index=options.index("កៅអី") if "កៅអី" in options else 0) if options else ""
    typed = right.text_input("…or type a word", key="explorer_typed").strip()
    word = typed or picked
    if word:
        mentions = view[view["_text"].str.contains(word, case=False, regex=False)]
        if len(mentions) == 0:
            st.info(f"No review contains **{word}**.")
        else:
            split = mentions["sentiment"].value_counts().reindex(indicators.SENTIMENTS).fillna(0).astype(int)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(f"Reviews with “{word}”", f"{len(mentions):,}", border=True)
            for column, label in zip([c2, c3, c4], indicators.SENTIMENTS):
                column.metric(f"{EMOJI[label]} {label}", f"{split[label]} ({split[label] / len(mentions):.0%})", border=True)
            show = st.multiselect("Show reviews that are", indicators.SENTIMENTS, default=["negative", "positive"],
                                  key="explorer_filter")
            selected = mentions[mentions["sentiment"].isin(show)]
            st.caption(f"{len(selected)} reviews" + (" (showing the first 30)" if len(selected) > 30 else ""))
            items = [f"<div style='margin:6px 0;font-family:{KHMER_FONT};font-size:1.05em'>{badge(r['sentiment'])} "
                     f"{highlight(r['_text'], word)}</div>" for _, r in selected.head(30).iterrows()]
            st.html("".join(items))      # plain HTML: review text is never read as markdown (no accidental lists)

    # ---------------------------------------------------------- 7. aspect summary
    st.subheader("Aspect summary (keyword matching)")
    st.caption("⚠️ This is **simple keyword matching**, not a trained aspect model: a review counts for an aspect when it "
               "contains one of the aspect's keywords, and the sentiment shown is the sentiment of the whole review. "
               "The keyword lists are in `app/aspects.py`.")
    aspect_rows = aspects.aspect_table(view["_text"], view["sentiment"])
    aspect_rows = aspect_rows[aspect_rows["reviews"] > 0]
    if len(aspect_rows) == 0:
        st.info("No review mentions any of the aspect keywords.")
    else:
        chart = aspect_rows.sort_values("% negative")
        st.plotly_chart(stacked_share_chart(chart, "aspect", 60 + 42 * len(chart)), width="stretch")
        with st.expander("Keywords per aspect"):
            st.dataframe(pd.DataFrame([{"aspect": a, "Khmer keywords": " · ".join(w["km"]),
                                        "English keywords": ", ".join(w["en"])} for a, w in aspects.ASPECTS.items()]),
                         hide_index=True, width="stretch")

    # ---------------------------------------------------------- 8. review table
    st.subheader("All reviews")
    f1, f2, f3 = st.columns([3, 2, 3])
    pick = f1.multiselect("Sentiment", indicators.SENTIMENTS, default=indicators.SENTIMENTS)
    min_conf = f2.slider("Minimum confidence", 0.5, 1.0, 0.5, 0.05)
    keyword = f3.text_input("Keyword search (e.g. កៅអី)")
    reviews = view[view["sentiment"].isin(pick) & (view["confidence"] >= min_conf)]
    if keyword:
        reviews = reviews[reviews[text_column].astype(str).str.contains(keyword, case=False, regex=False)]
    show_columns = [text_column, "sentiment", "confidence", "language"] + (["top words"] if model_key == "tfidf_logreg" else [])
    if category_column != "(none)":
        show_columns.insert(1, category_column)
    if date_column != "(none)":
        show_columns.insert(1, date_column)
    st.caption(f"{len(reviews):,} reviews" + (" · *top words* = the 3 words that pushed the TF-IDF model most towards "
                                               "its answer" if model_key == "tfidf_logreg" else ""))
    st.dataframe(reviews[show_columns], hide_index=True, width="stretch", height=420,
                 column_config={text_column: st.column_config.TextColumn(width="large"),
                                "top words": st.column_config.TextColumn(width="medium"),
                                "confidence": st.column_config.ProgressColumn(min_value=0.5, max_value=1.0, format="%.2f")})

    # ---------------------------------------------------------- 9. download
    st.subheader("Download the results")
    result = table.copy()
    added = ["sentiment", "confidence", "prob_negative", "prob_positive", "language"]
    for column in added:
        result[column] = data[column].reindex(table.index)          # empty rows get no prediction
    result["sentiment"] = result["sentiment"].astype(object).where(~is_empty, "(empty)")
    base = os.path.splitext(source_name)[0]
    d1, d2 = st.columns(2)
    d1.download_button("⬇️ CSV (opens in Excel with Khmer letters)", result.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{base}_sentiment.csv", mime="text/csv", width="stretch")
    excel = io.BytesIO()
    result.to_excel(excel, index=False, engine="openpyxl")
    d2.download_button("⬇️ Excel (.xlsx)", excel.getvalue(), file_name=f"{base}_sentiment.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", width="stretch")


with tab_file:
    render_file_dashboard()
