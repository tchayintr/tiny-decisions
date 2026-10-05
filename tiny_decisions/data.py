"""Download the data and recast public classification sets as decisions: (context, question, allowed answers) -> answer.

  data/raw/            downloaded files (a FineWeb-Edu shard for pretraining + the datasets below)
  data/decisions/      <task>.jsonl rows {"context", "label"[, "question"]} + tasks.json (questions, answers, role)

The held-out tasks are never used for the tokenizer, pretraining or decision training.
"""
import json
import random
import re
import urllib.request
from collections.abc import Callable

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from .config import DATA

RAW, OUT = DATA / "raw", DATA / "decisions"
CAP_TRAIN, CAP_EVAL, MAX_CHARS = 30_000, 2_000, 1_200   # rows kept per training / held-out task, characters per row
CONVERT = "refs/convert/parquet"
FINEWEB_SHARD = ("HuggingFaceFW/fineweb-edu", "sample/10BT/000_00000.parquet")
BANKING77_TEST = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv"


def get(repo: str, file: str, rev: str | None = None) -> str:
    return hf_hub_download(repo, file, repo_type="dataset", revision=rev, local_dir=RAW / repo.replace("/", "__"))


def names(path: str, col: str) -> list[str]:
    """ClassLabel names stored in a Hugging Face parquet file's schema metadata."""
    info = json.loads((pq.read_schema(path).metadata or {}).get(b"huggingface", b"{}"))
    feat = info.get("info", {}).get("features", {}).get(col, {})
    return feat.get("names") or feat.get("feature", {}).get("names")


def human(s: str) -> str:
    s = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", s.strip().rstrip("?"))  # CamelCase -> Camel Case
    return s.replace("_", " ").lower()


def pq_rows(repo: str, file: str, rev: str | None = None) -> tuple[pd.DataFrame, str]:
    path = get(repo, file, rev)
    return pd.read_parquet(path), path


def classlabel_task(repo: str, file: str, text_cols: list[str], label_col: str, rev: str | None = None,
                    rename: dict | None = None, drop=(), fmt: Callable | None = None) -> list[dict]:
    df, path = pq_rows(repo, file, rev)
    lab = names(path, label_col)
    rows = []
    for r in df.itertuples(index=False):
        name = lab[getattr(r, label_col)]
        if name in drop:
            continue
        ctx = fmt(r) if fmt else " ".join(str(getattr(r, c)) for c in text_cols)
        rows.append({"context": ctx, "label": (rename or {}).get(name, human(name))})
    return rows


NG20 = {"alt.atheism": "atheism", "comp.graphics": "computer graphics", "comp.os.ms-windows.misc": "windows",
        "comp.sys.ibm.pc.hardware": "pc hardware", "comp.sys.mac.hardware": "mac hardware", "comp.windows.x": "x window system",
        "misc.forsale": "for sale", "rec.autos": "cars", "rec.motorcycles": "motorcycles", "rec.sport.baseball": "baseball",
        "rec.sport.hockey": "hockey", "sci.crypt": "cryptography", "sci.electronics": "electronics", "sci.med": "medicine",
        "sci.space": "space", "soc.religion.christian": "christianity", "talk.politics.guns": "gun politics",
        "talk.politics.mideast": "middle east politics", "talk.politics.misc": "politics", "talk.religion.misc": "religion"}
LANGS = {"ar": "arabic", "bg": "bulgarian", "de": "german", "el": "greek", "en": "english", "es": "spanish", "fr": "french",
         "hi": "hindi", "it": "italian", "ja": "japanese", "nl": "dutch", "pl": "polish", "pt": "portuguese", "ru": "russian",
         "sw": "swahili", "th": "thai", "tr": "turkish", "ur": "urdu", "vi": "vietnamese", "zh": "chinese"}
YESNO = {"entailment": "yes", "not entailment": "no", "not_entailment": "no", "duplicate": "yes", "not_duplicate": "no",
         "equivalent": "yes", "not_equivalent": "no", "acceptable": "yes", "unacceptable": "no"}


def massive(label_col: str) -> list[dict]:
    df, path = pq_rows("AmazonScience/massive", "en-US/train/0000.parquet", CONVERT)
    lab = names(path, label_col)
    return [{"context": r.utt, "label": human(lab[getattr(r, label_col)])} for r in df.itertuples(index=False)]


def bitext(col: str) -> list[dict]:
    df = pd.read_csv(get("bitext/Bitext-customer-support-llm-chatbot-training-dataset",
                         "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"))
    return [{"context": r.instruction, "label": human(getattr(r, col))} for r in df.itertuples(index=False)]


def boolq() -> list[dict]:
    df, _ = pq_rows("google/boolq", "data/train-00000-of-00001.parquet")
    return [{"context": r.passage, "question": r.question.strip().capitalize() + "?", "label": "yes" if r.answer else "no"}
            for r in df.itertuples(index=False)]


def go_emotions() -> list[dict]:
    df, path = pq_rows("google-research-datasets/go_emotions", "simplified/train-00000-of-00001.parquet")
    lab = names(path, "labels")
    return [{"context": r.text, "label": lab[r.labels[0]]} for r in df.itertuples(index=False) if len(r.labels) == 1]


def banking77() -> list[dict]:
    """Banking77's official test set (PolyAI's CSV)."""
    path = RAW / "PolyAI-LDN__task-specific-datasets" / "test.csv"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(BANKING77_TEST, path)
    df = pd.read_csv(path)
    return [{"context": r.text, "label": human(r.category)} for r in df.itertuples(index=False)]


def sms_spam() -> list[dict]:
    df, path = pq_rows("ucirvine/sms_spam", "plain_text/train-00000-of-00001.parquet")
    spam = [{"context": r.sms.strip(), "label": "spam"} for r in df.itertuples(index=False) if r.label == 1]
    ham = [{"context": r.sms.strip(), "label": "not spam"} for r in df.itertuples(index=False) if r.label == 0]
    return spam + random.Random(0).sample(ham, len(spam))  # balanced, so chance is 50%


Q_INTENT = ["What does the user want?", "Which intent matches this request?", "Which action should the assistant take?"]
TASKS = {
    # ---------------- training tasks
    "clinc150": (Q_INTENT, lambda: classlabel_task("clinc/clinc_oos", "plus/train-00000-of-00001.parquet", ["text"], "intent", drop={"oos"})),
    "massive_intent": (Q_INTENT, lambda: massive("intent")),
    "massive_scenario": (["Which area of the assistant does this command belong to?", "What is this command about?"], lambda: massive("scenario")),
    "bitext_intent": (["What does the customer want?", "Which support workflow fits this request?"], lambda: bitext("intent")),
    "bitext_category": (["Which department should handle this request?", "Route this ticket to a team."], lambda: bitext("category")),
    "dbpedia": (["What kind of thing is this text about?", "Which category fits this description?"],
                lambda: classlabel_task("fancyzhx/dbpedia_14", "dbpedia_14/train-00000-of-00001.parquet", ["title", "content"], "label")),
    "yahoo": (["Which topic does this question belong to?", "Where should this question be posted?"],
              lambda: classlabel_task("community-datasets/yahoo_answers_topics", "yahoo_answers_topics/train-00000-of-00002.parquet",
                                      ["question_title", "question_content"], "topic")),
    "newsgroups": (["Which discussion group is this post from?", "What is this post about?"],
                   lambda: [{"context": r["text"], "label": NG20[r["label_text"]]} for r in map(json.loads, open(get("SetFit/20_newsgroups", "train.jsonl")))]),
    "sib200": (["What is the topic of this sentence?"],
               lambda: [{"context": r.text, "label": r.category} for r in pq_rows("Davlan/sib200", "eng_Latn/train/0000.parquet", CONVERT)[0].itertuples(index=False)]),
    "sst2": (["What is the sentiment of this sentence?", "Is this opinion positive or negative?"],
             lambda: classlabel_task("stanfordnlp/sst2", "data/train-00000-of-00001.parquet", ["sentence"], "label")),
    "tweet_sentiment": (["What is the tone of this tweet?", "How does the writer feel?"],
                        lambda: classlabel_task("cardiffnlp/tweet_eval", "sentiment/train-00000-of-00001.parquet", ["text"], "label")),
    "tweet_emotion": (["Which emotion does this tweet express?"],
                      lambda: classlabel_task("cardiffnlp/tweet_eval", "emotion/train-00000-of-00001.parquet", ["text"], "label")),
    "tweet_hate": (["Is this tweet hate speech?", "Should moderation flag this tweet as hateful?"],
                   lambda: classlabel_task("cardiffnlp/tweet_eval", "hate/train-00000-of-00001.parquet", ["text"], "label",
                                           rename={"non-hate": "not hateful", "hate": "hateful"})),
    "tweet_offensive": (["Is this tweet offensive?", "Should this post be hidden as offensive?"],
                        lambda: classlabel_task("cardiffnlp/tweet_eval", "offensive/train-00000-of-00001.parquet", ["text"], "label",
                                                rename={"non-offensive": "not offensive", "offensive": "offensive"})),
    "tweet_irony": (["Is this tweet ironic?"],
                    lambda: classlabel_task("cardiffnlp/tweet_eval", "irony/train-00000-of-00001.parquet", ["text"], "label",
                                            rename={"non_irony": "not ironic", "irony": "ironic"})),
    "go_emotions": (["Which emotion is expressed here?", "How is the writer feeling?"], go_emotions),
    "mnli": (["Given the premise, is the hypothesis true?"],
             lambda: classlabel_task("nyu-mll/multi_nli", "data/train-00000-of-00001.parquet", [], "label",
                                     rename={"entailment": "true", "neutral": "maybe", "contradiction": "false"},
                                     fmt=lambda r: f"Premise: {r.premise}\nHypothesis: {r.hypothesis}")),
    "qqp": (["Do these two questions ask the same thing?"],
            lambda: classlabel_task("nyu-mll/glue", "qqp/train-00000-of-00001.parquet", [], "label", rename=YESNO,
                                    fmt=lambda r: f"Question 1: {r.question1}\nQuestion 2: {r.question2}")),
    "mrpc": (["Do these two sentences mean the same thing?"],
             lambda: classlabel_task("nyu-mll/glue", "mrpc/train-00000-of-00001.parquet", [], "label", rename=YESNO,
                                     fmt=lambda r: f"Sentence 1: {r.sentence1}\nSentence 2: {r.sentence2}")),
    "rte": (["Does the first text imply the second?"],
            lambda: classlabel_task("nyu-mll/glue", "rte/train-00000-of-00001.parquet", [], "label", rename=YESNO,
                                    fmt=lambda r: f"Text: {r.sentence1}\nClaim: {r.sentence2}")),
    "cola": (["Is this sentence grammatically correct?"],
             lambda: classlabel_task("nyu-mll/glue", "cola/train-00000-of-00001.parquet", ["sentence"], "label", rename=YESNO)),
    "boolq": (None, boolq),  # the question comes with each example
    "language_id": (["Which language is this text written in?"],
                    lambda: [{"context": r.text, "label": LANGS[r.labels]} for r in pd.read_csv(get("papluca/language-identification", "train.csv")).itertuples(index=False)]),
    # ---------------- held-out tasks (never trained on)
    "eval:banking77": (["What is this customer's request about?"], banking77),
    "eval:trec": (["What type of answer is this question asking for?"],
                  lambda: classlabel_task("CogComp/trec", "default/test/0000.parquet", ["text"], "coarse_label", CONVERT,
                                          rename={"ABBR": "abbreviation", "ENTY": "entity", "DESC": "description",
                                                  "HUM": "person", "LOC": "location", "NUM": "number"})),
    "eval:sms_spam": (["Is this message spam?"], sms_spam),
    "eval:emotion": (["Which emotion does this text express?"],
                     lambda: classlabel_task("dair-ai/emotion", "split/test-00000-of-00001.parquet", ["text"], "label")),
    "eval:ag_news": (["Which news section does this article belong to?"],
                     lambda: classlabel_task("fancyzhx/ag_news", "data/test-00000-of-00001.parquet", ["text"], "label",
                                             rename={"World": "world", "Sports": "sports", "Business": "business",
                                                     "Sci/Tech": "science and technology"})),
}


def build() -> None:
    """Download everything and write data/decisions/."""
    OUT.mkdir(parents=True, exist_ok=True)
    print("fineweb-edu shard ->", get(*FINEWEB_SHARD), flush=True)
    meta = {}
    for name, (questions, load) in TASKS.items():
        role = "eval" if name.startswith("eval:") else "train"
        task = name.removeprefix("eval:")
        rows = [r for r in load() if str(r["context"]).strip()]
        for r in rows:
            r["context"] = re.sub(r"\s+", " ", str(r["context"])).strip()[:MAX_CHARS]
        cap = CAP_TRAIN if role == "train" else (CAP_EVAL if task != "banking77" else len(rows))
        rows = random.Random(0).sample(rows, min(cap, len(rows)))
        labels = sorted({r["label"] for r in rows})
        with open(OUT / f"{task}.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        meta[task] = {"role": role, "questions": questions, "answers": labels, "n": len(rows)}
        print(f"{role:5} {task:18} {len(rows):6} rows  {len(labels):3} answers  e.g. {labels[:4]}", flush=True)
    (OUT / "tasks.json").write_text(json.dumps(meta, indent=1))
