"""Synthetic training decisions: a bigger model (Qwen3.5-9B) writes realistic decisions across many domains and
decision types, in the Decisions API shape. The held-out task types are excluded from what it may write.

  -> data/decisions/synthetic.jsonl  rows {"context", "question", "answers", "label"}
"""
import json
import random
import re

from .config import BIG_LLM, DATA

N_DECISIONS = 12_000
EXAMPLES_PER_DECISION = 6
DOMAINS = [
    "an online store's customer support", "airline customer service", "a mobile carrier's support desk", "an internet provider's support",
    "insurance claims intake", "a SaaS product's support", "a company IT helpdesk", "an HR department", "recruiting and job applications",
    "a hospital front desk (scheduling only)", "a pharmacy's order desk", "a dental clinic's scheduling", "university admissions",
    "an online course platform", "a school's parent messages", "a smart home assistant", "a car's voice assistant", "a ride-hailing app",
    "a food delivery app", "restaurant reservations", "a hotel front desk", "a travel agency", "apartment rentals",
    "property maintenance requests", "an electricity and water utility", "a city permits office", "a tax office help desk",
    "a law firm's client intake", "social media content moderation", "an online marketplace's listing review", "a gaming community",
    "an email inbox assistant", "a calendar assistant", "expense report approvals", "company purchasing", "parcel shipping and tracking",
    "warehouse inventory", "factory quality control", "security alert triage", "cloud incident response", "a software bug tracker",
    "an automated code review bot", "app store review handling", "event ticketing", "a fitness coaching app", "a veterinary clinic",
    "farm sensor monitoring", "retail store operations", "a public library", "a museum's visitor services", "airport operations",
    "public transit operations", "a sports club's membership desk", "a coding assistant agent", "a web browsing agent",
]
TYPES = [
    "route the case to the right team or workflow", "choose the next action for an AI agent",
    "decide whether a human must confirm before the action runs", "set the priority or urgency", "pick the category of the request",
    "judge how satisfied the customer is", "check whether it breaks a policy", "decide whether to escalate",
    "pick which tool or API to call", "decide whether the request is in scope for this assistant", "assess the risk level",
    "identify what the person mainly wants", "decide whether it duplicates an existing ticket", "decide whether a follow-up is needed",
    "choose the best reply template", "approve or reject", "pick the right department", "pick the required response time",
]
ANSWER_COUNTS = [2, 2, 3, 3, 4, 4, 5, 6, 8]
HELD_OUT_WORDS = re.compile(r"\b(spam\w*|phish\w*|scam\w*|junk mail|type of question|question type|kind of question|kind of answer|type of answer)\b", re.I)
EXCLUDE = ("spam, scam or phishing detection; classifying what kind of answer a question asks for; recognizing emotions; "
           "categorizing news articles by topic; banking, card, top-up or money-transfer requests")
PROMPT = """Write training data for a small decision model.

Setting: {domain}
Decision type: {dtype}
Number of allowed answers: {k}

Invent ONE realistic decision that this system has to make, of the given type. Give:
- "question": the question the system asks itself (one short sentence)
- "answers": exactly {k} short, distinct allowed answers (1-4 words each)
- "examples": 6 different realistic inputs ("context": 1-3 sentences, written like a real message, log line, event or request) each with its correct "answer" copied exactly from "answers". Use as many different answers as possible and make some examples tricky.

Do not write about: {exclude}.
Reply with JSON only."""
SCHEMA = {"type": "object", "required": ["question", "answers", "examples"], "properties": {
    "question": {"type": "string"},
    "answers": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 10},
    "examples": {"type": "array", "minItems": EXAMPLES_PER_DECISION, "maxItems": EXAMPLES_PER_DECISION, "items": {
        "type": "object", "required": ["context", "answer"],
        "properties": {"context": {"type": "string"}, "answer": {"type": "string"}}}}}}


def keep(d: dict) -> list[dict]:
    """The usable examples of one generated decision: distinct answers, no held-out task type, answers from the list."""
    answers = [a.strip() for a in d["answers"] if a.strip()]
    if len(set(a.lower() for a in answers)) != len(answers) or len(answers) < 2:
        return []
    if HELD_OUT_WORDS.search(d["question"] + " | " + " | ".join(answers)):  # the instruction is not always obeyed
        return []
    good = [e for e in d["examples"] if e["answer"].strip() in answers and e["context"].strip()]
    return [{"context": e["context"].strip(), "question": d["question"].strip(), "answers": answers,
             "label": e["answer"].strip()} for e in good]


def generate(n: int = N_DECISIONS, mem: float = 0.4) -> None:
    """Write n decisions with Qwen3.5-9B (vLLM, JSON-constrained) → data/decisions/synthetic.jsonl."""
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    rng = random.Random(0)
    jobs = [(rng.choice(DOMAINS), rng.choice(TYPES), rng.choice(ANSWER_COUNTS)) for _ in range(n)]
    llm = LLM(model=BIG_LLM, max_model_len=4096, gpu_memory_utilization=mem, max_num_seqs=256, seed=0,
              limit_mm_per_prompt={"image": 0, "video": 0})
    msgs = [[{"role": "user", "content": PROMPT.format(domain=d, dtype=t, k=k, exclude=EXCLUDE)}] for d, t, k in jobs]
    sp = [SamplingParams(temperature=1.0, top_p=0.95, max_tokens=1200, seed=i,
                         structured_outputs=StructuredOutputsParams(json=SCHEMA)) for i in range(n)]
    outs = llm.chat(msgs, sp, use_tqdm=True, chat_template_kwargs={"enable_thinking": False})
    rows, kept = [], 0
    for o in outs:
        try:
            good = keep(json.loads(o.outputs[0].text))
        except json.JSONDecodeError:
            continue
        kept += bool(good)
        rows += good
    with open(DATA / "decisions" / "synthetic.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    print(f"kept {kept}/{n} decisions -> {len(rows)} examples")
