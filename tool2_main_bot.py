from __future__ import annotations

import asyncio
import csv
import io
import math
import os
import random
import time
from dataclasses import dataclass
from datetime import date
from typing import Iterable
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from urllib.request import Request, urlopen

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)


DEFAULT_SHEET_URL = os.getenv(
    "SHEET_URL",
    "https://docs.google.com/spreadsheets/d/e/2PACX-1vT0WLgPLd7IQCod4YfX5SnNqifk_EFZGPta_tMOe317KTOp5P4HeiF8RzuS8q_t7txSKR-tC3cC3OMr/pub?output=csv",
)

DEFAULT_KAPPA = 0.900
DEFAULT_CAP_NEG = -15.0
DEFAULT_M = 1
DEFAULT_MC = 8000

OCE, N_VALUE, KAPPA, CAP_NEG, M_VALUE, MC_VALUE = range(6)


@dataclass(frozen=True)
class Regression:
    a: float
    b: float


@dataclass(frozen=True)
class AuctionColumn:
    oce: float
    bids: tuple[float, ...]


@dataclass(frozen=True)
class TrainedModel:
    trained_on: str
    auctions: int
    total_bids: int
    oce2n: Regression
    oce2mu: Regression
    oce2sig: Regression
    sqrt_k: float
    g_p_lo: float
    g_p_hi: float
    lo_mu: float
    lo_sig: float
    hi_mu: float
    hi_sig: float
    avg_n: float


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def mean(values: Iterable[float]) -> float:
    values = tuple(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def std(values: Iterable[float]) -> float:
    values = tuple(values)
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / len(values))


def percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    index = (p / 100.0) * (len(sorted_values) - 1)
    lo = math.floor(index)
    hi = math.ceil(index)
    if lo == hi:
        return sorted_values[lo]
    frac = index - lo
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac


def looks_like_html(text: str) -> bool:
    sample = text.strip().lower()[:300]
    return (
        sample.startswith("<!doctype")
        or sample.startswith("<html")
        or sample.startswith("<head")
        or sample.startswith("<body")
        or "<table" in sample
        or "<meta " in sample
    )


def normalize_sheet_url(raw_url: str) -> str:
    url = (raw_url or "").strip()
    if not url:
        return ""

    try:
        parsed = urlparse(url)
    except ValueError:
        return url

    if not parsed.netloc.lower().endswith("docs.google.com"):
        return url

    path = parsed.path
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))

    if path.endswith("/pubhtml"):
        path = path[:-8] + "/pub"
        query["output"] = "csv"
    elif path.endswith("/pub"):
        query["output"] = "csv"
    elif path.endswith("/edit") or path.endswith("/htmlview"):
        base = path.rsplit("/", 1)[0]
        path = f"{base}/export"
        query.pop("usp", None)
        query["format"] = "csv"
    elif path.endswith("/export"):
        query["format"] = "csv"

    return urlunparse(parsed._replace(path=path, query=urlencode(query)))


def with_cache_bust(url: str) -> str:
    parsed = urlparse(url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query["_cb"] = str(int(time.time() * 1000))
    return urlunparse(parsed._replace(query=urlencode(query)))


def fetch_text(url: str) -> str:
    request = Request(
        with_cache_bust(normalize_sheet_url(url)),
        headers={"User-Agent": "Mozilla/5.0"},
    )
    with urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8", errors="replace")


def parse_csv(raw: str) -> list[list[str]]:
    clean = (
        raw.replace("\ufeff", "", 1)
        .removeprefix("```csv\n")
        .removeprefix("```\n")
        .removesuffix("\n```")
        .strip()
    )
    reader = csv.reader(io.StringIO(clean))
    rows: list[list[str]] = []
    for row in reader:
        if any(cell.strip() for cell in row):
            rows.append([cell.strip() for cell in row])
    return rows


def parse_sheet(raw_csv: str) -> list[AuctionColumn]:
    if looks_like_html(raw_csv):
        raise ValueError(
            "Google Sheets returned HTML instead of CSV. Use a published sheet URL or a link ending in output=csv."
        )

    rows = parse_csv(raw_csv)
    if len(rows) < 3:
        raise ValueError(
            f"Only {len(rows)} rows found after parsing. The response does not look like a multi-row CSV sheet."
        )

    oce_index = -1
    for index, row in enumerate(rows[:6]):
        first = row[0] if row else ""
        if "OCE" in first.upper():
            oce_index = index
            break
    if oce_index < 0:
        oce_index = 1

    header_row = rows[max(0, oce_index - 1)]
    oce_row = rows[oce_index]
    bid_rows = rows[oce_index + 1 :]

    auctions: list[AuctionColumn] = []
    for col in range(1, len(oce_row)):
        oce_raw = oce_row[col].replace(",", "").strip() if col < len(oce_row) else ""
        if not oce_raw:
            continue

        try:
            oce = float(oce_raw)
        except ValueError:
            continue
        if not math.isfinite(oce) or oce <= 0:
            continue

        _column_id = header_row[col].strip() if col < len(header_row) else str(col)
        bids: list[float] = []
        for row in bid_rows:
            if col >= len(row):
                continue
            value_text = row[col].strip().replace(",", "")
            if not value_text:
                continue
            try:
                value = float(value_text)
            except ValueError:
                continue
            if math.isfinite(value):
                bids.append(value)

        if len(bids) < 2:
            continue

        auctions.append(AuctionColumn(oce=oce, bids=tuple(sorted(bids))))

    if not auctions:
        raise ValueError("No valid auction columns found.")

    return auctions


def fit_dist(bids: list[float]) -> dict[str, float]:
    values = sorted(bids)
    q1 = percentile(values, 25)
    q3 = percentile(values, 75)
    iqr = q3 - q1
    lo_f = q1 - 1.5 * iqr
    hi_f = q3 + 1.5 * iqr

    low_out = [value for value in values if value < lo_f]
    high_out = [value for value in values if value > hi_f]
    core = [value for value in values if lo_f <= value <= hi_f]

    return {
        "core_mu": mean(core) if core else mean(values),
        "core_sig": std(core) if len(core) > 1 else max(std(values), 0.1),
        "p_lo": len(low_out) / len(values),
        "p_hi": len(high_out) / len(values),
        "lo_mu": mean(low_out) if low_out else mean(values) - 15.0,
        "lo_sig": std(low_out) if len(low_out) > 1 else 3.0,
        "hi_mu": mean(high_out) if high_out else mean(values) + 8.0,
        "hi_sig": std(high_out) if len(high_out) > 1 else 3.0,
        "spread": values[-1] - values[0],
    }


def lin_reg(xs: list[float], ys: list[float]) -> Regression:
    mx = mean(xs)
    my = mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / len(xs)
    var_x = sum((x - mx) ** 2 for x in xs) / len(xs)
    b = cov / var_x if var_x else 0.0
    return Regression(a=my - b * mx, b=b)


def build_model(auctions: list[AuctionColumn]) -> TrainedModel:
    per_fit = []
    for auction in auctions:
        fit = fit_dist(list(auction.bids))
        per_fit.append((auction, fit))

    log_oce = [math.log(auction.oce) for auction in auctions]
    oce2n = lin_reg(log_oce, [math.log(len(auction.bids)) for auction in auctions])
    oce2mu = lin_reg(log_oce, [fit["core_mu"] for _, fit in per_fit])
    oce2sig = lin_reg(log_oce, [fit["core_sig"] for _, fit in per_fit])

    spread_num = sum(fit["spread"] * math.sqrt(len(auction.bids)) for auction, fit in per_fit)
    spread_den = sum(len(auction.bids) for auction in auctions)

    all_bids = [bid for auction in auctions for bid in auction.bids]
    global_fit = fit_dist(all_bids)
    avg_n = mean(len(auction.bids) for auction in auctions)

    return TrainedModel(
        trained_on=date.today().isoformat(),
        auctions=len(auctions),
        total_bids=len(all_bids),
        oce2n=oce2n,
        oce2mu=oce2mu,
        oce2sig=oce2sig,
        sqrt_k=(spread_num / spread_den) if spread_den else 5.0,
        g_p_lo=global_fit["p_lo"],
        g_p_hi=global_fit["p_hi"],
        lo_mu=global_fit["lo_mu"],
        lo_sig=global_fit["lo_sig"],
        hi_mu=global_fit["hi_mu"],
        hi_sig=global_fit["hi_sig"],
        avg_n=avg_n or 1.0,
    )


def load_model_from_sheet(sheet_url: str) -> TrainedModel:
    raw_csv = fetch_text(sheet_url)
    auctions = parse_sheet(raw_csv)
    return build_model(auctions)


def mu_sig(model: TrainedModel, oce: float, n: int) -> tuple[float, float]:
    log_oce = math.log(oce)
    mu = model.oce2mu.a + model.oce2mu.b * log_oce
    base = model.oce2sig.a + model.oce2sig.b * log_oce
    sig = max(0.5, base * math.sqrt(n / model.avg_n))
    return mu, sig


def trunc_normal(mu: float, sig: float, lo: float, hi: float, tries: int = 15) -> float:
    sigma = max(sig, 1e-9)
    for _ in range(tries):
        value = random.gauss(mu, sigma)
        if lo <= value <= hi:
            return value
    return clamp(mu, lo, hi)


def sample_pct(model: TrainedModel, mu: float, sig: float, n: int) -> float:
    spread = model.sqrt_k * math.sqrt(n)
    low = mu - spread * 1.2
    high = mu + spread * 0.8
    roll = random.random()
    if roll < model.g_p_lo:
        return trunc_normal(model.lo_mu, model.lo_sig, low - 10.0, mu - 4.0)
    if roll < model.g_p_lo + model.g_p_hi:
        return trunc_normal(model.hi_mu, model.hi_sig, mu + 3.0, high + 10.0)
    return trunc_normal(mu, sig, low, high)


def gen_scenarios(
    model: TrainedModel,
    oce: float,
    n: int,
    mu: float,
    sig: float,
    count: int,
) -> list[list[float]]:
    scenarios: list[list[float]] = []
    for _ in range(count):
        bids = [oce * (1.0 + sample_pct(model, mu, sig, n) / 100.0) for _ in range(n)]
        bids.sort()
        scenarios.append(bids)
    return scenarios


def pred_order(
    model: TrainedModel,
    oce: float,
    n: int,
    mu: float,
    sig: float,
    sims: int = 2000,
) -> list[dict[str, float]]:
    per_rank: list[list[float]] = [[] for _ in range(n)]
    for _ in range(sims):
        draws = [sample_pct(model, mu, sig, n) for _ in range(n)]
        draws.sort()
        for index, value in enumerate(draws):
            per_rank[index].append(value)

    summary: list[dict[str, float]] = []
    for index, values in enumerate(per_rank):
        sorted_values = sorted(values)
        summary.append(
            {
                "rank": float(index + 1),
                "mu": mean(values),
                "sig": std(values),
                "p5": sorted_values[int(math.floor(sims * 0.05))],
                "p95": sorted_values[int(math.floor(sims * 0.95))],
            }
        )
    return summary


def comp_t(bids: list[float], oce: float, kappa: float) -> float:
    avg = mean(bids)
    x = 0.5 * avg + 0.2 * oce + 0.3 * (kappa * oce)
    s2 = sum((value - x) ** 2 for value in bids)
    return x - math.sqrt(s2 / len(bids))


def eval_b(
    my_bids: list[float],
    scenarios: list[list[float]],
    oce: float,
    kappa: float,
    upper: float,
) -> dict[str, float | list[float]]:
    epsilon = 1e-9
    wins = 0
    sum_win_bid = 0.0
    sum_t = 0.0

    for comp in scenarios:
        t_value = comp_t(comp + my_bids, oce, kappa)
        sum_t += t_value

        min_comp_above = math.inf
        for value in comp:
            if value > t_value + epsilon and value < min_comp_above:
                min_comp_above = value

        min_my_above = math.inf
        best_index = -1
        for index, value in enumerate(my_bids):
            if value > t_value + epsilon and value <= upper and value < min_my_above:
                min_my_above = value
                best_index = index

        if best_index < 0:
            continue
        if min_my_above <= min_comp_above + epsilon:
            wins += 1
            sum_win_bid += min_my_above

    scenario_count = len(scenarios)
    return {
        "bids": my_bids,
        "winRate": (wins / scenario_count) if scenario_count else 0.0,
        "avgWinBid": (sum_win_bid / wins) if wins else math.inf,
        "avgT": (sum_t / scenario_count) if scenario_count else math.nan,
    }


def normalize_bids(bids: list[float], low: float, high: float) -> list[float]:
    return sorted(clamp(value, low, high) for value in bids)


def bid_key(bids: list[float]) -> str:
    return "|".join(str(int(round(value * 10))) for value in bids)


def better(a: dict[str, float | list[float]], b: dict[str, float | list[float]]) -> bool:
    diff = float(a["winRate"]) - float(b["winRate"])
    if abs(diff) > 1e-9:
        return diff > 0
    return float(a["avgWinBid"]) < float(b["avgWinBid"])


def sort_candidates(items: list[dict[str, float | list[float]]]) -> list[dict[str, float | list[float]]]:
    return sorted(items, key=lambda item: (-float(item["winRate"]), float(item["avgWinBid"])))


def gen_cands(m: int, low: float, high: float, avg_t: float, extra_random: int = 300) -> list[list[float]]:
    unique: dict[str, list[float]] = {}
    span = high - low

    def add(raw_bids: list[float]) -> None:
        bids = normalize_bids(raw_bids, low, high)
        unique[bid_key(bids)] = bids

    target_bid = avg_t if low < avg_t < high else low + span * 0.25

    for offset in (-0.06, -0.04, -0.02, -0.01, 0.0, 0.01, 0.02, 0.04, 0.07, 0.10, 0.14):
        bids = [target_bid + (offset + index * 0.012) * span for index in range(m)]
        add(bids)

    for shift in (-0.10, -0.06, -0.03, 0.0, 0.03, 0.06, 0.10):
        bids = []
        for index in range(m):
            pos = clamp((index + 1) / (m + 1) + shift, 0.01, 0.99)
            bids.append(low + pos * span)
        add(bids)

    for _ in range(extra_random):
        bids = [low + random.random() * span for _ in range(m)]
        add(bids)

    return list(unique.values())


def refine(top: list[dict[str, float | list[float]]], m: int, low: float, high: float) -> list[list[float]]:
    unique: dict[str, list[float]] = {}
    span = high - low

    def add(raw_bids: list[float]) -> None:
        bids = normalize_bids(raw_bids, low, high)
        unique[bid_key(bids)] = bids

    for item in top:
        bids = list(item["bids"])  # type: ignore[arg-type]
        add(bids)

        for sigma in (0.005, 0.012, 0.022, 0.036):
            for _ in range(6):
                add([value + random.gauss(0.0, sigma * span) for value in bids])

        denom = max(1, m - 1)
        for shift in (-0.008, 0.008, -0.015, 0.015):
            add([value + shift * span * (1 + index / denom) for index, value in enumerate(bids)])

    return list(unique.values())


def solve_best_bid(
    model: TrainedModel,
    oce: float,
    n_value: int | None,
    kappa: float,
    cap_neg: float,
    m_value: int,
    mc_value: int,
) -> tuple[float, float]:
    n = n_value
    if n is None or n < 1:
        n = max(1, round(math.exp(model.oce2n.a + model.oce2n.b * math.log(oce))))

    lower = max(0.01, (1.0 + cap_neg / 100.0) * oce)
    upper = 1.1 * oce
    if lower >= upper:
        raise ValueError("Cap too high. Lower cap must stay below the 110% OCE ceiling.")

    mu, sig = mu_sig(model, oce, n)
    order = pred_order(model, oce, n, mu, sig, sims=2000)
    scenarios = gen_scenarios(model, oce, n, mu, sig, mc_value)

    mean_bids = [oce * (1.0 + item["mu"] / 100.0) for item in order]
    avg_t = comp_t(mean_bids + [lower + 0.3 * (upper - lower)], oce, kappa)

    phase1 = [
        eval_b(bids, scenarios, oce, kappa, upper)
        for bids in gen_cands(m_value, lower, upper, avg_t, extra_random=300)
    ]
    phase1 = sort_candidates(phase1)

    phase2 = [
        eval_b(bids, scenarios, oce, kappa, upper)
        for bids in refine(phase1[:40], m_value, lower, upper)
    ]
    phase2 = sort_candidates(phase2)

    merged: dict[str, dict[str, float | list[float]]] = {}
    for item in phase1:
        merged[bid_key(item["bids"])] = item  # type: ignore[arg-type]
    for item in phase2:
        key = bid_key(item["bids"])  # type: ignore[arg-type]
        prev = merged.get(key)
        if prev is None or better(item, prev):
            merged[key] = item

    all_candidates = sort_candidates(list(merged.values()))
    if not all_candidates:
        raise ValueError("No winning bid found. Try lowering the cap.")

    best = all_candidates[0]
    if float(best["winRate"]) <= 0:
        raise ValueError("No winning bid found. Try lowering the cap.")

    best_b1 = list(best["bids"])[0]  # type: ignore[arg-type]
    best_pct = ((best_b1 / oce) - 1.0) * 100.0
    win_rate = float(best["winRate"]) * 100.0
    return best_pct, win_rate


def get_session(context: ContextTypes.DEFAULT_TYPE) -> dict[str, object]:
    return context.user_data.setdefault("tool2_bot_session", {})


def clear_session(context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data.pop("tool2_bot_session", None)


def is_default(text: str) -> bool:
    return text.strip().lower() in {"default", "d"}


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Use /run to fetch the sheet, train the model, and enter the inputs step by step.\n"
        "Use /cancel to stop the current run."
    )


async def cancel_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    clear_session(context)
    await update.message.reply_text("Cancelled.")
    return ConversationHandler.END


async def run_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    clear_session(context)
    session = get_session(context)
    sheet_url = DEFAULT_SHEET_URL

    await update.message.reply_text("Fetching sheet data and training model...")
    try:
        model = await asyncio.to_thread(load_model_from_sheet, sheet_url)
    except Exception as exc:
        clear_session(context)
        await update.message.reply_text(f"Could not load the sheet: {exc}")
        return ConversationHandler.END

    session["model"] = model
    await update.message.reply_text(
        f"Model ready from {model.auctions} auctions and {model.total_bids} bids.\n"
        "Enter Official Cost Estimate (OCE)."
    )
    return OCE


async def oce_step(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    text = update.message.text.strip()
    try:
        oce = float(text)
    except ValueError:
        await update.message.reply_text("Enter a numeric OCE greater than 0.")
        return OCE

    if not math.isfinite(oce) or oce <= 0:
        await update.message.reply_text("Enter a numeric OCE greater than 0.")
        return OCE

    session = get_session(context)
    session["oce"] = oce
    await update.message.reply_text(
        "Enter number of bidders n (1-200), or type auto to predict it from OCE."
    )
    return N_VALUE


async def n_step(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    text = update.message.text.strip().lower()
    session = get_session(context)

    if text in {"auto", "a"}:
        session["n_value"] = None
    else:
        try:
            n_value = int(text)
        except ValueError:
            await update.message.reply_text("Enter an integer from 1 to 200, or type auto.")
            return N_VALUE
        if n_value < 1 or n_value > 200:
            await update.message.reply_text("Enter an integer from 1 to 200, or type auto.")
            return N_VALUE
        session["n_value"] = n_value

    await update.message.reply_text(
        f"Enter NPPI coefficient kappa (0.700-1.000). Type default for {DEFAULT_KAPPA:.3f}."
    )
    return KAPPA


async def kappa_step(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    text = update.message.text.strip()
    if is_default(text):
        kappa = DEFAULT_KAPPA
    else:
        try:
            kappa = float(text)
        except ValueError:
            await update.message.reply_text("Enter a number between 0.700 and 1.000.")
            return KAPPA

    if not math.isfinite(kappa) or not (0.7 <= kappa <= 1.0):
        await update.message.reply_text("Enter a number between 0.700 and 1.000.")
        return KAPPA

    session = get_session(context)
    session["kappa"] = kappa
    await update.message.reply_text(
        f"Enter lower cap percent vs OCE (-40 to 5). Type default for {DEFAULT_CAP_NEG:.0f}."
    )
    return CAP_NEG


async def cap_step(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    text = update.message.text.strip()
    if is_default(text):
        cap_neg = DEFAULT_CAP_NEG
    else:
        try:
            cap_neg = float(text)
        except ValueError:
            await update.message.reply_text("Enter a number from -40 to 5.")
            return CAP_NEG

    if not math.isfinite(cap_neg) or not (-40.0 <= cap_neg <= 5.0):
        await update.message.reply_text("Enter a number from -40 to 5.")
        return CAP_NEG

    session = get_session(context)
    session["cap_neg"] = cap_neg
    await update.message.reply_text(
        f"Enter number of my bids m (1-6). Type default for {DEFAULT_M}."
    )
    return M_VALUE


async def m_step(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    text = update.message.text.strip()
    if is_default(text):
        m_value = DEFAULT_M
    else:
        try:
            m_value = int(text)
        except ValueError:
            await update.message.reply_text("Enter an integer from 1 to 6.")
            return M_VALUE

    if m_value < 1 or m_value > 6:
        await update.message.reply_text("Enter an integer from 1 to 6.")
        return M_VALUE

    session = get_session(context)
    session["m_value"] = m_value
    await update.message.reply_text(
        f"Enter Monte Carlo scenarios (500-50000). Type default for {DEFAULT_MC}."
    )
    return MC_VALUE


async def mc_step(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    text = update.message.text.strip()
    if is_default(text):
        mc_value = DEFAULT_MC
    else:
        try:
            mc_value = int(text)
        except ValueError:
            await update.message.reply_text("Enter an integer from 500 to 50000.")
            return MC_VALUE

    if mc_value < 500 or mc_value > 50000:
        await update.message.reply_text("Enter an integer from 500 to 50000.")
        return MC_VALUE

    session = get_session(context)
    model = session.get("model")
    oce = session.get("oce")
    if not isinstance(model, TrainedModel) or not isinstance(oce, float):
        clear_session(context)
        await update.message.reply_text("Session expired. Run /run again.")
        return ConversationHandler.END

    await update.message.reply_text("Running optimizer...")
    try:
        best_pct, win_rate = await asyncio.to_thread(
            solve_best_bid,
            model,
            oce,
            session.get("n_value"),  # type: ignore[arg-type]
            float(session["kappa"]),
            float(session["cap_neg"]),
            int(session["m_value"]),
            mc_value,
        )
    except Exception as exc:
        clear_session(context)
        await update.message.reply_text(str(exc))
        return ConversationHandler.END

    clear_session(context)
    await update.message.reply_text(
        f"Best bid B1={best_pct:.2f}%, Win Rate={win_rate:.2f}%"
    )
    return ConversationHandler.END


def build_application() -> object:
    token = os.getenv("BOT_TOKEN")
    if not token:
        raise SystemExit("Set BOT_TOKEN before running this bot.")

    app = ApplicationBuilder().token(token).build()

    conversation = ConversationHandler(
        entry_points=[CommandHandler("run", run_command)],
        states={
            OCE: [MessageHandler(filters.TEXT & ~filters.COMMAND, oce_step)],
            N_VALUE: [MessageHandler(filters.TEXT & ~filters.COMMAND, n_step)],
            KAPPA: [MessageHandler(filters.TEXT & ~filters.COMMAND, kappa_step)],
            CAP_NEG: [MessageHandler(filters.TEXT & ~filters.COMMAND, cap_step)],
            M_VALUE: [MessageHandler(filters.TEXT & ~filters.COMMAND, m_step)],
            MC_VALUE: [MessageHandler(filters.TEXT & ~filters.COMMAND, mc_step)],
        },
        fallbacks=[CommandHandler("cancel", cancel_command)],
        allow_reentry=True,
    )

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("cancel", cancel_command))
    app.add_handler(conversation)
    return app


def main() -> None:
    application = build_application()
    application.run_polling()


if __name__ == "__main__":
    main()
