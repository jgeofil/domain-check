import asyncio
import csv
import io
import itertools
import random
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import aiofiles
import httpx
from wordfreq import iter_wordlist

VOWELS = "aeiou"
SONORANTS = "lrmnvwys"
SOFT_ONSETS = "lrmnvwystd"
LIQUID_NASAL_CODAS = "lrnm"

SONORANT_BRIDGES = {
    "ln",
    "lm",
    "rl",
    "rn",
    "rm",
    "lv",
    "rv",
    "ll",
    "nn",
    "sl",
    "sn",
    "sm",
}

ONSET_CLUSTERS = {"dr", "dw", "sl", "sm", "sn", "st", "sw", "tr", "tw"}
CODA_BRIDGES = {"ln", "lm", "rl", "rn", "rm", "ll", "nn"}

EUPHONIC_HIATUS = {
    "ae",
    "ai",
    "ao",
    "ea",
    "ei",
    "eo",
    "ia",
    "ie",
    "io",
    "oa",
    "oe",
    "oi",
}

SONORITY_WEIGHTS: dict[str, float] = {
    "a": 5.0,
    "e": 5.0,
    "i": 5.0,
    "o": 5.0,
    "u": 5.0,
    "w": 4.2,
    "y": 4.2,
    "l": 3.8,
    "r": 3.6,
    "m": 3.2,
    "n": 3.2,
    "v": 2.4,
    "s": 2.2,
    "d": 1.6,
    "t": 1.5,
}

TLDS = ("dev", "app")
TARGET_LENGTHS = (4, 5, 6)
SAMPLE_PER_LENGTH = 60  # 60 stems per length x 3 lengths = 180 stems (360 RDAP checks)
OUTPUT_CSV = Path("available_domains.csv")
DICT_PATH = Path("/usr/share/dict/words")
SEED = None  # Set an int (e.g., 42) for reproducible sampling


@dataclass(frozen=True)
class Candidate:
    stem: str
    length: int
    pattern: str
    score: float


type DomainResult = tuple[Candidate, str, str, int]
CSV_HEADERS = [
    "domain",
    "stem",
    "tld",
    "length",
    "pattern",
    "sonority_score",
    "rdap_status",
]


def load_english_lexicon() -> set[str]:
    lexicon: set[str] = {w.lower() for w in iter_wordlist("en") if len(w) in TARGET_LENGTHS and w.isalpha()}
    if DICT_PATH.is_file():
        with DICT_PATH.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                w = line.strip().lower()
                if len(w) in TARGET_LENGTHS and w.isalpha():
                    lexicon.add(w)
    return lexicon


def score_euphony(word: str) -> float | None:
    n = len(word)
    if n not in TARGET_LENGTHS:
        return None
    if any(word[i] == word[i + 1] == word[i + 2] for i in range(n - 2)):
        return None
    if any(bad in word for bad in ("yi", "iy", "wu", "uw", "vv", "ww", "yy", "ss", "rr", "tt", "dd")):
        return None

    for i in range(n - 1):
        a, b = word[i], word[i + 1]
        if a not in VOWELS and b not in VOWELS:
            allowed_clusters = ONSET_CLUSTERS if i == 0 else SONORANT_BRIDGES
            if f"{a}{b}" not in allowed_clusters:
                return None
        if a in VOWELS and b in VOWELS and f"{a}{b}" not in EUPHONIC_HIATUS:
            return None

    mean_sonority = sum(SONORITY_WEIGHTS.get(ch, 0.0) for ch in word) / n
    end_bonus = 0.25 if word[-1] in "aeou" else (0.15 if word[-1] in "lrn" else 0.0)
    diversity_bonus = 0.60 * (len(set(word)) / n)
    return round(mean_sonority + end_bonus + diversity_bonus, 3)


def generate_pool(lexicon: set[str]) -> dict[int, list[Candidate]]:
    bridges = tuple(sorted(SONORANT_BRIDGES))
    onsets = tuple(sorted(ONSET_CLUSTERS))
    coda_bridges = tuple(sorted(CODA_BRIDGES))
    hiatus = tuple(sorted(EUPHONIC_HIATUS))

    templates: list[tuple[str, tuple[Sequence[str], ...]]] = [
        # --- 4-Letter High-Sonority Templates ---
        # 1. CVCV (Two-syllable open legato: nalo, mira)
        ("CVCV", (SOFT_ONSETS, VOWELS, SONORANTS, "aeou")),
        # 2. VCCV (Vowel-initial sonorant bridge: alno, orva)
        ("VCCV", (VOWELS, bridges, "aeou")),
        # 3. VCVC (Vowel-initial liquid/nasal coda: alor, oren)
        ("VCVC", (VOWELS, SONORANTS, "aeo", LIQUID_NASAL_CODAS)),
        # 4. CVVC (Vowel-hiatus + liquid/nasal coda: nael, loen)
        ("CVVC", (SOFT_ONSETS, hiatus, LIQUID_NASAL_CODAS)),
        # 5. CVCC (Sonorant/liquid-nasal coda cluster: sorn, malm, verl)
        ("CVCC", (SOFT_ONSETS, "aeo", coda_bridges)),
        # 6. CCVC (Onset cluster + liquid/nasal coda: slen, tral, drel)
        ("CCVC", (onsets, "aeo", LIQUID_NASAL_CODAS)),
        # 7. CCVV (Onset cluster + euphonic hiatus: slea, dreo, stao)
        ("CCVV", (onsets, hiatus)),
        # 8. VVCV (Vowel-hiatus onset open stem: aelo, iora)
        ("VVCV", (hiatus, SONORANTS, "aeou")),
        # --- 5-Letter High-Sonority Templates ---
        # 9. CVVCV (Vowel-hiatus bridges: naelo, loema)
        ("CVVCV", (SOFT_ONSETS, hiatus, SONORANTS, "aeou")),
        # 10. VCVCV (Vowel-initial open stems: aleno, eliro)
        ("VCVCV", (VOWELS, SONORANTS, VOWELS, SONORANTS, "aeou")),
        # 11. CVCCV (Sonorant-sonorant bridges: solna, melna)
        ("CVCCV", (SOFT_ONSETS, VOWELS, bridges, "aeou")),
        # 12. CVCVC (Liquid/nasal coda stems: lorel, nalor)
        ("CVCVC", (SOFT_ONSETS, VOWELS, SONORANTS, "aeo", LIQUID_NASAL_CODAS)),
        # --- 6-Letter High-Sonority Templates ---
        # 13. VCVCVC (Vowel-initial alternating: aloren, elimar)
        ("VCVCVC", (VOWELS, SONORANTS, VOWELS, SONORANTS, "aeo", LIQUID_NASAL_CODAS)),
        # 14. CVCCVC (Sonorant-bridge closed: solven, melnor)
        ("CVCCVC", (SOFT_ONSETS, VOWELS, bridges, "aeo", LIQUID_NASAL_CODAS)),
        # 15. CVVCVC (Vowel-hiatus + liquid/nasal coda: naelor, saelen)
        ("CVVCVC", (SOFT_ONSETS, hiatus, SONORANTS, "aeo", LIQUID_NASAL_CODAS)),
        # 16. VCCVCV (Vowel-initial sonorant bridge: arnelo, orvela)
        ("VCCVCV", (VOWELS, bridges, VOWELS, SONORANTS, "aeou")),
        # 17. CVCVCV (Three-syllable open legato: nolira, velaro)
        ("CVCVCV", (SOFT_ONSETS, VOWELS, SONORANTS, VOWELS, SONORANTS, "aeou")),
    ]

    seen: set[str] = set()
    by_length: dict[int, list[Candidate]] = {length: [] for length in TARGET_LENGTHS}

    for pattern, slots in templates:
        for parts in itertools.product(*slots):
            stem = "".join(parts)
            if stem in seen or stem in lexicon:
                continue
            seen.add(stem)
            score = score_euphony(stem)
            if score is not None:
                by_length[len(stem)].append(Candidate(stem=stem, length=len(stem), pattern=pattern, score=score))
    return by_length


async def check_domain(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    candidate: Candidate,
    tld: str,
    retries: int = 2,
) -> tuple[Candidate, str, str, int]:
    domain = f"{candidate.stem}.{tld}"
    url = f"https://pubapi.registry.google/rdap/domain/{domain}"
    async with sem:
        for attempt in range(retries + 1):
            try:
                resp = await client.get(url)
                if resp.status_code == 429 and attempt < retries:
                    await asyncio.sleep(1.5 * (attempt + 1))
                    continue
                await asyncio.sleep(0.15)
                return candidate, tld, domain, resp.status_code
            except httpx.HTTPError:
                if attempt == retries:
                    return candidate, tld, domain, -1
                await asyncio.sleep(0.5)
    return candidate, tld, domain, -1


def _format_csv_row(row: Sequence[object]) -> str:
    """Serialize a single CSV row using RFC 4180 line endings."""
    buf = io.StringIO(newline="")
    csv.writer(buf).writerow(row)
    return buf.getvalue()


async def file_writer(
    queue: asyncio.Queue[DomainResult | None],
    output_path: Path = OUTPUT_CSV,
) -> None:
    """Stream queued domain results into a temp CSV and atomically replace output_path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f".{output_path.name}.tmp")
    completed = False
    try:
        # Open the temp file once and flush each row as it arrives
        async with aiofiles.open(tmp_path, mode="w", encoding="utf-8", newline="") as f:
            await f.write(_format_csv_row(CSV_HEADERS))
            await f.flush()

            while True:
                item = await queue.get()
                try:
                    if item is None:  # Sentinel value signals all workers are done
                        completed = True
                        break
                    cand, tld, domain, status = item
                    await f.write(
                        _format_csv_row([
                            domain,
                            cand.stem,
                            tld,
                            cand.length,
                            cand.pattern,
                            f"{cand.score:.3f}",
                            status,
                        ])
                    )
                    await f.flush()
                finally:
                    queue.task_done()

        if completed:
            tmp_path.replace(output_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _format_status(code: int) -> str:
    return "ERR (-1)" if code == -1 else str(code)


async def main() -> None:
    queue: asyncio.Queue[DomainResult | None] = asyncio.Queue()

    rng = random.Random(SEED)
    lexicon = load_english_lexicon()
    pools = generate_pool(lexicon)

    sample: list[Candidate] = []
    for length in TARGET_LENGTHS:
        pool = pools[length]
        sample.extend(rng.sample(pool, min(SAMPLE_PER_LENGTH, len(pool))))

    available: list[DomainResult] = []
    status_counts: Counter[int] = Counter()
    sem = asyncio.Semaphore(4)
    writer_task = asyncio.create_task(file_writer(queue, OUTPUT_CSV))
    completed_normally = False

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            tasks = [asyncio.create_task(check_domain(client, sem, cand, tld)) for cand in sample for tld in TLDS]
            try:
                for fut in asyncio.as_completed(tasks):
                    cand, tld, domain, status = await fut
                    status_counts[status] += 1
                    if writer_task.done():
                        await writer_task
                        raise RuntimeError("file_writer exited unexpectedly")
                    if status == 404:
                        item = (cand, tld, domain, status)
                        available.append(item)
                        await queue.put(item)
            finally:
                pending = [t for t in tasks if not t.done()]
                for t in pending:
                    t.cancel()
                if pending:
                    await asyncio.gather(*pending, return_exceptions=True)
        completed_normally = True
    finally:
        if completed_normally:
            await queue.put(None)
            await writer_task
        else:
            writer_task.cancel()
            await asyncio.gather(writer_task, return_exceptions=True)

    pool_summary = ", ".join(f"{len(pools[length]):,} ({length}-char)" for length in TARGET_LENGTHS)
    status_summary = ", ".join(f"{_format_status(code)}: {count}" for code, count in sorted(status_counts.items()))
    print(
        f"Dictionary words excluded: {len(lexicon):,} | "
        f"Valid pool: {pool_summary}\n"
        f"Checked: {len(sample) * len(TLDS)} domains ({status_summary or 'none'}) | "
        f"Available (404): {len(available)} -> saved to {OUTPUT_CSV.resolve()}\n"
    )
    print(f"{'DOMAIN':<15} {'LEN':<4} {'PATTERN':<8} {'SONORITY'}")
    print("-" * 39)
    for cand, _, domain, _ in available:
        print(f"{domain:<15} {cand.length:<4} {cand.pattern:<8} {cand.score:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
