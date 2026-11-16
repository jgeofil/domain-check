import asyncio
import csv
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from domain_check import phono


class TestFileWriter(unittest.IsolatedAsyncioTestCase):
    async def test_streams_incrementally_and_swaps_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "nested" / "available.csv"
            tmp_path = out_path.with_name(f".{out_path.name}.tmp")
            queue: asyncio.Queue[phono.DomainResult | None] = asyncio.Queue()

            c1 = phono.Candidate(stem="naelo", length=5, pattern="CVVCV", score=4.1)
            c2 = phono.Candidate(stem="aleno", length=5, pattern="VCVCV", score=4.9)

            writer_task = asyncio.create_task(phono.file_writer(queue, out_path))
            await queue.put((c1, "dev", "naelo.dev", 404))
            await queue.join()

            self.assertFalse(out_path.exists())
            self.assertTrue(tmp_path.exists())
            partial_rows = list(
                csv.reader(
                    io.StringIO(tmp_path.read_text(encoding="utf-8"), newline="")
                )
            )
            self.assertEqual(
                partial_rows,
                [
                    phono.CSV_HEADERS,
                    ["naelo.dev", "naelo", "dev", "5", "CVVCV", "4.100", "404"],
                ],
            )

            await queue.put((c2, "app", "aleno.app", 404))
            await queue.put(None)
            await writer_task

            self.assertFalse(tmp_path.exists())
            self.assertTrue(out_path.exists())
            raw_bytes = out_path.read_bytes()
            self.assertEqual(
                raw_bytes,
                (
                    b"domain,stem,tld,length,pattern,sonority_score,rdap_status\r\n"
                    b"naelo.dev,naelo,dev,5,CVVCV,4.100,404\r\n"
                    b"aleno.app,aleno,app,5,VCVCV,4.900,404\r\n"
                ),
            )

    async def test_empty_queue_writes_header_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "available.csv"
            queue: asyncio.Queue[phono.DomainResult | None] = asyncio.Queue()

            writer_task = asyncio.create_task(phono.file_writer(queue, out_path))
            await queue.put(None)
            await writer_task

            self.assertEqual(
                out_path.read_bytes(),
                b"domain,stem,tld,length,pattern,sonority_score,rdap_status\r\n",
            )

    async def test_preserves_existing_file_and_cleans_temp_on_cancel(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "available.csv"
            tmp_path = out_path.with_name(f".{out_path.name}.tmp")
            out_path.write_text("existing,content\n", encoding="utf-8")

            queue: asyncio.Queue[phono.DomainResult | None] = asyncio.Queue()
            cand = phono.Candidate(stem="naelo", length=5, pattern="CVVCV", score=4.1)

            writer_task = asyncio.create_task(phono.file_writer(queue, out_path))
            await queue.put((cand, "dev", "naelo.dev", 404))
            await queue.join()

            writer_task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await writer_task

            self.assertEqual(out_path.read_text(encoding="utf-8"), "existing,content\n")
            self.assertFalse(tmp_path.exists())


class TestPoolGeneration(unittest.TestCase):
    def test_generates_4_5_and_6_letter_patterns(self) -> None:
        pools = phono.generate_pool({"nalo"})
        self.assertEqual(set(pools.keys()), {4, 5, 6})
        patterns_4 = {c.pattern for c in pools[4]}
        self.assertEqual(
            patterns_4,
            {"CVCV", "VCCV", "VCVC", "CVVC", "CVCC", "CCVC", "CCVV", "VVCV"},
        )
        stems_4 = {c.stem for c in pools[4]}
        self.assertNotIn("nalo", stems_4)
        self.assertIn("sorn", stems_4)
        self.assertIn("drel", stems_4)
        self.assertIn("slea", stems_4)
        self.assertTrue(all(len(c.stem) == 4 and c.length == 4 for c in pools[4]))
        self.assertTrue(all(len(c.stem) == 5 and c.length == 5 for c in pools[5]))
        self.assertTrue(all(len(c.stem) == 6 and c.length == 6 for c in pools[6]))

    def test_score_euphony_accepts_4_letter_and_rejects_invalid_lengths(self) -> None:
        self.assertIsNotNone(phono.score_euphony("nael"))
        self.assertIsNotNone(phono.score_euphony("drel"))
        self.assertIsNotNone(phono.score_euphony("slea"))
        self.assertIsNotNone(phono.score_euphony("sorn"))
        self.assertIsNone(phono.score_euphony("llao"))
        self.assertIsNone(phono.score_euphony("nel"))
        self.assertIsNone(phono.score_euphony("naelora"))
        self.assertIsNone(phono.score_euphony("alll"))


class TestMainOutputFlow(unittest.IsolatedAsyncioTestCase):
    async def test_writes_404s_in_arrival_order_and_reports_statuses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "available.csv"
            c_slow = phono.Candidate(stem="naelo", length=5, pattern="CVVCV", score=4.9)
            c_fast = phono.Candidate(stem="nael", length=4, pattern="CVVC", score=4.1)
            c_taken = phono.Candidate(
                stem="aloren", length=6, pattern="VCVCVC", score=4.3
            )
            c_err = phono.Candidate(
                stem="melnor", length=6, pattern="CVCCVC", score=4.2
            )

            pools = {4: [c_fast], 5: [c_slow], 6: [c_taken, c_err]}

            async def fake_check_domain(
                _client: object,
                _sem: asyncio.Semaphore,
                candidate: phono.Candidate,
                tld: str,
            ) -> phono.DomainResult:
                delays = {
                    "naelo.dev": 0.04,
                    "nael.dev": 0.005,
                    "aloren.dev": 0.01,
                    "melnor.dev": 0.02,
                }
                statuses = {
                    "naelo.dev": 404,
                    "nael.dev": 404,
                    "aloren.dev": 200,
                    "melnor.dev": -1,
                }
                domain = f"{candidate.stem}.{tld}"
                await asyncio.sleep(delays[domain])
                return candidate, tld, domain, statuses[domain]

            stdout = io.StringIO()
            with (
                patch.object(phono, "OUTPUT_CSV", out_path),
                patch.object(phono, "TLDS", ("dev",)),
                patch.object(phono, "SAMPLE_PER_LENGTH", 2),
                patch.object(phono, "load_english_lexicon", return_value=set()),
                patch.object(phono, "generate_pool", return_value=pools),
                patch.object(phono, "check_domain", side_effect=fake_check_domain),
                patch("sys.stdout", stdout),
            ):
                await phono.main()

            with out_path.open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.reader(f))

            self.assertEqual(
                rows,
                [
                    phono.CSV_HEADERS,
                    ["nael.dev", "nael", "dev", "4", "CVVC", "4.100", "404"],
                    ["naelo.dev", "naelo", "dev", "5", "CVVCV", "4.900", "404"],
                ],
            )
            output = stdout.getvalue()
            self.assertIn("ERR (-1): 1, 200: 1, 404: 2", output)
            self.assertIn("1 (4-char), 1 (5-char), 2 (6-char)", output)

    async def test_propagates_writer_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            blocker_file = Path(tmp) / "not_a_dir"
            blocker_file.write_text("x", encoding="utf-8")
            bad_out_path = blocker_file / "available.csv"

            cand = phono.Candidate(stem="nael", length=4, pattern="CVVC", score=4.9)
            pools = {4: [cand], 5: [], 6: []}

            async def fake_check_domain(
                _client: object,
                _sem: asyncio.Semaphore,
                candidate: phono.Candidate,
                tld: str,
            ) -> phono.DomainResult:
                await asyncio.sleep(0.01)
                return candidate, tld, f"{candidate.stem}.{tld}", 404

            with (
                patch.object(phono, "OUTPUT_CSV", bad_out_path),
                patch.object(phono, "TLDS", ("dev",)),
                patch.object(phono, "SAMPLE_PER_LENGTH", 1),
                patch.object(phono, "load_english_lexicon", return_value=set()),
                patch.object(phono, "generate_pool", return_value=pools),
                patch.object(phono, "check_domain", side_effect=fake_check_domain),
                self.assertRaises(FileExistsError),
            ):
                await phono.main()
