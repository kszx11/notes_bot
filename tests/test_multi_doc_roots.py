from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from notes_bot.config import load_config
from notes_bot.note_view import get_note_excerpt, safe_doc_path
from notes_bot.scanner import iter_files


class MultiDocRootsTests(unittest.TestCase):
    def test_named_multi_roots_scan_and_resolve_virtual_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            work_root = base / "work_notes"
            personal_root = base / "personal_notes"
            work_root.mkdir()
            personal_root.mkdir()
            (work_root / "ops.md").write_text("work line\n", encoding="utf-8")
            (personal_root / "journal.md").write_text("one\ntwo\nthree\n", encoding="utf-8")

            config_path = base / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "doc_roots:",
                        f"  - name: work",
                        f"    path: {work_root}",
                        f"  - name: personal",
                        f"    path: {personal_root}",
                        'include_ext: [".md", ".txt"]',
                        f'data_dir: "{base / "data"}"',
                        f'index_dir: "{base / "data" / "index"}"',
                        f'manifest_path: "{base / "data" / "manifest.sqlite"}"',
                        f'chat_history_path: "{base / "data" / "chat_history.jsonl"}"',
                    ]
                ),
                encoding="utf-8",
            )

            cfg = load_config(config_path)
            discovered = sorted(iter_files(cfg.doc_roots, cfg.include_ext), key=lambda item: item.rel_path)

            self.assertEqual([item.rel_path for item in discovered], ["personal/journal.md", "work/ops.md"])
            self.assertEqual(safe_doc_path(cfg.doc_roots, "work/ops.md"), work_root / "ops.md")

            excerpt = get_note_excerpt(
                doc_roots=cfg.doc_roots,
                rel_path="personal/journal.md",
                start_line=2,
                end_line=2,
                context_before=1,
                context_after=1,
            )
            self.assertEqual(excerpt["root_name"], "personal")
            self.assertEqual(excerpt["root_rel_path"], "journal.md")
            self.assertEqual(excerpt["display_start_line"], 1)
            self.assertEqual(excerpt["display_end_line"], 3)


if __name__ == "__main__":
    unittest.main()
