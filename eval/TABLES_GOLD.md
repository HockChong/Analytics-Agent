# Table answer key: how to verify it

`eval/tables_gold.json` is a **draft** answer key for 10 tables in CICT AR FY2025, built
by `python -m eval.prefill_tables_gold` from what the extractors found. Until a person has
checked it against the PDF it proves nothing. Scored on the draft, each extractor gets
close to 100% on its own mistakes.

## Setup
```
make fetch build
python -m eval.prefill_tables_gold      # writes the draft + eval/table_check/*.png
```

## For each table (about 5 minutes each)
1. Open `eval/table_check/<table>.png`. Every draft cell is boxed and numbered:
   **green** = both extractors agree, **red** = check this first.
2. Open the PDF at that page (`page` in the JSON) next to it.
3. For each cell in `cells`, compare with the printed table:
   - `row`: the row label exactly as printed, plus any parent heading needed to make it
     unambiguous ("Management fees – Base component"). Note references are ignored.
   - `column`: the column header as printed, top to bottom, joined with ` | `
     ("Group | 2025 | $’000"). Scoring checks that every word of your column appears in
     the extractor's column, so include the words that matter (year, Group/Trust, unit).
   - `value`: exactly as printed, including brackets for negatives ("(7,180)").
   - Fix anything that is wrong. **Delete cells that are not table values** (for example
     the "31" picked up from "31 Dec 2024" in a header).
4. **Add the rows no extractor found.** Look for numbers in the table that have no box.
   Copy an existing cell, change `row`, `column` and `value`, and set `bbox` to the
   number's position (run `python scripts/draw_blocks.py` or ask Claude to look up the
   word's bbox). Rows missed by every extractor are exactly what this key must catch.
5. Write anything unusual in `notes`, then set `"verified": true`.

You can delete the `check` field once a cell is done; it is only a hint.

## Score
```
python -m eval.score_tables            # verified tables only → eval/results/table_extractors.md
```
