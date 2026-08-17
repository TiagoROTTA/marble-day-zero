# Sweep report

Tier swept: **all** · 20 restaurants · 6 failed (non-empty `error`).

Every figure below is read from the cached `data/output/<slug>-costed.json` files; no part of this report runs the pipeline. `--` means there was nothing to measure, never that a measurement was skipped.

Failed runs (their rows below are still shown, and are incomplete):

- `bubbys-tribeca` — ended at stage menu_extracted instead of costed
- `di-an-di` — ended at stage ingested instead of costed
- `fonda-park-slope` — ended at stage ingested instead of costed
- `junoon` — ended at stage ingested instead of costed
- `tacombi` — ended at stage ingested instead of costed
- `veselka` — ended at stage ingested instead of costed

## Headline

Pooled over 519 plates at or above 50% ingredient coverage, the median implied food cost is **8.2%** (mean 10.1%); 1.7% of plates fall inside the 28%-33% industry band. That is a clear miss against the band, and it is reported as measured.

The band is *actual* food cost: total purchases over total revenue, weighted by what actually sells and inclusive of waste, spoilage, over-portioning and staff meals. The figure above is an unweighted median of *theoretical* plate cost over every priced menu line, high-margin sides and drinks included. Theoretical plate cost always sits below actual food cost, and the unweighted median widens the gap further. The two are not like-for-like; see `data/output/findings.md` for what the harness does and does not establish.

## Per restaurant

| restaurant | cuisine | format | items | item conf | alias | norm | llm | unmatched | coverage | median FC% | review |
|---|---|---|---|---|---|---|---|---|---|---|---|
| adda-indian-canteen | indian | pdf | 23 | 0.91 | 84 | 0 | 3 | 13 | 0.95 | 11.8% | 29 |
| au-zaatar | middle-eastern | html | 160 | 0.90 | 176 | 1 | 1 | 15 | 0.92 | 6.4% | 304 |
| balaboosta | middle-eastern | html | 17 | 0.95 | 81 | 0 | 0 | 3 | 0.97 | 10.2% | 28 |
| bubbys-tribeca | diner-american | html | 164 | 0.88 | 0 | 0 | 0 | 0 | -- | -- | 0 |
| clinton-st-baking-company | diner-american | html | 76 | 0.87 | 111 | 0 | 0 | 13 | 0.85 | 10.3% | 86 |
| court-street-grocers | american-sandwich | html | 29 | 0.90 | 47 | 0 | 0 | 42 | 0.56 | 6.4% | 78 |
| di-an-di | vietnamese | pdf | 0 | -- | 0 | 0 | 0 | 0 | -- | -- | 0 |
| fonda-park-slope | mexican | html | 0 | -- | 0 | 0 | 0 | 0 | -- | -- | 0 |
| hanoi-house | vietnamese | html | 24 | 0.88 | 84 | 1 | 0 | 8 | 0.86 | 11.9% | 35 |
| joes-pizza-carmine | italian-pizza | image | 5 | 0.91 | 12 | 0 | 0 | 0 | 1.00 | 19.2% | 2 |
| junoon | indian | html | 0 | -- | 0 | 0 | 0 | 0 | -- | -- | 0 |
| kanoyama | japanese-sushi | html | 62 | 0.83 | 88 | 0 | 1 | 22 | 0.81 | 15.2% | 100 |
| los-tacos-no1-chelsea | mexican | html | 16 | 0.89 | 26 | 1 | 1 | 1 | 0.92 | 18.1% | 18 |
| madame-vo | vietnamese | html | 60 | 0.94 | 145 | 0 | 0 | 1 | 0.89 | 11.2% | 51 |
| mamouns-falafel | middle-eastern | html | 39 | 0.88 | 67 | 0 | 0 | 4 | 0.99 | -- | 20 |
| rubirosa | italian | html | 139 | 0.46 | 145 | 2 | 1 | 31 | 0.85 | -- | 113 |
| scarrs-pizza | italian-pizza | html | 33 | 0.89 | 51 | 0 | 0 | 5 | 0.81 | 5.5% | 60 |
| sushi-yasuda | japanese-sushi | pdf | 20 | 0.75 | 48 | 0 | 0 | 0 | 0.96 | 13.3% | 29 |
| tacombi | mexican | pdf | 0 | -- | 0 | 0 | 0 | 0 | -- | -- | 0 |
| veselka | ukrainian-diner | pdf | 0 | -- | 0 | 0 | 0 | 0 | -- | -- | 0 |

`coverage` is the mean over costable plates. `median FC%` is over plates with coverage >= 50%; the industry band is 28%-33%. `unmatched` counts every match with a null `sku_id`, whichever pass produced it — an honest catalog gap, not a failure of the matcher.

## Cross-cut: menu format

| format | n restaurants | mean extraction confidence | mean coverage |
|---|---|---|---|
| html | 14 | 0.86 | 0.86 |
| image | 1 | 0.91 | 1.00 |
| pdf | 5 | 0.83 | 0.95 |

- **image** has n=1, below the 3-restaurant floor: its mean describes those 1 restaurant(s), not the format. Read it as an anecdote.
- Measured coverage ranking: image (1.00, n=1) > pdf (0.95, n=5) > html (0.86, n=14).
- That **contradicts** the expected html > pdf > image ordering. The data says image > pdf > html; the expectation was wrong, and the honest finding is more interesting than the predicted one.

## Cross-cut: cuisine

| cuisine | n restaurants | mean unmatched-ingredient rate |
|---|---|---|
| american-sandwich | 1 | 47.2% |
| diner-american | 2 | 10.5% |
| indian | 2 | 13.0% |
| italian | 1 | 17.3% |
| italian-pizza | 2 | 4.5% |
| japanese-sushi | 2 | 9.9% |
| mexican | 3 | 3.4% |
| middle-eastern | 3 | 5.7% |
| ukrainian-diner | 1 | -- |
| vietnamese | 3 | 4.6% |

- Highest unmatched rate: **american-sandwich** at 47.2% (n=1). Lowest: **mexican** at 3.4% (n=3).
- Spread of 43.7% between them. The expectation was that cuisines the 324-SKU catalog under-serves would show materially higher unmatched rates; on this sweep that holds.
- n < 3 for: american-sandwich (n=1), italian (n=1), indian (n=2), diner-american (n=2), japanese-sushi (n=2), italian-pizza (n=2). Those rows are single restaurants wearing a cuisine label, not a measurement of the cuisine.

## Cost and latency

| metric | value |
|---|---|
| LLM calls | 199 |
| input tokens (total) | 1,301,899 |
| of which read from cache | 837,551 (64.3%) |
| of which written to cache | 91,119 |
| fresh input tokens (full rate) | 373,229 |
| output tokens | 768,805 |
| mean wall clock per restaurant | 272.1s |
| median wall clock per restaurant | 169.4s |
| estimated cost, full sweep | $22.07 |
| estimated cost per restaurant | $1.10 |

Priced at claude-opus-5 list rates ($5.00 in / $25.00 out / $0.50 cache read / $6.25 cache write per million tokens; see `PRICING_PER_MTOK` at the top of `scripts/sweep_report.py` for the source and date). Cache reads are priced at the cached rate, which is the whole point of putting the catalog behind one `cache_control` block: 64.3% of input tokens billed at a tenth of the input rate.

## Most-frequently-uncosted ingredients

| ingredient | plates affected |
|---|---|
| roasted nori sheets (no conversion from each to lb) | 41 |
| canned san marzano whole tomatoes (no conversion from oz to each) | 27 |
| peeled garlic cloves (no conversion from each to lb) | 21 |
| house red wine (no conversion from fl_oz to each) | 20 |
| sesame tahini (no conversion from fl_oz to lb) | 16 |
| clover honey (no conversion from fl_oz to lb) | 15 |
| house white wine (no conversion from fl_oz to each) | 15 |
| plain whole-milk yogurt (no conversion from fl_oz to lb) | 14 |
| seeded hero roll (no catalog sku matched) | 12 |
| yellow onion (no conversion from each to lb) | 11 |
| sea urchin roe (uni) tray (no conversion from oz to each) | 11 |
| flat-leaf parsley (no conversion from oz to bunch) | 10 |
| fresh cilantro (no conversion from oz to bunch) | 10 |
| well vodka (no conversion from fl_oz to each) | 10 |
| rice paper wrappers (no conversion from each to lb) | 8 |

This is the catalog shopping list, ordered by leverage: one SKU added to `data/catalog/skus.json` for the top name lifts every plate that mentions it.
