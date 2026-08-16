# Restaurant corpus — provenance and honesty notes

20 real NYC-area independent restaurants, snapshotted once on **2026-08-08** and committed
to disk. Nothing in the demo pipeline fetches the network: every downstream step
(005 loader, 017 food-cost harness, 018 sweep) reads only from this directory.

None of these restaurants is a Marble customer. Marble's published logo wall
(A&W, The Original Pancake House, The Kati Roll Company, Sophie's Cuban, Kokomo NYC,
Holy Shred, Dead Letter No. 9, JJ's Sports Bar, Spare Birdie, Good Folks GTX) was
deliberately excluded so the demo reads as "here is a restaurant you could sell to
tomorrow", not "here is one you already have".

## Layout

```
data/restaurants/
  index.json                 flat array of {slug, tier} — the enumeration order
  <slug>/source.html|pdf|jpg the raw menu artifact, fetched once
  <slug>/meta.json           metadata the pipeline cannot infer from the artifact
```

## Which fields are real and which are estimated

**Fetched / verified real — do not treat these as approximations:**

| Field | Provenance |
|---|---|
| `url` | the exact page the artifact was fetched from |
| `source.{html,pdf,jpg}` | fetched once with `curl -sL -A "Mozilla/5.0" <url>`, byte-for-byte as served |
| `snapshot_date` | the date of that fetch (2026-08-08 for the whole corpus) |
| `menu_format` | determined by inspecting the fetched bytes, not guessed from the URL |
| `name`, `neighborhood`, `cuisine`, `service_style`, `price_tier` | read off the restaurant's own site |

**Observed after the fact — declared in each `meta.json` under `"observed_fields"`:**

The corpus was first built with `popular_times_index` and `review_count` estimated. They
were then **replaced with real Google Maps data** on 2026-08-08, read from the listing's
accessibility tree (each hourly bar exposes an exact `Taux de fréquentation de N %` label,
so a day's traffic is the sum of its hourly values, normalised so the busiest day = 1.0).

| Field | Provenance | Coverage |
|---|---|---|
| `popular_times_index` | summed hourly popular-times percentages, Mon→Sun, busiest = 1.0 | **13 of 20** |
| `review_count` | read off the Google Maps listing | **15 of 20** |

Two caveats worth stating rather than hiding:

- **Google omits closed days from the chart.** A venue closed one day renders six day
  blocks, not seven, which silently breaks positional day mapping. `sushi-yasuda` returned
  six blocks and its closed weekday was not verified, so its `popular_times_index` remains
  the estimate.
- **Not every listing publishes popular times.** `adda-indian-canteen` and `balaboosta`
  have none, so theirs remain estimates too.

**Still estimated — declared in each `meta.json` under `"estimated_fields"`:**

| Field | Why it is an estimate |
|---|---|
| `seats` | estimated from the venue's format and published photos, not counted. All 20. |
| `popular_times_index` | only where Google publishes none, or where a closed day made the mapping unsafe: `adda-indian-canteen`, `balaboosta`, `sushi-yasuda`. |
| `review_count` | only `balaboosta`, whose listing returned no parsable count. |

This follows the project rule: **fake the data sources, never the reasoning.** Where a
signal could be observed it now is; where it could not, the record says so in-band via
`estimated_fields` so no downstream step can mistake an estimate for a measurement.
The real data was not cosmetic — Los Tacos No. 1 peaks **midweek**, not at the weekend
(Chelsea Market office and tourist lunch trade), which no judgement call would have guessed.

## Deliberate difficulty spread

The corpus is not a set of easy wins. Parsing quality is meant to vary, because the
degradation is the interesting finding (Steps 017/018) and the honest failure modes are
what Step 020's recording shows.

| Format | Count | Notes |
|---|---|---|
| `html` | 14 | includes 2 that carry no prices at all, and 1 whose entire menu lives in schema.org JSON-LD inside a `<script>` tag |
| `pdf` | 5 | 3 with a real text layer, 2 with **no text layer at all** |
| `image` | 1 | menu published only as a bitmap |

Known hard cases, by slug:

- **`joes-pizza-carmine`** — menu exists only as a 382×1287 bitmap. Vision-only input. Tier A.
- **`di-an-di`**, **`tacombi`** — PDFs whose pages are images; `pypdf` extracts 0 and 1
  characters respectively. Any text-only PDF path returns nothing here; these need vision.
- **`sushi-yasuda`** — PDF with a text layer, but the multi-column layout separates dish
  names from prices, so most items extract without a price.
- **`veselka`** — 36k characters of good text, but the display font mangles glyphs
  (`Bo/t_tled`, `D/r.initialaft`). Real OCR-ish noise in a PDF that is not scanned.
- **`rubirosa`**, **`mamouns-falafel`** — full dish lists and descriptions, **no prices
  anywhere on the page**. Food-cost percentage is not computable from the menu alone;
  the correct behaviour is to flag low confidence, not to invent a price.
- **`los-tacos-no1-chelsea`** — strip the `<script>` tags and the page has ~1.5k characters
  of navigation and nothing else. The entire priced menu is JSON-LD `MenuItem` markup.
  A loader that naively removes scripts before extraction will see an empty menu here.
- **`clinton-st-baking-company`** — prices are bare integers with no currency symbol
  (`Pancakes ... 20`), adjacent to other bare integers in descriptions.

## Tier A — the five worked examples

| Slug | Cuisine | Format |
|---|---|---|
| `joes-pizza-carmine` | italian-pizza | image |
| `adda-indian-canteen` | indian | pdf |
| `veselka` | ukrainian-diner | pdf |
| `los-tacos-no1-chelsea` | mexican | html (JSON-LD) |
| `madame-vo` | vietnamese | html |

Two clean HTML menus, two PDFs, one image-only menu — the mix the demo needs.

## Cuisine spread across all 20

italian-pizza 2 · italian 1 · indian 2 · vietnamese 3 · mexican 3 · diner-american 2 ·
american-sandwich 1 · japanese-sushi 2 · middle-eastern 3 · ukrainian-diner 1

## Re-fetching

Don't, during a recording. If a snapshot ever needs refreshing, fetch once, verify the
artifact actually contains menu text (a JS shell that renders client-side is a failed
snapshot, not a snapshot), and bump `snapshot_date`.
