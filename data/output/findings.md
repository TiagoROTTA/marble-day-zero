# Findings

Written against the 20-restaurant corpus in `data/restaurants/`, of which 14
reached the `costed` stage and 12 produced a food-cost ratio. Every number below
comes from the cached `data/output/<slug>-costed.json` files, the regenerated
`data/output/sweep-report.md`, or a hand check of a specific plate.

---

## 1. The food-cost validation returns a red result, and the benchmark is what is wrong

Pooled over 519 plates at or above 50% ingredient coverage, the median implied
food cost is **8.2%** (mean 10.1%, p10 3.4%, p90 19.0%). Only 1.7% of plates
land inside the 28-33% industry band and only 8.1% inside the wider 20-45%
plausible band. Per restaurant the medians run from joes-pizza-carmine at 19.2%
down to scarrs-pizza at 5.5%. That is a clear miss and it is reported as
measured.

The obvious upstream causes were checked and ruled out. Portions are not too
small: across the 741 recipes with weight-denominated components the median
plate carries 237 g of ingredients priced by weight, before the `each` and
`fl_oz` lines that are excluded from that arithmetic. Catalog prices are not
wrong: chicken breast is $3.15/lb and ribeye $12.95/lb in
`data/catalog/skus.json`, both plausible NYC wholesale. Conversions are not
broken: mean plate coverage runs 0.81-1.00 on thirteen of the fourteen costed
restaurants after per-SKU density and pack-weight conversions were added, the
one exception being `court-street-grocers` at 0.56, which is a catalog gap
rather than a conversion gap (finding 8).

The comparison itself is unsound. The 28-33% figure is *actual* food cost: total
food purchases divided by total revenue, weighted by what actually sells and
inclusive of waste, spoilage, over-portioning and staff meals. What this
pipeline computes is an *unweighted median of theoretical plate cost* over every
priced menu line. Theoretical plate cost always sits below actual food cost --
that distinction is standard in the industry -- and taking an unweighted median
across a menu full of cheap high-margin lines widens the gap substantially.
Au Za'atar alone contributes 197 of the 519 plates, and its menu is thick with
$8 rice sides and $12 juice.

The spread across restaurants supports that reading in part. Restaurants of
normal single-serving entrees sit at the top -- los-tacos-no1-chelsea 18.1%,
kanoyama 15.2%, sushi-yasuda 13.3%, hanoi-house 11.9%, madame-vo 11.2% -- and
the platter-and-sides menu sits at the bottom, au-zaatar at 6.4%. The bottom of
the table is not purely composition, though. `court-street-grocers` at 6.4% is a
catalog gap rather than a menu shape (finding 8), and `scarrs-pizza` at 5.5% is
a decomposition error: its whole 20-inch pies come back with per-slice
quantities -- 6 oz of flour and 3.5 oz of mozzarella against a $33 price --
which is finding 3's mechanism, not menu composition.

One caution about that ordering, because it changed. An earlier revision of this
file cited `joes-pizza-carmine` at 2.4% as the strongest example of the low end
and read it as menu composition. It was a bug: `plate_cost` was divided by
`yield_qty` while `menu_price` is the price of the whole menu line, so an
8-slice pie reported the cost of one slice against the price of eight. With the
division removed, `joes-pizza-carmine` costs 19.2% and is the *highest*
restaurant in the corpus -- a 17-point move on unchanged recipes -- and its
"Classic Cheese Pie 8 slices" is $24 against 14 oz of flour and 12 oz of
mozzarella, $4.62. The corpus median moved 7.8% to 8.2% with it. The
composition argument holds on the spread that remains, but one of the three
examples it used to rest on was a broken calculation.

## 2. A hand-checked plate confirms the arithmetic is right and the metric is the problem

Au Za'atar's "Vermicelli Rice" is on the menu at $8.00. The decomposer emits
4 oz long grain white rice, 0.5 oz bean thread glass noodles, 0.4 oz unsalted
butter and 0.05 oz kosher salt, all four of which cost, giving coverage 1.00 and
a plate cost of **$0.39** -- 4.8% implied food cost. Checked by hand against the
catalog, $0.39 is correct. Rice really does cost that, and an $8 rice side
really does run about 5% food cost. There is no bug to find on this plate; the
plate is simply not the kind of line the industry benchmark is computed over.

What the harness therefore *does* validate: that menu prices are transcribed
correctly, that recipe quantities are dimensionally sane, that per-SKU unit
conversions resolve, and that individual plate costs survive being checked by
hand. What it does not validate is a restaurant-level food-cost percentage,
because that needs sales mix and this build has none.

## 3. Large shared platters are genuinely under-costed, and the decomposer's quantities are why

Au Za'atar's "Mixed Grill Taweel" is priced at $318 and decomposes to 19
components totalling 1.80 kg of weight-denominated ingredients -- 4 oz chicken
breast, 4 oz NY strip, 5 oz boneless leg of lamb, 8 oz lamb shoulder, 4 oz
shrimp, and so on -- costing **$18.95**, or 6.0%. Its sibling "Mixed Grill
Platter Royal" at $212 costs $14.36 (6.8%). A platter that serves eight should
carry roughly eight times that meat.

The mechanism is the quantities the decomposer writes, not the `yield_qty`
field. `plate_cost` is the cost of the whole sold unit -- the menu line the
price is attached to -- and `yield_qty` does not enter it at all; the per-portion
figure is carried separately as `cost_per_serving`. A platter is therefore
under-costed exactly when the decomposer emits a single-serving ingredient list
against a group-serving price, which is what both platters above do. Across the
corpus **844 of 879 recipes carry `yield_qty: 1.0`**, including both platters;
only 10 recipes were assigned a yield of 8, 9 a yield of 4, and 6 a yield of 6.
The decomposer almost never estimates servings per dish, and the estimate would
not change the cost if it did.

The same failure shows up outside the platter menus, and it is inconsistent
rather than systematic. `scarrs-pizza` decomposes "Original (20-inch)", a whole
$33 pie, to 6 oz of flour and 3.5 oz of mozzarella -- a slice's worth, giving
3.5%. `joes-pizza-carmine` decomposes the equivalent dish correctly:
"Classic Cheese Pie 8 slices" comes back as 14 oz of flour and 12 oz of
mozzarella, $4.62 against $24. Same dish, same model, two different readings of
what the price covers. This is a real second-order under-costing, but it is not
the dominant term -- it affects tens of plates, not the 519 that set the median.

## 4. Two corpus PDFs are scanned images, and the loader refuses rather than guesses

`di-an-di` and `tacombi` both stop at the `ingested` stage with the same
message from `src/tools/snapshot.py`:

> `di-an-di\source.pdf yielded 0 chars of text: it is a scanned image with no text layer. Re-snapshot di-an-di as .jpg and set menu_format="image".`

That is not a bug, it is the intended refusal: pypdf returns an empty string,
and extracting nothing while reporting success would have put two silent
zero-item restaurants into the sweep. The message names the fix, and the fix is
mechanical -- the one image-format restaurant already in the corpus,
`joes-pizza-carmine`, extracted 5 items at 0.91 confidence and 1.00 coverage.

## 5. The two menus that broke at extraction broke on output tokens

`fonda-park-slope` and `junoon` both fail at `extract_menu` with an empty tool
payload:

> `ValidationError: 2 validation errors for MenuExtraction / items: Field required [type=missing, input_value={}]`

That empty `{}` is the signature of the model hitting its output ceiling
mid-structure; the console reported a `max_tokens` stop reason on the same runs.
`llm_max_tokens` was raised from 16000 to 32000 and thinking was disabled on
`extract_menu` in response; both restaurants remain uncosted. For scale, the longest menus
that did succeed are au-zaatar at 160 items and bubbys-tribeca at 164.

## 6. `bubbys-tribeca` failed on merchandise, not on food

`bubbys-tribeca` extracted 164 menu items at 0.88 confidence and decomposed 125
of them, then stopped with:

> `decomposition failed for 40 item(s): Acqua Panna Bottle (1L), Pellegrino Bottle (1L), Variety Coffee, Harney & Sons, Iced Tea, Arnold Palmer, Hibiscus Iced Tea, ... Bubby's Apron, Bubby's Pie or Brunch Cookbook, Doerfler's Maple Syrup`

An apron and a cookbook have no ingredient decomposition, and neither does a
sealed 1L bottle of Acqua Panna. The failure is correct per line and wrong as a
policy: a restaurant menu that sells retail goods and packaged drinks should
route those lines around the decomposer instead of failing the whole run 125
recipes in. This is the cheapest fix in the list.

## 7. Two restaurants publish no prices at all, so 14 costed runs yield only 12 ratios

`mamouns-falafel` (39 items) and `rubirosa` (139 items) both have `price: None`
on **every** extracted item, so neither contributes a food-cost ratio despite
costing cleanly -- mamouns-falafel reaches 0.99 mean coverage. Rubirosa's mean
item confidence of **0.46** is the extractor correctly flagging that it could
not attach prices: "Piatto di Antipasti", "Salumi e Formaggi" and "Pasta e
Fagioli" all come back at 0.50 confidence with a null price. The distinction
matters for the headline, which is over 12 restaurants and 519 plates, not 14
and not 20.

## 8. The catalog gap is concentrated in one restaurant; the conversion gap is everywhere

`court-street-grocers` has an unmatched-ingredient rate of **47.2%** -- 42 of 89
ingredients found no SKU -- against 3.4% across the three Mexican restaurants.
The unmatched list is an American deli in miniature: `seeded hero roll`,
`ciabatta roll`, `corned beef`, `heritage ham`, `pepperjack cheese, sliced`,
`sauerkraut`, `full sour pickles`, `hoagie spread`, `ajvar`, `comeback sauce`,
plus a full vegan line (`vegan american cheese`, `vegan mushroom black bean
patty`, `vegan comeback sauce`). `kanoyama` is the same story in Japanese: 22 of
111 unmatched, including `monkfish liver (ankimo)`, `salmon roe (ikura)`,
`anago sea eel`, `gobo burdock root` and `dried kanpyo gourd strips`. A
324-SKU catalog built around general NYC wholesale does not carry a deli counter
or a sushi case.

Missing SKUs are not, however, the largest source of lost coverage. The top of
the corpus-wide uncosted tally is dominated by *conversions*, not absences:
`roasted nori sheets (no conversion from each to lb)` on 41 plates, `canned san
marzano whole tomatoes (no conversion from oz to each)` on 27, `peeled garlic
cloves (no conversion from each to lb)` on 21, `house red wine (no conversion
from fl_oz to each)` on 20. Those four SKUs exist and are priced; they are
dropped because the catalog carries no density or pack weight for them. Adding
four numbers to `data/catalog/skus.json` lifts 109 plates.

## 9. The menu-format cross-cut came out backwards, and survivorship is why

Measured mean coverage ranks **image 1.00 (1 restaurant) > pdf 0.95 (2 of the 5
PDFs completed) > html 0.86 (11 of the 14 HTML menus completed)**, which
contradicts the expected html > pdf > image ordering that
`scripts/sweep_report.py` states up front. The honest reading is survivorship,
not format quality: the one image restaurant is `joes-pizza-carmine`, a 5-item
pizza menu that is trivially easy, and the two surviving PDFs are
`adda-indian-canteen` and `sushi-yasuda`, while the three PDFs that did not
complete (`di-an-di`, `tacombi`, `veselka`) never produced a row at all.
Format difficulty shows up here as a *failure to complete*, which this cross-cut
cannot see because it averages only over completions.

Group sizes make most of the cuisine cross-cut anecdote rather than trend. Of
the ten cuisine labels, only `mexican` (n=3), `middle-eastern` (n=3) and
`vietnamese` (n=3) meet the 3-restaurant floor; `american-sandwich`, `italian`
and `ukrainian-diner` are n=1 and `indian`, `diner-american`, `japanese-sushi`
and `italian-pizza` are n=2. The 47.2% unmatched rate for `american-sandwich`
above is one restaurant, `court-street-grocers`, wearing a cuisine label. It is
worth acting on as a catalog observation; it is not a measurement of American
sandwich shops.

## 10. What the sweep costs to run

The full 20-restaurant sweep cost **$22.07 by list-rate estimate, $1.10 per
restaurant**, over 199 LLM calls, 1.30M input tokens (64.3% read from cache) and
769K output tokens, at a median 169s of wall clock per restaurant. That
per-restaurant figure is the number that matters for an onboarding pipeline, and
the 64.3% cache-read share is what keeps it there: the 324-SKU catalog sits
behind a single `cache_control` block, so most input tokens bill at a tenth of
the input rate.
