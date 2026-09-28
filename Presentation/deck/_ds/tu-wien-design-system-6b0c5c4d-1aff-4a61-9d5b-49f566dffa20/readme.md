# TU Wien Design System

A design system reconstructed from the **official TU Wien Office presentation templates**. It carries the university's colour scheme, type scale, slide geometry, logo files and accessibility rules, packaged so an agent can produce on-brand TU Wien slides, documents and HTML mock-ups.

---

## 1 · Context

**Technische Universität Wien (TU Wien)** is a public technical university in Vienna, founded 1815, with roughly 28,000 students across eight faculties in engineering, computer science and natural sciences. Its mission statement — and the sentence that governs the whole brand — is **„Technik für Menschen"** (*Technology for people*).

The brand's defining characteristic is not a visual flourish but a constraint: **accessibility**. The TU Wien template manual devotes more pages to contrast ratios, alt texts, reading order and minimum type sizes than to anything else. Every colour pair in this system ships with its measured contrast ratio, and 18 pt is treated as a hard floor for slide body copy.

### Surfaces represented here

| Surface | Status | Source |
|---|---|---|
| **Presentation (PowerPoint 16:9)** | Fully reconstructed | `TUW_Musterpraesentation.potx` — 33 slide layouts, 28 sample slides |
| **Presentation authoring rules** | Fully documented | `TUW_Praesentation_Manual.potx` — 56 slides of guidance |
| Word / Excel templates | **Not provided** | Referenced by the manual (`TUW_Arbeitsmappe_Muster`) but not supplied |
| Website / web app | **Not provided** | No web source was attached; nothing here is inferred from tuwien.at |

### Sources given to this project

- **Local codebase:** `TUW_Vorlage_Powerpoint/` containing
  - `TUW_Musterpraesentation.potx` — the sample presentation with every slide layout
  - `TUW_Praesentation_Manual.potx` — the authoring and accessibility manual
  - Both are archived at `sources/` in this project, with their extracted media in `sources/media/`.
- **Corporate design wiki:** <https://colab.tuwien.ac.at/spaces/CD/pages/32836503/Corporate+Design> — **inaccessible**: the page is behind TU Wien's Confluence login and returned no design content. Nothing in this system comes from it.
- Public reference used only to confirm the primary hex value and the motto: tuwien.at brand listings.

Everything below is taken from the two `.potx` files unless explicitly flagged.

---

## 2 · Content fundamentals

**Language.** German (Austrian, `de-AT`) is the primary language of the templates. Austrian spelling and vocabulary throughout. English versions exist at the university but are not in the source; if you write English, keep the same register.

**Register: formal, institutional, second-person plural.** The manual addresses the reader as **„Sie"**, never „du". It opens with *„Sehr geehrte Kolleg_innen"* and stays in that mode. Instructions are phrased as polite requests, not commands:

> „Bitte verwenden Sie die vorgesehenen Platzhalter…"
> „Bitte beachten Sie, dass PowerPoint keine Silbentrennung bietet."

The word **„bitte"** appears on almost every instruction slide. Directness without it reads as un-TU-Wien.

**We / you.** The institution speaks as **„wir"** when stating values (*„Mit barrierefrei gestalteten Dokumenten schließen wir niemanden aus"*), and as **„Sie"** when guiding the reader. First-person singular never appears.

**Gender-inclusive spelling is mandatory and uses the underscore**, not the asterisk or colon: `Kolleg_innen`, `Leser_innen`, `Autor_in`, `Benutzer_innen`. Reproduce this exactly — it is a formal TU Wien convention.

**Casing.** Sentence case for headings and titles; German noun capitalisation applies as normal. No all-caps for emphasis, no title case. Slide titles are descriptive nouns or noun phrases: *„Barrierefreie Nutzung der Farben"*, *„Tipps zur Diagrammgestaltung (1 von 2)"*. Multi-part topics are numbered `(1 von 2)`, `(2 von 2)`.

**Typographic detail.** German quotation marks are low-high: **„Titelfolie mit Bild"**. Bullet fragments are sentence-cased and usually end without a full stop; full sentences take one. The ellipsis is a real character `…`. Thousands separator is a dot (`29.600`), decimal separator a comma.

**Emoji: never.** Not one appears in either file. Do not introduce them.

**Tone in three words:** precise, considerate, unhurried. It explains *why* before *how* — the manual spends three slides on why accessibility matters before showing a single dialog box. Copy is generous with reasons and short on adjectives. There is no marketing voice here at all: no superlatives, no exclamation marks, no rhetorical questions.

**Boilerplate you will reuse verbatim:**

- Footer: `<Presentation title> | <Month Year>` — e.g. `Musterpräsentation | September 2021`
- Copyright placeholder: `© Copyright- oder Quellenhinweis, …`
- Contact line: `Telefon: +43 1 58801 …`, mail as `vorname.nachname@tuwien.ac.at`
- Closing URL: `www.tuwien.at`

---

## 3 · Visual foundations

### Colour

One primary does nearly all the work: **TU Blau `#006699`**. It is simultaneously the accent colour, the second dark text colour, the footer band, the bullet colour and the table header. Everything else is a tint of it or a neutral.

| Role | Value | Token |
|---|---|---|
| Primary | `#006699` | `--tuw-blue` |
| Tint 60 | `#72ADD5` | `--tuw-blue-60` |
| Tint 40 | `#A6D5EC` | `--tuw-blue-40` |
| Tint 10 | `#DFF2FD` | `--tuw-blue-10` |
| Neutral grey | `#646363` | `--tuw-grey` |
| Blue-grey | `#5485AB` | `--tuw-blue-grey` |
| Body text | `#000000` | `--tuw-black` |
| Surface | `#FFFFFF` | `--tuw-white` |

Two alternative themes swap the primary only: **TU Grün `#007E71`** and **TU Magenta `#BA4682`**. Apply with `data-tuw-theme="green" | "magenta"`. Blue is the default and should stay the default unless a unit has been assigned another.

**Cleared text combinations** (from the manual, measured):

- White on TU Blau — **6.25:1** — normal text, large text, graphics ✅
- White on Grau — **5.99:1** — large text and graphics
- White on Blaugrau — **3.95:1** — large text (18 pt+) and graphics only
- White on Grün 1 — **4.97:1**; white on Magenta 1 — **4.90:1**

Each combination is also valid inverted. **The manual states plainly that no other pairing is accessible** — do not invent tinted text colours. Colour must never be the only carrier of meaning (explicitly called out for traffic-light and negative-value displays).

### Typography

**Arial. That is the whole type system.** The Office theme sets both major and minor fonts to Arial; there is no display face, no serif, no monospace. Two weights: Regular 400 and Bold 700. Emphasis is bold or italic — **underline is forbidden** because it reads as a hyperlink. No letterspacing, no all-caps.

Slide ramp (PowerPoint pt → px on the 1280×720 canvas):

| Use | pt | px |
|---|---|---|
| Title-slide headline | 54 | 72 |
| Quote | 40 | 53.33 |
| Slide title | 32 | 42.67 |
| Slide title, "Logo klein" | 28 | 37.33 |
| Body level 1 | 22 | 29.33 |
| Body level 2 | 20 | 26.67 |
| Body levels 3–5 | 18 | 24 |
| Table cell | 16 | 21.33 |
| Footer / slide number | 12 | 16 |
| Source note | 8 | 10.67 |

18 pt is the stated accessibility floor for content. The 8 pt source note is the only exception and carries no content. Paragraph line spacing is 90 % in PowerPoint terms, which is `line-height: 1.08` in CSS (`--leading-tight`).

### Layout

The slide is 1280×720 with a strict, published guide grid:

- Outer margin **49 px** — logo, footer text and slide number all sit on it.
- Content column starts at **x = 137 px**, width **975 px** (1092 px on the title master).
- Standard title baseline block at **y = 163 px**, body well at **y = 276 px, height 325 px**.
- "Logo klein" variants move the title to **y = 49** and the body to **y = 163, height 438** — used when a slide genuinely needs more text.
- The blue band spans full width at **y = 655, height 65**.
- Big logo **204×77**, small square mark **51×51**, both at 49/49.

Layouts come in matched pairs (large logo / small logo) and cover: title, title with image, title with sub-logo, agenda, section heading, one/two/three content, two-column running text, text + image (portrait and landscape), table, chart, video, quote, image, contact and blank. Four-column layouts do not exist.

### Backgrounds, imagery, surfaces

Backgrounds are **flat white** or **flat `#006699`** — there is no third background colour, no gradient, no texture, no pattern, no illustration set. The only "full-bleed" treatment is a photograph occupying the top 637 px of a title slide, with the white band beneath carrying the title.

Photography is **documentary**: the campus building at Karlsplatz, researchers at real instruments. Cool-leaning, natural colour, no grain, no duotone, no filter. Images fill their placeholder edge-to-edge and are cropped to fit — square corners, no shadow, no border, no scrim. Text is never set over a busy photo; it sits in the white band below, or on the flat blue. Every image needs an alt text and a credit in the 8 pt note.

### Shape, depth, motion

- **Corner radius: 0.** Nothing in the system is rounded except the logo mark itself, whose rounded square is part of the artwork.
- **Shadows: none.** No drop shadows, no inner shadows, no elevation model. Surfaces separate by fill and by 1 px `#006699` rules.
- **Cards** in the PowerPoint sense do not exist. The nearest equivalent is a `#DFF2FD` tinted block or a 1 px blue-outlined block. Neither has radius or shadow.
- **Gradients: none**, anywhere.
- **Transparency:** used in exactly one place — table row banding at `rgba(0,102,153,.4)`. No blur, no glass, no protection gradients; TU Wien solves the text-on-image problem by not putting text on images.
- **Borders:** 1 px `#006699` for table gridlines, 2 px for the total-row rule and the header/body divide.

**Motion.** The templates specify no slide transitions and no build animations. The manual's only animation guidance is compositional: *„Bei der Animation von Bildern sollten diese am rechten oberen Eck ausgerichtet werden."* For HTML work, treat motion as functional only — 120 ms linear colour changes on hover, nothing decorative, no easing curves with overshoot, no bounce.

**Interaction states** (not defined by the source; the convention below is the documented extension used by the `Button` component):
- Hover — **darken** the fill or move to the solid blue fill. Never lighten, never add a shadow.
- Press — colour change only. No scale, no translate.
- Focus — 2 px solid `#006699` ring, 2 px offset.
- Disabled — 40 % opacity, no cursor change beyond `default`.
- Links — `#006699`, underlined **on hover only**, because a persistent underline is the template's link signal.

---

## 4 · Iconography

**There is no TU Wien icon set in the provided sources.** No icon font, no SVG sprite, no PNG icon library ships in either `.potx`. This is a genuine absence, not an oversight of this extraction.

What the templates use instead:

- **A single Unicode glyph as the only recurring ornament:** the black square **■** (Arial), coloured `#006699` and sized to 90 % of its line's text size, used as the bullet at every list level. Levels 6–9 fall back to `•`, but those levels are never reached in practice. This square is effectively the brand's icon.
- **Numerals in circles** for step-by-step callouts in the manual (annotation graphics, not a reusable system).
- **The logo mark** as the only piece of brand artwork.
- **No emoji, ever.**

**Recommendation for new work:** if a deliverable genuinely needs icons, do not draw them. Use a neutral, uniform-stroke open-source set from a CDN — **Lucide** (1.5–2 px stroke, square terminals, 24 px grid) sits closest to the flat, geometric, unrounded character of this brand — and colour every glyph `#006699` or `#000000` at 3:1 minimum contrast. **This is a substitution, not a TU Wien standard; flag it whenever you use it.**

Available brand artwork in `assets/`:

| File | Use |
|---|---|
| `logo-tuw-full.svg` / `.png` | Full lockup, colour — 204×77 on slides |
| `logo-tuw-full-white.svg` / `.png` | Full lockup, white knockout — on photos and blue |
| `logo-tuw-mark.svg` / `.png` | Square mark — 51×51, "Logo klein" layouts |
| `photo-title-hero.jpg` | Main building, Karlsplatz — title slides |
| `photo-wide-1.jpg`, `photo-wide-2.jpg`, `photo-large-1.png`, `photo-large-2.png` | Laboratory imagery for image layouts |
| `sublogo-example-1.png`, `sublogo-example-2.png` | Sub-logo placement examples from the template |

Never redraw, recolour, outline or reconstruct the mark. The templates also permit a **sub-logo** (a faculty or institute mark) on dedicated title layouts; it must fill its placeholder with no white margin, and PNG is the recommended format.

---

## 5 · Components

All components are plain React, styled from CSS custom properties, exported under the design-system namespace.

**`components/brand/`**
- **`Logo`** — the official lockup, full or square mark, colour or white knockout.
- **`UrlLockup`** — the white `www.tuwien.at` line in the title-slide blue band.

**`components/slide/`**
- **`Slide`** — the 1280×720 frame; owns background, logo slot and footer band. Variants `default`, `compact`, `accent`, `photo`, `blank`.
- **`SlideTitle`** — headline at 54 / 40 / 32 / 28 pt.
- **`SlideBody`** — the positioned content well, 1–3 columns.
- **`SlideFooter`** — the blue band with running footer and slide number.
- **`CopyrightNote`** — the 8 pt source line.

**`components/content/`**
- **`BulletList`** — square-bullet body copy, levels 1–5.
- **`AgendaList`** — two-level contents list.
- **`DataTable`** — both sanctioned table treatments, banding and total row.
- **`Quote`** — the centred statement for the blue quote slide.
- **`ContactBlock`** — the closing contact details.
- **`Button`** — see below.

### Intentional additions

- **`Button`** — the PowerPoint source defines no interactive controls, but any HTML deliverable needs one. It is built strictly from cleared brand values: square corners, white on `#006699`, no shadow, colour-only hover. Treat it as a house extension, not a TU Wien standard.

Nothing else has been invented. Families a general-purpose design system would normally carry — inputs, tabs, dialogs, toasts, avatars, badges — are **absent because the source defines none**. Ask before adding them.

---

## 6 · Index

```
readme.md                    this file
SKILL.md                     agent entry point
styles.css                   global CSS entry — @imports only
thumbnail.html               homepage tile

tokens/
  colors.css                 palette, semantic aliases, alternative themes
  typography.css             Arial stack, weights, slide + document ramps
  spacing.css                8px scale plus the template's real page guides
  slide-grid.css             1280x720 geometry, verbatim from the master
  elevation.css              radii, shadows, borders, motion (all near-zero)

components/
  brand/     Logo, UrlLockup
  slide/     Slide, SlideTitle, SlideBody, SlideFooter, CopyrightNote
  content/   BulletList, AgendaList, DataTable, Quote, ContactBlock, Button

guidelines/                  16 specimen cards: Colors, Type, Spacing, Brand
slides/                      10 standalone slide layouts as static HTML
ui_kits/presentation/        interactive click-through of the ten layouts
templates/tuw-presentation/  copy-and-edit 9-slide deck (Design Component)
assets/                      logos, mark, photography, sub-logo examples
sources/                     the original .potx files and their extracted media
```

---

## 7 · Known gaps

- **No webfont binaries.** The templates rely on locally installed Arial. `--font-sans` falls back to Helvetica Neue and Liberation Sans (metric-compatible). If TU Wien licenses a webfont for digital use, supply the files and the `@font-face` rules can be added.
- **No icon set** — see §4.
- **The corporate design wiki could not be read** (login-gated). Print rules, stationery, signage, sub-brand architecture and any wordmark guidance beyond the presentation templates are therefore missing.
- **No web or application source**, so no web UI kit exists. The one UI kit here recreates the presentation surface, which is the only product the sources document.
- Word and Excel templates referenced by the manual were not supplied.
