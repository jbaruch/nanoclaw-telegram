---
name: recommend-books
description: Recommend audiobooks based on Baruch's reading history and preferences. Analyzes reading patterns, filters unread titles by genre and rating, identifies series continuations, checks for new releases from favorite authors, and suggests similar authors or highly-rated unread titles from books-library.csv. Use when Baruch asks for book recommendations, "what to read next", wants something similar to a specific author, or asks about his unread queue.
---

# Audiobook Recommendation Skill

## Input

Baruch may ask with or without specifics:
- "Посоветуй книгу" (general)
- "Что-нибудь похожее на Jeremy Robinson"
- "Хочу что-то легкое / фантастику / триллер"
- "Что у меня непрочитанного?"

## Step 1: Load library

Read `/workspace/group/books-library.csv`. Key fields:
- `Title`, `Author`, `Genre`, `Series Name`, `Series Sequence`
- `Read Status`: "Finished" / "Unread" / "Reading"
- `Ave. Rating`: float, higher = better rated
- `Purchase Date`: when Baruch bought it
- `Duration`: listen time
- `Description` / `Summary`: plot summary

## Step 2: Analyze reading patterns

Derive all counts and current state directly from the CSV — do not rely on hardcoded stats.

**Positive signals** (from "Finished" books):
- Favorite authors (by count)
- Preferred genres and subgenres (by count)
- Themes and styles from Description/Summary: action thrillers, sci-fi military, cozy Scottish crime, monster/creature action, epic fantasy, spy fiction, cozy mystery, etc.
- Series he's completing — strong signal of sustained interest

**Negative signals** (the main indicator of what didn't work):
- `Read Status = "Reading"` — **strongest negative signal**. Identify all such books from the CSV at runtime. Avoid recommending the same author, genre, or style as any currently-in-progress title.
- Unread books bought before 2024 = bought, never started, forgot — mild negative. Pattern: mostly business/self-help (7 Habits, Hooked, High Performance Habits) → this entire genre is low priority
- Never recommend something from a genre that has multiple abandoned/long-forgotten books

## Step 3: Check new releases (web search required)

My training cutoff is stale — always search the web for new books from top authors before recommending.

Derive the top authors list from the CSV (highest "Finished" count). The following are illustrative examples of likely high-priority authors — confirm against the CSV at runtime:
- Jeremy Robinson, Craig Alanson, JD Kirk, Daniel Silva, Jason Anspach + Nick Cole
- John Scalzi, Harlan Coben, Brandon Sanderson, Nelson DeMille, James S.A. Corey
- Any author Baruch specifically mentions

For each top author, search: `"[Author name]" new book 2025 OR 2026 audiobook`

Cross-reference results against books-library.csv — if new book is already in library, skip. If it's not in library and fits his taste → recommend it (flag as "not yet in your library, available on Audible").

## Step 4: Generate recommendations

**If "what to read next" / unread queue:**
- Filter `Read Status = Unread`
- Prioritize: continuing an in-progress series > high-rated (≥4.3) > matching favorite genres
- Flag series continuations clearly: "Book 5 of [Series] — you've finished 1-4"

**If "something like X":**
- Find books by same author, same genre/subgenre, or same series style
- Check both Finished (for comparison) and Unread (for suggestions)

**If general recommendation:**
- Mix: 1-2 unread books that match top genres, 1 wildcard from a less-explored genre with high rating

## Step 5: Format response

Keep it tight — 3-5 recommendations max. For each, write a **targeted pitch**, not a book summary:
- Connect to something specific Baruch already likes ("Батчер но в Риме", "как Агата Кристи только жестче")
- Flag relevant facts: series length, whether finished, narrator quality if notable
- Be honest about weaknesses ("первые 2 книги медленные", "автор ещё не закончил серию")

Example format:
```
*[Title]* — [Author]
[1-2 sentence targeted pitch tied to Baruch's taste]
[Duration] | ⭐ [Rating] | [Series note if relevant]
```

If a recommendation doesn't fit ("уже читал", "не закончена"), pivot immediately to alternatives — don't just say "okay".

Reply in Russian if Baruch asked in Russian.
