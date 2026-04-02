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

**From "Finished" books — build preference profile:**
- Favorite authors (by finished count)
- Preferred genres and subgenres (by finished count)
- Themes and styles from Description/Summary
- Series he's completing (strong signal of sustained interest)

**Filtering rules — apply before generating any recommendations:**
- `Read Status = "Reading"` → **strongest exclusion signal**: skip any author, genre, or style matching an in-progress book
- `Purchase Date` before 2024 + `Read Status = "Unread"` → identify dominant genres of these books at runtime; treat those genres as low priority
- Genre with multiple abandoned/long-forgotten unread titles → exclude entirely

## Step 3: Check new releases (web search required)

My training cutoff is stale — always search the web for new books from top authors before recommending.

Derive the top authors list from the CSV (highest "Finished" count), plus any author Baruch specifically mentions.

For each top author, search: `"[Author name]" new book 2025 OR 2026 audiobook`

Cross-reference results against books-library.csv — if the new book is already in the library, skip. If it's not in the library and fits his taste → recommend it (flag as "not yet in your library, available on Audible").

## Step 4: Generate recommendations

| Request type | Logic |
|---|---|
| "What to read next" / unread queue | Filter `Read Status = Unread`. Priority: continuing an in-progress series > high-rated (≥4.3) > matching favorite genres. Flag series continuations: "Book 5 of [Series] — you've finished 1-4" |
| "Something like X" | Find books by same author, same genre/subgenre, or same series style. Check both Finished (for comparison) and Unread (for suggestions) |
| General recommendation | Mix: 1-2 unread books matching top genres + 1 wildcard from a less-explored genre with high rating |

## Step 5: Format response

Keep it tight — 3-5 recommendations max. For each, write a **targeted pitch**, not a book summary:
- Connect to something specific Baruch already likes ("Батчер но в Риме", "как Агата Кристи только жестче")
- Flag relevant facts: series length, whether finished, narrator quality if notable
- Be honest about weaknesses ("первые 2 книги медленные", "автор ещё не закончил серию")

```
*[Title]* — [Author]
[1-2 sentence targeted pitch tied to Baruch's taste]
[Duration] | ⭐ [Rating] | [Series note if relevant]
```

If a recommendation doesn't fit ("уже читал", "не закончена"), pivot immediately to alternatives — don't just say "okay".

Reply in Russian if Baruch asked in Russian.
