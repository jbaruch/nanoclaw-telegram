---
name: trakt-watch-history
description: Fetch Trakt.tv watch history (shows, movies, ratings) for TV/movie recommendations. Runs host-side via run_host_script. Use when the user asks for show recommendations, what to watch, or wants to see their watch history.
---

# Trakt Watch History

Run via host: `mcp__nanoclaw__run_host_script(script: "trakt-watch-history.py")`

The script returns JSON:
```json
{
  "shows": [{"title", "year", "trakt_id", "slug", "episodes_watched", "last_watched", "rating"}],
  "movies": [{"title", "year", "trakt_id", "slug", "last_watched", "rating"}],
  "stats": {"total_shows", "total_movies", "rated"},
  "fetched_at": "ISO timestamp"
}
```

Use the watch history + ratings to recommend shows/movies. Higher-rated shows indicate preferences. Genre patterns emerge from the collection.
