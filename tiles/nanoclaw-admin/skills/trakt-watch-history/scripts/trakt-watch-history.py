#!/usr/bin/env python3
"""
Fetch Trakt.tv watch history. Outputs JSON to stdout.
Credentials from environment: TRAKT_CLIENT_ID, TRAKT_ACCESS_TOKEN.

Returns:
  {
    "shows": [{"title", "year", "trakt_id", "episodes_watched", "last_watched", "rating"}],
    "movies": [{"title", "year", "trakt_id", "last_watched", "rating"}],
    "ratings": {"show-slug": rating, ...},
    "fetched_at": "ISO timestamp"
  }
"""
import json, os, sys, urllib.request
from datetime import datetime, timezone

CLIENT_ID = os.environ.get("TRAKT_CLIENT_ID", "")
ACCESS_TOKEN = os.environ.get("TRAKT_ACCESS_TOKEN", "")

if not CLIENT_ID or not ACCESS_TOKEN:
    print(json.dumps({"error": "TRAKT_CLIENT_ID and TRAKT_ACCESS_TOKEN required"}))
    sys.exit(1)

HEADERS = {
    "Content-Type": "application/json",
    "trakt-api-version": "2",
    "trakt-api-key": CLIENT_ID,
    "Authorization": f"Bearer {ACCESS_TOKEN}",
    "User-Agent": "NanoClaw/1.0",
}


def api_get(path):
    req = urllib.request.Request(
        f"https://api.trakt.tv{path}",
        headers=HEADERS,
    )
    return json.loads(urllib.request.urlopen(req).read())


# Watched shows (with play counts)
watched_shows = api_get("/users/me/watched/shows")
shows = []
for entry in watched_shows:
    show = entry.get("show", {})
    eps = sum(
        sum(1 for e in season.get("episodes", []))
        for season in entry.get("seasons", [])
    )
    shows.append({
        "title": show.get("title"),
        "year": show.get("year"),
        "trakt_id": show.get("ids", {}).get("trakt"),
        "slug": show.get("ids", {}).get("slug"),
        "episodes_watched": eps,
        "last_watched": entry.get("last_watched_at"),
    })

# Watched movies
watched_movies = api_get("/users/me/watched/movies")
movies = []
for entry in watched_movies:
    movie = entry.get("movie", {})
    movies.append({
        "title": movie.get("title"),
        "year": movie.get("year"),
        "trakt_id": movie.get("ids", {}).get("trakt"),
        "slug": movie.get("ids", {}).get("slug"),
        "last_watched": entry.get("last_watched_at"),
    })

# Ratings (shows and movies)
ratings = {}
for item in api_get("/users/me/ratings/shows"):
    slug = item.get("show", {}).get("ids", {}).get("slug")
    if slug:
        ratings[slug] = item.get("rating")
for item in api_get("/users/me/ratings/movies"):
    slug = item.get("movie", {}).get("ids", {}).get("slug")
    if slug:
        ratings[slug] = item.get("rating")

# Attach ratings to shows/movies
for s in shows:
    s["rating"] = ratings.get(s["slug"])
for m in movies:
    m["rating"] = ratings.get(m["slug"])

# Sort by last watched (most recent first)
shows.sort(key=lambda x: x.get("last_watched") or "", reverse=True)
movies.sort(key=lambda x: x.get("last_watched") or "", reverse=True)

result = {
    "shows": shows,
    "movies": movies,
    "stats": {
        "total_shows": len(shows),
        "total_movies": len(movies),
        "rated": len(ratings),
    },
    "fetched_at": datetime.now(timezone.utc).isoformat(),
}

print(json.dumps(result, indent=2))
