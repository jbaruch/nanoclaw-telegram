# Operator carve-out from `coding-policy: dependency-management`'s
# "Pin versions or use a lock file" — the base image and apt-installed
# ffmpeg are deliberately floating tags, NOT pinned digests/versions.
# Rationale: this sidecar is rebuilt only on operator demand (no CI
# build trigger, no scheduled rebuild), and the operator has explicitly
# opted out of manual base-image maintenance. Floating `python:3.13-slim`
# + unpinned ffmpeg lets each rebuild pick up Debian / Python security
# patches without requiring a manifest update; the cost is that a
# rebuild may behave differently from the prior one. We accept that
# tradeoff for runtime infrastructure (OS + decoder) but NOT for the
# Python application dependency below (audible-cli), where a version
# bump can silently break the script contract.
FROM python:3.14-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Pin audible-cli — the 5-week-old image silently shipped 0.3.3
# (latest at its build time), which had removed the `decrypt`
# subcommand the script was calling. A floating dep would re-create
# that surprise on every rebuild. 0.3.3 is the version the live retry
# on 2026-05-23 validated end-to-end with the ffmpeg-based decrypt
# path; bumping requires re-validating the download / decrypt flow.
RUN pip install --no-cache-dir audible-cli==0.3.3

WORKDIR /app
COPY backup.py enrich.py /app/

ENTRYPOINT ["python3", "/app/backup.py"]
