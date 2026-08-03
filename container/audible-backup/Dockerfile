# Base image: pinned at the minor version (`3.14-slim`), NOT by digest.
# Renewal is scanner-tracked per `coding-policy: dependency-management`
# Freshness: `.github/dependabot.yml` runs the `docker` ecosystem against
# this directory weekly and opens the bump PR — that is how #738 landed.
# The tag still floats at PATCH level between bumps, which is what carries
# Debian/Python security fixes into a rebuild without a manifest edit.
#
# A minor bump here can break the audible-cli pin below (see #895 and the
# comment on that line), so the two move together under one gate: CI builds
# this image on every PR, including Dependabot's.
FROM python:3.14-slim

# ffmpeg is unversioned under an approved exception: `coding-policy:
# dependency-management` OS-Package Runtime Carve-Out (0.3.131), with
# `nanoclaw-host: os-package-floating` (0.1.55) as the authority-of-record. Debian's
# archive serves only the current version, so a literal `ffmpeg=<version>`
# pin stops resolving at the next security update and fails every build —
# the pin schedules an outage rather than freezing behaviour. The renewal
# mechanism is the rebuild cadence: this image is rebuilt on every deploy
# (step 2a-bis) and on every PR in CI, so it tracks current Debian stable
# continuously. deploy.sh step 3b-quater fails the deploy if the base stops
# being pinned or a package outside the recorded set appears.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Pin audible-cli — an earlier image silently shipped 0.3.3 (latest at
# its build time), which had removed the `decrypt` subcommand the script
# was calling. A floating dep would re-create that surprise on every
# rebuild. Bumping requires re-validating the download / decrypt flow.
#
# THIS PIN IS COUPLED TO THE BASE IMAGE ABOVE. audible-cli declares an
# upper `requires-python` bound, so a base-image minor bump can make the
# pinned version uninstallable — and pip reports it as "no matching
# distribution", which reads like a yanked release rather than an
# interpreter mismatch. That is #895: Dependabot #738 moved the base
# 3.13 → 3.14 on 2026-07-20 while 0.3.3 declared `<3.14`. Nothing built
# this image in CI or on deploy, so the break stayed invisible until a
# host prune removed the last image built under 3.13.
#
# 0.4.0 declares `<3.15,>=3.11`, so the SAME failure returns on
# `python:3.15-slim`. What catches it next time is the image build in CI
# (.github/workflows/ci.yml) plus deploy.sh step 2a-bis — a base bump that
# breaks this pin now fails at PR time instead of at backup time.
#
# Renewal is scanner-tracked, not memory: `.github/renovate.json`'s
# `custom.regex` manager watches this `audible-cli==<version>` literal
# (pypi datasource) and opens a bump PR, the same mechanism the agent
# image's npm globals use. CI builds this image on that PR, so a bump
# that breaks the recipe is red before it merges.
#
# 0.4.0 validated against every CLI surface backup.py drives: `api`,
# `activation-bytes`, and `download` with --asin / --output-dir /
# --aax-fallback / --cover / --cover-size / --chapter / --no-confirm.
RUN pip install --no-cache-dir audible-cli==0.4.0

WORKDIR /app
COPY backup.py enrich.py /app/

ENTRYPOINT ["python3", "/app/backup.py"]
