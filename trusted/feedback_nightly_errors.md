# Feedback: Report blocking errors from nightly housekeeping

**Rule**: If any nightly step fails in a way that requires host action, report it immediately — do NOT stay silent.

Examples of blockable failures to always report:
- Sync scripts failing due to missing host modules (e.g. reclaim-tripit-timezones-sync)
- run_host_script returning errors
- Git backup failures

Default silence rule applies to clean runs only. Broken = report.

Added: 2026-04-01
