# Webhook modes and task results implementation plan

**Goal:** Support ASR, subtitle translation with ASR fallback, and subtitle-only Webhook processing; expose completed task subtitles and a dedicated setup guide.

**Architecture:** Extend the existing config enum without changing saved `asr` / `subtitle` values. Use `subtitle_only` for strict processing and return a skipped response when no source exists. Reuse authenticated task details for a mounted result page and extend existing query-string navigation with results and guide views.

## Tasks

- [x] Add failing API tests for strict mode with external/internal/missing/invalid sources, config round trip, and existing subtitle skips.
- [x] Extend config, dispatch, selector labels, and README. Verify focused API tests.
- [x] Add a task result page using task ID, original and translated segments, loading/error/empty states, safe rendering, and return navigation.
- [x] Add a standalone Webhook guide page, sidebar/settings links, three-mode explanation, path mapping, HMAC requirements, request example, and response troubleshooting.
- [x] Verify result rendering and guide navigation in Chromium, including history/reload and mobile widths; run full pytest and diff checks.

## Validation

Full suite: 204 passed, 1 deselected. Chromium verified selected-task results, safe subtitle rendering, refresh, back navigation, guide-to-settings navigation, three-option config serialization and layouts at 320/390/768/1280/1920 pixels. JS syntax and diff checks passed. Narrow navigation was adjusted to two rows after detecting overflow at 320 pixels.

Independent review identified embedded tracks that exist but are empty or cannot be extracted. Selection now extracts and validates each candidate in an isolated temporary directory before choosing it. Regression coverage includes empty, unextractable and later usable streams in both translation modes. Real ffprobe/ffmpeg extraction was also verified with a generated MKV and a signed subtitle-only Webhook request. Follow-up review found no remaining important issues.

No changes to user config.json and no model downloads required.
