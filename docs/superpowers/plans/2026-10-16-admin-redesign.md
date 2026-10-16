# Admin Workspace Redesign

**Goal:** Replace the visually busy admin dashboard with a calm, readable workspace, preserving all current subtitle operations.

**Architecture:** Keep existing API calls and DOM IDs. Add admin-only CSS and interaction helpers, leaving shared home/login styles intact. Reorganize the admin HTML into a sidebar, compact task overview, media workspace, and tabbed settings dialog.

**Scope:** Responsive layouts from 320px to desktop; accessible navigation, buttons and dialogs; all current engine choices, task controls, escaping and file actions retained. Do not modify local config.json or model/backend behavior.

- [x] Reorganize the HTML shell, navigation and overview; add an admin-only stylesheet and SVG icon sprite.
- [x] Convert the existing settings groups into labeled tabs, retaining form IDs and save behavior.
- [x] Improve keyboard dismissal, focus restoration and background scroll locking for overlays.
- [x] Verify settings navigation and existing interactions with tests; inspect browser renders with empty/populated tasks and long media paths at desktop/mobile widths.
- [x] Run pytest, inspect diff, and leave changes ready for user review.

Validation: 181 pytest tests passed (1 integration test deselected). Headless Chromium checks with mock API responses covered populated and empty libraries, search, selection, settings serialization across tabs, keyboard interactions, unique IDs, 30 active tasks, and viewport widths 320/390/768/1280/1440/1920px. Preview screenshots are under ignored `tmp/admin-redesign/`.

## Follow-up: page navigation and public home

User preference: task center and settings should use dedicated views instead of a drawer/modal; also redesign the public home page.

- [x] Convert media, tasks and settings into mounted page sections with query-string navigation, active sidebar links and browser history.
- [x] Keep folder hashes and unsaved configuration values across view changes; refresh active task view without clearing selected tasks.
- [x] Redesign the bilingual homepage with consistent branding, responsive typography, a subtitle illustration, and feature summaries.
- [x] Verify real browser navigation/back/refresh, form saving, language switching and mobile layouts; run all regression tests.

Follow-up validation: 183 pytest tests passed (1 deselected); JavaScript syntax checks passed. Real Chromium with mock API data verified bilingual persistence, settings navigation/back/reload, folder hash preservation, unsaved edits, invalid hidden-panel fields, configuration serialization, search/selection, and generation dialog dismissal. Home, task and settings views checked at 320–1920px without horizontal overflow or JavaScript exceptions.
