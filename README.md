# Braidwater — Landing Page

Static landing page for Braidwater ("Clearer Minds for the Planetary Transition"), ready to deploy on Netlify.

## Structure

- `index.html` — the page (markup, styles, and component logic)
- `assets/js/` — page runtime (`dc-runtime.js`), image placeholder component, design-system bundle
- `assets/vendor/` — React 18.3.1 + ReactDOM (self-hosted, no CDN dependency)
- `assets/fonts/` — Barlow / Barlow Condensed (self-hosted woff2)
- `netlify.toml` — Netlify config (no build step; publishes repo root)

## Deploy

On Netlify: **Add new site → Import from Git**, pick this repo, branch `main`.
Leave the build command empty and the publish directory as `.` (already set in `netlify.toml`).

## Local preview

```bash
python3 -m http.server 8080
```

Then open http://localhost:8080.

## Notes

- Forms (intake, steward, newsletter, calculator email) are front-end only right now and don't send data anywhere yet.
- Image slots show placeholders until real photography is added.
