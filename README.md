# AI Social Media Analyzer

A single-file Streamlit app that helps creators, societies, businesses and social media managers review captions, posters, engagement numbers, comments and hashtags, and draft improved content.

It runs in two clearly labeled modes:

- **AI mode** (optional): uses the Google Gemini API through the official `google-genai` SDK for text and image analysis.
- **Local mode** (default, no key needed): simple rule-based text analysis and basic Pillow image checks. It never pretends an AI model produced a result.

## Features

- **Dashboard**: posts analyzed, average engagement rate, average content score, sentiment distribution, recent history, charts. Every figure is labeled *Calculated*, *Estimate*, *Rule-based* or *AI-generated*.
- **Caption Analyzer**: length vs. platform, opening hook, call to action, readability (Flesch), basic text checks, hashtag ideas, strengths/weaknesses, improved caption.
- **Poster Analyzer**: JPG/PNG/WEBP upload with preview, dimensions, brightness, contrast and edge-detail checks, aspect-ratio guidance. Readability, hierarchy, typography and layout feedback require AI vision.
- **Engagement Analyzer**: engagement rate by reach (or, clearly labeled, by followers), interaction breakdown, reach-to-impressions ratio, charts, rule-based recommendations. No benchmark comparisons.
- **Sentiment Analyzer**: paste comments or upload a CSV; positive/neutral/negative counts, percentages, themes, charts.
- **Hashtag Analyzer**: extraction, case-insensitive duplicate detection, topic grouping, copyable lists. No popularity or trend data.
- **AI Content Assistant**: alternative captions, short/long, professional/friendly, calls to action, content ideas, post descriptions, content calendar draft. Local mode only offers simple templates for some tasks.
- **Reports & History**: download CSV or TXT reports generated in memory.

## Project structure

```
ai-social-media-analyzer/
├── app.py            # all application logic, CSS and helpers
├── requirements.txt
└── README.md
```

## Prerequisites

- Python 3.10 or newer
- `pip`
- Optional: a Gemini API key from Google AI Studio (https://aistudio.google.com/apikey)

## Installation

### Windows (PowerShell)

```powershell
cd ai-social-media-analyzer
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If activation is blocked, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use `.venv\Scripts\activate.bat` in Command Prompt.

### macOS / Linux

```bash
cd ai-social-media-analyzer
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Enabling AI mode (optional)

The app reads these names from **environment variables first, then Streamlit secrets**:

| Name | Required | Purpose |
|---|---|---|
| `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) | Yes, for AI mode | Gemini API key |
| `GEMINI_MODEL` | No | Model name. Default: `gemini-2.5-flash` |

**Environment variable**

```bash
# macOS / Linux
export GEMINI_API_KEY="your-key"
# Windows PowerShell
$env:GEMINI_API_KEY = "your-key"
```

**Streamlit secrets** (local): create `.streamlit/secrets.toml` next to `app.py`:

```toml
GEMINI_API_KEY = "your-key"
```

This is your own local configuration, not part of the project deliverable. Never commit it; add `.streamlit/` to `.gitignore`.

Once a key is found, a **Use AI analysis** toggle appears in the sidebar. It is off by default. While it is on, text and images you analyze are sent to Google's Gemini API when you press an analyze/generate button.

## Running the app

```bash
streamlit run app.py
```

Open the URL shown in the terminal (usually http://localhost:8501).

### Deploying on Streamlit Community Cloud

1. Upload the three files to a GitHub repository (the GitHub web UI's "Add file → Upload files" works).
2. In Streamlit Community Cloud, create a new app pointing to `app.py`.
3. In the app's **Settings → Secrets**, paste `GEMINI_API_KEY = "your-key"`.

## How to use each analyzer

1. **Caption**: choose a platform, paste the caption, press *Analyze caption*.
2. **Poster**: upload an image (max 5 MB), press *Analyze poster*.
3. **Engagement**: enter metrics from your platform's analytics. Provide reach for the main rate; otherwise only the follower-based rate is shown.
4. **Sentiment**: paste one comment per line, or upload a CSV (max 2 MB) and pick the comment column.
5. **Hashtags**: enter existing hashtags and/or a caption.
6. **Assistant**: pick a task, describe the topic, press *Generate*.
7. **Reports & History**: choose one analysis or the whole session, pick CSV or TXT, download.

**Engagement formula:** `(Likes + Comments + Shares + Saves) / Reach × 100`. The fallback uses Followers in place of Reach and is labeled as such. Zero or missing denominators show "n/a".

## Local fallback mode

Without an API key (or with the toggle off), the app uses:

- word counts, a Flesch readability estimate, regex checks for calls to action and basic text issues;
- a small word-list sentiment method with simple negation handling;
- keyword-based hashtag ideas;
- Pillow pixel statistics for images (no OCR, no color or typography detection).

These are heuristics for guidance. The scores are checklists, not predictions of real performance.

## Security and privacy

- API keys are read only from the environment or Streamlit secrets, never hardcoded, displayed, logged, or included in reports or error messages.
- Content is sent to Google only when AI mode is on and you press a button. Review Google's terms and privacy policy for the Gemini API before using sensitive content.
- Uploaded files are validated (type, size, decodable image) and kept in memory; they are never executed or written to disk. CSV report cells beginning with `=`, `+`, `-` or `@` are escaped to prevent spreadsheet formula injection.
- All history is session-only (`st.session_state`). There is no database, login or permanent storage, and this MVP is not hardened for production use.

## Known limitations

- Local mode cannot judge tone, audience fit, sarcasm, other languages, or design quality.
- Flesch scoring is English-only and approximate.
- Platform character limits and image shapes are approximate and change over time; verify against current platform documentation.
- AI sentiment is limited to the first 100 comments per run; local mode to 5,000.
- AI output can be wrong. Review it, especially text read from images.
- The dark navy/purple theme is applied with CSS and looks best in Streamlit's dark mode; some built-in widgets keep their default styling.
- History disappears when the session ends or the page is fully reloaded.

## Troubleshooting

| Problem | Fix |
|---|---|
| `streamlit: command not found` | Activate the virtual environment, or run `python -m streamlit run app.py` |
| No AI toggle in the sidebar | The key was not found. Check the variable name, restart the app after setting it, and confirm `google-genai` is installed |
| "API key was rejected" / HTTP 400 | Recreate the key and check for stray spaces or quotes |
| "Rate limit or quota reached" | Wait and retry, or switch AI off to use local mode |
| "Configured model is unavailable" | Unset `GEMINI_MODEL` or set a model your key can use |
| CSV shows "malformed" | Save as UTF-8 CSV with a header row and a comment column |
| Image won't load | Use a real JPG, PNG or WEBP under 5 MB |
| Port already in use | `streamlit run app.py --server.port 8502` |

## Extending later

Everything lives in `app.py`: add a page function, register it in `PAGES` and `PAGE_FUNCS`, and call `call_ai()` for new AI tasks (it returns parsed JSON or raises `AIError`).
