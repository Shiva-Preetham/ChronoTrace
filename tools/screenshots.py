"""Screenshot every preloaded scenario at desktop and phone width into docs/screenshots/.

  python -m chronotrace serve            # in another terminal
  python tools/screenshots.py [http://127.0.0.1:8000]
"""

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
OUT = Path(__file__).resolve().parents[1] / "docs" / "screenshots"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except Exception:  # bundled Chromium not downloaded: use an installed Edge or Chrome
            try:
                browser = p.chromium.launch(channel="msedge")
            except Exception:
                browser = p.chromium.launch(channel="chrome")
        cases = browser.new_page().request.get(f"{BASE}/api/cases").json()
        for label, size in (("desktop", (1280, 900)), ("phone", (390, 844))):
            page = browser.new_page(viewport={"width": size[0], "height": size[1]})
            errors: list[str] = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            for c in cases:
                page.goto("about:blank")
                page.goto(f"{BASE}/#{c['id']}")
                page.wait_for_selector("#v-headline:not(:empty)")
                page.wait_for_function("document.getElementById('vid-packing').readyState >= 2")
                overflow = page.evaluate("document.documentElement.scrollWidth > window.innerWidth")
                if overflow:
                    errors.append(f"{c['id']}: horizontal scroll at {size[0]} px")
                path = OUT / f"{c['id']}_{label}.png"
                page.screenshot(path=str(path), full_page=True)
                print(f"{path.name}: {page.inner_text('#v-status')}")
            if errors:
                sys.exit("Problems: " + "; ".join(errors))
        browser.close()


if __name__ == "__main__":
    main()
