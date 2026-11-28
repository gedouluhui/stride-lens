"""Real isolated-browser regression checks. No shared profile or screenshots.

Install the optional browser extra and a Playwright browser, or select an already
installed browser with --channel msedge. A missing runtime is an error, not a skip.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def run(channel):
    import torch
    from playwright.sync_api import sync_playwright
    from stridelens import capture, write_report

    checks = []
    def require(name, condition):
        if not condition:
            raise AssertionError(name)
        checks.append(name)

    x = torch.arange(4096, dtype=torch.float32).reshape(64, 64)
    snapshot = capture({'large': x, 'broadcast': x[:1, :4].expand(4, 4),
                        'empty': x[:0], 'scalar': torch.tensor(2.),
                        '</script><script>window.badLabel=true</script>': x[:1, :1]})
    artifacts = ROOT / 'artifacts'
    artifacts.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='browser-check-', dir=artifacts) as folder:
        folder = Path(folder)
        report = folder / 'report.html'
        write_report(snapshot, report)
        with sync_playwright() as p:
            kwargs = {'headless': True}
            if channel:
                kwargs['channel'] = channel
            browser = p.chromium.launch(**kwargs)
            try:
                for width in (360, 1280):
                    context = browser.new_context(viewport={'width': width, 'height': 900},
                                                  reduced_motion='no-preference', accept_downloads=True)
                    page = context.new_page()
                    errors, external = [], []
                    page.on('pageerror', lambda error: errors.append(str(error)))
                    page.on('request', lambda req: external.append(req.url) if req.url.startswith(('http:', 'https:')) else None)
                    try:
                        page.goto(report.as_uri())
                        prefix = str(width) + ':'
                        require(prefix+'actual_viewport', page.evaluate('innerWidth') == width)
                        grid = page.locator('#logical-grid')
                        require(prefix+'bounded_page', grid.locator('button').count() == 64)
                        basis = page.get_by_role('columnheader', name='Evidence basis')
                        require(prefix+'basis_header_accessible', basis.count() == 1)
                        visible_pairs = [
                            pair for pair in snapshot['pairs']
                            if pair['a'] == 'large' or pair['b'] == 'large'
                        ]
                        rows = page.locator('#pair-body tr')
                        require(prefix+'basis_visible_in_rows', all(
                            rows.nth(i).locator('td.pair-basis').inner_text() == pair['basis']
                            for i, pair in enumerate(visible_pairs)))
                        require(prefix+'index_before_grid', page.locator('#index-input').bounding_box()['y'] < grid.bounding_box()['y'])
                        field = page.locator('#index-input')
                        field.fill('63, 63'); field.press('Enter')
                        chosen = grid.locator('[aria-pressed="true"]')
                        require(prefix+'off_initial_page_visible', chosen.count() == 1 and chosen.get_attribute('data-index') == '[63,63]')
                        require(prefix+'correct_address', page.locator('#address-result').inner_text() == 'S1 + 16380 B')
                        require(prefix+'one_grid_tab_stop', grid.locator('button[tabindex="0"]').count() == 1)
                        before = page.locator('#page-range').inner_text()
                        page.locator('#page-prev').click()
                        require(prefix+'previous_page_changes_range', page.locator('#page-range').inner_text() != before)
                        selected = grid.locator('button[tabindex="0"]'); before_index = selected.get_attribute('data-index')
                        selected.focus(); selected.press('ArrowRight')
                        require(prefix+'arrow_moves_selected_index', grid.locator('[aria-pressed="true"]').get_attribute('data-index') != before_index)
                        require(prefix+'visible_names_in_accessible_names', page.evaluate("""() => Array.from(document.querySelectorAll('#logical-grid button')).every(b => (b.getAttribute('aria-label') || '').replace(/\\s+/g,' ').includes(b.innerText.replace(/\\s+/g,' ').trim()))"""))
                        before_address = page.locator('#address-result').inner_text()
                        field.fill('99, 0'); field.press('Enter')
                        require(prefix+'invalid_preserves_address', page.locator('#index-error').inner_text() != '' and page.locator('#address-result').inner_text() == before_address)
                        page.locator('#tensor-select').select_option(label='empty')
                        require(prefix+'empty_disabled', field.is_disabled() and grid.locator('button').count() == 0)
                        page.locator('#tensor-select').select_option(label='scalar')
                        field.fill(''); field.press('Enter')
                        require(prefix+'scalar', grid.locator('button').count() == 1 and page.locator('#index-error').inner_text() == '')
                        page.locator('#tensor-select').select_option(index=4)
                        require(prefix+'unsafe_label_is_text', page.evaluate('window.badLabel === undefined'))
                        page.locator('#tensor-select').select_option(label='broadcast')
                        field.fill('3, 3'); field.press('Enter')
                        require(prefix+'broadcast_aliases', grid.locator('.alias').count() == 4)
                        before_motion = page.locator('.primary').evaluate('(b)=>getComputedStyle(b).transitionDuration')
                        require(prefix+'motion_control_exists', any(float(v.strip().rstrip('s')) > 0 for v in before_motion.split(',')))
                        page.emulate_media(reduced_motion='reduce')
                        reduced = page.locator('.primary').evaluate('(b)=>getComputedStyle(b).transitionDuration')
                        require(prefix+'reduced_motion', all(float(v.strip().rstrip('s')) == 0 for v in reduced.split(',')))
                        require(prefix+'no_page_overflow', page.evaluate('document.documentElement.scrollWidth <= innerWidth'))
                        with page.expect_download() as event:
                            page.locator('#download-json').click()
                        download = event.value
                        destination = folder / f'download-{width}.json'
                        download.save_as(destination)
                        require(prefix+'download_name', download.suggested_filename == 'stridelens-snapshot.json')
                        require(prefix+'download_disk_contents', json.loads(destination.read_text(encoding='utf-8')) == snapshot)
                        require(prefix+'no_page_errors', not errors)
                        require(prefix+'no_external_requests', not external)
                    finally:
                        context.close()
            finally:
                browser.close()
        return {'passed': True, 'checks': checks, 'channel': channel or 'chromium',
                'html_sha256': hashlib.sha256(report.read_bytes()).hexdigest(),
                'method': 'fresh isolated headless browser contexts; actual keyboard, paging and disk download; no screenshots'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--channel', default=None)
    args = parser.parse_args()
    try:
        result = run(args.channel)
    except AssertionError as exc:
        print(json.dumps({'passed': False, 'assertion': str(exc)}))
        return 1
    except Exception as exc:
        print(json.dumps({'passed': False, 'runtime_error': type(exc).__name__,
                          'detail': str(exc)}, ensure_ascii=True))
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
