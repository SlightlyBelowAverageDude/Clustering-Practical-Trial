"""Apply the P0 documentation update to the execution roadmap (.docx) without touching anything else.

Edits (all in word/document.xml, plus one hyperlink relationship):
  1. after the "Checkboxes:" note, insert a shaded "Phase Completion and Documentation Rules" box with a link to the
     Markdown phase index (reports/roadmap/README.md on GitHub, branch Aiman-testing)
  2. tick the P0 checkbox content controls listed in --check (w14:checked 0 -> 1 and glyph U+2610 -> U+2612);
     every other checkbox is left exactly as it was
  3. after the P0 "DONE WHEN" paragraph, insert a small P0 status table that points to the P0 report
Usage: python docs/roadmap/source/update_roadmap_docx.py SRC.docx OUT.docx --check P0-001 P0-002 ... --status-json status.json
"""
import argparse, json, re, shutil, tempfile, zipfile
from pathlib import Path
from xml.sax.saxutils import escape

REPO = 'https://github.com/SlightlyBelowAverageDude/Clustering-Practical-Trial/blob/Aiman-testing/'
W = 9582  # text width in dxa (A4, 1162 margins)


def run(text, bold=False, color=None, size=None, link=False):
    rpr = ''
    if bold: rpr += '<w:b/>'
    if color: rpr += f'<w:color w:val="{color}"/>'
    if link: rpr += '<w:u w:val="single"/>'
    if size: rpr += f'<w:sz w:val="{size}"/>'
    rpr = f'<w:rPr>{rpr}</w:rPr>' if rpr else ''
    return f'<w:r>{rpr}<w:t xml:space="preserve">{escape(text)}</w:t></w:r>'


def para(runs, style='SmallText', after=None):
    sp = f'<w:spacing w:after="{after}"/>' if after is not None else ''
    return f'<w:p><w:pPr><w:pStyle w:val="{style}"/>{sp}</w:pPr>{"".join(runs)}</w:p>'


def hyperlink(rid, text):
    return f'<w:hyperlink r:id="{rid}" w:history="1">{run(text, color="0563C1", link=True)}</w:hyperlink>'


def cell(content, width, fill=None):
    shd = f'<w:shd w:val="clear" w:color="auto" w:fill="{fill}"/>' if fill else ''
    return f'<w:tc><w:tcPr><w:tcW w:type="dxa" w:w="{width}"/>{shd}</w:tcPr>{content}</w:tc>'


BORDERS = '<w:tblBorders>' + ''.join(f'<w:{b} w:val="single" w:sz="4" w:space="0" w:color="BFBFBF"/>' for b in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV')) + '</w:tblBorders>'


def rules_box(rid_index, rid_template):
    rules = [
        'Every phase (P0–P20, including optional P5A) ends with its own Markdown completion report, written from the template reports/roadmap/PHASE_REPORT_TEMPLATE.md.',
        'The report lives in reports/roadmap/phases/<PHASE_ID>/ as <PHASE_ID>_REPORT.md, next to that phase\'s artifacts/ and source/ folders.',
        'Every report states the work actually done, the measured findings, their interpretation, the limitations and the exact paths of all scripts, configurations, manifests, scores, assignments, figures and logs.',
        'Reports name every incomplete, failed, deferred or missing experiment explicitly; negative results are kept.',
        'Each phase receives a completion decision (COMPLETE / PARTIAL / BLOCKED / DEFERRED) in its report and in the phase index before the next phase begins.',
        'Results are validated (recomputed from saved outputs by a script) before a phase or a checkbox is marked complete.',
        'Section 12 of each report (presentation-ready summary) is the only source for later progress slides; every slide number must trace back to a report and an artifact.',
        'Results are committed to GitHub (branch Aiman-testing). Secrets (.env, API keys), raw image collections and model weights or other oversized artifacts are never committed; their paths and SHA-256 hashes are recorded instead.',
    ]
    inner = para([run('PHASE COMPLETION AND DOCUMENTATION RULES', bold=True, color='147849', size=22)], style='Normal', after=80)
    inner += para([run('These rules apply to every work package below. The task lists, dates, checkboxes and DONE WHEN criteria are unchanged.')], after=60)
    for i, r in enumerate(rules, 1):
        inner += para([run(f'{i}.  ', bold=True), run(r)], after=40)
    inner += para([run('Phase index (status of every phase, links to reports): ', bold=True), hyperlink(rid_index, 'reports/roadmap/README.md')], after=40)
    inner += para([run('Report template: ', bold=True), hyperlink(rid_template, 'reports/roadmap/PHASE_REPORT_TEMPLATE.md')], after=0)
    tbl = (f'<w:tbl><w:tblPr><w:tblW w:type="dxa" w:w="{W}"/><w:jc w:val="center"/>{BORDERS}<w:tblLayout w:type="fixed"/>'
           f'<w:tblCellMar><w:top w:w="100" w:type="dxa"/><w:left w:w="140" w:type="dxa"/><w:bottom w:w="100" w:type="dxa"/><w:right w:w="140" w:type="dxa"/></w:tblCellMar>'
           f'<w:tblLook w:val="0000" w:firstRow="0" w:lastRow="0" w:firstColumn="0" w:lastColumn="0" w:noHBand="1" w:noVBand="1"/></w:tblPr>'
           f'<w:tblGrid><w:gridCol w:w="{W}"/></w:tblGrid><w:tr><w:tc><w:tcPr><w:tcW w:type="dxa" w:w="{W}"/><w:shd w:val="clear" w:color="auto" w:fill="FFF8E6"/></w:tcPr>{inner}</w:tc></w:tr></w:tbl>')
    return tbl + '<w:p><w:pPr><w:spacing w:after="0"/></w:pPr></w:p>'


def status_table(status, rid_report):
    cols = [1100, 1500, 6982]
    head = '<w:tr>' + ''.join(cell(para([run(h, bold=True)], after=0), w, 'EAF3ED') for h, w in zip(['Task', 'Status', 'Evidence'], cols)) + '</w:tr>'
    body = ''.join('<w:tr>' + cell(para([run(r['task'])], after=0), cols[0]) + cell(para([run(r['status'], bold=True)], after=0), cols[1])
                   + cell(para([run(r['evidence'])], after=0), cols[2]) + '</w:tr>' for r in status['rows'])
    tbl = (f'<w:tbl><w:tblPr><w:tblW w:type="dxa" w:w="{W}"/><w:jc w:val="center"/>{BORDERS}<w:tblLayout w:type="fixed"/>'
           f'<w:tblCellMar><w:top w:w="40" w:type="dxa"/><w:left w:w="100" w:type="dxa"/><w:bottom w:w="40" w:type="dxa"/><w:right w:w="100" w:type="dxa"/></w:tblCellMar>'
           f'<w:tblLook w:val="0000" w:firstRow="1" w:lastRow="0" w:firstColumn="0" w:lastColumn="0" w:noHBand="1" w:noVBand="1"/></w:tblPr>'
           f'<w:tblGrid>' + ''.join(f'<w:gridCol w:w="{w}"/>' for w in cols) + f'</w:tblGrid>{head}{body}</w:tbl>')
    title = para([run(f"P0 STATUS: {status['phase_status']}", bold=True, color='147849'), run(f"  ({status['date']}, commit recorded in the report)  "),
                  run('Report: ', bold=True), hyperlink(rid_report, 'reports/roadmap/phases/P0/P0_REPORT.md')], after=60)
    note = para([run(status['note'])], after=120)
    return title + tbl + note


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('src'); ap.add_argument('out'); ap.add_argument('--check', nargs='*', default=[])
    ap.add_argument('--status-json', required=True); a = ap.parse_args()
    status = json.loads(Path(a.status_json).read_text())
    tmp = Path(tempfile.mkdtemp())
    with zipfile.ZipFile(a.src) as z:
        names = z.namelist(); z.extractall(tmp)
    doc = (tmp / 'word/document.xml').read_text(encoding='utf8')
    rels_p = tmp / 'word/_rels/document.xml.rels'; rels = rels_p.read_text(encoding='utf8')
    ids = [int(i) for i in re.findall(r'Id="rId(\d+)"', rels)]
    rid = {k: f'rId{max(ids) + n}' for n, k in enumerate(['index', 'template', 'report'], 1)}
    targets = {'index': 'reports/roadmap/README.md', 'template': 'reports/roadmap/PHASE_REPORT_TEMPLATE.md', 'report': 'reports/roadmap/phases/P0/P0_REPORT.md'}
    rels = rels.replace('</Relationships>', ''.join(
        f'<Relationship Id="{rid[k]}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink" Target="{REPO}{targets[k]}" TargetMode="External"/>'
        for k in rid) + '</Relationships>')
    n_before = len(re.findall(r'<w14:checkbox>', doc)); checked_before = doc.count('<w14:checked w14:val="1"/>')

    # 1. rules box after the "Checkboxes:" note
    i = doc.index('Checkboxes: '); e = doc.index('</w:p>', i) + len('</w:p>')
    doc = doc[:e] + rules_box(rid['index'], rid['template']) + doc[e:]

    # 2. tick the requested P0 checkboxes
    for tag in a.check:
        assert tag.startswith('P0-'), 'only P0 boxes may be ticked in P0'
        pat = re.compile(r'(<w:sdt><w:sdtPr><w:alias w:val="' + re.escape(tag) + r'"/>.*?</w:sdt>)', re.S)
        m = pat.search(doc); assert m, tag
        blk = m.group(1)
        new = blk.replace('<w14:checked w14:val="0"/>', '<w14:checked w14:val="1"/>').replace('<w:t>☐</w:t>', '<w:t>☒</w:t>')
        assert new != blk and new.count('☒') == 1
        doc = doc[:m.start()] + new + doc[m.end():]

    # 3. P0 status table after P0's DONE WHEN
    i = doc.index('P0  |  Freeze a fair comparison protocol'); j = doc.index('DONE WHEN', i); e = doc.index('</w:p>', j) + len('</w:p>')
    doc = doc[:e] + status_table(status, rid['report']) + doc[e:]

    n_after = len(re.findall(r'<w14:checkbox>', doc)); checked_after = doc.count('<w14:checked w14:val="1"/>')
    assert n_after == n_before, 'checkbox controls lost'
    assert checked_after == checked_before + len(a.check)
    (tmp / 'word/document.xml').write_text(doc, encoding='utf8'); rels_p.write_text(rels, encoding='utf8')
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
        for n in names:  # original member order; no directory entries
            if not n.endswith('/'):
                z.write(tmp / n, n)
    shutil.rmtree(tmp)
    print(json.dumps({'checkbox_controls': n_after, 'checked_before': checked_before, 'checked_after': checked_after, 'ticked': a.check}))


if __name__ == '__main__':
    main()
