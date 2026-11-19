"""Record one completed roadmap phase in the Word roadmap, in place: tick that phase's checkbox controls and insert a
status table after the phase's DONE WHEN line. Every other checkbox, paragraph and style is left untouched.

Usage (repo root):
  python docs/roadmap/source/add_phase_status.py docs/roadmap/Clustering_Execution_Roadmap_W8_to_Final.docx P1 \
      --check P1-001 P1-002 --status-json docs/roadmap/source/p1_status.json
status json: {"phase_status": "...", "date": "...", "report": "reports/roadmap/phases/P1/P1_REPORT.md",
              "rows": [{"task": "P1-001", "status": "Complete", "evidence": "..."}], "note": "..."}
"""
import argparse, json, re, shutil, sys, tempfile, zipfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from update_roadmap_docx import REPO, W, BORDERS, run, para, hyperlink, cell  # noqa: E402  same building blocks as the P0 update


def status_block(phase, st, rid):
    cols = [1100, 1500, 6982]
    head = '<w:tr>' + ''.join(cell(para([run(h, bold=True)], after=0), w, 'EAF3ED') for h, w in zip(['Task', 'Status', 'Evidence'], cols)) + '</w:tr>'
    body = ''.join('<w:tr>' + cell(para([run(r['task'])], after=0), cols[0]) + cell(para([run(r['status'], bold=True)], after=0), cols[1])
                   + cell(para([run(r['evidence'])], after=0), cols[2]) + '</w:tr>' for r in st['rows'])
    tbl = (f'<w:tbl><w:tblPr><w:tblW w:type="dxa" w:w="{W}"/><w:jc w:val="center"/>{BORDERS}<w:tblLayout w:type="fixed"/>'
           f'<w:tblCellMar><w:top w:w="40" w:type="dxa"/><w:left w:w="100" w:type="dxa"/><w:bottom w:w="40" w:type="dxa"/><w:right w:w="100" w:type="dxa"/></w:tblCellMar>'
           f'<w:tblLook w:val="0000" w:firstRow="1" w:lastRow="0" w:firstColumn="0" w:lastColumn="0" w:noHBand="1" w:noVBand="1"/></w:tblPr>'
           f'<w:tblGrid>' + ''.join(f'<w:gridCol w:w="{w}"/>' for w in cols) + f'</w:tblGrid>{head}{body}</w:tbl>')
    title = para([run(f"{phase} STATUS: {st['phase_status']}", bold=True, color='147849'), run(f"  ({st['date']})  "),
                  run('Report: ', bold=True), hyperlink(rid, st['report'])], after=60)
    return title + tbl + para([run(st['note'])], after=120)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('docx'); ap.add_argument('phase'); ap.add_argument('--check', nargs='*', default=[])
    ap.add_argument('--status-json', required=True); a = ap.parse_args()
    st = json.loads(Path(a.status_json).read_text())
    tmp = Path(tempfile.mkdtemp())
    with zipfile.ZipFile(a.docx) as z:
        names = z.namelist(); z.extractall(tmp)
    doc = (tmp / 'word/document.xml').read_text(encoding='utf8')
    rels_p = tmp / 'word/_rels/document.xml.rels'; rels = rels_p.read_text(encoding='utf8')
    assert f'{a.phase} STATUS:' not in doc, f'{a.phase} status table already present'
    ids = [int(i) for i in re.findall(r'Id="rId(\d+)"', rels)]
    rid = f'rId{max(ids) + 1}'
    rels = rels.replace('</Relationships>', f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink" '
                                            f'Target="{REPO}{st["report"]}" TargetMode="External"/></Relationships>')
    n0, c0 = doc.count('<w14:checkbox>'), doc.count('<w14:checked w14:val="1"/>')
    for tag in a.check:
        assert tag.startswith(a.phase + '-'), f'{tag} is not a {a.phase} box'
        m = re.search(r'<w:sdt><w:sdtPr><w:alias w:val="' + re.escape(tag) + r'"/>.*?</w:sdt>', doc, re.S); assert m, tag
        new = m.group(0).replace('<w14:checked w14:val="0"/>', '<w14:checked w14:val="1"/>').replace('<w:t>☐</w:t>', '<w:t>☒</w:t>')
        assert new != m.group(0), f'{tag} already ticked'
        doc = doc[:m.start()] + new + doc[m.end():]
    hm = re.search(r'<w:t>' + re.escape(a.phase) + r'  \|  [^<]*</w:t>', doc); assert hm, a.phase
    j = doc.index('DONE WHEN', hm.end()); e = doc.index('</w:p>', j) + len('</w:p>')
    doc = doc[:e] + status_block(a.phase, st, rid) + doc[e:]
    assert doc.count('<w14:checkbox>') == n0 and doc.count('<w14:checked w14:val="1"/>') == c0 + len(a.check)
    (tmp / 'word/document.xml').write_text(doc, encoding='utf8'); rels_p.write_text(rels, encoding='utf8')
    out = Path(a.docx).with_suffix('.tmp.docx')
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
        for n in names:
            if not n.endswith('/'):
                z.write(tmp / n, n)
    out.replace(a.docx); shutil.rmtree(tmp)
    print(json.dumps({'checkbox_controls': n0, 'checked_before': c0, 'checked_after': c0 + len(a.check), 'ticked': a.check}))


if __name__ == '__main__':
    main()
