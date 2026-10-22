"""Content-based PDF review signatures, independent of PDF timestamps/object IDs."""
from pathlib import Path
import hashlib
import pymupdf as fitz


def signature(path):
    h=hashlib.sha256();fonts=[];clipped=[]
    with fitz.open(path) as document:
        pages=len(document)
        for i,page in enumerate(document):
            pix=page.get_pixmap(matrix=fitz.Matrix(1,1),alpha=False)
            h.update(f'{i}:{pix.width}:{pix.height}:{pix.n}:'.encode());h.update(pix.samples)
            for block in page.get_text('dict')['blocks']:
                for line in block.get('lines',[]):
                    for span in line['spans']:
                        if span['font'].startswith('DejaVu'):
                            fonts.append(span['size'])
                            if not page.rect.contains(fitz.Rect(span['bbox'])):
                                clipped.append(dict(page=i+1,text=span['text']))
    return dict(pages=pages,render_sha256=h.hexdigest(),minimum_figure_font_pt=min(fonts) if fonts else None,
                all_figure_fonts_at_least_18=all(s>=17.99 for s in fonts),clipped_figure_text=clipped)


def review_issues(paper,review):
    issues=[]
    for name in ('manuscript.pdf','supplement.pdf'):
        expected=review.get('reviewed_documents',{}).get(name)
        if not expected:
            issues.append('missing content-based visual review: '+name);continue
        path=Path(paper)/name
        if not path.exists():issues.append('missing compiled document: '+name);continue
        actual=signature(path)
        if actual['pages']!=expected.get('pages') or actual['render_sha256']!=expected.get('render_sha256'):
            issues.append('compiled PDF differs from reviewed rendering: '+name)
        if not actual['all_figure_fonts_at_least_18']:
            issues.append('compiled figure font below 18 pt: '+name)
        if actual['clipped_figure_text']:
            issues.append('compiled figure text outside page: '+name)
    return issues
