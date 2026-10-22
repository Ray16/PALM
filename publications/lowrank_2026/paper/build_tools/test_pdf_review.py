"""A stale visual-review flag must not certify changed rendered PDF content."""
from pathlib import Path
import pymupdf as fitz
from matplotlib.font_manager import findfont
from pdf_review import signature,review_issues


def make_pdf(path,text='Reviewed text',fontsize=18):
    doc=fitz.open();page=doc.new_page(width=300,height=200)
    page.insert_font(fontname='DejaVuSans',fontfile=findfont('DejaVu Sans'))
    page.insert_text((30,60),text,fontname='DejaVuSans',fontsize=fontsize)
    doc.save(path);doc.close()


def reviewed_pair(tmp_path):
    for name in ['manuscript.pdf','supplement.pdf']:make_pdf(tmp_path/name)
    return dict(passed=True,reviewed_documents={name:signature(tmp_path/name) for name in ['manuscript.pdf','supplement.pdf']})


def test_changed_pdf_rejected_despite_old_passed_flag(tmp_path):
    review=reviewed_pair(tmp_path)
    assert review_issues(tmp_path,review)==[]
    changed=tmp_path/'changed.pdf';make_pdf(changed,text='Unreviewed replacement');changed.replace(tmp_path/'manuscript.pdf')
    assert any('differs from reviewed' in s for s in review_issues(tmp_path,review))


def test_metadata_change_keeps_review_valid(tmp_path):
    review=reviewed_pair(tmp_path)
    path=tmp_path/'manuscript.pdf';doc=fitz.open(path);doc.set_metadata({'title':'Changed metadata only'});new=tmp_path/'metadata.pdf';doc.save(new);doc.close();new.replace(path)
    assert review_issues(tmp_path,review)==[]


def test_small_figure_font_rejected(tmp_path):
    review=reviewed_pair(tmp_path)
    path=tmp_path/'small.pdf';make_pdf(path,fontsize=16);path.replace(tmp_path/'manuscript.pdf')
    assert any('below 18 pt' in s for s in review_issues(tmp_path,review))
