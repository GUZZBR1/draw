from draw import export
from tools import vectorize


def test_vectorizer_uses_shared_uniform_a4_scale():
    scale = export.sheet_scale(paper="A4")

    assert vectorize.SHEET_SCALE == scale
    assert vectorize.PAPER_W_MM == scale.effective_width_mm
    assert vectorize.PAPER_H_MM == scale.effective_height_mm


def test_vectorizer_svg_preserves_aspect_ratio_and_effective_sheet_size(tmp_path):
    output = tmp_path / "scale.svg"
    vectorize.write_svg(
        [{"points": [(0.0, 0.0), (850.0, 1200.0)], "closed": False}],
        output,
        epsilon=1.5,
        thinning="zhangsuen",
    )

    svg = output.read_text(encoding="utf-8")
    assert 'width="210.00mm" height="296.47mm"' in svg
    assert 'viewBox="0 0 210.00 296.47"' in svg
    assert 'points="0.00,0.00 210.00,296.47"' in svg
