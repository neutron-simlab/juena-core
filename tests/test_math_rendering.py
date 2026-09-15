"""Representative contracts for the extracted equation renderer."""

from juena_core.ui import math_rendering


def test_normalize_math_markdown_rewrites_aliases() -> None:
    assert math_rendering.normalize_math_markdown(
        r"Inline \(a^2+b^2=c^2\); block \[x=1\]"
    ) == "Inline $a^2+b^2=c^2$; block \n$$\nx=1\n$$"


def test_normalize_math_markdown_skips_code() -> None:
    content = "Inline \\(x\\) and `\\(y\\)`\n```tex\n\\[z\\]\n```"
    assert math_rendering.normalize_math_markdown(content) == (
        "Inline $x$ and `\\(y\\)`\n```tex\n\\[z\\]\n```"
    )


def test_repair_latex_delimiters_wraps_bare_equation() -> None:
    assert math_rendering.normalize_math_markdown(r"\frac{a}{b}") == r"$\frac{a}{b}$"


def test_command_reference_prose_is_not_treated_as_equation() -> None:
    content = r"Use \frac in LaTeX when formatting fractions."
    assert math_rendering.content_contains_math_markup(content) is False
    assert math_rendering.normalize_math_markdown(content) == content


def test_trailing_equation_fragment_is_repaired() -> None:
    content = (
        r"$\frac{d\sigma}{d\Omega}=|f|^{2}$ = "
        r"\left(\frac{m}{2\pi\hbar^{2}}\right)^{2}."
    )
    assert math_rendering.normalize_math_markdown(content) == (
        r"$\frac{d\sigma}{d\Omega}=|f|^{2}$ "
        r"$= \left(\frac{m}{2\pi\hbar^{2}}\right)^{2}$."
    )
