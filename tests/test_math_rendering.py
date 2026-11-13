"""Tests for equation-markup detection and normalization helpers."""

import juena_core.ui.math_rendering as math_rendering


def test_normalize_math_markdown_rewrites_inline_alias() -> None:
    assert math_rendering.normalize_math_markdown(r"Equation: \(a^2+b^2=c^2\)") == (
        r"Equation: $a^2+b^2=c^2$"
    )


def test_normalize_math_markdown_rewrites_block_alias() -> None:
    assert math_rendering.normalize_math_markdown(r"Equation:\[a^2+b^2=c^2\]Done") == (
        "Equation:\n$$\na^2+b^2=c^2\n$$\nDone"
    )


def test_normalize_math_markdown_preserves_existing_dollar_syntax() -> None:
    content = "Inline $x^2$ and block $$\ny=x+1\n$$"
    assert math_rendering.normalize_math_markdown(content) == content


def test_normalize_math_markdown_skips_inline_code_and_fenced_code() -> None:
    content = "Inline \\(x\\) and `\\(y\\)`\n```tex\n\\[z\\]\n```"
    assert math_rendering.normalize_math_markdown(content) == (
        "Inline $x$ and `\\(y\\)`\n```tex\n\\[z\\]\n```"
    )


def test_normalize_math_markdown_leaves_unmatched_aliases_literal() -> None:
    content = r"Broken inline \(x^2 + y^2 and broken block \[z^2"
    assert math_rendering.normalize_math_markdown(content) == content


def test_content_contains_math_markup_ignores_code_only_math() -> None:
    assert math_rendering.content_contains_math_markup(r"`\(a+b\)`") is False


def test_content_contains_math_markup_detects_alias_or_dollar_syntax() -> None:
    assert math_rendering.content_contains_math_markup(r"Equation: \(a+b\)") is True
    assert math_rendering.content_contains_math_markup("Equation: $a+b$") is True


def test_repair_latex_delimiters_inserts_missing_opening_dollar() -> None:
    content = r"1. \frac{4\pi R^{3}}{3},\frac{j_{1}(qR)}{qR}$"
    assert math_rendering.normalize_math_markdown(content) == (
        r"1. $\frac{4\pi R^{3}}{3},\frac{j_{1}(qR)}{qR}$"
    )


def test_repair_latex_delimiters_inserts_missing_closing_dollar() -> None:
    content = r"2. $\displaystyle \frac{d\sigma}{d\Omega}= \left(\frac{\mu R^{3}}{3\hbar^{2}}\right)^{2}"
    assert math_rendering.normalize_math_markdown(content) == (
        r"2. $\displaystyle \frac{d\sigma}{d\Omega}= \left(\frac{\mu R^{3}}{3\hbar^{2}}\right)^{2}$"
    )


def test_repair_latex_delimiters_wraps_bare_equation_line() -> None:
    content = r"\frac{a}{b}"
    assert math_rendering.normalize_math_markdown(content) == r"$\frac{a}{b}$"


def test_content_contains_math_markup_detects_broken_latex_lines() -> None:
    assert math_rendering.content_contains_math_markup(r"1. \frac{4\pi R^{3}}{3}$") is True
    assert math_rendering.content_contains_math_markup(
        r"$\displaystyle \frac{d\sigma}{d\Omega}= \left(\frac{\mu R^{3}}{3\hbar^{2}}\right)^{2}"
    ) is True


def test_content_contains_math_markup_does_not_treat_latex_prose_reference_as_equation() -> None:
    assert math_rendering.content_contains_math_markup(r"Use \frac in LaTeX when formatting fractions.") is False


def test_normalize_math_markdown_does_not_wrap_command_reference_prose() -> None:
    content = r"\frac is a LaTeX command for fractions."
    assert math_rendering.normalize_math_markdown(content) == content


def test_normalize_math_markdown_keeps_fully_delimited_equation_list_item() -> None:
    content = (
        r"1. $\psi^{(+)}(\mathbf{r}) = e^{i\mathbf{k}_i\cdot\mathbf{r}} + "
        r"\int d^3r'\, G^{(+)}(\mathbf{r},\mathbf{r}')\, V(\mathbf{r}')\,"
        r"\psi^{(+)}(\mathbf{r}')$"
    )
    assert math_rendering.normalize_math_markdown(content) == content


def test_normalize_math_markdown_wraps_trailing_equation_fragment_after_math_block() -> None:
    content = (
        r"2. $\frac{d\sigma}{d\Omega}=|f(\mathbf{k}_f,\mathbf{k}_i)|^{2}$ = "
        r"\left(\frac{m}{2\pi\hbar^{2}}\right)^{2}\! "
        r"\bigl|\tilde V(\mathbf{q})\bigr|^{2}."
    )
    assert math_rendering.normalize_math_markdown(content) == (
        r"2. $\frac{d\sigma}{d\Omega}=|f(\mathbf{k}_f,\mathbf{k}_i)|^{2}$ "
        r"$= \left(\frac{m}{2\pi\hbar^{2}}\right)^{2}\! "
        r"\bigl|\tilde V(\mathbf{q})\bigr|^{2}$."
    )


def test_normalize_math_markdown_keeps_dollars_out_of_bracket_display_block() -> None:
    # A correct \[ ... \] block whose second line starts with \sigma must not
    # come back wrapped in $...$, which renders as red LaTeX source.
    content = (
        "The inverse-variance weighted (IVW) mean\n"
        "\\[\n"
        r"I_{\text{c}}(q)=\frac{\sum_k I_k(q)/\sigma_k^2(q)}{\sum_k 1/\sigma_k^2(q)},\qquad" "\n"
        r"\sigma_{\text{c}}^2(q)=\frac{1}{\sum_k 1/\sigma_k^2(q)} ." "\n"
        "\\]\n"
        "If the two runs have identical counting statistics"
    )
    assert math_rendering.normalize_math_markdown(content) == (
        "The inverse-variance weighted (IVW) mean\n"
        "$$\n"
        r"I_{\text{c}}(q)=\frac{\sum_k I_k(q)/\sigma_k^2(q)}{\sum_k 1/\sigma_k^2(q)},\qquad" "\n"
        r"\sigma_{\text{c}}^2(q)=\frac{1}{\sum_k 1/\sigma_k^2(q)} ." "\n"
        "$$\n"
        "If the two runs have identical counting statistics"
    )


def test_normalize_math_markdown_leaves_dollar_display_block_unchanged() -> None:
    content = (
        "the error bar improves by $\\sqrt{2}$:\n"
        "$$\n"
        r"\sigma_{\text{c}} \approx \frac{\sigma}{\sqrt{2}} ." "\n"
        "$$\n"
        "The code below implements the general IVW formula."
    )
    assert math_rendering.normalize_math_markdown(content) == content


def test_normalize_math_markdown_still_repairs_bare_equation_after_display_block() -> None:
    content = "$$\n\\sigma^2 = \\frac{1}{N}\n$$\n\\frac{a}{b}"
    assert math_rendering.normalize_math_markdown(content) == (
        "$$\n\\sigma^2 = \\frac{1}{N}\n$$\n$\\frac{a}{b}$"
    )


def test_normalize_math_markdown_wraps_trailing_bare_expression_after_math_block() -> None:
    content = (
        r"3. $\frac{d\sigma}{d\Omega}= "
        r"\left(\frac{m V_{0}\, \pi^{3/2} a^{3}}{2\pi\hbar^{2}}\right)^{2}$ "
        r"e^{-q^{2}a^{2}/2}, \qquad q = 2k\sin\frac{\theta}{2}."
    )
    assert math_rendering.normalize_math_markdown(content) == (
        r"3. $\frac{d\sigma}{d\Omega}= "
        r"\left(\frac{m V_{0}\, \pi^{3/2} a^{3}}{2\pi\hbar^{2}}\right)^{2}$ "
        r"$e^{-q^{2}a^{2}/2}, \qquad q = 2k\sin\frac{\theta}{2}$."
    )
