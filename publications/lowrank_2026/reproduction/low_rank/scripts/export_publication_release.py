"""Build a namespaced release tree without changing active experiment sources.

Only lowrank.* is installed. Legacy top-level packages are retained in the
research worktree but not exported, avoiding collisions with HuggingFace datasets
and unrelated features/eval/splitters packages. Source tests are transformed by
the same import mapping and must pass before this tree is a release candidate.
"""
from __future__ import annotations
import ast
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import tokenize

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "paper/mlst_2026/release_candidate_namespaced"
PACKAGES = ("splitters", "datasets", "features", "eval")


def publication_docs(relative, text):
    """Align documentation without changing experiment code or numeric behaviour."""
    relative = str(relative)
    if relative == 'lowrank/__init__.py':
        end = text.index('"""', 3) + 3
        text = ('"""Capacity-controlled splitting through low-rank similarity factors.\n\n'
                'The plain method uses compact fold sums. Optional kernel powers and\n'
                'capped neighbour groups control local contamination, with additional\n'
                'neighbour-construction work and conditional group preservation.\n'
                'Support modules and the registry live inside lowrank.*.\n"""' + text[end:])
    if relative == 'lowrank/nystrom.py':
        text = text.replace('Pick landmark rows by k-means++ (D^2-weighted) sampling on the GPU.',
            'Spread landmarks using squared similarity-distance weights on the tensor device.\n\n'
            'The legacy kmeans++ name denotes (1-max_similarity)^2 sampling,\n'
            'rather than ordinary squared Hilbert-distance k-means++ sampling.')
        start = text.index('Tighter-sketch variants')
        end = text.index('"""', start)
        text = text[:start] + text[end:]
    if relative == 'lowrank/grouped.py':
        start = text.index('The sum objective')
        end = text.index('  1. ``must_link_groups``', start)
        text = text[:start] + ('Capped groups encourage strongly related entities to remain together.\n'
                'A dense cut and nearest-neighbour contamination measure different\n'
                'properties; grouping restricts assignments while retaining the\n'
                'factor-space objective, subject to capacity repair.\n\n') + text[end:]
    if relative == 'lowrank/target_gap.py':
        text = text.replace('GapCalibrator (fits gap ≈ b + a·alpha from probe splits)',
                            'GapCalibrator (isotonic piecewise fit; optional linear summary)')
        text = text.replace('The gap is measured with the same estimator as',
            'This public API measures train-minus-test score gaps. The manuscript\n'
            'also studies a separately defined random-relative test difficulty,\n'
            'using the generic GapCalibrator to interpolate those observations.\n'
            'These definitions must not be interchanged.\n\n'
            'The train-minus-test gap is measured with the same estimator as')
    if relative == 'splitters/__init__.py':
        text = text.replace('"metric", "leakage", "imbalance", "runtime_s"',
                            '"metric", "leakage", "split_counts", "runtime_s"')
    if relative == 'splitters/base.py':
        text = text.replace('``splits`` and ``names`` must be the same length (2 or 3). ``epsilon`` is the\n    (1 +/- epsilon) balance corridor. ``seed`` seeds all method randomness.',
            '``splits`` and ``names`` must be the same length (2 or 3). ``seed``\n'
            'seeds method randomness. ``epsilon`` is shared balance metadata; the\n'
            'low-rank methods use exact counts unless allowed_size_imbalance or\n'
            'balance_slack is explicitly supplied to split().')
        text = text.replace('``diagnostics`` always carries ``method``, ``n``, ``metric``, ``runtime_s``,\n    ``imbalance``, ``leakage`` and the realized ``split_fractions``; individual\n    methods add their own keys (``km1``, ``cut``, ``moves``, ``rank``, ...).',
            '``diagnostics`` carries ``method``, ``n``, ``runtime_s``,\n'
            '``split_counts`` and ``split_fractions``. Methods add optional keys\n'
            'such as metric, leakage, factor_leakage, rank and broken_groups.\n'
            'An imbalance key is not universally returned.')
    if relative == 'lowrank/optimize.py':
        text = text.replace('    Alternates: block sums from current labels,',
            '    The legacy epsilon argument is retained for call compatibility and\n'
            '    is unused here; balance_slack controls the actual corridor.\n\n'
            '    Alternates: block sums from current labels,', 1)
    return text


def namespace(text):
    text = re.sub(r"(?m)^(\s*)from (splitters|datasets|features|eval)(?=\.|\s)",
                  lambda m: m[1] + "from lowrank." + m[2], text)
    pattern = re.compile(r"(?m)^(\s*)import (splitters|datasets|features|eval)(\.[\w.]+)?(?: as (\w+))?(\s*(?:\#.*)?)$")

    def module(m):
        indent, root, suffix, alias, rest = m.groups()
        if alias:
            return indent + "import lowrank." + root + (suffix or "") + " as " + alias + rest
        result = indent + "from lowrank import " + root
        if suffix:
            result += "\n" + indent + "import lowrank." + root + suffix
        return result + rest

    text = pattern.sub(module, text)
    tokens = []
    for token in tokenize.generate_tokens(io.StringIO(text).readline):
        if token.type == tokenize.STRING:
            try:
                value = ast.literal_eval(token.string)
            except (SyntaxError, ValueError):
                value = None
            if isinstance(value, str) and re.fullmatch(r"(?:splitters|datasets|features|eval)(?:\.[A-Za-z_]\w*)+", value):
                token = token._replace(string=repr("lowrank." + value))
        tokens.append(token)
    result = tokenize.untokenize(tokens)
    tree = ast.parse(result)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module and node.module.split(".")[0] in PACKAGES:
            raise ValueError("unnamespaced import remains")
        if isinstance(node, ast.Import) and any(alias.name.split(".")[0] in PACKAGES for alias in node.names):
            raise ValueError("unnamespaced import remains")
    return result


def main():
    if DEST.exists():
        raise RuntimeError("namespaced candidate already exists; preserve or inspect before regenerating")
    (DEST / "lowrank").mkdir(parents=True)
    manifest = []
    for package in ("lowrank", *PACKAGES, "tests"):
        for path in sorted((ROOT / package).rglob("*.py")):
            relative = path.relative_to(ROOT)
            target = DEST / (relative if package in ("lowrank", "tests") else Path("lowrank") / relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            original = path.read_text()
            output = publication_docs(relative, namespace(original))
            target.write_text(output)
            manifest.append(dict(source=str(relative), target=str(target.relative_to(DEST)),
                source_sha256=hashlib.sha256(original.encode()).hexdigest(),
                exported_sha256=hashlib.sha256(output.encode()).hexdigest()))
    init = DEST / "lowrank/__init__.py"
    with init.open("a") as out:
        out.write("\n# Public entry point; all support packages live inside lowrank.\n"
            "from .splitters import split, SplitSpec, SplitResult, list_splitters\n"
            "__all__ += ['split', 'SplitSpec', 'SplitResult', 'list_splitters']\n")
    project = (ROOT / "pyproject.toml").read_text().replace(
        'include = ["lowrank*", "splitters*", "datasets*", "features*", "eval*"]', 'include = ["lowrank*"]')
    project = project.replace('Graph-free, linear-memory leakage-minimizing dataset splitting via Nyström factorization',
                              'Capacity-controlled similarity splitting via low-rank kernel factors')
    (DEST / "pyproject.toml").write_text(project)
    for name in ("LICENSE", "CHANGELOG.md"):
        shutil.copy2(ROOT / name, DEST / name)
    (DEST / "README.md").write_text('''# lowrank-split

Low-rank similarity splitting with specified fold sizes. Dense similarity is
approximated by Nyström factors; integer powers and optional capped groups
provide controls over global and local overlap. Lower similarity is not a
guarantee of a realistic deployment test.

```python
from lowrank import split, SplitSpec

result = split("lowrank", feature_data,
               SplitSpec([8, 1, 1], ["train", "val", "test"], seed=0),
               score_exact_leakage=False)
```

Default low-rank fold counts are exact; ratios rounding to an empty fold raise.
Set `allowed_size_imbalance` explicitly to allow a corridor. Group preservation
is conditional: caps, missing graph edges and capacity repair can split related
entities. Diagnostics retain `broken_groups` rather than claim hard exclusion.
Optional exact leakage diagnostics can dominate runtime and are disabled above.

Install with `pip install .`; `.[molecules]` adds RDKit and `.[ann]` adds the
approximate neighbour backend. PyTorch is required; it uses CUDA when visible
and CPU otherwise. The NN metric defaults to CPU and accepts an explicit device.

All support modules are under `lowrank.splitters`, `lowrank.datasets`,
`lowrank.features` and `lowrank.eval`. The wheel installs no top-level package
named `datasets`, `features`, `eval` or `splitters`.

Dataset loaders are optional research helpers. Public raw datasets and the
manuscript's frozen reproduction sources are distributed separately; this
package does not bundle input datasets or author publication identifiers.
''')
    # Include the final public-API append in the exported digest.
    for entry in manifest:
        entry["exported_sha256"] = hashlib.sha256((DEST / entry["target"]).read_bytes()).hexdigest()
    (DEST / "namespace_export_manifest.json").write_text(json.dumps(manifest, indent=2)+'\n')
    print("Exported", len(manifest), "Python files to", DEST)


if __name__ == "__main__":
    main()
