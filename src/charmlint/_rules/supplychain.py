"""SUPPLYCHAIN rules — dependency and release hygiene.

A charm deploys more than its own source: everything it pulls in from
elsewhere, at build time or at deploy time. These rules check that the
charm records where each of those artefacts comes from and which
versions it accepts.
"""

import ast
import dataclasses
import functools
import pathlib
import re
from collections import Counter
from collections.abc import Iterator
from typing import Any

from .. import _ast, _dependencies, _yaml
from .. import _models as models
from ._base import Rule

_PEP508_RE = re.compile(
    r"""
    ^\s*
    (?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)   # the distribution name
    \s*
    (?:\[(?P<extras>[^\]]*)\])?            # optional [extras], e.g. ops[tracing]
    \s*
    (?P<rest>.*)                           # the version specifier and/or a PEP 508 marker
    """,
    re.VERBOSE | re.DOTALL,
)


@dataclasses.dataclass(frozen=True)
class _OpsDependency:
    """A charm's declared ``ops`` dependency, and where it was declared.

    Every rule about the ``ops`` dependency works from one of
    these, so the charm's packaging is parsed once and interrogated
    many times rather than each rule re-deriving the answer it happens
    to need. Keeping the specifier and the section rather than a
    collapsed verdict is what lets a later rule ask a question the
    current two do not — whether there is an upper bound, whether the
    pinned version is an LTS, whether the section it was found in is
    one a charm ought to be using.
    """

    #: The version specifier exactly as written, e.g. ``">=2.23,<4"``.
    #: Empty when the dependency is declared with no specifier at all.
    specifier: str
    #: Extras requested alongside it, e.g. ``("tracing",)``.
    extras: tuple[str, ...]
    #: The file it was declared in, relative to the charm directory.
    source: str
    #: Where within that file, as a dotted path for ``pyproject.toml``
    #: (``"project.dependencies"``) or the file name for a flat
    #: requirements file. Names a place a charm author can go and look.
    section: str
    #: 1-based line, or ``None`` for a ``pyproject.toml`` match —
    #: ``tomllib`` discards positions, so a finding there anchors to
    #: the file.
    line: int | None

    @property
    def is_unpinned(self) -> bool:
        """Whether it carries no version constraint at all."""
        return not self.specifier or self.specifier == "*"

    @property
    def is_exact(self) -> bool:
        """Whether it is pinned to a single version with ``==``."""
        return self.specifier.startswith("==")

    @property
    def is_test_only(self) -> bool:
        """Whether this declaration is the testing harness rather than the runtime.

        ``ops[testing]`` pulls in ``ops.testing``, which runs in the
        test environment and never ships in the charm. Neither pinning
        rule has anything useful to say about it: a test harness wants
        the version the tests were written against, not a range chosen
        so security fixes flow into production.
        """
        return "testing" in self.extras

    @property
    def where(self) -> str:
        """A phrase naming the section, for a diagnostic message.

        Empty for a flat requirements file, where the section *is* the
        file the diagnostic already points at, and naming it again
        would only pad the message. For ``pyproject.toml``, where
        there is no line to anchor to and a charm can declare ``ops``
        in any of half a dozen places, it is the only thing telling
        the author which one to go and edit.
        """
        if self.section == self.source:
            return ""
        return f" in `{self.section}`"


def _parse_pep508(
    entry: str, source: str, section: str, line: int | None = None
) -> _OpsDependency | None:
    """Parse a PEP 508 requirement string, returning ``None`` if it isn't ``ops``."""
    match = _PEP508_RE.match(entry)
    if match is None or _dependencies.normalise(match.group("name")) != "ops":
        return None
    # Drop any environment marker: `ops>=2.23; python_version < "3.12"`
    # constrains when the dependency applies, not which versions satisfy it.
    specifier = match.group("rest").split(";", 1)[0].strip()
    return _OpsDependency(
        specifier=specifier,
        extras=_split_extras(match.group("extras")),
        source=source,
        section=section,
        line=line,
    )


def _parse_poetry(value: Any, source: str, section: str) -> _OpsDependency:
    """Parse a Poetry dependency value (bare string, or table with ``version``)."""
    if isinstance(value, dict):
        specifier = str(value.get("version", ""))
        extras = value.get("extras")
        extras = tuple(str(e) for e in extras) if isinstance(extras, list) else ()
    else:
        specifier = str(value)
        extras = ()
    return _OpsDependency(
        specifier=specifier.strip(),
        extras=extras,
        source=source,
        section=section,
        line=None,
    )


def _split_extras(extras: str | None) -> tuple[str, ...]:
    """Split the bracketed extras of a PEP 508 requirement into a tuple."""
    if not extras:
        return ()
    stripped = (part.strip() for part in extras.split(","))
    return tuple(part for part in stripped if part)


def _find_ops_in_pyproject(data: dict[str, Any]) -> _OpsDependency | None:
    """Look for an ``ops`` dependency across the common pyproject.toml layouts.

    *data* is the charm's parsed ``pyproject.toml``, read once by the
    linter core. ``None`` means no ``ops`` dependency appears in any
    known location.
    """
    source = "pyproject.toml"
    for declaration in _dependencies.pyproject_declarations(data):
        if declaration.name != "ops":
            continue
        if declaration.requirement is not None:
            dep = _parse_pep508(declaration.requirement, source, declaration.section)
        else:
            dep = _parse_poetry(declaration.poetry_value, source, declaration.section)
        if dep is not None and not dep.is_test_only:
            return dep
    return None


def _find_ops_in_requirements(
    requirements: pathlib.Path, name: str = "requirements.txt"
) -> _OpsDependency | None:
    """Look for an ``ops`` line in a ``requirements.txt``-style file.

    *name* is what the diagnostic calls the file. It defaults to
    ``requirements.txt`` because that is the file almost every charm
    uses, but the ``python`` plugin lets a charm name its requirements
    files itself, and a finding should point at the file the charm
    actually declared rather than at one it does not have.
    """
    try:
        text = requirements.read_text()
    except OSError:
        return None
    for lineno, requirement in _dependencies.requirement_lines(text):
        dep = _parse_pep508(requirement, name, name, lineno)
        if dep is not None and not dep.is_test_only:
            return dep
    return None


#: charmcraft plugins that resolve dependencies from ``pyproject.toml``
#: plus a lock file. Any ``requirements.txt`` alongside one of these is a
#: generated artefact, not a hand-written declaration, so the pinning
#: rules must not read it.
#:
#: This is a narrow guard, and deliberately so. Across the corpus it
#: changes the outcome for two charms — the ``kubernetes`` and ``machine``
#: charms of ``opensearch-dashboards-operator``, which export a
#: ``requirements.txt`` from ``poetry.lock`` (``--only main``) purely so
#: the pins are visible inside the built ``.charm``. That export pins
#: ``ops`` exactly, which is what a lock file is for; without this guard
#: SUPPLYCHAIN-006 would flag it.
_LOCKFILE_PLUGINS = frozenset({"poetry", "uv"})

#: Every charmcraft plugin that builds the charm's Python environment,
#: and so decides where the charm's dependencies are declared. The rest
#: of the plugins a charm uses — ``nil``, ``dump``, ``reactive`` — put
#: files in the payload without resolving anything, and say nothing
#: about where ``ops`` is declared.
_DEPENDENCY_PLUGINS = _LOCKFILE_PLUGINS | frozenset({"charm", "python"})


def _dependency_part(metadata: models.Yaml) -> models.Yaml | None:
    """Return the ``parts`` entry that builds the charm's Python environment.

    A charm has at most one. Across the corpus, 329 of 655 charms
    declare exactly one part using a dependency-resolving plugin and
    **none declares two**; the sets are disjoint as well as unique
    (``uv`` 198, ``poetry`` 79, ``charm`` 52). The remaining 326 either
    declare ``parts`` using only ``nil``, ``dump`` or ``reactive``, or
    declare no ``parts`` at all — so "no dependency-resolving part" is
    the common case, not an error, and the caller falls back to the
    default ``requirements.txt`` for it.
    """
    for _, part in metadata.get("parts").items():
        plugin = part.get("plugin").value
        if isinstance(plugin, str) and plugin in _DEPENDENCY_PLUGINS:
            return part
    return None


def _requirements_files(context: models.CharmContext) -> tuple[tuple[pathlib.Path, str], ...]:
    """Return the requirements files the charm's build plugin actually reads.

    Each entry is the path to read and the name to call it in a
    diagnostic. An empty tuple means the plugin resolves dependencies
    from ``pyproject.toml`` and a lock file, so there is no hand-written
    requirements file to read at all.
    """
    part = _dependency_part(context.metadata)
    default = ((context.charm_dir / "requirements.txt", "requirements.txt"),)
    if part is None:
        return default
    plugin = part.get("plugin").value
    if plugin in _LOCKFILE_PLUGINS:
        return ()
    if plugin == "python":
        # The python plugin names its own requirements files, and a
        # charm using it need not call them requirements.txt.
        declared = tuple(
            (context.charm_dir / str(entry.value), str(entry.value))
            for entry in part.get("python-requirements").elements or ()
            if entry.value
        )
        if declared:
            return declared
    return default


@functools.cache
def _find_ops_requirements(
    files: tuple[tuple[pathlib.Path, str], ...],
) -> _OpsDependency | None:
    """Scan the charm's requirements files once, for both pinning rules."""
    for path, name in files:
        if not path.is_file():
            continue
        dep = _find_ops_in_requirements(path, name)
        if dep is not None:
            return dep
    return None


def _find_ops_dep(context: models.CharmContext) -> _OpsDependency | None:
    """Return the charm's runtime ``ops`` dependency, or ``None``.

    Both pinning rules share this scan and stop at the first hit that
    is not the test harness. Declaring ``ops`` more than once is the
    norm rather than the exception — 245 of the 417 corpus charms that
    declare it at all do so twice or more, almost always runtime
    ``ops`` alongside ``ops[testing]`` in a test group — so
    :attr:`_OpsDependency.is_test_only` declarations are stepped over
    rather than returned. ``pyproject.toml`` wins over the requirements
    files when both are present, and which requirements files those are
    is decided by the charm's build plugin — see
    :func:`_requirements_files`.
    """
    if context.pyproject is not None:
        dep = _find_ops_in_pyproject(context.pyproject)
        if dep is not None:
            return dep

    return _find_ops_requirements(_requirements_files(context))


class OciImageMissingUpstreamSource(Rule):
    """Flag an ``oci-image`` resource with no ``upstream-source``."""

    category = "SUPPLYCHAIN"
    number = 1
    name = "oci-image-missing-upstream-source"
    description = "oci-image resource declared without an 'upstream-source'"
    default_severity = models.Severity.INFO

    def check(self, context: models.CharmContext) -> list[models.Diagnostic]:
        candidates = [
            (name, resource)
            for name, resource in context.metadata.get("resources").items()
            if resource.get("type").value == "oci-image" and not resource.get("upstream-source")
        ]
        if not candidates:
            return []

        # `upstream-source` names the image the resource is built from, so
        # it has nothing to say about an image this repository builds
        # itself: for those the build recipe is the record of provenance.
        local = _locally_built_images(context.charm_dir)
        diagnostics: list[models.Diagnostic] = []
        for name, resource in candidates:
            stem = _normalise_image_name(name)
            for suffix in _IMAGE_SUFFIXES:
                stem = stem.removesuffix(suffix)
            if stem in local or _normalise_image_name(name) in local:
                continue
            diagnostics.append(
                self.diagnostic(
                    f"OCI image resource '{name}' has no 'upstream-source', so there is "
                    "no record of which image it is built from",
                    # A split-metadata charm can declare `resources` in
                    # metadata.yaml, where the diagnostic — and any noqa
                    # silencing it — then belongs.
                    path=resource.source,
                    line=resource.line,
                    fix_hint=(
                        f"Add 'upstream-source' under resources.{name}, naming the image "
                        "the resource is built from (e.g. 'ghcr.io/canonical/foo:1.2.3')"
                    ),
                )
            )
        return diagnostics


class OpsDependencyUnpinned(Rule):
    """Flag an ``ops`` dependency with no version specifier."""

    category = "SUPPLYCHAIN"
    number = 5
    name = "ops-dependency-unpinned"
    description = "ops dependency has no version specifier"
    default_severity = models.Severity.WARNING

    def check(self, context: models.CharmContext) -> list[models.Diagnostic]:
        dep = _find_ops_dep(context)
        if dep is None or not dep.is_unpinned:
            return []
        return [
            self.diagnostic(
                f"ops dependency has no version specifier{dep.where} — "
                "charms should pin a supported range so dependency "
                "resolvers do not silently pull a major bump",
                path=dep.source,
                line=dep.line,
                fix_hint="Add a version range, e.g. `ops>=2.23,<4`",
            )
        ]


class OpsDependencyExactlyPinned(Rule):
    """Flag an ``ops`` dependency pinned with ``==``."""

    category = "SUPPLYCHAIN"
    number = 6
    name = "ops-dependency-exactly-pinned"
    description = "ops dependency pinned with `==`"
    default_severity = models.Severity.INFO

    def check(self, context: models.CharmContext) -> list[models.Diagnostic]:
        dep = _find_ops_dep(context)
        if dep is None or not dep.is_exact:
            return []
        return [
            self.diagnostic(
                f"ops dependency is exactly pinned (`==`){dep.where} — "
                "prefer a version range so security fixes flow in "
                "without a manual bump",
                path=dep.source,
                line=dep.line,
                fix_hint="Replace the `==` pin with a range, e.g. `ops>=2.23,<4`",
            )
        ]


class SnapInstallUnpinned(Rule):
    """Flag snap-library ``add`` and ``ensure`` calls without a revision.

    A requested revision does not prevent later automatic refreshes.
    """

    category = "SUPPLYCHAIN"
    number = 7
    name = "snap-install-unpinned"
    description = "snap installation does not specify a revision"
    default_severity = models.Severity.WARNING
    reference_url = (
        "https://canonical.com/juju/docs/ops/latest/howto/run-workloads-with-a-charm-machines/"
    )

    def check(self, context: models.CharmContext) -> list[models.Diagnostic]:
        diagnostics: list[models.Diagnostic] = []
        for module in context.charm_sources():
            imports = _ast.Imports.of(module)
            for operation in _snap_library_calls(module, imports):
                call = operation.call
                if operation.hold or not _missing_snap_revision(call):
                    continue
                diagnostics.append(
                    self.diagnostic(
                        "snap installation does not specify a revision",
                        path=module.path,
                        line=call.lineno,
                        fix_hint="Specify `revision=` using a snap library version that supports it",
                    )
                )
        return diagnostics


class SnapRevisionWithoutHold(Rule):
    """Flag revision-selected snaps that may auto-refresh without a hold.

    Recognizes snap-library holds in charm-owned Python source. Ambiguous
    hold targets are left alone; external refresh policies are not inferred.
    """

    category = "SUPPLYCHAIN"
    number = 8
    name = "snap-revision-without-hold"
    description = "snap revision selected without a refresh hold"
    default_severity = models.Severity.WARNING
    reference_url = (
        "https://canonical.com/juju/docs/ops/latest/howto/run-workloads-with-a-charm-machines/"
    )

    def check(self, context: models.CharmContext) -> list[models.Diagnostic]:
        installs: list[tuple[models.Module, ast.Call, str | None]] = []
        holds: set[str | None] = set()
        for module in context.charm_sources():
            imports = _ast.Imports.of(module)
            for operation in _snap_library_calls(module, imports):
                call = operation.call
                if operation.hold:
                    holds.add(operation.name)
                elif not _missing_snap_revision(call):
                    installs.append((module, call, operation.name))
            for call in module.walk(ast.Call):
                target = _ast.call_target(call, imports)
                library, _, method = (target or "").rpartition(".")
                if _SNAP_LIBRARY.fullmatch(library) and method == "hold_refresh":
                    days = call.args[0] if call.args else _ast.keyword(call, "days")
                    forever = call.args[1] if len(call.args) > 1 else _ast.keyword(call, "forever")
                    removes_hold = (
                        isinstance(days, ast.Constant)
                        and days.value == 0
                        and (
                            forever is None
                            or (isinstance(forever, ast.Constant) and forever.value is False)
                        )
                    )
                    if not removes_hold:
                        holds.add(None)
                elif library == "charmlibs.snap" and method in {
                    "install",
                    "refresh",
                    "ensure_installed",
                }:
                    if not _missing_snap_revision(call):
                        name = call.args[0] if call.args else _ast.keyword(call, "snap")
                        installs.append((module, call, _literal_snap_name(name)))
        diagnostics: list[models.Diagnostic] = []
        for module, call, name in installs:
            # An unresolved hold target might cover this snap.
            if holds and (name is None or None in holds or name in holds):
                continue
            diagnostics.append(
                self.diagnostic(
                    "snap revision may change on automatic refresh; no refresh hold found",
                    path=module.path,
                    line=call.lineno,
                    fix_hint="Hold automatic refreshes after installation using the snap library",
                )
            )
        return diagnostics


_IMAGE_SUFFIXES = ("-image", "_image")
_SNAP_LIBRARY = re.compile(r"^(?:charmlibs|charms\.operator_libs_linux\.v\d+)\.snap$")


@dataclasses.dataclass(frozen=True)
class _SnapCall:
    call: ast.Call
    name: str | None
    hold: bool = False


def _snap_library_calls(module: models.Module, imports: _ast.Imports) -> Iterator[_SnapCall]:
    """Find installs and holds, with literal snap names where available."""
    scopes = (
        module.tree,
        *module.walk(ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda),
    )
    for scope in scopes:
        caches: set[str] = set()
        snaps: dict[str, ast.expr] = {}
        nodes = sorted(
            _scope_nodes(scope),
            key=lambda node: (getattr(node, "lineno", 0), getattr(node, "col_offset", 0)),
        )
        bindings = Counter(
            node.id
            for node in nodes
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store | ast.Del)
        )
        for node in nodes:
            if isinstance(node, ast.Assign | ast.AnnAssign):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                value = node.value
                is_cache = value is not None and _snap_cache(value, imports)
                is_snap = value is not None and _snap_from_cache(value, caches, imports)
                for target in targets:
                    if not isinstance(target, ast.Name) or bindings[target.id] != 1:
                        continue
                    name = target.id
                    if is_cache:
                        caches.add(name)
                    if is_snap and isinstance(value, ast.Subscript):
                        snaps[name] = value.slice
            elif isinstance(node, ast.Call):
                operation = _snap_library_operation(node, imports, caches, snaps)
                if operation is not None:
                    yield operation


def _literal_snap_name(node: ast.expr | None) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _scope_nodes(scope: ast.AST) -> Iterator[ast.AST]:
    """Walk one lexical scope without entering nested scopes."""
    for child in ast.iter_child_nodes(scope):
        if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.Lambda):
            continue
        yield child
        yield from _scope_nodes(child)


def _snap_cache(node: ast.AST, imports: _ast.Imports) -> bool:
    if not isinstance(node, ast.Call):
        return False
    target = _ast.call_target(node, imports)
    if target is None:
        return False
    module, _, name = target.rpartition(".")
    return name == "SnapCache" and bool(_SNAP_LIBRARY.fullmatch(module))


def _snap_from_cache(node: ast.AST, caches: set[str], imports: _ast.Imports) -> bool:
    return isinstance(node, ast.Subscript) and (
        _snap_cache(node.value, imports) or _ast.dotted_name(node.value) in caches
    )


def _snap_library_operation(
    call: ast.Call, imports: _ast.Imports, caches: set[str], snaps: dict[str, ast.expr]
) -> _SnapCall | None:
    target = _ast.call_target(call, imports)
    function_hold = target == "charmlibs.snap.hold"
    hold = function_hold or (isinstance(call.func, ast.Attribute) and call.func.attr == "hold")
    if hold:
        duration_index = 1 if function_hold else 0
        duration = (
            call.args[duration_index]
            if len(call.args) > duration_index
            else _ast.keyword(call, "duration")
        )
        if (
            isinstance(duration, ast.Constant)
            and isinstance(duration.value, int | float)
            and duration.value <= 0
        ):
            return None
    elif not _snap_library_install(call, imports, caches, snaps):
        return None
    receiver = call.func.value if isinstance(call.func, ast.Attribute) else None
    if isinstance(receiver, ast.Subscript) and _snap_from_cache(receiver, caches, imports):
        name = receiver.slice
    elif isinstance(receiver, ast.Name) and receiver.id in snaps:
        name = snaps[receiver.id]
    elif not hold or function_hold:
        name = call.args[0] if call.args else _ast.keyword(call, "snap_names")
        if name is None and function_hold:
            name = _ast.keyword(call, "snap")
    else:
        name = None
    return _SnapCall(call, _literal_snap_name(name), hold=hold)


def _snap_library_install(
    call: ast.Call, imports: _ast.Imports, caches: set[str], snaps: dict[str, ast.expr]
) -> bool:
    """Recognize install-capable calls with unambiguous arguments."""
    if any(keyword.arg is None for keyword in call.keywords) or any(
        isinstance(argument, ast.Starred) for argument in call.args
    ):
        return False
    target = _ast.call_target(call, imports)
    module, _, method = (target or "").rpartition(".")
    if _SNAP_LIBRARY.fullmatch(module) and method in {"add", "ensure"}:
        state_index = 1
        accepts_string_state = True
        accepts_enum_state = method == "add"
        max_unambiguous_args = 5
    elif (
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "ensure"
        and (
            _ast.dotted_name(call.func.value) in snaps
            or _snap_from_cache(call.func.value, caches, imports)
        )
    ):
        method = "ensure"
        state_index = 0
        accepts_string_state = False
        accepts_enum_state = True
        max_unambiguous_args = 4
    else:
        return False
    # Older v2 libraries place the revision before newer versions do.
    if len(call.args) > max_unambiguous_args:
        return False
    state = call.args[state_index] if len(call.args) > state_index else _ast.keyword(call, "state")
    if state is None:
        if method != "add":
            return False
    elif not _install_state(
        state,
        imports,
        strings=accepts_string_state,
        enums=accepts_enum_state,
    ):
        return False
    return True


def _missing_snap_revision(call: ast.Call) -> bool:
    revision = _ast.keyword(call, "revision")
    return revision is None or (
        isinstance(revision, ast.Constant) and revision.value in (None, "", 0)
    )


def _install_state(state: ast.expr, imports: _ast.Imports, *, strings: bool, enums: bool) -> bool:
    if isinstance(state, ast.Constant):
        return strings and state.value in ("present", "latest")
    if not enums:
        return False
    target = imports.resolve(state)
    module, _, value = (target or "").rpartition(".SnapState.")
    return value in {"Present", "Latest"} and bool(_SNAP_LIBRARY.fullmatch(module))


def _normalise_image_name(name: object) -> str:
    """Fold a rock, directory, or resource name to one comparable form."""
    return str(name).replace("_", "-").lower()


def _locally_built_images(charm_dir: pathlib.Path) -> set[str]:
    """Return the names of images this repository builds for itself.

    A rock or Dockerfile in the charm directory, or up to two levels
    below it, names an image the charm's own CI builds and uploads. So
    does one up to two levels below the repository root, since in a
    monorepo the rock is often a sibling of the charm (``app/charm/``
    beside ``app/rockcraft.yaml``) or sits at the top of the repository.
    The candidate names are the rock's declared ``name`` and the
    directory holding the build recipe, since the ``foo_rock/``,
    ``rock/`` and ``foo_rocks/<component>/`` layouts are all common.
    """
    charm_dir = charm_dir.resolve()
    names: set[str] = set()
    for root in {charm_dir, _repo_root(charm_dir)}:
        for depth in ("", "*/", "*/*/"):
            for rockcraft in root.glob(f"{depth}rockcraft.yaml"):
                try:
                    declared = _yaml.load(rockcraft).get("name")
                except _yaml.FileLoadError:
                    declared = models.Yaml.absent(rockcraft.name)
                if isinstance(declared.value, str):
                    names.add(_normalise_image_name(declared.value))
                names.add(
                    _normalise_image_name(
                        rockcraft.parent.name.removesuffix("_rock").removesuffix("-rock")
                    )
                )
            for dockerfile in root.glob(f"{depth}Dockerfile"):
                names.add(_normalise_image_name(dockerfile.parent.name))
    return names


def _repo_root(charm_dir: pathlib.Path) -> pathlib.Path:
    """Return the root of the git repository holding the charm.

    Falls back to the charm directory itself when it is not in a git
    repository, so the search never wanders further than it would have.
    """
    for directory in (charm_dir, *charm_dir.parents):
        if (directory / ".git").exists():
            return directory
    return charm_dir
