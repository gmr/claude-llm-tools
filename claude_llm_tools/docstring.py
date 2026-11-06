"""Parse Google style docstrings into tool descriptions"""

import dataclasses
import re

ARGUMENT_SECTIONS = frozenset({'Args', 'Arguments', 'Parameters'})
REMOVED_SECTIONS = ARGUMENT_SECTIONS | {
    'Raises',
    'Return',
    'Returns',
    'Yields',
}

ARGUMENT_PATTERN = re.compile(r'^\**(\w+)\s*(?:\([^)]*\))?:\s*(.*)$')
LIST_ITEM_PATTERN = re.compile(r'^([-*+]|\d+[.)])\s')
SECTION_PATTERN = re.compile(r'^(\w+):$')


@dataclasses.dataclass(frozen=True)
class Docstring:
    """A docstring split into its description and argument descriptions"""

    description: str | None
    arguments: dict[str, str]


def parse(docstring: str | None) -> Docstring:
    """Parse a cleaned docstring, as returned by ``inspect.getdoc``.

    The description is the docstring without its argument, return, yield,
    and raise sections, with lines wrapped within a paragraph joined.

    """
    if not docstring:
        return Docstring(description=None, arguments={})
    kept, sections = _split_sections(docstring.splitlines())
    arguments: dict[str, str] = {}
    for name in ARGUMENT_SECTIONS & sections.keys():
        arguments.update(_parse_arguments(sections[name]))
    description = '\n'.join(_unwrap(kept))
    description = re.sub(r'\n{3,}', '\n\n', description).strip()
    return Docstring(description=description or None, arguments=arguments)


def _unwrap(lines: list[str]) -> list[str]:
    """Join each wrapped line onto the line before it"""
    unwrapped: list[str] = []
    for line in lines:
        if unwrapped and _continues(unwrapped[-1], line):
            unwrapped[-1] = f'{unwrapped[-1]} {line.strip()}'
        else:
            unwrapped.append(line)
    return unwrapped


def _continues(previous: str, line: str) -> bool:
    """Whether a line continues the paragraph or list item before it.

    Blank lines, indented lines, list items, and section headers start
    their own line.

    """
    return not any(
        (
            _stands_alone(previous),
            _stands_alone(line),
            LIST_ITEM_PATTERN.match(line),
        )
    )


def _stands_alone(line: str) -> bool:
    """Whether a line is blank, indented, or a section header"""
    return (
        not line.strip()
        or line[0].isspace()
        or SECTION_PATTERN.match(line.rstrip()) is not None
    )


def _split_sections(
    lines: list[str],
) -> tuple[list[str], dict[str, list[str]]]:
    """Separate the removed sections from the lines that are kept"""
    kept: list[str] = []
    sections: dict[str, list[str]] = {}
    current: list[str] | None = None
    for line in lines:
        header = SECTION_PATTERN.match(line.rstrip())
        if header and header.group(1) in REMOVED_SECTIONS:
            current = sections.setdefault(header.group(1), [])
        elif current is not None and (not line.strip() or line[0].isspace()):
            current.append(line)
        else:
            current = None
            kept.append(line)
    return kept, sections


def _parse_arguments(lines: list[str]) -> dict[str, str]:
    """Map each argument name to its description, joining continued lines"""
    arguments: dict[str, list[str]] = {}
    entry_indent: int | None = None
    name: str | None = None
    for line in filter(str.strip, lines):
        indent = len(line) - len(line.lstrip())
        entry_indent = indent if entry_indent is None else entry_indent
        match = ARGUMENT_PATTERN.match(line.strip())
        if match and indent <= entry_indent:
            name = match.group(1)
            arguments[name] = [match.group(2)]
        elif name is not None:
            arguments[name].append(line.strip())
    return {
        key: ' '.join(filter(None, parts)) for key, parts in arguments.items()
    }
