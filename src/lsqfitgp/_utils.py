import textwrap

def append_to_docstring(docs, doctail):
    doctail = textwrap.dedent(doctail)
    dedocs = textwrap.dedent(docs)
    lineend = docs.find('\n')
    indented_lineend = dedocs.find('\n')
    indent = docs[:indented_lineend - lineend]
    return textwrap.indent(dedocs + doctail, indent)
