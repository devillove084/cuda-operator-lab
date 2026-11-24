"""Render the bundled LeetGPU statements for the terminal and Markdown pages."""
from html.parser import HTMLParser
from pathlib import Path
import re
import sys


class ProblemText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.preformatted = False
        self.list_types = []
        self.links = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag in ("h1", "h2", "h3", "h4"):
            self.parts.append("\n\n" + "#" * max(2, int(tag[1])) + " ")
        elif tag == "pre":
            self.preformatted = True
            self.parts.append("\n\n~~~text\n")
        elif tag in ("ul", "ol"):
            self.list_types.append(tag)
            self.parts.append("\n")
        elif tag == "li":
            bullet = "1. " if self.list_types and self.list_types[-1] == "ol" else "- "
            self.parts.append("\n" + "  " * max(0, len(self.list_types) - 1) + bullet)
        elif tag in ("p", "div", "table", "tr"):
            self.parts.append("\n")
        elif tag == "br":
            self.parts.append("\n")
        elif tag == "a":
            url = attributes.get("href")
            self.links.append(url)
            if url:
                self.parts.append("[")
        elif tag == "img":
            self.parts.append(f"\n\n![{attributes.get('alt', '图示')}]({attributes['src']})\n\n")
        elif tag == "code" and not self.preformatted:
            self.parts.append("\x60")
        elif tag == "strong":
            self.parts.append("**")
        elif tag == "em":
            self.parts.append("*")
        elif tag in ("sup", "sub"):
            self.parts.append("<" + tag + ">")
        elif tag in ("td", "th"):
            self.parts.append(" | ")

    def handle_endtag(self, tag):
        if tag == "pre":
            self.preformatted = False
            self.parts.append("\n~~~\n\n")
        elif tag == "a" and self.links:
            url = self.links.pop()
            if url:
                self.parts.append(f"]({url})")
        elif tag == "code" and not self.preformatted:
            self.parts.append("\x60")
        elif tag == "strong":
            self.parts.append("**")
        elif tag == "em":
            self.parts.append("*")
        elif tag in ("sup", "sub"):
            self.parts.append("</" + tag + ">")
        elif tag in ("ul", "ol"):
            if self.list_types:
                self.list_types.pop()
            self.parts.append("\n")
        elif tag in ("p", "div", "li", "table", "tr", "h1", "h2", "h3", "h4"):
            self.parts.append("\n")

    def handle_data(self, data):
        self.parts.append(data if self.preformatted else re.sub(r"\s+", " ", data))


def render_html(html):
    # Keep tabular problem data as HTML, which GitHub Markdown can display directly.
    tables = []

    def store(match):
        tables.append(match[0])
        return f"\nCUDA_OPERATOR_TABLE_{len(tables)}\n"

    source = re.sub(r"(?is)<table\b[^>]*>.*?</table>", store, html)
    document = ProblemText()
    document.feed(source)
    text = "".join(document.parts).strip()
    for index, table in enumerate(tables, start=1):
        text = text.replace(f"CUDA_OPERATOR_TABLE_{index}", "\n\n" + table + "\n\n")
    # Only trim excessive block separators. Code indentation remains in the fences.
    text = re.sub(r"\n[ \t]+\n", "\n\n", text)
    return re.sub(r"\n{3,}", "\n\n", text)


def render(path):
    html = Path(path).read_text()
    # The Markdown generator extracts diagrams. The terminal fallback keeps their text.
    return render_html(html)


if __name__ == "__main__":
    print(render(sys.argv[1]))
