from html.parser import HTMLParser
from pathlib import Path
import sys


class ProblemText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.preformatted = False

    def handle_starttag(self, tag, attrs):
        if tag in ("h1", "h2", "h3", "h4"):
            self.parts.append("\n\n" + "#" * int(tag[1]) + " ")
        elif tag == "pre":
            self.preformatted = True
            self.parts.append("\n\n~~~\n")
        elif tag == "li":
            self.parts.append("\n- ")
        elif tag in ("p", "div", "table", "tr"):
            self.parts.append("\n")
        elif tag == "br":
            self.parts.append("\n")
        elif tag == "code" and not self.preformatted:
            self.parts.append("\x60")
        elif tag in ("td", "th"):
            self.parts.append(" | ")

    def handle_endtag(self, tag):
        if tag == "pre":
            self.preformatted = False
            self.parts.append("\n~~~\n\n")
        elif tag == "code" and not self.preformatted:
            self.parts.append("\x60")
        elif tag in ("p", "div", "li", "ul", "ol", "table", "tr", "h1", "h2", "h3", "h4"):
            self.parts.append("\n")

    def handle_data(self, data):
        self.parts.append(data if self.preformatted else " ".join(data.splitlines()))



def render(path):
    document = ProblemText()
    document.feed(Path(path).read_text())
    return "".join(document.parts).strip()


if __name__ == "__main__":
    print(render(sys.argv[1]))
