"""Conservative incremental text release for Hermes tool responses."""


class HermesTextStream:
    """Release ordinary text while withholding potential protocol markers.

    Once a marker starts, leave the rest for the existing complete parser.  This
    deliberately handles only the unambiguous prefix; tools are never executed
    from a partial token stream.
    """

    MARKERS = ("<tool_call>", "<function", "<think", "</think", "<|im_", "<｜｜tool")

    def __init__(self):
        self.pending = ""
        self.sent = ""
        self.blocked = False

    def feed(self, delta):
        if self.blocked:
            return ""
        self.pending += delta
        cut = len(self.pending)
        for marker in self.MARKERS:
            index = self.pending.find(marker)
            if index >= 0:
                cut = min(cut, index)
                self.blocked = True
            else:
                for size in range(1, min(len(marker) - 1, len(self.pending)) + 1):
                    if self.pending.endswith(marker[:size]):
                        cut = min(cut, len(self.pending) - size)
        safe = self.pending[:cut]
        if not self.sent:
            safe = safe.lstrip()
        # Complete parser strips outer whitespace; keep a potentially final tail.
        safe = safe.rstrip()
        if not safe:
            if self.blocked:
                self.pending = ""
            return ""
        # Retain the whitespace until it is known to be internal content.
        consumed = len(self.pending[:cut].rstrip())
        self.pending = "" if self.blocked else self.pending[consumed:]
        self.sent += safe
        return safe
