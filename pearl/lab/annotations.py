class Param:
    def __init__(self, typ, editable: bool = True, range_start=None, range_end=None, isFilePath : bool | None = None,
                 choices=None, default=None, disc: str | None = None):
        self.typ = typ
        self.editable = editable
        self.range_start = range_start
        self.range_end = range_end
        self.choices = choices
        self.default = default
        self.isFilePath = isFilePath
        self.disc = disc

    def __str__(self):
        return f"Person(Type={self.typ}, editable={self.editable}, range=({self.range_start}, {self.range_end}), isFilePath={self.isFilePath}, choices={self.choices}, default={self.default}, disc={self.disc})"
