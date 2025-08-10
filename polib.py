import os
from typing import List

class POEntry:
    def __init__(self, msgid: str, msgstr: str = "", obsolete: bool = False):
        self.msgid = msgid
        self.msgstr = msgstr
        self.obsolete = obsolete

class POFile(list):
    def untranslated_entries(self) -> List[POEntry]:
        return [e for e in self if not e.msgstr]

    def translated_entries(self) -> List[POEntry]:
        return [e for e in self if e.msgstr]

    def fuzzy_entries(self) -> List[POEntry]:
        return []

    def save(self, path: str) -> None:
        with open(path, 'w', encoding='utf-8') as f:
            f.write('msgid ""\nmsgstr ""\n')
            for e in self:
                if e.msgid:
                    f.write(f"\nmsgid \"{e.msgid}\"\nmsgstr \"{e.msgstr}\"\n")


def pofile(data: str) -> POFile:
    if os.path.exists(data):
        with open(data, 'r', encoding='utf-8') as f:
            content = f.read()
    else:
        content = data

    entries: List[POEntry] = []
    msgid = None
    msgstr = ""
    for line in content.splitlines():
        line = line.strip()
        if line.startswith('msgid '):
            if msgid is not None:
                entries.append(POEntry(msgid.strip('"'), msgstr.strip('"')))
            msgid = line[6:].strip()
            msgstr = ""
        elif line.startswith('msgstr '):
            msgstr = line[7:].strip()
    if msgid is not None:
        entries.append(POEntry(msgid.strip('"'), msgstr.strip('"')))
    return POFile(entries)
