import re

def methods(text, classname):
    # Preserve offsets while hiding comments and string/character literals.
    pattern = r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\''
    masked = re.sub(pattern, lambda m: ''.join('\n' if c=='\n' else ' ' for c in m.group()), text)
    found = {}
    for match in re.finditer(r'^[^\n;{}]*\b'+re.escape(classname)+r'::(~?\w+)\s*\(', masked, re.M):
        start = match.start()
        brace = masked.index('{', match.end())
        depth = 1
        end = brace+1
        while depth:
            depth += (masked[end]=='{') - (masked[end]=='}')
            end += 1
        name = match.group(1)
        if name in found: raise ValueError('Overloaded method requires explicit signature: '+name)
        found[name] = text[start:end]
    return found
