import json
import os
import subprocess


DEFAULT_KERNEL = "/Applications/Wolfram.app/Contents/MacOS/WolframKernel"


def cellular_automaton(expression, kernel=DEFAULT_KERNEL, timeout=30):
    if not os.path.exists(kernel):
        raise FileNotFoundError(kernel)

    code = f'WriteString[$Output, ExportString[CellularAutomaton{expression}, "JSON"]]; Quit[]'
    result = subprocess.run(
        [kernel, "-noprompt", "-run", code],
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, "LC_ALL": "C", "LANG": "C"},
    )
    return json.loads(result.stdout)
