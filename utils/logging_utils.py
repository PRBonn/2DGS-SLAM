import rich

_log_styles = {
    "SLAM": "bold green",
    "GUI": "bold magenta",
    "Eval": "bold red",
    "Loop": "dim",
    "Viser": "cyan",
}

def get_style(tag):
    if tag in _log_styles.keys():
        return _log_styles[tag]
    return "bold blue"


def Log(*args, tag="SLAM"):
    style = get_style(tag)
    rich.print(f"[{style}]{tag}:[/{style}]", *args)
