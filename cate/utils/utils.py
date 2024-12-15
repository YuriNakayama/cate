from __future__ import annotations

recursive_dict = dict[str, str] | dict[str, "recursive_dict"]


def dict_flatten(
    d: recursive_dict, parent_key: str = "", sep: str = "-"
) -> dict[str, str]:
    items: list[tuple[str, str]] = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(dict_flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
