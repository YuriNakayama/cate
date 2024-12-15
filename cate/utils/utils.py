from __future__ import annotations

from omegaconf import DictConfig, DictKeyType

recursive_dict = dict[str, str] | dict[str, "recursive_dict"] | DictConfig


def dict_flatten(
    d: recursive_dict, parent_key: DictKeyType | str = "", sep: str = "-"
) -> dict[str, str]:
    items: list[tuple[str, str]] = []
    for k, v in d.items():
        new_key = str(parent_key) + sep + str(k) if parent_key else k
        if isinstance(v, DictConfig):
            items.extend(dict_flatten(v, new_key, sep=sep).items())
        else:
            items.append((str(new_key), str(v)))
    return dict(items)
