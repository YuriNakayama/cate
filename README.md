# CATE

## command

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env
uv sync --all-extras --dev
source .venv/bin/activate
ipython kernel install --user --name=cate
```

```bash
pytest 
```

## commit

[gitmoji.dev](https://gitmoji.dev/)

|         Emoji         | code                    | Mean                                    |
| :-------------------: | :---------------------- | :-------------------------------------- |
|      :sparkles:       | `:sparkles:`            | Introduce new features.                 |
|         :bug:         | `:bug:`                 | Fix a bug.                              |
|        :books:        | `:books:`               | Add or update documentation.            |
|         :art:         | `:art:`                 | Improve structure / format of the code. |
|       :recycle:       | `:recycle:`             | Refactor code.                          |
|         :zap:         | `:zap:`                 | Improve performance                     |
|  :white_check_mark:   | `:white_check_mark:`    | Add, update, or pass tests.             |
| :construction_worker: | `:construction_worker:` | Add or update CI build system.          |
|       :wrench:        | `:wrench:`              | Add or update configuration files.      |
|   :heavy_plus_sign:   | `:heavy_plus_sign:`     | Add a dependency.                       |
