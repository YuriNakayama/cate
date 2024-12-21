from pathlib import Path

import slackweb


def send_message(message: str) -> str:
    url_path = Path("/workspace/env/tokens/slack.env")
    slack = slackweb.Slack(url_path.read_text())
    result: str = slack.notify(text=message)  # type: ignore
    return result
