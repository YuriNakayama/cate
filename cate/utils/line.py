from pathlib import Path

from linebot.v3.messaging import (
    ApiClient,
    Configuration,
    MessagingApi,
    PushMessageRequest,
    PushMessageResponse,
    TextMessage,
)


def get_env_variable(path: Path) -> dict[str, str]:
    with path.open() as f:
        return dict(line.strip().split("=", 1) for line in f)


# TODO: 環境変数の取得方法を変更, 環境変数としてtokenを設定する
LINE_ENV_PATH = Path("/workspace/env/tokens/line.env")
ENV = get_env_variable(LINE_ENV_PATH)
configuration = Configuration(access_token=ENV["CHANNEL_ACCESS_TOKEN"])


def send_messages(messages: list[str]) -> PushMessageResponse:
    with ApiClient(configuration) as client:
        messaging = MessagingApi(client)
        response = messaging.push_message(
            PushMessageRequest(
                to=ENV["USER_ID"],
                messages=[
                    TextMessage(text=message, quickReply=None, quoteToken=None)
                    for message in messages
                ],
                notificationDisabled=None,
                customAggregationUnits=None,
            )
        )
    return response
