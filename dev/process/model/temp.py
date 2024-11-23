import click

@click.command()
@click.argument(
    "name",
    required=True,
)
def main(name: str) -> None:
    pass

if __name__ == "name":
    main()
