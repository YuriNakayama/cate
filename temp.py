import click


@click.group()
def main() -> None:
    pass


@main.command()
@click.option("--name", default="default", help="The person to greet.")
def num_rank(name: str) -> None:
    print(f"Hello {name}, world!")


main()
