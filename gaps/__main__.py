try:
    from gaps.cli import *
except:
    try:
        from .cli import *
    except:
        from cli import *

if __name__ == "__main__":
    cli()
