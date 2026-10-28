"""统一命令入口：``python -m flashengine``（等价于 ``python -m scripts.framework``）。"""

from scripts.framework import main


if __name__ == "__main__":
    raise SystemExit(main())
