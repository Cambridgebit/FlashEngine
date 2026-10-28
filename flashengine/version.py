"""统一发布身份：不导入任何设备依赖，纯标准库。

`VERSION` 跟随主项目 `pyproject.toml` 的发布版本（本文件是运行期读取点，改版本时两处
一起改）；`UPSTREAM_VERSION` 记录本仓库合并的统一框架基线版本，便于追溯
`flashengine/runtime.py`、`scripts/framework.py` 这套入口的来源发布版。
"""

VERSION = "2.1.0"
SERVICE_NAME = "flashengine"
UPSTREAM_VERSION = "1.0.0"
