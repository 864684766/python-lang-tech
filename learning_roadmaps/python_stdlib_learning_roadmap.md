# Python 标准库学习路线

装上解释器就有、要 `import`、不用 pip。json、路径、复制、环境、时间、日志。程序员基础。

**不需要 API Key。** 前置：[Python 类型](./python_basics_learning_roadmap.md)。建议先或同时看 [语言](./python_lang_learning_roadmap.md) 的 `with`（读文件那节会用到）。

## 怎么跑

在**仓库根目录**执行：

```bash
python python_stdlib/ps_run_all.py

python python_stdlib/ps_0_json.py
python python_stdlib/ps_1_pathlib.py
python python_stdlib/ps_2_glob.py
python python_stdlib/ps_3_copy.py
python python_stdlib/ps_4_os_sys.py
python python_stdlib/ps_5_datetime.py
python python_stdlib/ps_6_logging.py
```

## 顺序

每课文件头和运行开头都有「是什么 / 用在 / 不用」。先看总览再跑。

| 节 | 文件 | 是什么 | 什么时候用 | 不用 | 盯住的反例 |
|----|------|--------|------------|------|------------|
| ⓪ | `ps_0_json.py` | dict ↔ JSON 文本 | HTTP body、配置、工具参数、模型输出 | 对象只在本进程里用 | 单引号、尾逗号；`set` 不能直接 dumps |
| ① | `ps_1_pathlib.py` | 打开「已经知道名字」的那一份 | 读 `SKILL.md`、用 `__file__` 定位仓库根 | 不知道文件名、要按样子找一批 | 字符串 `+` 拼路径；文件不存在还读 |
| ② | `ps_2_glob.py` | 按 `*` / `**` 一次列出「这类文件」 | 扫全部 `pb_1*.py`；`7_1` 扫所有 `SKILL.md` | 已经知道「就这一份」 | 以为 `*` 会进子目录 |
| ③ | `ps_3_copy.py` | `=` / 浅拷贝 / `deepcopy` | 图 state / messages 要改副本 | 不可变的 str/int；用 `=` 当备份 | `dict.copy` 后改嵌套 `messages` |
| ④ | `ps_4_os_sys.py` | 环境变量、import 路径、命令行 | 读 API Key、`sys.path.insert`、`argv` | 拼文件路径（用 pathlib） | `os.environ["KEY"]` 没设就 KeyError |
| ⑤ | `ps_5_datetime.py` | 墙上几点、上海时区、秒表；`datetime` 和 `time` 各用各的 | UTC 存库、给用户看上海时间、TTL、打耗时 | async 里 `time.sleep`；`perf_counter` 当几点 | naive 和 aware 比较；`replace` 当换算 |
| ⑥ | `ps_6_logging.py` | 按级别打日志；handler 决定打到终端还是文件 | 服务关键路径；线上关 DEBUG；容器 stdout | 课堂演示用 `print_ok` 即可 | 只 `print`；先 f-string 再打日志 |

`python_basics/pb_11_array.py` 只点过 json/pathlib 的名字；`pb_5` 的 `.copy()` 只覆盖一维 list。本节把每天用的标准库跑透。

下一步：[FastAPI](./fastapi_learning_roadmap.md)。
