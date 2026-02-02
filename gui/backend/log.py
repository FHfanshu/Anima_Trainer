"""
日志模块
参考 mikazuki/log.py
"""

import logging
import sys

try:
    from rich.console import Console
    from rich.logging import RichHandler
    from rich.theme import Theme

    console = Console(
        log_time=True,
        log_time_format='%H:%M:%S',
        theme=Theme({
            'traceback.border': 'black',
            'traceback.border.syntax_error': 'black',
            'inspect.value.border': 'black',
        }),
    )
    
    handler = RichHandler(
        show_time=True,
        omit_repeated_times=False,
        show_level=True,
        show_path=False,
        markup=False,
        rich_tracebacks=True,
        log_time_format='%H:%M:%S',
        level=logging.INFO,
        console=console,
    )
    
    log = logging.getLogger('anima-trainer')
    log.setLevel(logging.DEBUG)
    
    # 清除已有处理器
    while log.hasHandlers() and len(log.handlers) > 0:
        log.removeHandler(log.handlers[0])
    
    log.addHandler(handler)

except ModuleNotFoundError:
    # 回退到标准日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    log = logging.getLogger('anima-trainer')
