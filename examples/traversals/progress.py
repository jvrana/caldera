import time
from typing import Union

from rich.progress import Progress


def safe_update(progress: Progress, every: Union[float, int] = 0.25):
    """Safely update a progress bar so that it displays properly in a jupyter
    notebook.

    :param progress:
    :param every: Refresh every nth second.
    :return:
    """
    last_refresh = {}

    def wrapped(task_id, *args, **kwargs):
        last_refresh.setdefault(task_id, time.time())
        dt = time.time() - last_refresh[task_id]
        if dt > every:
            last_refresh[task_id] = time.time()
            kwargs["refresh"] = True
        return progress.update(task_id, *args, **kwargs)

    return wrapped
