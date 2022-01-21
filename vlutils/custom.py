from rich.progress import Progress, TaskID, Task


class RichProgress(Progress):
    def get_task(self, taskID: TaskID) -> Task:
        with self._lock:
            return self._tasks[taskID]

    def __enter__(self) -> "RichProgress":
        self.start()
        return self
