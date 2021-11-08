import time
import humanize

class LogTime:
    def __init__(self, verbose=True, **humanize_kwargs) -> None:
        if "minimum_unit" not in humanize_kwargs.keys():
            humanize_kwargs["minimum_unit"] = 'microseconds'
        self.humanize_kwargs = humanize_kwargs
        self.elapsed = None
        self.verbose = verbose

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        """
        Exceptions are captured in *args, we’ll handle none, since failing can be timed anyway
        """
        self.elapsed = time.time() - self.start
        self.elapsed_str = humanize.precisedelta(self.elapsed, **self.humanize_kwargs)
        if self.verbose:
            print(f"Time Elapsed: {self.elapsed_str}")