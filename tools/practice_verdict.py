"""Verdicts shared by the per-case correctness runners."""
import sys


class SubmissionError(Exception):
    pass


class Judge:
    def __init__(self, total):
        self.total = total
        self.passed = 0

    def run(self, number, evaluate, arguments, synchronize):
        try:
            result = evaluate(*arguments)
            synchronize()
            return result
        except Exception as error:
            raise SubmissionError(f"RE #{number} · {self.passed}/{self.total}\n{error}") from error

    def wrong(self, number, reason):
        raise SubmissionError(f"WA #{number} · {self.passed}/{self.total}\n{reason}")

    def accept(self, number):
        self.passed += 1
        print(f"AC #{number}", flush=True)

    def finish(self):
        print(f"AC · {self.passed}/{self.total} 用例通过", flush=True)


def fail(error):
    print(error, file=sys.stderr, flush=True)
    return 10
