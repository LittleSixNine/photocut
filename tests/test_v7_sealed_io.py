import json
import threading

import pytest

from photocut.algorithms.v7.sealed_io import write_json_new_fsync


def test_exclusive_create_and_canonical_json(tmp_path):
    p = tmp_path / "x.json"
    write_json_new_fsync(p, {"b": 1, "a": 2})
    assert json.loads(p.read_text()) == {"a": 2, "b": 1}
    with pytest.raises(FileExistsError):
        write_json_new_fsync(p, {"other": 1})


def test_concurrent_writers_only_one_succeeds(tmp_path):
    p = tmp_path / "x.json"
    outcomes = []
    def run(i):
        try:
            write_json_new_fsync(p, {"i": i})
            outcomes.append(True)
        except FileExistsError:
            outcomes.append(False)
    threads = [threading.Thread(target=run, args=(i,)) for i in range(8)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert outcomes.count(True) == 1
