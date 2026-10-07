import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts import export_parquet


def batches(*sizes):
    start = 0
    for n in sizes:
        yield pa.record_batch({"id": list(range(start, start + n))})
        start += n


def test_streams_batches_into_one_file(tmp_path):
    path = tmp_path / "t.parquet"
    assert export_parquet.write_parquet(batches(3, 4, 5), path) == 12
    assert pq.read_table(path).column("id").to_pylist() == list(range(12))
    assert not path.with_suffix(".parquet.tmp").exists()


def test_failed_export_leaves_no_file(tmp_path):
    def broken():
        yield from batches(3)
        raise ConnectionError("dropped")

    path = tmp_path / "t.parquet"
    with pytest.raises(ConnectionError):
        export_parquet.write_parquet(broken(), path)
    assert not path.exists()


def test_empty_result_is_an_error(tmp_path):
    with pytest.raises(RuntimeError, match="no rows"):
        export_parquet.write_parquet(iter([]), tmp_path / "t.parquet")


def test_count_and_export_read_the_same_snapshot():
    sql = export_parquet.snapshot_sql("p.scrm.t", "COUNT(*) AS n")
    assert sql == "SELECT COUNT(*) AS n FROM `p.scrm.t` FOR SYSTEM_TIME AS OF @as_of"
