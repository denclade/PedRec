"""Comparison values (doc/experiments): protocol parsing, merging and tables."""
import math

from pedrec.tools.experiments.collect_results import parse_protocol, protocol_rows, merge_rows, comparison_table

V1_PROTOCOL = "doc/diss_experiment_protocols/experiment_pedrec_p2d3d_c_o_h36m_sim_mebow_protocol.md"


def test_parse_v1_protocol():
    results = parse_protocol(V1_PROTOCOL)
    last = results["R2E5"]
    assert last["H36M"]["MPJPE"] == 64.37 and last["COCO"]["PCK@0.05"] == 53.72
    assert last["H36M"]["PCJP"] == 0.8  # protocol column MRCJP
    assert math.isclose(last["COCO"]["Body phi deg"], math.degrees(0.18))  # radians in the protocol
    assert "Val Loss" not in last["COCO"] and "R1E1" in results


def test_rows_merge_and_compare():
    rows = protocol_rows(V1_PROTOCOL, "v1-diss", "v1", "mebow", "last", "note")
    assert all(row["checkpoint"] == "R2E5 (last)" for row in rows)
    updated = dict(rows[0], value="1")
    merged = merge_rows(rows, [updated])
    assert len(merged) == len(rows) and updated in merged
    table = comparison_table(merged, ["H36M MPJPE", "XYZ unknown"])
    assert "| H36M MPJPE | 64.37 |" in table and "XYZ" not in table
