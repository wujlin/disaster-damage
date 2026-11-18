# Palu mapping ledger

This worktree records a Palu mapping **decision packet** for Jlin. It does not
freeze a Palu mapping, does not compute \(U\), and does not treat transfer
across hazards as shown.

The source of truth for Palu ground reference is the official Paulik et al.
(2019) 17-page PDF plus the publisher xlsx Buildings sheet under
`/mnt/data_hdd/datasets/paulik_palu_2018/`. Those raw files stay outside git.
The official grade system in the PDF is DL0-DL3, not the earlier DS1/DS2/DS3
candidate schemes. Those DS schemes must not be applied as the mapping.

Wind events continue to use the already-frozen StEER wind package in
[`eval_protocol.md`](eval_protocol.md). That protocol's Palu paragraph is left
unchanged in this worktree (it still says full text not obtained). File
obtainment is not a freeze.

## Files

- Ledger: [`configs/palu_crosswalk/ledger.yaml`](../configs/palu_crosswalk/ledger.yaml)
- Decision packet: [`palu_crosswalk_decision_packet.md`](palu_crosswalk_decision_packet.md)
- Validator and checks: `src/mvbridge/crosswalk/`
- Synthetic fixture generator: `src/mvbridge/crosswalk/fixtures.py`

## Validate

From the repository root, CPU only:

```bash
PYTHONPATH=src python -m mvbridge palu-crosswalk --packet-out docs/palu_crosswalk_decision_packet.md
PYTHONPATH=src python -m mvbridge palu-crosswalk --write-synthetic-fixture /tmp/synthetic_palu_ground.json
```

`--ground-table` may point at an authorized JSON table with `ground_records`
and `footprints`. Real Palu microdata must stay outside git. The command
refuses tables that include model predictions or \(U\) / conflict fields.

Paulik 2019 files are inventoried on WSA. Palu mapping remains **PENDING**
Jlin adjudication of official Y from the DL0-DL3 definitions. The packet can
be submitted to Jlin; it is not itself a freeze.
