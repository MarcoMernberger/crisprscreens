import subprocess
from pathlib import Path
import types

import pytest

from crisprscreens.core import mageck
from crisprscreens.services import io


class FakeCompleted:
    def __init__(self, stdout='', stderr='', returncode=0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def test_mageck_pathway_calls_and_collects(tmp_path, monkeypatch):
    # create dummy gene ranking and gmt files
    gr = tmp_path / 'rank.rnk'
    gr.write_text('G1\t1\nG2\t2\n')
    gmt = tmp_path / 'pathways.gmt'
    gmt.write_text('P1\t\tG1\tG2\n')

    out = tmp_path / 'out'
    out.mkdir()

    # create a dummy output file that the function should detect
    (out / 'myprefix.somepathway.txt').write_text('dummy')

    def fake_run(cmd, capture_output, text, shell):
        return FakeCompleted(stdout='ok', stderr='')

    monkeypatch.setattr(subprocess, 'run', fake_run)

    res = mageck.mageck_pathway(
        gene_ranking=gr,
        gmt_file=gmt,
        out_dir=out,
        prefix='myprefix',
        method='gsea'
    )

    assert 'stdout' in res and 'stderr' in res
    assert any('myprefix' in p for p in res['outputs'])


def test_mageck_plot_calls_and_collects(tmp_path, monkeypatch):
    # create dummy gene and sgrna summaries
    g = tmp_path / 'gene.txt'
    g.write_text('Gene\nG1\n')
    s = tmp_path / 'sgrna.txt'
    s.write_text('sgRNA\n')

    out = tmp_path / 'out'
    out.mkdir()
    (out / 'plotmyprefix.png').write_text('img')

    def fake_run(cmd, capture_output, text, shell):
        return FakeCompleted(stdout='ok', stderr='')

    monkeypatch.setattr(subprocess, 'run', fake_run)

    res = mageck.mageck_plot(gene_summary=g, sgrna_summary=s, out_dir=out, prefix='plotmyprefix')
    assert 'stdout' in res and 'stderr' in res
    assert any('plotmyprefix' in p for p in res['outputs'])

    # service wrapper
    res2 = io.mageck_plot(gene_summary=g, sgrna_summary=s, output_dir=out, prefix='plotmyprefix_service')
    # since no files created with that prefix, outputs may be empty but function should return dict
    assert isinstance(res2, dict)
