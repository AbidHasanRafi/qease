"""analysis.__init__"""
from qease.analysis.calibration import an_calsize, an_conditional, an_crc_main, an_crc_resample, an_riskhead
from qease.analysis.common import prepare_analysis_state, run_block
from qease.analysis.dataset import an_hist, an_phase0, an_probe_stats, an_run_config
from qease.analysis.difficulty import an_budget, an_failures, an_qual
from qease.analysis.efficiency import an_compute_equiv, an_gen, an_latency, an_matched_compute
from qease.analysis.export import an_finalize
from qease.analysis.fidelity import an_dataset, an_fidelity


def run_analyze():
    """Stage 'analyze': every table and figure of the paper, exported to CSV / LaTeX / PDF."""
    prepare_analysis_state()
    for _name, _fn in [("run_config", an_run_config), ("probe_stats", an_probe_stats), ("phase0", an_phase0), ("training_curves", an_hist),
                       ("fidelity+significance+faithfulness", an_fidelity), ("per_dataset", an_dataset), ("crc_main+headline", an_crc_main),
                       ("crc_validity_resampled", an_crc_resample), ("calibration_size", an_calsize), ("conditional_validity", an_conditional),
                       ("risk_head", an_riskhead), ("budget_vs_difficulty", an_budget), ("qualitative", an_qual), ("failure_cases", an_failures),
                       ("generation_accuracy", an_gen), ("latency", an_latency), ("compute_equivalent", an_compute_equiv), ("compute_matched_vs_published", an_matched_compute), ("finalize", an_finalize)]:
        run_block(_name, _fn)
