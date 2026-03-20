"""
PyPipeGraph2 job wrappers for MAGeCK method comparison.
"""

from pypipegraph2 import (
    Job,
    MultiFileGeneratingJob,
    FunctionInvariant,
    ParameterInvariant,
)
from pandas import DataFrame
from pathlib import Path
from typing import List, Union, Optional, Dict, Callable
from crisprscreens.core.method_comparison import (
    leave_one_replicate_out_analysis,
    analyze_sgrna_coherence,
    analyze_control_false_positives,
    permutation_test_analysis,
    compare_mageck_methods,
    compare_rankings_simple,
)
from crisprscreens.core.mageck import mageck_test, mageck_mle


def leave_one_replicate_out_job(
    count_table: Union[Path, str],
    control_ids: List[str],
    treatment_ids: List[str],
    output_dir: Union[Path, str],
    prefix: str,
    run_mageck_func: Callable,
    method_params: Dict = {},
    top_n_list: List[int] = [50, 100, 200],
    gene_col: str = "id",
    dependencies: List[Job] = [],
) -> MultiFileGeneratingJob:
    """
    Create PyPipeGraph job for leave-one-replicate-out analysis.

    This job performs leave-one-replicate-out consistency analysis to assess
    the stability and reproducibility of a MAGeCK analysis method.

    Parameters
    ----------
    count_table : Path or str
        Path to MAGeCK count table
    control_ids : List[str]
        List of control sample IDs (e.g., ['Total_Rep1', 'Total_Rep2', 'Total_Rep3'])
    treatment_ids : List[str]
        List of treatment sample IDs (e.g., ['Sort_Rep1', 'Sort_Rep2', 'Sort_Rep3'])
    output_dir : Path or str
        Directory to save results
    prefix : str
        Prefix for output files and run name
    run_mageck_func : Callable
        Function to run MAGeCK analysis. Should accept:
        - count_table, control_ids, treatment_ids, out_dir, prefix, **method_params
    method_params : Dict
        Additional parameters for MAGeCK run (e.g., {'paired': True, 'norm_method': 'median'})
    top_n_list : List[int]
        List of top-N values to test for overlap/Jaccard
    gene_col : str
        Gene identifier column name
    dependencies : List[Job]
        Job dependencies

    Returns
    -------
    MultiFileGeneratingJob
        Job that creates leave-one-out comparison results

    Examples
    --------
    >>> from crisprscreens.core.mageck import mageck_test
    >>> job = leave_one_replicate_out_job(
    ...     count_table="results/counts.txt",
    ...     control_ids=["Total_Rep1", "Total_Rep2", "Total_Rep3"],
    ...     treatment_ids=["Sort_Rep1", "Sort_Rep2", "Sort_Rep3"],
    ...     output_dir="results/comparison/RRA_leave_one_out",
    ...     prefix="RRA_paired",
    ...     run_mageck_func=mageck_test,
    ...     method_params={'paired': True, 'norm_method': 'median'},
    ... )
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define output files
    output_files = [
        output_dir / f"{prefix}_leave_one_out_comparison.tsv",
    ]

    # Add per-run outputs
    n_replicates = min(len(control_ids), len(treatment_ids))
    for i in range(n_replicates):
        run_name = f"{prefix}_leave_out_rep{i+1}"
        output_files.append(
            output_dir / f"leave_out_rep{i+1}" / f"{run_name}.gene_summary.tsv"
        )

    def __run(output_files):
        leave_one_replicate_out_analysis(
            count_table=count_table,
            control_ids=control_ids,
            treatment_ids=treatment_ids,
            output_dir=output_dir,
            prefix=prefix,
            run_mageck_func=run_mageck_func,
            method_params=method_params,
            top_n_list=top_n_list,
            gene_col=gene_col,
        )

    job = MultiFileGeneratingJob(output_files, __run).depends_on(dependencies)

    job.depends_on(
        FunctionInvariant(leave_one_replicate_out_analysis),
        ParameterInvariant(
            (
                count_table,
                control_ids,
                treatment_ids,
                str(output_dir),
                prefix,
                method_params,
                top_n_list,
                gene_col,
            )
        ),
    )

    return job


def sgrna_coherence_job(
    gene_summary: Union[Path, str],
    sgrna_summary: Union[Path, str],
    output_file: Union[Path, str],
    top_n: int = 100,
    gene_col: str = "Gene",
    sgrna_gene_col: str = "Gene",
    sgrna_col: str = "sgrna",
    dependencies: List[Job] = [],
) -> Job:
    """
    Create PyPipeGraph job for sgRNA coherence analysis.

    Analyzes how consistently multiple sgRNAs targeting the same gene
    show similar effects (direction and magnitude).

    Parameters
    ----------
    gene_summary : Path or str
        MAGeCK gene summary file
    sgrna_summary : Path or str
        MAGeCK sgRNA summary file
    output_file : Path or str
        Output file for coherence results
    top_n : int
        Number of top genes to analyze
    gene_col : str
        Gene column name in gene summary
    sgrna_gene_col : str
        Gene column name in sgRNA summary
    sgrna_col : str
        sgRNA identifier column
    dependencies : List[Job]
        Job dependencies

    Returns
    -------
    Job
        FileGeneratingJob that creates coherence analysis

    Examples
    --------
    >>> job = sgrna_coherence_job(
    ...     gene_summary="results/mle/gene_summary.tsv",
    ...     sgrna_summary="results/mle/sgrna_summary.tsv",
    ...     output_file="results/comparison/MLE_sgrna_coherence.tsv",
    ...     top_n=200,
    ... )
    """
    from pypipegraph2 import FileGeneratingJob

    def __run(output_file):
        coherence_df = analyze_sgrna_coherence(
            gene_summary=gene_summary,
            sgrna_summary=sgrna_summary,
            top_n=top_n,
            gene_col=gene_col,
            sgrna_gene_col=sgrna_gene_col,
            sgrna_col=sgrna_col,
        )
        coherence_df.to_csv(output_file, sep="\t", index=False)

    job = FileGeneratingJob(output_file, __run).depends_on(dependencies)

    job.depends_on(
        FunctionInvariant(analyze_sgrna_coherence),
        ParameterInvariant(
            (
                gene_summary,
                sgrna_summary,
                top_n,
                gene_col,
                sgrna_gene_col,
                sgrna_col,
            )
        ),
    )

    return job


def control_false_positive_job(
    gene_summary: Union[Path, str],
    control_sgrnas: Union[Path, str],
    output_file: Union[Path, str],
    top_n_list: List[int] = [50, 100, 200, 500],
    gene_col: str = "id",
    control_prefix: str = "Non-Targeting",
    dependencies: List[Job] = [],
) -> Job:
    """
    Create PyPipeGraph job for control sgRNA false-positive analysis.

    Checks how many control sgRNAs (non-targeting) appear in the top-N
    significant hits, which indicates false-positive rate.

    Parameters
    ----------
    gene_summary : Path or str
        MAGeCK gene summary file
    control_sgrnas : Path or str
        File containing control sgRNA identifiers
    output_file : Path or str
        Output file for FP analysis
    top_n_list : List[int]
        List of top-N values to check
    gene_col : str
        Gene identifier column name
    control_prefix : str
        Prefix used for control genes (e.g., "Non-Targeting")
    dependencies : List[Job]
        Job dependencies

    Returns
    -------
    Job
        FileGeneratingJob that creates FP analysis

    Examples
    --------
    >>> job = control_false_positive_job(
    ...     gene_summary="results/rra/gene_summary.tsv",
    ...     control_sgrnas="cache/input/control_sgrnas.txt",
    ...     output_file="results/comparison/RRA_control_fp.tsv",
    ... )
    """
    from pypipegraph2 import FileGeneratingJob

    def __run(output_file):
        fp_df = analyze_control_false_positives(
            gene_summary=gene_summary,
            control_sgrnas=control_sgrnas,
            top_n_list=top_n_list,
            gene_col=gene_col,
            control_prefix=control_prefix,
        )
        fp_df.to_csv(output_file, sep="\t", index=False)

    job = FileGeneratingJob(output_file, __run).depends_on(dependencies)

    job.depends_on(
        FunctionInvariant(analyze_control_false_positives),
        ParameterInvariant(
            (
                gene_summary,
                control_sgrnas,
                tuple(top_n_list),
                gene_col,
                control_prefix,
            )
        ),
    )

    return job


def permutation_test_job(
    count_table: Union[Path, str],
    control_ids: List[str],
    treatment_ids: List[str],
    output_dir: Union[Path, str],
    prefix: str,
    run_mageck_func: Callable,
    method_params: Dict = {},
    n_permutations: int = 5,
    permutation_types: List[str] = ["within_sample"],
    dependencies: List[Job] = [],
) -> MultiFileGeneratingJob:
    """
    Create PyPipeGraph job for permutation testing.

    Runs negative control experiments by permuting count data to assess
    false positive rates and method robustness.

    Parameters
    ----------
    count_table : Path or str
        Original count table
    control_ids : List[str]
        Control sample IDs
    treatment_ids : List[str]
        Treatment sample IDs
    output_dir : Path or str
        Output directory
    prefix : str
        Prefix for output files
    run_mageck_func : Callable
        Function to run MAGeCK
    method_params : Dict
        MAGeCK parameters
    n_permutations : int
        Number of permutations to run
    permutation_types : List[str]
        Types of permutations: 'within_sample', 'swap_labels'
    dependencies : List[Job]
        Job dependencies

    Returns
    -------
    MultiFileGeneratingJob
        Job that creates permutation test results

    Examples
    --------
    >>> from crisprscreens.core.mageck import mageck_mle
    >>> job = permutation_test_job(
    ...     count_table="results/counts.txt",
    ...     control_ids=["Total_Rep1", "Total_Rep2", "Total_Rep3"],
    ...     treatment_ids=["Sort_Rep1", "Sort_Rep2", "Sort_Rep3"],
    ...     output_dir="results/comparison/MLE_permutations",
    ...     prefix="MLE",
    ...     run_mageck_func=mageck_mle,
    ...     method_params={'design_matrix': 'design.tsv'},
    ...     n_permutations=5,
    ... )
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define output files
    output_files = []
    for perm_type in permutation_types:
        for i in range(n_permutations):
            perm_name = f"{prefix}_perm_{perm_type}_{i+1}"
            perm_dir = output_dir / f"permutation_{perm_type}_{i+1}"
            output_files.append(perm_dir / f"{perm_name}.gene_summary.tsv")

    def __run(output_files):
        permutation_test_analysis(
            count_table=count_table,
            control_ids=control_ids,
            treatment_ids=treatment_ids,
            output_dir=output_dir,
            prefix=prefix,
            run_mageck_func=run_mageck_func,
            method_params=method_params,
            n_permutations=n_permutations,
            permutation_types=permutation_types,
        )

    job = MultiFileGeneratingJob(output_files, __run).depends_on(dependencies)

    job.depends_on(
        FunctionInvariant(permutation_test_analysis),
        ParameterInvariant(
            (
                count_table,
                control_ids,
                treatment_ids,
                str(output_dir),
                prefix,
                method_params,
                n_permutations,
                tuple(permutation_types),
            )
        ),
    )

    return job


def mageck_method_comparison_job(
    count_table: Union[Path, str],
    output_dir: Union[Path, str],
    control_ids: List[str],
    treatment_ids: List[str],
    full_design_matrix: Optional[Union[Path, str]] = None,
    control_sgrnas: Optional[Union[Path, str]] = None,
    methods: Optional[Dict[str, Dict]] = None,
    top_n_list: List[int] = [50, 100, 200],
    run_leave_one_out: bool = True,
    run_coherence: bool = True,
    run_control_fp: bool = True,
    run_permutation: bool = True,
    n_permutations: int = 5,
    dependencies: List[Job] = [],
) -> MultiFileGeneratingJob:
    """
    Create comprehensive PyPipeGraph job for MAGeCK method comparison.

    This job performs a complete comparison between different MAGeCK analysis
    methods (e.g., RRA vs MLE, different normalization strategies) using:
    1. Leave-one-replicate-out consistency analysis
    2. sgRNA coherence analysis
    3. Control sgRNA false-positive checks
    4. Permutation tests

    Parameters
    ----------
    count_table : Path or str
        MAGeCK count table
    output_dir : Path or str
        Output directory for all comparison results
    control_ids : List[str]
        Control sample IDs (e.g., ['Total_Rep1', 'Total_Rep2', 'Total_Rep3'])
    treatment_ids : List[str]
        Treatment sample IDs (e.g., ['Sort_Rep1', 'Sort_Rep2', 'Sort_Rep3'])
    full_design_matrix : Path or str, optional
        Full design matrix file (if needed for MLE)
    control_sgrnas : Path or str, optional
        Path to control sgRNAs file (for false-positive analysis)
    methods : Dict[str, Dict], optional
        Dictionary of method configurations. If None, uses default RRA and MLE.
        Format:
        {
            'RRA_paired_median': {
                'run_func': mageck_test,
                'params': {'paired': True, 'norm_method': 'median'},
                'gene_col': 'id',
            },
            'MLE_batch': {
                'run_func': mageck_mle,
                'params': {'design_matrix': 'design.tsv', 'norm_method': 'median'},
                'gene_col': 'Gene',
            }
        }
    top_n_list : List[int]
        List of top-N values to test
    run_leave_one_out : bool
        Whether to run leave-one-replicate-out analysis
    run_coherence : bool
        Whether to run sgRNA coherence analysis
    run_control_fp : bool
        Whether to run control false-positive checks
    run_permutation : bool
        Whether to run permutation tests
    n_permutations : int
        Number of permutations for permutation tests
    dependencies : List[Job]
        Job dependencies

    Returns
    -------
    MultiFileGeneratingJob
        Job that creates comprehensive comparison results

    Examples
    --------
    >>> from crisprscreens.core.mageck import mageck_test, mageck_mle
    >>>
    >>> methods = {
    ...     'RRA_paired_median': {
    ...         'run_func': mageck_test,
    ...         'params': {
    ...             'paired': True,
    ...             'norm_method': 'median',
    ...             'control_sgrnas': 'controls.txt'
    ...         },
    ...         'gene_col': 'id',
    ...     },
    ...     'MLE_batch_median': {
    ...         'run_func': mageck_mle,
    ...         'params': {
    ...             'design_matrix': 'design_batch.tsv',
    ...             'norm_method': 'median',
    ...             'control_sgrnas': 'controls.txt'
    ...         },
    ...         'gene_col': 'Gene',
    ...     }
    ... }
    >>>
    >>> job = mageck_method_comparison_job(
    ...     count_table="results/mageck_count/counts.txt",
    ...     control_ids=["Total_Rep1", "Total_Rep2", "Total_Rep3"],
    ...     treatment_ids=["Sort_Rep1", "Sort_Rep2", "Sort_Rep3"],
    ...     full_design_matrix="incoming/design_matrix.tsv",
    ...     output_dir="results/method_comparison",
    ...     control_sgrnas="cache/input/control_sgrnas.txt",
    ...     methods=methods,
    ...     top_n_list=[50, 100, 200],
    ...     run_leave_one_out=True,
    ...     run_coherence=True,
    ...     run_control_fp=True,
    ...     run_permutation=True,
    ...     n_permutations=5,
    ... )
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Default methods if not provided
    if methods is None:
        from crisprscreens.core.mageck import mageck_test, mageck_mle

        methods = {
            "RRA_paired_median": {
                "run_func": mageck_test,
                "params": {"paired": True, "norm_method": "median"},
                "gene_col": "id",
            },
            "RRA_unpaired_median": {
                "run_func": mageck_test,
                "params": {"paired": False, "norm_method": "median"},
                "gene_col": "id",
            },
            "MLE": {
                "run_func": mageck_mle,
                "params": {
                    "design_matrix": str(
                        Path(count_table).parent / "design_matrix.tsv"
                    )
                },
                "gene_col": "Gene",
            },
        }

    # Define output files
    output_files = [
        output_dir / "method_comparison_summary.tsv",
    ]

    # Add method-specific outputs
    for method_name in methods.keys():
        method_dir = output_dir / method_name

        if run_leave_one_out:
            output_files.append(
                method_dir / f"{method_name}_leave_one_out_comparison.tsv"
            )

        if run_coherence:
            output_files.append(
                method_dir / f"{method_name}_sgrna_coherence.tsv"
            )

        if run_control_fp and control_sgrnas:
            output_files.append(
                method_dir / f"{method_name}_control_false_positives.tsv"
            )

    def __run(
        output_files,
        count_table=count_table,
        output_dir=output_dir,
        control_ids=control_ids,
        full_design_matrix=full_design_matrix,
        treatment_ids=treatment_ids,
        control_sgrnas=control_sgrnas,
        methods=methods,
        top_n_list=top_n_list,
        run_leave_one_out=run_leave_one_out,
        run_coherence=run_coherence,
        run_control_fp=run_control_fp,
        run_permutation=run_permutation,
        n_permutations=n_permutations,
    ):
        compare_mageck_methods(
            count_table=count_table,
            output_dir=output_dir,
            control_ids=control_ids,
            treatment_ids=treatment_ids,
            full_design_matrix=full_design_matrix,
            control_sgrnas=control_sgrnas,
            methods=methods,
            top_n_list=top_n_list,
            run_leave_one_out=run_leave_one_out,
            run_coherence=run_coherence,
            run_control_fp=run_control_fp,
            run_permutation=run_permutation,
            n_permutations=n_permutations,
        )

    job = MultiFileGeneratingJob(output_files, __run).depends_on(dependencies)
    print(
        count_table,
        control_ids,
        treatment_ids,
        str(output_dir),
        control_sgrnas,
        str(top_n_list),
        run_leave_one_out,
        run_coherence,
        run_control_fp,
        run_permutation,
        n_permutations,
    )
    job.depends_on(
        FunctionInvariant(compare_mageck_methods),
        ParameterInvariant(
            f"{str(output_dir)}_method_comparison",
            (
                str(count_table),
                control_ids,
                treatment_ids,
                str(output_dir),
                control_sgrnas,
                str(top_n_list),
                run_leave_one_out,
                run_coherence,
                run_control_fp,
                run_permutation,
                n_permutations,
            ),
        ),
    )

    return job


def compare_rankings_simple_job(
    frames: Dict[str, DataFrame],
    specs: Dict[str, Dict[str, str]],
    outdir: str,
    run_prefix: str,
    top_x: int = 100,
    fdr_thresh: Optional[float] = 0.05,
    ascending: Optional[
        Dict[str, bool]
    ] = None,  # per condition: True if smaller is better
    rbo_p: float = 0.9,
    make_combined_plot: bool = True,
    dependencies: List[Job] = [],
) -> MultiFileGeneratingJob:
    """
    Create a PyPipeGraph job to compare gene rankings from different methods.

    This job compares gene rankings produced by different MAGeCK methods
    using overlap metrics like Jaccard index and Rank-Biased Overlap (RBO).

    Parameters
    ----------
    frames : Dict[str, pd.DataFrame]
        Dictionary of gene summary DataFrames keyed by method name
    specs : Dict[str, Dict[str, str]]
        Specifications for each method including 'gene_col' and 'score_col'
    outdir : str
        Output directory for comparison results
    run_prefix : str
        Prefix for output files
    top_x : int
        Number of top genes to consider for overlap analysis
    fdr_thresh : float, optional
        FDR threshold to filter significant genes
    ascending : Dict[str, bool], optional
        Dictionary indicating if lower scores are better for each method
    rbo_p : float
        Parameter p for RBO calculation (0 < p < 1)
    make_combined_plot : bool
        Whether to create a combined plot of the comparisons
    dependencies : List[Job]
        Job dependencies

    Returns
    -------
    MultiFileGeneratingJob
        Job that creates ranking comparison results
    """

    def __dump(
        output_files,
        frames=frames,
        specs=specs,
        outdir=outdir,
        run_prefix=run_prefix,
        top_x=top_x,
        fdr_thresh=fdr_thresh,
        ascending=ascending,
        rbo_p=rbo_p,
        make_combined_plot=make_combined_plot,
    ):
        compare_rankings_simple(
            frames=frames,
            specs=specs,
            outdir=outdir,
            run_prefix=run_prefix,
            top_x=top_x,
            fdr_thresh=fdr_thresh,
            ascending=ascending,
            rbo_p=rbo_p,
            make_combined_plot=make_combined_plot,
        )

    outfiles = [Path(outdir) / f"{run_prefix}.ranking_similarity_long.csv"]
    return MultiFileGeneratingJob(outfiles, __dump).depends_on(dependencies)
