import pandas as pd  # type: ignore
import re
import shutil
import subprocess
import rpy2.robjects as ro
from typing import Dict, Optional, Union, List, Callable, Any, Iterable, Tuple
from pandas import DataFrame  # type: ignore
from pathlib import Path


def combine_comparisons(
    mageck_results: Dict[str, DataFrame],
    combine_on: Union[str, Dict[str, str]] = "id",
    how: str = "inner",
) -> DataFrame:
    keys = list(mageck_results.keys())
    combine_on_columns = combine_on
    if isinstance(combine_on, str):
        combine_on_columns = dict([(key, combine_on) for key in keys])
    for key in mageck_results:
        mageck_results[key] = mageck_results[key].rename(
            columns={
                c: f"{key}|{c}"
                for c in mageck_results[key].columns
                if c != combine_on_columns[key]
            }
        )
    combine_on_columns = combine_on
    if isinstance(combine_on, str):
        combine_on_columns = dict([(key, combine_on) for key in keys])
    merged_frames = mageck_results[keys[0]].merge(
        mageck_results[keys[1]],
        left_on=combine_on_columns[keys[0]],
        right_on=combine_on_columns[keys[1]],
    )
    if len(keys) > 2:
        for key2 in keys[2:]:
            print(key2)
            merged_frames = merged_frames.merge(
                mageck_results[key2],
                left_on=combine_on_columns[keys[0]],
                right_on=combine_on_columns[key2],
            )
    return merged_frames


def filter_multiple_mageck_comparisons(
    combined_frame: DataFrame,
    comparisons_to_filter: List[str],
    fdr_threshold: Union[float, Dict[str, float]] = 0.05,
    change_threshold: Union[float, Dict[str, float]] = 1.0,
    z_thresholds: Optional[Union[float, Dict[str, float]]] = None,
    direction: str = "both",  # "both", "pos", "neg"
    require_all: bool = True,  # AND (True) vs OR (False)
) -> DataFrame:
    """
    Filter combined MAGeCK RRA and/or MLE output frames.

    This function automatically detects whether a comparison column comes
    from MAGeCK-MLE (beta, z, wald-fdr) or MAGeCK-RRA (lfc, fdr).

    Expected patterns (before suffixing by combine_comparisons):
        MLE:
            <condition>|beta
            <condition>|z
            <condition>|fdr
            <condition>|wald-fdr

        RRA:
            neg|lfc, pos|lfc
            neg|fdr, pos|fdr

    Suffixing:
        When merging, columns become for example:
            neg|lfc_mycomparison
            high_T21|beta_othercomparison

    The function automatically detects whether a comparison is MLE-like
    (beta, z, wald-fdr) or RRA-like (lfc, fdr) based on column names.
    """

    df = combined_frame

    def _to_dict(val, keys):
        """convert scalar thresholds to dicts keyed by comparison names"""
        if isinstance(val, dict):
            return val
        return {k: val for k in keys}

    def find_matching(pattern_list):
        """find columns that match any pattern + the comparison suffix"""
        regexes = [re.compile(key + p + r"$") for p in pattern_list]
        matches = []
        for col in df.columns:
            if any(r.search(col) for r in regexes):
                matches.append(col)
        return matches

    fdr_thr = _to_dict(fdr_threshold, comparisons_to_filter)
    change_thr = _to_dict(change_threshold, comparisons_to_filter)
    z_thr = (
        _to_dict(z_thresholds, comparisons_to_filter)
        if z_thresholds is not None
        else None
    )

    # Initialize global mask:
    if require_all:
        # require_all=True → logical AND
        global_mask = pd.Series(True, index=df.index)
    else:
        # require_all=False → logical OR
        global_mask = pd.Series(False, index=df.index)

    # regex patterns help us identify relevant columns, even after suffixing
    fdr_patterns = [
        r"\|fdr",  # RRA and MLE fdr after merging
        r"\|wald-fdr",  # MLE wald-fdr
    ]  # FDR columns (MLE and RRA)
    eff_patterns = [
        r"\|beta",  # MLE effect size
        r"\|lfc",  # RRA effect size
    ]  # effect size columns
    z_patterns = [r"\|z_"]  # Z score columns (MLE only)
    for key in comparisons_to_filter:
        # loop over comparisons
        eff_cols = find_matching(eff_patterns)
        fdr_cols = find_matching(fdr_patterns)
        z_cols = find_matching(z_patterns)
        if not fdr_cols or not eff_cols:
            # no valid columns for this comparison
            raise ValueError(f"No fdr or effect column found for key={key}.")

        if len(fdr_cols) > 1:
            fdr_cols = [x for x in fdr_cols if "wald-fdr" in x]
        fdr_col = fdr_cols[0]
        eff_col = eff_cols[0]
        z_col = z_cols[0] if z_cols else None

        # build filter mask for this comparison
        mask = pd.Series(True, index=df.index)
        if fdr_thr[key] is not None:
            mask = df[fdr_col] <= fdr_thr[key]  # FDR filter
        if direction == "both":
            mask &= df[eff_col].abs() >= change_thr[key]  # effect size filter
        elif direction == "pos":
            mask &= df[eff_col] >= change_thr[key]
        elif direction == "neg":
            mask &= df[eff_col] <= -change_thr[key]

        else:
            raise ValueError(
                "direction must be one of: 'both', 'pos', or 'neg'"
            )
        df["test"] = df[eff_col] >= change_thr[key]
        if z_col is not None and z_thr is not None and key in z_thr:
            mask &= df[z_col].abs() >= z_thr[key]  # Z-filter (MLE only)

        # Combine mask depending on AND/OR logic
        if require_all:
            global_mask &= mask
        else:
            global_mask |= mask
    return df[global_mask].copy()


def mageck_count(
    sgrna_list: Union[Path, str],
    samples: dict,
    out_dir: Union[Path, str],
    prefix: str,
    control_sgrnas=Optional[: Union[Path, str]],
    norm_method: str = "median",
    pdf_report: bool = False,
    other_parameter: Dict[str, str] = [],
):

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    if isinstance(list(samples.values())[1], list):
        fastqs = " ".join(f"{",".join(fastq)}" for fastq in samples.values())
    else:
        fastqs = " ".join(fastq for fastq in samples.values())

    sample_labels = ",".join(sample_name for sample_name in samples.keys())

    command_parameters = [
        "-l",
        sgrna_list,
        "--fastq",
        fastqs,
        "--sample-label",
        sample_labels,
        "-n",
        f"{out_dir}/{prefix}",
    ]

    if control_sgrnas is not None:
        command_parameters.extend(["--control-sgrna", str(control_sgrnas)])

    if norm_method is not None:
        command_parameters.extend(["--norm-method", str(norm_method)])
    else:
        command_parameters.extend(["--norm-method", "none"])
    if pdf_report:
        command_parameters.append("--pdf-report")

    command_parameters.extend(other_parameter)

    command = ["mageck count"]
    command.extend(command_parameters)
    cmd = " ".join(command)

    print(cmd)

    mageck_count = subprocess.run(
        cmd, capture_output=True, text=True, shell=True
    )
    print(mageck_count.stdout)
    print(mageck_count.stderr)

    # if pdf_report:
    ro.r("rmarkdown::render")(f"{out_dir}/{prefix}.count_report.Rmd")

    count_txt = Path(f"{out_dir}/{prefix}.count.txt")
    count_tsv = Path(f"{out_dir}/{prefix}.count.tsv")
    if count_txt.is_file():
        shutil.copy(count_txt, count_tsv)


def mageck_count2(
    sgrna_list: Union[Path, str],
    samples: dict,
    out_dir: Union[Path, str],
    prefix: str,
    count_table: Optional[Union[Path, str]] = None,
    control_sgrnas=Optional[: Union[Path, str]],
    norm_method: str = "median",
    pdf_report: bool = False,
    other_parameter: Dict[str, str] = [],
):

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    if count_table is None:
        if isinstance(list(samples.values())[1], list):
            fastqs = " ".join(
                f"{",".join(fastq)}" for fastq in samples.values()
            )
        else:
            fastqs = " ".join(fastq for fastq in samples.values())

        sample_labels = ",".join(sample_name for sample_name in samples.keys())

        command_parameters = [
            "-l",
            sgrna_list,
            "--fastq",
            fastqs,
            "--sample-label",
            sample_labels,
            "-n",
            f"{out_dir}/{prefix}",
        ]
    else:
        command_parameters = [
            "-k",
            str(count_table),
            "-n",
            f"{out_dir}/{prefix}",
        ]
    if control_sgrnas is not None:
        command_parameters.extend(["--control-sgrna", str(control_sgrnas)])

    if norm_method is not None:
        command_parameters.extend(["--norm-method", str(norm_method)])
    else:
        command_parameters.extend(["--norm-method", "none"])
    if pdf_report:
        command_parameters.append("--pdf-report")

    command_parameters.extend(other_parameter)

    command = ["mageck count"]
    command.extend(command_parameters)
    cmd = " ".join(command)

    print(cmd)

    mageck_count = subprocess.run(
        cmd, capture_output=True, text=True, shell=True
    )
    print(mageck_count.stdout)
    print(mageck_count.stderr)

    # if pdf_report:
    ro.r("rmarkdown::render")(f"{out_dir}/{prefix}.count_report.Rmd")

    count_txt = Path(f"{out_dir}/{prefix}.count.txt")
    count_tsv = Path(f"{out_dir}/{prefix}.count.tsv")
    if count_txt.is_file():
        shutil.copy(count_txt, count_tsv)


def mageck_test(
    count_table: Union[Path, str],
    treatment_ids: List[str],
    control_ids: List[str],
    out_dir: Union[Path, str],
    prefix: str,
    control_sgrnas: Optional[Union[Path, str]],
    norm_method: str = None,
    paired: bool = False,
    pdf_report: bool = False,
    other_parameter: Dict[str, str] = [],
):

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    command_parameters = [
        "-k",
        str(count_table),
        "-t",
        ",".join(treatment_ids),
        "-c",
        ",".join(control_ids),
        "-n",
        f"{out_dir}/{prefix}",
    ]

    if control_sgrnas is not None:
        command_parameters.extend(["--control-sgrna", str(control_sgrnas)])

    if norm_method is not None:
        command_parameters.extend(["--norm-method", str(norm_method)])

    if paired:
        command_parameters.append("--paired")

    if pdf_report:
        command_parameters.append("--pdf-report")
    for k in other_parameter:
        command_parameters.extend([k, other_parameter[k]])

    command = ["mageck test"]
    command.extend(command_parameters)
    cmd = " ".join(command)
    print(cmd)

    mageck_count = subprocess.run(
        cmd, capture_output=True, text=True, shell=True
    )
    print(mageck_count.stdout)
    print(mageck_count.stderr)

    # if pdf_report:
    ro.r("rmarkdown::render")(f"{out_dir}/{prefix}.report.Rmd")

    gene_summary_txt = Path(f"{out_dir}/{prefix}.gene_summary.txt")
    gene_summary_tsv = Path(f"{out_dir}/{prefix}.gene_summary.tsv")

    sgrna_summary_txt = Path(f"{out_dir}/{prefix}.sgrna_summary.txt")
    sgrna_summary_tsv = Path(f"{out_dir}/{prefix}.sgrna_summary.tsv")

    if gene_summary_txt.is_file():
        shutil.copy(gene_summary_txt, gene_summary_tsv)
    if sgrna_summary_txt.is_file():
        shutil.copy(sgrna_summary_txt, sgrna_summary_tsv)


def mageck_mle(
    count_table: Union[Path, str],
    design_matrix: Union[Path, str],
    out_dir: Union[Path, str],
    prefix: str,
    control_sgrnas: Optional[Union[Path, str]] = None,
    norm_method: str = None,
    other_parameter: Dict[str, str] = [],
):

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    command_parameters = [
        "-k",
        str(count_table),
        "-d",
        str(design_matrix),
        "-n",
        f"{str(out_dir)}/{str(prefix)}",
    ]

    if control_sgrnas is not None:
        command_parameters.extend(["--control-sgrna", str(control_sgrnas)])

    if norm_method is not None:
        command_parameters.extend(["--norm-method", str(norm_method)])

    command_parameters.extend(other_parameter)

    command = ["mageck mle"]
    command.extend(command_parameters)
    cmd = " ".join(command)
    print(cmd)

    mageck_count = subprocess.run(
        cmd, capture_output=True, text=True, shell=True
    )
    print(mageck_count.stdout)
    print(mageck_count.stderr)

    gene_summary_txt = Path(f"{out_dir}/{prefix}.gene_summary.txt")
    gene_summary_tsv = Path(f"{out_dir}/{prefix}.gene_summary.tsv")

    sgrna_summary_txt = Path(f"{out_dir}/{prefix}.sgrna_summary.txt")
    sgrna_summary_tsv = Path(f"{out_dir}/{prefix}.sgrna_summary.tsv")

    if gene_summary_txt.is_file():
        shutil.copy(gene_summary_txt, gene_summary_tsv)
    if sgrna_summary_txt.is_file():
        shutil.copy(sgrna_summary_txt, sgrna_summary_tsv)


def split_frame_to_control_and_query(
    mageck_frame: DataFrame,
    control_prefix: str,
    id_col: Optional[str] = None,
    name_column: str = "Name",
    sgRNA_column: str = "sgRNA",
    infer_genes: Optional[Callable] = None,
) -> Dict[str, DataFrame]:

    if id_col is None:
        mageck_frame["id"] = "s_" + mageck_frame.index.astype(str)
    else:
        mageck_frame["id"] = mageck_frame[id_col].str.replace(
            r"\s+", "_", regex=True
        )

    if infer_genes is not None:
        mageck_frame[name_column] = infer_genes(mageck_frame)
    else:
        mageck_frame[name_column] = mageck_frame[name_column].str.replace(
            r"\s+", "_", regex=True
        )
    control_rows = mageck_frame[
        mageck_frame[name_column].str.startswith(control_prefix)
    ].index
    df_control = mageck_frame.loc[control_rows][["id"]].copy()
    df_query = mageck_frame[["id", sgRNA_column, name_column]].copy()
    return {"control": df_control, "query": df_query}


def mageck_pathway(
    gene_ranking: Union[Path, str],
    gmt_file: Union[Path, str],
    out_dir: Union[Path, str],
    prefix: str = "pathway",
    method: str = "gsea",
    single_ranking: bool = False,
    output_prefix: Optional[str] = None,
    sort_criteria: str = "neg",
    keep_tmp: bool = False,
    ranking_column: Optional[Union[str, int]] = None,
    ranking_column_2: Optional[Union[str, int]] = None,
    pathway_alpha: Optional[float] = None,
    permutation: Optional[int] = None,
    other_parameter: Dict[str, str] = [],
):
    """
    Wrapper for `mageck pathway` subcommand.

    Required:
      - gene_ranking: gene ranking file
      - gmt_file: GMT pathways file

    Optional parameters mirror MAGeCK CLI options.

    Returns a dict with stdout/stderr and list of output files in out_dir
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    command_parameters = [
        "--gene-ranking",
        str(gene_ranking),
        "--gmt-file",
        str(gmt_file),
    ]

    # output prefix
    out_pref = output_prefix if output_prefix is not None else prefix
    command_parameters.extend(["-n", f"{out_dir}/{out_pref}"])

    # method
    if method is not None:
        command_parameters.extend(["--method", str(method)])

    if single_ranking:
        command_parameters.append("--single-ranking")

    if sort_criteria is not None:
        command_parameters.extend(["--sort-criteria", str(sort_criteria)])

    if keep_tmp:
        command_parameters.append("--keep-tmp")

    if ranking_column is not None:
        command_parameters.extend(["--ranking-column", str(ranking_column)])
    if ranking_column_2 is not None:
        command_parameters.extend(["--ranking-column-2", str(ranking_column_2)])
    if pathway_alpha is not None:
        command_parameters.extend(["--pathway-alpha", str(pathway_alpha)])
    if permutation is not None:
        command_parameters.extend(["--permutation", str(permutation)])

    for k in other_parameter:
        command_parameters.extend([k, other_parameter[k]])

    command = ["mageck pathway"]
    command.extend(command_parameters)
    cmd = " ".join(command)
    print(cmd)

    proc = subprocess.run(cmd, capture_output=True, text=True, shell=True)
    print(proc.stdout)
    print(proc.stderr)

    # Collect output files with provided prefix
    outputs = list(Path(out_dir).glob(f"{out_pref}*"))

    return {
        "cmd": cmd,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "outputs": [str(p) for p in outputs],
    }


def mageck_plot(
    gene_summary: Optional[Union[Path, str]] = None,
    sgrna_summary: Optional[Union[Path, str]] = None,
    out_dir: Union[Path, str] = ".",
    prefix: str = "plot",
    other_parameter: Dict[str, str] = [],
):
    """
    Generic wrapper for `mageck plot` subcommand.

    It will pass any provided files to the CLI and collect generated plots.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    command_parameters = []
    if gene_summary is not None:
        command_parameters.extend(["-k", str(gene_summary)])
    if sgrna_summary is not None:
        command_parameters.extend(["-s", str(sgrna_summary)])

    command_parameters.extend(["-n", f"{out_dir}/{prefix}"])

    for k in other_parameter:
        command_parameters.extend([k, other_parameter[k]])

    command = ["mageck plot"]
    command.extend(command_parameters)
    cmd = " ".join(command)
    print(cmd)

    proc = subprocess.run(cmd, capture_output=True, text=True, shell=True)
    print(proc.stdout)
    print(proc.stderr)

    outputs = list(Path(out_dir).glob(f"{prefix}*"))
    return {
        "cmd": cmd,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "outputs": [str(p) for p in outputs],
    }


def combine_gene_info_with_mageck_output(
    df_mageck: DataFrame,
    df_genes: DataFrame,
    name_column_mageck: str = "id",
    name_column_genes: str = "Gene",
    how: str = "left",
    columns_to_add: List[str] = [
        "gene_stable_id",
        "name",
        "chr",
        "start",
        "stop",
        "strand",
        "biotype",
    ],
) -> DataFrame:
    """
    Combine gene information dataframe with MAGeCK output dataframe.

    Parameters:
    - df_mageck: DataFrame containing MAGeCK results.
    - df_genes: DataFrame containing gene information.
    - name_column_mageck: Column name in df_mageck to match with df_genes.
    - name_column_genes: Column name in df_genes to match with df_mageck.

    Returns:
    - Merged DataFrame with gene information added to MAGeCK results.
    """
    df_genes = df_genes.drop_duplicates(keep="first", subset=name_column_genes)
    merged_df = df_mageck.merge(
        df_genes[columns_to_add + [name_column_genes]],
        left_on=name_column_mageck,
        right_on=name_column_genes,
        how=how,
    )
    merged_df.drop(columns=[name_column_genes], inplace=True)
    return merged_df


def get_significant_genes(
    df_mageck: DataFrame,
    fdr_column: str = "pos|fdr",
    fdr_threshold: float = 0.05,
    logfc_column: str = "pos|lfc",
    logfc_or_beta_threshold: float = 1.0,
    direction: str = "both",  # "both", "pos", "neg"
) -> DataFrame:
    """
    get_significant_genes filters the gene summary table for significant hits.

        Parameters
    ----------
    df_mageck : DataFrame
        Mageck output dataframe, gene summary table.
    fdr_column : str, optional
        False Discovery Rate column, by default "pos|fdr"
    fdr_threshold : float, optional
        maximum FDR, by default 0.05
    logfc_column : str, optional
        logFC column, by default "pos|lfc"
    logfc_threshold : float, optional
        minimum fold change or beta, by default 1.0
    direction : str, optional
        positive or nagative selection, by default "both"

    Returns
    -------
    DataFrame
        Filtered dataframe with significant genes only.
    """
    if direction == "both":
        sig_genes = df_mageck[
            (df_mageck[fdr_column] <= fdr_threshold)
            & (df_mageck[logfc_column].abs() >= logfc_or_beta_threshold)
        ]
    elif direction == "pos":
        sig_genes = df_mageck[
            (df_mageck[fdr_column] <= fdr_threshold)
            & (df_mageck[logfc_column] >= logfc_or_beta_threshold)
        ]
        sig_genes = sig_genes.sort_values(by=logfc_column, ascending=False)
    elif direction == "neg":
        sig_genes = df_mageck[
            (df_mageck[fdr_column] <= fdr_threshold)
            & (df_mageck[logfc_column] <= -logfc_or_beta_threshold)
        ]
        sig_genes = sig_genes.sort_values(by=logfc_column, ascending=True)
    else:
        raise ValueError("direction must be one of: 'both', 'pos', or 'neg'")
    return sig_genes


# def filter_mageck_counts(
#     count_table: DataFrame,
#     paired_samples_before_after: Dict[str, str],
#     filter_thresholds: Dict[str, int] = {5},
#     min_baseline_above_threshold: int = 1,
#     samples_to_excluide: Optional[List[str]] = None,

# ) -> DataFrame:
#     """
#     filter_mageck_counts filters the mageck count table based on a minimum count threshold
#     in the baseline samples of paired samples.

#     Parameters
#     ----------
#     count_table : DataFrame
#         Mageck count table dataframe.
#     filter_threshold : int, optional
#         minimum count threshold, by default 10
#     paired_samples : Dict[str, str]
#         dictionary of paired samples, by default {}

#     Returns
#     -------
#     DataFrame
#         Filtered mageck count table dataframe.
#     """
#     df = count_table.copy()
#     # identify count columns
#     for baseline, cond in paired_samples_before_after.items():
#         if baseline not in df.columns or cond not in df.columns:
#             raise ValueError(
#                 f"Paired sample columns {baseline} and {cond} not found in count table."
#             )
#     baseline_cols = list(paired_samples_before_after.keys())
#     mask = df[baseline_cols].ge(filter_threshold).sum(axis=1) >= min_baseline_above_threshold
#     return df[mask].copy()


def filter_mageck_counts(
    df: pd.DataFrame,
    conditions: Dict[str, Any],
    baseline_samples: Iterable[str],
    aggregations: Optional[Dict[str, Tuple[List[str], Callable]]] = None,
    exclude_samples: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Filter an *annotated* MAGeCK-style count table.

    Assumptions
    ----------
    - `df` contains raw counts for samples (e.g. columns Total1..3, Sorted1..3)
    - `df` may also contain derived columns (logCPM_*, M_*, A_* etc.)
    - You want to filter by:
        1) arbitrary column-based thresholds (A/M/logCPM/etc.)
        2) baseline detectability: at least N baseline samples have raw counts >= min_count
        3) potenmtially aggregations of multiple columns (e.g. min(A_rep*), mean(M_rep*), etc.)

    Parameters
    ----------
    df : pd.DataFrame
        Annotated count table.
    conditions : Dict[str, Any]
        Dict defining filtering rules. Supported keys:

            Column thresholds (simple):
            - "col_filters": list of dicts, each with:
                {"col": <column name>, "op": one of {"<","<="," >",">=","==","!="}, "value": number}
                Example:
                {"col_filters": [{"col": "Total_Rep1", "op": ">=", "value": 2}]}

            Baseline detectability:
            - "baseline_min_count": int (e.g. 20)
            - "baseline_min_n": int (e.g. 2)  # at least this many baseline samples pass the count
                Example:
                {"baseline_min_count": 20, "baseline_min_n": 2}

            Missing handling:
            - "na_policy": "drop" or "keep" (default "drop")
                "drop": rows with NA in any used filter metric are removed
    baseline_samples : List[str]
        Names of baseline sample columns (raw counts) like
        ["Total1","Total2","Total3"].
    aggregations : Optional[Dict[str, Tuple[List[str], Callable]]], optional
        Optional dict defining new aggregated columns to create prior to filtering.
        You can specify in aggregations a new column name, a list of existing
        columns to aggregate and a function to use for aggregation.
        Supported functions are any pandas DataFrame aggregation functions like
        min, max, mean, median, etc., by default None
    exclude_samples : Optional[Iterable[str]], optional
        Samples to exclude (both from raw counts and any derived columns
        mentioning them), by default None

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame
    """

    def _apply_op(series: pd.Series, op: str, value: Any) -> pd.Series:
        if op == "<":
            return series < value
        if op == "<=":
            return series <= value
        if op == ">":
            return series > value
        if op == ">=":
            return series >= value
        if op == "==":
            return series == value
        if op == "!=":
            return series != value
        raise ValueError(f"Unsupported op: {op}")

    def _aggregate(frame: pd.DataFrame, how: str) -> pd.Series:
        how = how.lower()
        if how == "min":
            return frame.min(axis=1)
        if how == "max":
            return frame.max(axis=1)
        if how == "mean":
            return frame.mean(axis=1)
        if how == "median":
            return frame.median(axis=1)
        raise ValueError(f"Unsupported aggregation reduce: {how}")

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    na_policy = conditions.get("na_policy", "drop")
    if na_policy not in {"drop", "keep"}:
        raise ValueError("conditions['na_policy'] must be 'drop' or 'keep'")
    exclude_samples = set(exclude_samples or [])
    drop_excluded_columns = len(exclude_samples) > 0
    baseline_samples = [s for s in baseline_samples if s not in exclude_samples]
    if len(baseline_samples) == 0:
        raise ValueError("No baseline_samples left after excluding samples.")
    out = df.copy()
    out[baseline_samples].apply(pd.to_numeric, errors="raise")

    # Drop any derived columns that reference excluded sample names
    if drop_excluded_columns:
        raw_excl = []
        for col in out.columns:
            for s in exclude_samples:
                if s in col:
                    raw_excl.append(col)
        out = out.drop(columns=raw_excl, errors="ignore")

    # Build filter mask
    mask = pd.Series(True, index=out.index)

    used_cols_for_na: set[str] = set()

    # Baseline detectability
    baseline_min_count = conditions.get("baseline_min_count", None)
    baseline_min_n = conditions.get("baseline_min_n", None)
    if baseline_min_count is not None or baseline_min_n is not None:
        if baseline_min_count is None or baseline_min_n is None:
            raise ValueError(
                "Provide BOTH 'baseline_min_count' and 'baseline_min_n' if using baseline detectability."
            )

        missing = [c for c in baseline_samples if c not in out.columns]
        if missing:
            raise ValueError(f"Baseline raw count columns not found: {missing}")

        # number of baseline samples above threshold
        n_ok = (out[baseline_samples] >= int(baseline_min_count)).sum(axis=1)
        used_cols_for_na.update(baseline_samples)
        mask &= n_ok >= int(baseline_min_n)

    # aggregations
    if aggregations:
        for new_col, (cols, how) in aggregations.items():
            missing = [c for c in cols if c not in out.columns]
            if missing:
                raise ValueError(
                    f"Columns for aggregation '{new_col}' not found: {missing}"
                )
            frame = out[list(cols)].apply(pd.to_numeric, errors="coerce")
            used_cols_for_na.update(cols)
            out[new_col] = _aggregate(frame, how)

    # Simple column filters
    for f in conditions.get("col_filters", []) or []:
        col = f["col"]
        op = f["op"]
        val = f["value"]
        if col not in out.columns:
            raise ValueError(
                f"Column not found for col_filters: {col}. Do you need to aggregate first?"
            )
        s = pd.to_numeric(out[col], errors="coerce")
        used_cols_for_na.add(col)
        mask &= _apply_op(s, op, val)

    # NA policy
    if na_policy == "drop" and used_cols_for_na:
        any_na = out[list(used_cols_for_na)].isna().any(axis=1)
        mask &= ~any_na

    filtered = out.loc[mask].reset_index(drop=True)
    return filtered
