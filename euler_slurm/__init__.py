"""euler-slurm: SLURM jobs as typed, versioned JSON, explained in plain words.

>>> from euler_slurm import list_jobs, get_job, diagnose, explain
>>> explain("QOSMaxGRESPerUser").label
'At your GPU limit'

See the README for the CLI (``euler-slurm jobs|job|diagnose|explain|capabilities``).
"""

__version__ = "0.2.0"

from ._parse import Gpus, expand_hostlist, expand_ranges, parse_duration  # noqa: E402
from .diagnose import (  # noqa: E402
    Diagnosis,
    Finding,
    Verdict,
    diagnose,
    diagnose_job,
    diagnose_logs,
)
from .explain import (  # noqa: E402
    Explanation,
    catalog,
    explain,
    explain_exit,
    explain_job,
    explain_reason,
    explain_state,
)
from .jobs import (  # noqa: E402
    Job,
    JobListing,
    JobNotFound,
    SlurmClient,
    get_job,
    list_jobs,
)
from .report import capability_report  # noqa: E402
from .runner import (  # noqa: E402
    Runner,
    SlurmCommandError,
    SlurmError,
    SlurmTimeout,
    SlurmUnavailable,
    SubprocessRunner,
)

__all__ = [
    "Diagnosis",
    "Explanation",
    "Finding",
    "Gpus",
    "Job",
    "JobListing",
    "JobNotFound",
    "Runner",
    "SlurmClient",
    "SlurmCommandError",
    "SlurmError",
    "SlurmTimeout",
    "SlurmUnavailable",
    "SubprocessRunner",
    "Verdict",
    "__version__",
    "capability_report",
    "catalog",
    "diagnose",
    "diagnose_job",
    "diagnose_logs",
    "expand_hostlist",
    "expand_ranges",
    "explain",
    "explain_exit",
    "explain_job",
    "explain_reason",
    "explain_state",
    "get_job",
    "list_jobs",
    "parse_duration",
]
