"""SLURM in plain words: pending reasons, job states and exit codes.

Every explanation says what the code means and what to do about it (a fix,
usually a command). ``<job_id>`` in a fix is replaced when the explanation is
made for a job (:meth:`Explanation.for_job`).

A code this module does not know gets an honest generic explanation
(``known: false``) that points to where the meaning can be looked up. It never
guesses from the code's name.

Every explanation also has a ``category`` from a small closed set per kind
(:data:`REASON_CATEGORIES`, :data:`STATE_CATEGORIES`, :data:`EXIT_CATEGORIES`),
so tools can group codes: a ``limit`` clears by itself, a ``request`` never
starts as it is. :func:`catalog` returns everything as data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Tuple

from ._parse import (
    LONG_STATES,
    PENDING_STATES,
    STATE_CODES,
    TERMINAL_STATES,
    parse_state,
)

if TYPE_CHECKING:  # pragma: no cover
    from .jobs import Job

SEVERITIES = ("error", "warning", "info")

#: The category of an explanation of a code euler-slurm does not know
#: (``known: false``). It is in none of the closed category sets.
UNKNOWN_CATEGORY = "unknown"


@dataclass(frozen=True)
class Explanation:
    """What a SLURM code means, and what to do."""

    #: ``pending_reason``, ``state`` or ``exit_code``.
    kind: str
    #: The code as euler-slurm knows it (``Priority``, ``TIMEOUT``, ``137``).
    code: str
    #: A few words: "Waiting for its turn".
    label: str
    #: What it means, in one to three sentences.
    meaning: str
    #: What to do (often a command).
    fix: str
    #: ``error`` (needs action), ``warning`` (worth a look), ``info``.
    severity: str = "info"
    #: What kind of code it is, from the closed set of its kind
    #: (:data:`REASON_CATEGORIES`, :data:`STATE_CATEGORIES`,
    #: :data:`EXIT_CATEGORIES`); ``unknown`` for a code euler-slurm does not know.
    category: str = UNKNOWN_CATEGORY
    #: Pending reasons only: the job will not start unless someone acts.
    blocked: bool = False
    #: ``False``: euler-slurm does not know this code (generic text).
    known: bool = True
    #: Extra detail SLURM appended (``UnavailableNodes:eu-g1-[001-004]``).
    detail: Optional[str] = None
    #: The text that was explained, as given.
    raw: Optional[str] = None

    def for_job(self, job_id: Any) -> "Explanation":
        """The same explanation with ``<job_id>`` filled in."""
        text = str(job_id)
        return replace(
            self,
            meaning=self.meaning.replace("<job_id>", text),
            fix=self.fix.replace("<job_id>", text),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "code": self.code,
            "known": self.known,
            "label": self.label,
            "meaning": self.meaning,
            "fix": self.fix,
            "severity": self.severity,
            "category": self.category,
            "blocked": self.blocked,
            "detail": self.detail,
            "raw": self.raw,
        }


class _Entry(NamedTuple):
    #: One of :data:`REASON_CATEGORIES`, :data:`STATE_CATEGORIES` or
    #: :data:`EXIT_CATEGORIES` (by the table the entry is in).
    category: str
    label: str
    meaning: str
    fix: str
    severity: str = "info"


# ---------------------------------------------------------------------------
# Categories: closed sets, code → a few plain words
# ---------------------------------------------------------------------------

#: What kind of wait a pending reason is.
REASON_CATEGORIES: Dict[str, str] = {
    "queue": "Waiting for its turn or for free resources",
    "dependency": "Waits for another job, or was broken by it",
    "schedule": "Waits for a set time, reservation or license",
    "hold": "On hold until someone releases it",
    "limit": "At a usage limit; starts by itself",
    "request": "Does not fit the limits; never starts as is",
    "budget": "Compute budget used up",
    "nodes": "Nodes, partition or cluster services unavailable",
    "startup": "Its nodes are being prepared",
    "requeued": "An earlier run ended; it waits to run again",
    "other": "Something else",
}

#: Where in its life a job is, by its state.
STATE_CATEGORIES: Dict[str, str] = {
    "waiting": "Waiting to start",
    "held": "Held until someone releases it",
    "running": "Running or finishing",
    "paused": "Paused; keeps its nodes",
    "succeeded": "Finished successfully",
    "failed": "Ended with a failure",
    "interrupted": "Stopped to make room for another job",
    "cancelled": "Cancelled or removed",
}

#: How a program ended, by its exit code.
EXIT_CATEGORIES: Dict[str, str] = {
    "success": "Success",
    "error": "The program reported an error",
    "environment": "A command could not be run",
    "signal": "Ended by a signal",
    "memory": "Killed for using too much memory",
}


# ---------------------------------------------------------------------------
# Pending reasons (squeue %r / state_reason, scontrol Reason=)
# ---------------------------------------------------------------------------

_JOBS_LIMIT = _Entry(
    "limit",
    "At the running-jobs limit",
    "You (or your group) already have as many jobs running or queued as the "
    "QOS or account allows. It starts by itself when one of them ends.",
    "Wait, or cancel jobs you no longer need (`squeue --me`, then `scancel "
    "<id>`). `sacctmgr show qos format=Name,MaxJobsPerUser,MaxSubmitJobsPerUser` "
    "and `sacctmgr show assoc user=$USER format=Account,MaxJobs,GrpJobs` show "
    "the limits.",
    "warning",
)
_GPU_USER_LIMIT = _Entry(
    "limit",
    "At your GPU limit",
    "Your running jobs already use as many GPUs as your QOS allows per user "
    "(or per account). It starts by itself when some of your GPUs are released.",
    "Wait, or cancel or shrink other GPU jobs of yours. `sacctmgr show qos "
    "format=Name,MaxTRESPerUser%40,MaxTRESPerAccount%40` shows the limit.",
    "warning",
)
_GPU_GROUP_LIMIT = _Entry(
    "limit",
    "Your group is at its GPU limit",
    "Your group (account) is already using all the GPUs its share allows. This "
    "counts your colleagues' jobs too, so it can happen even when you run "
    "nothing. It starts by itself when some are released.",
    "Wait. `squeue --account=<account>` shows who in the group uses them; "
    "`sacctmgr show assoc user=$USER format=Account,GrpTRES%40` the limit. "
    "Fewer GPUs per job can start sooner.",
    "warning",
)
_GPU_PER_JOB = _Entry(
    "request",
    "Too many GPUs for one job",
    "It asks for more GPUs than one job may use (or fewer than it must), so it "
    "will never start as it is.",
    "Cancel it (`scancel <job_id>`) and resubmit with fewer GPUs "
    "(`--gpus`/`--gres=gpu:N`). `sacctmgr show qos format=Name,MaxTRESPerJob%40,"
    "MinTRESPerJob%40` shows the limit.",
    "error",
)
_CPU_LIMIT = _Entry(
    "limit",
    "At the CPU limit",
    "You or your group already use as many CPUs as the QOS or account allows. "
    "It starts by itself when some are released.",
    "Wait, or ask for fewer CPUs (`--cpus-per-task`). `sacctmgr show assoc "
    "user=$USER format=Account,GrpTRES%40` and `sacctmgr show qos "
    "format=Name,MaxTRESPerUser%40` show the limits.",
    "warning",
)
_CPU_PER_JOB = _Entry(
    "request",
    "Too many CPUs for one job",
    "It asks for more CPUs than one job may use, so it will never start as it is.",
    "Cancel it (`scancel <job_id>`) and resubmit with fewer CPUs "
    "(`--cpus-per-task`, `--ntasks`).",
    "error",
)
_MEMORY_LIMIT = _Entry(
    "limit",
    "At the memory limit",
    "You or your group already use as much memory as the QOS or account allows. "
    "It starts by itself when some is released.",
    "Wait, or ask for less memory (`--mem`, `--mem-per-cpu`).",
    "warning",
)
_MEMORY_PER_JOB = _Entry(
    "request",
    "Too much memory for one job",
    "It asks for more memory than one job may use, so it will never start as it is.",
    "Cancel it (`scancel <job_id>`) and resubmit with a smaller `--mem` "
    "(or `--mem-per-cpu`).",
    "error",
)
_NODE_LIMIT = _Entry(
    "limit",
    "At the node limit",
    "You or your group already use as many nodes as the QOS or account allows. "
    "It starts by itself when some are released.",
    "Wait, or ask for fewer nodes (`--nodes`).",
    "warning",
)
_NODES_PER_JOB = _Entry(
    "request",
    "Too many nodes for one job",
    "It asks for more nodes than one job may use, so it will never start as it is.",
    "Cancel it (`scancel <job_id>`) and resubmit with fewer `--nodes`.",
    "error",
)
_WALLTIME_TOO_LONG = _Entry(
    "request",
    "Time limit too long",
    "It asks for more time (`--time`) than its QOS or account allows per job, "
    "so it will never start as it is.",
    "Lower the limit of the waiting job: `scontrol update JobId=<job_id> "
    "TimeLimit=HH:MM:SS` (users may shorten it), or cancel and resubmit. "
    "`sacctmgr show qos format=Name,MaxWall` shows the limit.",
    "error",
)
_BUDGET = _Entry(
    "budget",
    "Compute budget used up",
    "The account has used up its compute budget (CPU, GPU or billing minutes) "
    "for now. It will not start until the budget is reset or raised.",
    "Ask the account's owner (your PI) or cluster support; `sshare -U` shows "
    "the account's usage. Another account you may use: `sacctmgr show assoc "
    "user=$USER format=Account`.",
    "error",
)
_RUN_MINUTES = _Entry(
    "limit",
    "Too much work running at once",
    "Your group's running jobs together hold as many resource-minutes "
    "(resources × time left) as allowed. It starts as running jobs progress or end.",
    "Wait. Jobs with a shorter `--time` count less against this limit and can "
    "start sooner.",
    "warning",
)
_GENERIC_LIMIT = _Entry(
    "limit",
    "At a usage limit",
    "One of the limits of your QOS or account (jobs, CPUs, GPUs, memory or "
    "nodes in use) is reached. It starts by itself when usage drops.",
    "Wait. `sacctmgr show assoc user=$USER format=Account,QOS,GrpTRES%40,MaxJobs` "
    "and `sacctmgr show qos` show the limits.",
    "warning",
)
_SUBMIT_LIMIT = _Entry(
    "limit",
    "At the queued-jobs limit",
    "You (or your group) already have as many jobs submitted as allowed.",
    "Wait for some to finish, or cancel ones you no longer need (`squeue --me`, "
    "`scancel <id>`).",
    "warning",
)

REASONS: Dict[str, _Entry] = {
    "None": _Entry(
        "queue",
        "About to be scheduled",
        "SLURM has not looked at it yet. It gets a reason, or starts, within a "
        "minute or two.",
        "Nothing to do.",
    ),
    "Priority": _Entry(
        "queue",
        "Waiting for its turn",
        "Jobs with a higher priority are ahead of it in the queue. Priority grows "
        "the longer a job waits, and is lower for users and groups who used a lot "
        "recently (fair share).",
        "Nothing is wrong. `squeue --start -j <job_id>` shows SLURM's start "
        "estimate and `sprio -j <job_id>` its priority. Smaller requests (fewer "
        "GPUs, a shorter `--time`) can start earlier by filling gaps (backfill).",
    ),
    "Resources": _Entry(
        "queue",
        "Next in line, waiting for free resources",
        "It is at the front of the queue, but no node has the GPUs, CPUs or memory "
        "it asks for free right now. It starts when running jobs end.",
        "Nothing is wrong; wait. If it waits long, ask for less (fewer GPUs, less "
        "`--mem`) or another GPU type: `sinfo -o '%P %G %D %T'` shows what exists "
        "and what is idle.",
    ),
    "Dependency": _Entry(
        "dependency",
        "Waiting for another job",
        "It was submitted with `--dependency` and starts once the job it depends "
        "on has finished (in the way the dependency requires).",
        "Nothing to do. `scontrol show job <job_id> | grep -i dependency` shows "
        "which job it waits for.",
    ),
    "DependencyNeverSatisfied": _Entry(
        "dependency",
        "The job it depends on failed",
        "The job it waits for ended in a way the dependency does not accept (for "
        "`afterok`: it failed or was cancelled), so this job will never start.",
        "Cancel it (`scancel <job_id>`), fix and rerun the job it depended on, then "
        "resubmit. `sbatch --kill-on-invalid-dep=yes` makes SLURM cancel such "
        "jobs for you.",
        "error",
    ),
    "BeginTime": _Entry(
        "schedule",
        "Scheduled for later",
        "It was submitted with a start time (`--begin`) that has not come yet.",
        "Wait, or let it start as soon as possible: `scontrol update "
        "JobId=<job_id> StartTime=now`.",
    ),
    "JobHeldUser": _Entry(
        "hold",
        "Held by you",
        "It was put on hold (`scontrol hold`, or `sbatch --hold`) and will not "
        "start until it is released.",
        "Release it: `scontrol release <job_id>`.",
        "error",
    ),
    "JobHeldAdmin": _Entry(
        "hold",
        "Held by an administrator",
        "A cluster administrator put it on hold. Only they can release it.",
        "Ask your cluster support why, quoting job <job_id>.",
        "error",
    ),
    "launch failed requeued held": _Entry(
        "hold",
        "Start failed; put back and held",
        "SLURM tried to start it, but the launch failed on the node (often a node "
        "or file system problem, or a working directory or output path that does "
        "not exist), so it put the job back in the queue and held it.",
        "Check that its working directory and output paths exist (`scontrol show "
        "job <job_id>`), then `scontrol release <job_id>`. If it fails again, "
        "resubmit avoiding that node: `sbatch --exclude=<node> …`.",
        "error",
    ),
    "JobHoldMaxRequeue": _Entry(
        "hold",
        "Held after too many restarts",
        "It was requeued as often as the cluster allows and is now held.",
        "Find out why it keeps failing (`euler-slurm diagnose <job_id>`), then "
        "`scontrol release <job_id>` or resubmit.",
        "error",
    ),
    "ReqNodeNotAvail": _Entry(
        "nodes",
        "Nodes it needs are unavailable",
        "A node it needs is down, drained (taken out, e.g. for maintenance) or "
        "reserved, often for an upcoming maintenance window. The nodes SLURM lists "
        "are the ones it could not use; if you did not ask for specific nodes, "
        "that is not a problem by itself.",
        "If you used `--nodelist`/`-w`, remove it or choose other nodes. "
        "`sinfo -R` shows why nodes are down, `scontrol show reservation` upcoming "
        "maintenance; a `--time` that ends before it lets the job start now: "
        "`scontrol update JobId=<job_id> TimeLimit=HH:MM:SS`.",
        "warning",
    ),
    "Nodes required for job are DOWN, DRAINED or reserved for jobs in higher "
    "priority partitions": _Entry(
        "nodes",
        "Nodes it needs are down, drained or reserved",
        "The nodes that could run it are down, drained (taken out, e.g. for "
        "maintenance) or kept for jobs of higher-priority partitions.",
        "Usually just wait. `sinfo -R` shows why nodes are down; a smaller request "
        "or another partition may start sooner.",
        "warning",
    ),
    "NodeDown": _Entry(
        "nodes",
        "A node it needs is down",
        "A node it was assigned or asked for is down. It starts when the node is "
        "back, or SLURM finds other nodes.",
        "Wait; if it lasts, cancel and resubmit without `--nodelist` (or with "
        "`--exclude=<node>`).",
        "warning",
    ),
    "Reservation": _Entry(
        "schedule",
        "Waiting for its reservation",
        "It was submitted to a reservation (`--reservation`) that has not started "
        "yet, or whose nodes are busy.",
        "Wait for the reservation; `scontrol show reservation` shows when it starts.",
    ),
    "PartitionDown": _Entry(
        "nodes",
        "Partition is down",
        "Its partition is down (often maintenance). It starts when an administrator "
        "brings the partition back.",
        "Wait, or move it to another partition: `scontrol update JobId=<job_id> "
        "Partition=<other>` (`sinfo -s` lists them).",
        "warning",
    ),
    "PartitionInactive": _Entry(
        "nodes",
        "Partition does not take jobs",
        "Its partition is inactive: it accepts no jobs right now.",
        "Move it to another partition (`scontrol update JobId=<job_id> "
        "Partition=<other>`) or ask cluster support.",
        "warning",
    ),
    "PartitionNodeLimit": _Entry(
        "request",
        "Node count outside the partition's limits",
        "It asks for more (or fewer) nodes than its partition allows per job, or "
        "than the partition has up right now.",
        "Resubmit with a `--nodes` inside the limits (`scontrol show partition "
        "<partition>` shows MinNodes/MaxNodes), or use another partition.",
        "error",
    ),
    "PartitionTimeLimit": _Entry(
        "request",
        "Time limit too long for the partition",
        "It asks for more time (`--time`) than its partition allows, so it will "
        "never start there.",
        "Lower it: `scontrol update JobId=<job_id> TimeLimit=HH:MM:SS` (within the "
        "partition's limit, `sinfo -o '%P %l'`), or cancel and resubmit to a "
        "partition with a longer limit.",
        "error",
    ),
    "PartitionConfig": _Entry(
        "request",
        "Request does not fit the partition",
        "Its request (account, QOS, or resources) is not allowed in this "
        "partition, so it will never start there.",
        "Cancel it and resubmit to a partition that allows it (`scontrol show "
        "partition` lists what each one allows).",
        "error",
    ),
    "BadConstraints": _Entry(
        "request",
        "No node matches the request",
        "No node can satisfy what it asks for together (features, GPU type, CPUs, "
        "memory per node), so it will never start.",
        "Cancel it and resubmit with a request some node satisfies: `sinfo -o "
        "'%P %c %m %G %f'` lists CPUs, memory, GPUs and features per node type.",
        "error",
    ),
    "InvalidAccount": _Entry(
        "request",
        "Account not allowed",
        "It was submitted under an account you may not use (or that does not "
        "exist), so it will never start.",
        "Cancel it and resubmit with `--account=<account>`; `sacctmgr show assoc "
        "user=$USER format=Account,Partition,QOS` lists the accounts you may use.",
        "error",
    ),
    "InvalidQOS": _Entry(
        "request",
        "QOS not allowed",
        "It was submitted with a QOS you may not use (or that does not exist), so "
        "it will never start.",
        "Cancel it and resubmit with `--qos=<qos>`; `sacctmgr show assoc user=$USER "
        "format=Account,QOS%40` lists the QOS you may use.",
        "error",
    ),
    "AccountNotAllowed": _Entry(
        "request",
        "Account not allowed in this partition",
        "The partition does not accept jobs of its account.",
        "Resubmit to another partition, or with another `--account` (`sacctmgr show "
        "assoc user=$USER format=Account,Partition`).",
        "error",
    ),
    "QOSNotAllowed": _Entry(
        "request",
        "QOS not allowed in this partition",
        "The partition does not accept jobs with its QOS.",
        "Resubmit with another `--qos` or to another partition (`scontrol show "
        "partition <partition>` shows AllowQos).",
        "error",
    ),
    "QOSMaxJobsPerUserLimit": _JOBS_LIMIT,
    "QOSMaxJobsPerAccount": _JOBS_LIMIT,
    "QOSGrpJobsLimit": _JOBS_LIMIT,
    "QOSJobLimit": _JOBS_LIMIT,
    "AssocMaxJobsLimit": _JOBS_LIMIT,
    "AssocGrpJobsLimit": _JOBS_LIMIT,
    "AssociationJobLimit": _JOBS_LIMIT,
    "QOSMaxSubmitJobPerUserLimit": _SUBMIT_LIMIT,
    "QOSGrpSubmitJobsLimit": _SUBMIT_LIMIT,
    "AssocMaxSubmitJobLimit": _SUBMIT_LIMIT,
    "AssocGrpSubmitJobsLimit": _SUBMIT_LIMIT,
    "QOSMaxGRESPerUser": _GPU_USER_LIMIT,
    "QOSMaxGRESPerAccount": _GPU_USER_LIMIT,
    "QOSGrpGRES": _GPU_GROUP_LIMIT,
    "AssocGrpGRES": _GPU_GROUP_LIMIT,
    "QOSMaxGRESPerJob": _GPU_PER_JOB,
    "QOSMinGRES": _GPU_PER_JOB,
    "AssocMaxGRESPerJob": _GPU_PER_JOB,
    "QOSMaxCpuPerUserLimit": _CPU_LIMIT,
    "QOSGrpCpuLimit": _CPU_LIMIT,
    "AssocGrpCpuLimit": _CPU_LIMIT,
    "QOSMaxCpuPerJobLimit": _CPU_PER_JOB,
    "AssocMaxCpuPerJobLimit": _CPU_PER_JOB,
    "QOSGrpMemLimit": _MEMORY_LIMIT,
    "AssocGrpMemLimit": _MEMORY_LIMIT,
    "QOSMaxMemoryPerUser": _MEMORY_LIMIT,
    "QOSMaxMemoryPerJob": _MEMORY_PER_JOB,
    "QOSMaxNodePerUserLimit": _NODE_LIMIT,
    "QOSGrpNodeLimit": _NODE_LIMIT,
    "AssocGrpNodeLimit": _NODE_LIMIT,
    "QOSMaxNodePerJobLimit": _NODES_PER_JOB,
    "AssocMaxNodesPerJobLimit": _NODES_PER_JOB,
    "QOSMaxWallDurationPerJobLimit": _WALLTIME_TOO_LONG,
    "AssocMaxWallDurationPerJobLimit": _WALLTIME_TOO_LONG,
    "QOSGrpCPUMinutesLimit": _BUDGET,
    "AssocGrpCPUMinutesLimit": _BUDGET,
    "QOSGrpBillingMinutes": _BUDGET,
    "AssocGrpBillingMinutes": _BUDGET,
    "AssocGrpGRESMinutes": _BUDGET,
    "QOSGrpCPURunMinutesLimit": _RUN_MINUTES,
    "AssocGrpCPURunMinutesLimit": _RUN_MINUTES,
    "AssocGrpGRESRunMinutes": _RUN_MINUTES,
    "QOSGrpBilling": _GENERIC_LIMIT,
    "AssocGrpBilling": _GENERIC_LIMIT,
    "QOSResourceLimit": _GENERIC_LIMIT,
    "AssociationResourceLimit": _GENERIC_LIMIT,
    "QOSUsageThreshold": _Entry(
        "limit",
        "QOS usage threshold reached",
        "The QOS has used as much of its fair share as its usage threshold allows; "
        "it starts when usage decays.",
        "Wait, or submit with another `--qos` you may use.",
        "warning",
    ),
    "JobArrayTaskLimit": _Entry(
        "limit",
        "Array runs its maximum at once",
        "Its job array already runs as many tasks at the same time as allowed "
        "(`--array=…%N`, or the cluster's limit). The next task starts when one ends.",
        "Nothing to do. To allow more at once: `scontrol update JobId=<job_id> "
        "ArrayTaskThrottle=<N>`.",
    ),
    "Licenses": _Entry(
        "schedule",
        "Waiting for a license",
        "A software license it asks for (`--licenses`) is in use; it starts when "
        "one is free.",
        "Wait; `scontrol show licenses` shows how many are in use.",
    ),
    "Cleaning": _Entry(
        "startup",
        "Cleaning up an earlier attempt",
        "SLURM is still cleaning up an earlier run of this job (it was requeued). "
        "It starts right after.",
        "Nothing to do.",
    ),
    "Prolog": _Entry(
        "startup",
        "Preparing its node",
        "Its node runs the cluster's start-up script (prolog). It starts in a moment.",
        "Nothing to do.",
    ),
    "SchedDefer": _Entry(
        "queue",
        "Scheduling deferred",
        "The scheduler was told to look at new jobs later (cluster setting "
        "`defer`). It is considered in the next scheduling pass.",
        "Nothing to do.",
    ),
    "BurstBufferResources": _Entry(
        "queue",
        "Waiting for burst buffer space",
        "The burst buffer space it asks for is in use.",
        "Wait, or ask for less burst buffer space.",
    ),
    "BurstBufferStageIn": _Entry(
        "startup",
        "Staging in files",
        "Its input files are being copied into the burst buffer before it starts.",
        "Nothing to do.",
    ),
    "BurstBufferOperation": _Entry(
        "nodes",
        "Burst buffer operation failed",
        "A burst buffer operation for this job failed.",
        "Check the job's burst buffer directives (`scontrol show job <job_id>`) "
        "and ask cluster support if it repeats.",
        "error",
    ),
    "FrontEndDown": _Entry(
        "nodes",
        "Front end down",
        "The front-end node it needs is down. It starts when it is back.",
        "Wait; ask cluster support if it lasts.",
        "warning",
    ),
    "DeadLine": _Entry(
        "schedule",
        "Cannot finish before its deadline",
        "It cannot start in time to finish before its `--deadline`.",
        "Cancel it and resubmit with a later `--deadline` or a shorter `--time`.",
        "error",
    ),
    "SystemFailure": _Entry(
        "nodes",
        "SLURM or a node failed",
        "A failure of SLURM itself or of a node affected the job.",
        "Resubmit; ask cluster support if it repeats, quoting job <job_id>.",
        "warning",
    ),
    "JobLaunchFailure": _Entry(
        "requeued",
        "Start failed",
        "SLURM could not start it on its node (a node or file system problem, or a "
        "missing working directory).",
        "Check that the working directory and output paths exist, then resubmit "
        "(with `--exclude=<node>` if it was one node).",
        "error",
    ),
    "NonZeroExitCode": _Entry(
        "requeued",
        "Ended with an error",
        "Its batch script exited with a non-zero exit code.",
        "Read the end of its error log: `euler-slurm diagnose <job_id>`.",
        "error",
    ),
    "TimeLimit": _Entry(
        "requeued",
        "Reached its time limit",
        "It ran until its time limit (`--time`) and was stopped.",
        "Ask for more time or save checkpoints and resume (see `euler-slurm "
        "explain TIMEOUT`).",
        "error",
    ),
    "InactiveLimit": _Entry(
        "requeued",
        "Stopped for inactivity",
        "An interactive job (salloc/srun) was idle longer than the cluster allows.",
        "Start a new allocation; keep the session that holds it alive (tmux/screen).",
        "warning",
    ),
}

#: Descriptions squeue can print in place of a code (``state_desc``).
_REASON_LOOKUP: Dict[str, str] = {key.lower(): key for key in REASONS}

# ---------------------------------------------------------------------------
# Job states
# ---------------------------------------------------------------------------

STATES: Dict[str, _Entry] = {
    "PENDING": _Entry(
        "waiting",
        "Waiting to start",
        "It waits in the queue. The pending reason says why.",
        "`euler-slurm job <job_id>` explains the reason in plain words.",
    ),
    "CONFIGURING": _Entry(
        "waiting",
        "Nodes are being prepared",
        "Its nodes are allocated and being booted or configured. It starts in a moment.",
        "Nothing to do.",
    ),
    "RUNNING": _Entry(
        "running",
        "Running",
        "It is running on its node(s).",
        "Follow its output with `tail -f` on its stdout file (`euler-slurm job "
        "<job_id>` shows the path).",
    ),
    "COMPLETING": _Entry(
        "running",
        "Finishing",
        "It has ended; SLURM is cleaning up its processes on the node(s).",
        "Nothing to do; it disappears from the queue shortly.",
    ),
    "STAGE_OUT": _Entry(
        "running",
        "Staging out files",
        "It has ended; its files are being copied out of the burst buffer.",
        "Nothing to do.",
    ),
    "SIGNALING": _Entry(
        "running",
        "Being signaled",
        "SLURM is delivering a signal to it (e.g. before its time limit, as "
        "`--signal` asked).",
        "Nothing to do.",
    ),
    "RESIZING": _Entry(
        "running",
        "Changing size",
        "Its allocation is changing size.",
        "Nothing to do.",
    ),
    "SUSPENDED": _Entry(
        "paused",
        "Suspended",
        "It was paused (by an administrator, or preemption by suspension). It "
        "keeps its nodes but does not run; its time does not count.",
        "Wait for it to resume; ask cluster support if it stays suspended.",
        "warning",
    ),
    "STOPPED": _Entry(
        "paused",
        "Stopped",
        "Its processes were stopped (SIGSTOP); it keeps its nodes.",
        "Resume it: `scontrol resume <job_id>` (if you stopped it), or ask "
        "cluster support.",
        "warning",
    ),
    "REQUEUED": _Entry(
        "waiting",
        "Put back in the queue",
        "It was put back in the queue (after a failure, preemption, or "
        "`scontrol requeue`) and will run again.",
        "Nothing to do; it waits for its turn again.",
    ),
    "REQUEUE_FED": _Entry(
        "waiting",
        "Requeued by the federation",
        "A federated cluster put it back in the queue.",
        "Nothing to do.",
    ),
    "REQUEUE_HOLD": _Entry(
        "held",
        "Requeued and held",
        "It was put back in the queue and held: it will not run until released.",
        "`scontrol release <job_id>`.",
        "warning",
    ),
    "SPECIAL_EXIT": _Entry(
        "held",
        "Held after a special exit",
        "It was requeued after exiting with the cluster's special exit code and is held.",
        "Find out why it exited (`euler-slurm diagnose <job_id>`), then "
        "`scontrol release <job_id>`.",
        "warning",
    ),
    "RESV_DEL_HOLD": _Entry(
        "held",
        "Held: its reservation was deleted",
        "The reservation it was submitted to was deleted, so it is held.",
        "Move it out of the reservation and release it: `scontrol update "
        "JobId=<job_id> Reservation=` then `scontrol release <job_id>`.",
        "warning",
    ),
    "COMPLETED": _Entry(
        "succeeded",
        "Completed",
        "It finished, and its batch script exited with code 0.",
        "Nothing to do; its outputs are where the script wrote them.",
    ),
    "FAILED": _Entry(
        "failed",
        "Failed",
        "Its batch script exited with a non-zero exit code.",
        "Read the end of its error log; `euler-slurm diagnose <job_id>` reads it "
        "for you and names the likely cause.",
        "error",
    ),
    "TIMEOUT": _Entry(
        "failed",
        "Hit its time limit",
        "It ran until its time limit (`--time`) and SLURM stopped it (SIGTERM, "
        "then SIGKILL after a grace period). Work since the last checkpoint is lost.",
        "Ask for more time (`#SBATCH --time=…`, up to the partition's limit: "
        "`sinfo -o '%P %l'`), or make it resumable: save checkpoints regularly and "
        "resume from the last one. `#SBATCH --signal=B:USR1@300` warns the script "
        "5 minutes before the limit so it can save one.",
        "error",
    ),
    "OUT_OF_MEMORY": _Entry(
        "failed",
        "Ran out of memory",
        "It used more CPU memory (RAM) than it asked for, and was killed. (GPU "
        "memory is separate: that shows as `CUDA out of memory` in the log.)",
        "Ask for more (`#SBATCH --mem=64G`, or `--mem-per-cpu=`), or use less: "
        "fewer DataLoader workers, smaller prefetch, memory-mapped data. `sacct -j "
        "<job_id> -o JobID,ReqMem,MaxRSS` shows what it asked for and used.",
        "error",
    ),
    "NODE_FAIL": _Entry(
        "failed",
        "A node failed",
        "A node it ran on failed (hardware, network or system problem). This is "
        "not caused by your code.",
        "Resubmit it (or `scontrol requeue <job_id>`); if the same node fails "
        "again, avoid it: `sbatch --exclude=<node>`. `#SBATCH --requeue` lets "
        "SLURM resubmit it automatically.",
        "error",
    ),
    "BOOT_FAIL": _Entry(
        "failed",
        "A node failed to boot",
        "A node allocated to it did not boot. This is not caused by your code.",
        "Resubmit it; tell cluster support if it repeats.",
        "error",
    ),
    "DEADLINE": _Entry(
        "failed",
        "Missed its deadline",
        "It could not finish before its `--deadline`, so SLURM ended it.",
        "Resubmit with a later `--deadline` or a shorter `--time`.",
        "error",
    ),
    "PREEMPTED": _Entry(
        "interrupted",
        "Preempted",
        "A job with a higher priority (often on a shared or 'scavenger' "
        "partition) needed its resources, so SLURM stopped it.",
        "Resubmit it, or use a partition or QOS that is not preemptible. For long "
        "runs, checkpoint often and add `#SBATCH --requeue` so SLURM restarts it "
        "after preemption.",
        "warning",
    ),
    "CANCELLED": _Entry(
        "cancelled",
        "Cancelled",
        "It was cancelled (`scancel`) by you, an administrator, or a tool acting "
        "for you.",
        "If you did not cancel it, `sacct -j <job_id> -o JobID,State%30` names who "
        "did (`CANCELLED by <uid>`; `id -nu <uid>` turns the uid into a name). "
        "Resubmit if it was a mistake.",
        "warning",
    ),
    "REVOKED": _Entry(
        "cancelled",
        "Revoked",
        "A federated sibling cluster started the job, so this copy was removed.",
        "Nothing to do.",
    ),
}

# ---------------------------------------------------------------------------
# Exit codes
# ---------------------------------------------------------------------------

#: Linux signal names (exit code 128 + n means "killed by signal n").
SIGNAL_NAMES: Dict[int, str] = {
    1: "SIGHUP",
    2: "SIGINT",
    3: "SIGQUIT",
    4: "SIGILL",
    5: "SIGTRAP",
    6: "SIGABRT",
    7: "SIGBUS",
    8: "SIGFPE",
    9: "SIGKILL",
    10: "SIGUSR1",
    11: "SIGSEGV",
    12: "SIGUSR2",
    13: "SIGPIPE",
    14: "SIGALRM",
    15: "SIGTERM",
    16: "SIGSTKFLT",
    17: "SIGCHLD",
    18: "SIGCONT",
    19: "SIGSTOP",
    20: "SIGTSTP",
    21: "SIGTTIN",
    22: "SIGTTOU",
    23: "SIGURG",
    24: "SIGXCPU",
    25: "SIGXFSZ",
    26: "SIGVTALRM",
    27: "SIGPROF",
    28: "SIGWINCH",
    29: "SIGIO",
    30: "SIGPWR",
    31: "SIGSYS",
}

EXIT_CODES: Dict[int, _Entry] = {
    0: _Entry(
        "success", "Success", "The program exited with 0: success.", "Nothing to do."
    ),
    1: _Entry(
        "error",
        "General error",
        "The program reported an error and exited with 1. For Python this usually "
        "means an exception that nothing caught.",
        "The last traceback near the end of the error log names it "
        "(`euler-slurm diagnose <job_id>`).",
        "error",
    ),
    2: _Entry(
        "error",
        "Usage error",
        "Exit code 2 usually means a wrong command-line argument (Python's "
        "argparse exits with 2) or a misused shell command.",
        "Look for `error:` near the end of the log and check the arguments in "
        "your job script.",
        "error",
    ),
    126: _Entry(
        "environment",
        "Command not executable",
        "A command in the job script exists but could not be executed (no execute "
        "permission, or not a program).",
        "`chmod +x <script>`, or run it through its interpreter (`bash script.sh`, "
        "`python script.py`).",
        "error",
    ),
    127: _Entry(
        "environment",
        "Command not found",
        "A command in the job script was not found. Jobs do not run your "
        "interactive shell's setup, so `module load`, `conda activate` or a "
        "virtualenv must be set up in the job script itself.",
        "Check the program name, and add the environment setup to the job script "
        "(`module load …`, `source .venv/bin/activate`).",
        "error",
    ),
    130: _Entry(
        "signal",
        "Interrupted (SIGINT)",
        "It was interrupted with SIGINT, as Ctrl-C does.",
        "If you did not interrupt it, check what sent SIGINT (an `srun` you "
        "stopped, or a wrapper script).",
        "warning",
    ),
    134: _Entry(
        "signal",
        "Aborted (SIGABRT)",
        "The program stopped itself after an internal error: a failed C/C++ "
        "assertion, a CUDA error, or `terminate called after throwing …`.",
        "The lines just before the end of the error log name the error.",
        "error",
    ),
    135: _Entry(
        "signal",
        "Bus error (SIGBUS)",
        "Invalid memory access in native code. In ML jobs often shared memory "
        "(`/dev/shm`) running out with many DataLoader workers, or a memory-mapped "
        "file that changed while being read.",
        "Use fewer DataLoader workers or ask for more memory; do not modify files "
        "while a job reads them.",
        "error",
    ),
    136: _Entry(
        "signal",
        "Arithmetic error (SIGFPE)",
        "Native code did an invalid arithmetic operation (e.g. integer division by zero).",
        "The error log shows where; often a bug in a compiled extension.",
        "error",
    ),
    137: _Entry(
        "signal",
        "Killed (SIGKILL)",
        "It was forcibly stopped with SIGKILL. On SLURM that is most often because "
        "it used more memory than it asked for (the out-of-memory killer), or "
        "because it was cancelled or hit its time limit and did not stop after "
        "SIGTERM.",
        "`sacct -j <job_id> -o JobID,State,ReqMem,MaxRSS` tells which: "
        "OUT_OF_MEMORY (or MaxRSS near ReqMem) means ask for more `--mem`; "
        "CANCELLED or TIMEOUT mean it was stopped.",
        "error",
    ),
    139: _Entry(
        "signal",
        "Segmentation fault (SIGSEGV)",
        "Native code (C/C++/CUDA, a compiled extension) crashed reading invalid "
        "memory. In ML jobs often a library built for another version (PyTorch vs "
        "CUDA or the driver, an extension compiled elsewhere) or a bug in a custom op.",
        "Rebuild compiled extensions in the job's environment and check package "
        "versions (`euler-suite doctor`); `python -X faulthandler …` shows where it "
        "crashed.",
        "error",
    ),
    141: _Entry(
        "signal",
        "Broken pipe (SIGPIPE)",
        "It wrote to a pipe whose reader had gone (e.g. `… | head`).",
        "Usually harmless in shell pipelines; check the pipeline in the job script.",
        "warning",
    ),
    143: _Entry(
        "signal",
        "Terminated (SIGTERM)",
        "It was asked to stop with SIGTERM. SLURM sends it when a job is cancelled, "
        "reaches its time limit, or is preempted.",
        "`sacct -j <job_id> -o JobID,State` says which (CANCELLED, TIMEOUT or PREEMPTED).",
        "warning",
    ),
    255: _Entry(
        "error",
        "Exit status 255",
        "255 is what `exit(-1)` produces, and what `ssh` returns when the connection fails.",
        "Look for the error just before the end of the log.",
        "error",
    ),
}

# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------

_REASON_CODE = re.compile(
    r"^([A-Za-z][A-Za-z0-9_]*)(?:\s*[,:]\s*_?\s*(.*))?$", re.DOTALL
)
_LAUNCH_FAILED = "launch failed requeued held"


def split_reason(raw: str) -> Tuple[str, Optional[str]]:
    """``ReqNodeNotAvail, UnavailableNodes:eu-g1-[001-004]`` → code and detail.

    Also strips the parentheses squeue wraps reasons in, and reads scontrol's
    underscore form (``launch_failed_requeued_held``).
    """
    text = re.sub(r"^\((.*)\)$", r"\1", raw.strip()).strip()
    if text.lower().replace("_", " ").startswith(_LAUNCH_FAILED):
        return _LAUNCH_FAILED, None
    if text.lower() in _REASON_LOOKUP:
        return text, None
    match = _REASON_CODE.match(text)
    if match is None:
        return text, None
    detail = (match.group(2) or "").strip() or None
    return match.group(1), detail


def explain_reason(raw: str) -> Explanation:
    """A pending reason (``Priority``, ``QOSMaxGRESPerUser``, …) in plain words."""
    code, detail = split_reason(raw)
    key = _REASON_LOOKUP.get(code.lower())
    if key is None:
        return Explanation(
            kind="pending_reason",
            code=code,
            known=False,
            label=f"Unrecognised reason {code!r}",
            meaning=(
                f"SLURM gives “{raw.strip()}” as the reason. euler-slurm has "
                "no explanation for this code, so it does not guess one."
            ),
            fix=(
                "`scontrol show job <job_id>` shows the whole job record, and `man "
                "squeue` (section JOB REASON CODES) lists the codes of your SLURM "
                "version. If it stays unclear, ask your cluster support and quote "
                "the job id and this reason."
            ),
            severity="info",
            category=UNKNOWN_CATEGORY,
            detail=detail,
            raw=raw,
        )
    entry = REASONS[key]
    return Explanation(
        kind="pending_reason",
        code=key,
        label=entry.label,
        meaning=entry.meaning,
        fix=entry.fix,
        severity=entry.severity,
        category=entry.category,
        blocked=entry.severity == "error",
        detail=detail,
        raw=raw,
    )


def signal_name(signal: int) -> Optional[str]:
    return SIGNAL_NAMES.get(signal)


#: ``ExitCode=0:125``: Slurm's out-of-memory marker (``SIG_OOM`` = 253 = 128 + 125),
#: not a real signal.
SLURM_OOM_SIGNAL = 125


def explain_exit(exit_code: Optional[int], signal: Optional[int] = None) -> Explanation:
    """An exit status: ``explain_exit(1)``, ``explain_exit(0, 9)`` (= 137)."""
    effective = 128 + signal if signal else exit_code
    raw = f"{exit_code}:{signal}" if signal else str(exit_code)
    if signal == SLURM_OOM_SIGNAL:
        entry = STATES["OUT_OF_MEMORY"]
        return Explanation(
            kind="exit_code",
            code=f"{exit_code or 0}:{signal}",
            label="Killed for using too much memory",
            meaning=(
                "SLURM records `0:125` for jobs it stopped because they used more "
                "memory than they asked for (its out-of-memory marker, not a real "
                "signal). " + entry.meaning
            ),
            fix=entry.fix,
            severity="error",
            category="memory",
            raw=raw,
        )
    if effective is None:
        return Explanation(
            kind="exit_code",
            code="unknown",
            known=False,
            label="Exit code unknown",
            meaning="SLURM did not record how the job's script exited.",
            fix="`sacct -j <job_id> -o JobID,State,ExitCode` shows what was recorded.",
            category=UNKNOWN_CATEGORY,
            raw=raw,
        )
    entry = EXIT_CODES.get(effective)
    if entry is not None:
        return Explanation(
            kind="exit_code",
            code=str(effective),
            label=entry.label,
            meaning=entry.meaning,
            fix=entry.fix,
            severity=entry.severity,
            category=entry.category,
            raw=raw,
        )
    if effective > 128 and signal_name(effective - 128):
        name = signal_name(effective - 128)
        return Explanation(
            kind="exit_code",
            code=str(effective),
            label=f"Killed by signal {effective - 128} ({name})",
            meaning=(
                f"Exit code {effective} means the program was ended by signal "
                f"{effective - 128} ({name}): 128 + the signal number."
            ),
            fix="The last lines of the error log usually show what sent or caused it.",
            severity="error",
            category="signal",
            raw=raw,
        )
    return Explanation(
        kind="exit_code",
        code=str(effective),
        known=False,
        label=f"Exit code {effective}",
        meaning=(
            f"The program chose to exit with {effective}. SLURM gives this code no "
            "meaning of its own; the program (or its documentation) does."
        ),
        fix="Look for an error message near the end of the job's log.",
        severity="error" if effective else "info",
        category="error",
        raw=raw,
    )


def _user_label(uid: str) -> str:
    try:
        import pwd

        return f"{pwd.getpwuid(int(uid)).pw_name} (uid {uid})"
    except (ImportError, KeyError, ValueError, OverflowError):
        return f"uid {uid}"


def explain_state(
    state: str,
    *,
    exit_code: Optional[int] = None,
    signal: Optional[int] = None,
    cancelled_by: Optional[str] = None,
) -> Explanation:
    """A job state (long or short code) in plain words.

    For ``FAILED`` the exit code is explained too; for ``CANCELLED`` who
    cancelled it, when SLURM recorded that.
    """
    parsed = parse_state(state)
    name = parsed.state
    cancelled_by = cancelled_by or parsed.cancelled_by
    entry = STATES.get(name)
    if entry is None:
        return Explanation(
            kind="state",
            code=name,
            known=False,
            label=f"Unrecognised state {name!r}",
            meaning=(
                f"SLURM reports the state “{state.strip()}”. euler-slurm "
                "does not know it, so it does not guess what it means."
            ),
            fix=(
                "`man squeue` (section JOB STATE CODES) lists the states of your "
                "SLURM version; `scontrol show job <job_id>` shows the whole record."
            ),
            category=UNKNOWN_CATEGORY,
            raw=state,
        )
    label, meaning, fix = entry.label, entry.meaning, entry.fix
    if name == "FAILED" and (exit_code is not None or signal):
        exit_explanation = explain_exit(exit_code, signal)
        code = exit_explanation.code
        label = f"{label} (exit code {code})"
        meaning = f"{meaning} Exit code {code}: {exit_explanation.meaning}"
        if exit_explanation.known:
            fix = exit_explanation.fix
    if name == "CANCELLED" and cancelled_by:
        meaning = f"{meaning} SLURM recorded who: {_user_label(cancelled_by)}."
    return Explanation(
        kind="state",
        code=name,
        label=label,
        meaning=meaning,
        fix=fix,
        severity=entry.severity,
        category=entry.category,
        raw=state,
    )


_EXIT_TEXT = re.compile(
    r"^(?:exit(?:[ _-]?code)?[\s:=]*)?(\d+)(?::(\d+))?$", re.IGNORECASE
)


def explain(text: str) -> Explanation:
    """Explain whatever SLURM printed: a state, a pending reason or an exit code.

    ``explain("PD")``, ``explain("QOSMaxGRESPerUser")``, ``explain("137")``,
    ``explain("0:9")``, ``explain("CANCELLED by 12345")``.
    """
    raw = str(text).strip()
    match = _EXIT_TEXT.match(raw)
    if match:
        exit_code, signal = int(match.group(1)), int(match.group(2) or 0)
        return replace(explain_exit(exit_code, signal or None), raw=raw)
    if raw in REASONS:  # exact reason codes first: `DeadLine` is not `DEADLINE`
        return explain_reason(raw)
    parsed = parse_state(raw)
    if parsed.state in LONG_STATES:
        return explain_state(raw, cancelled_by=parsed.cancelled_by)
    return explain_reason(raw)


def explain_job(job: "Job") -> Explanation:
    """The one explanation that matters most for ``job``: why it waits, or how it ended."""
    if job.pending and job.state_reason:
        explanation = explain_reason(job.state_reason)
    else:
        explanation = explain_state(
            job.state,
            exit_code=job.exit_code,
            signal=job.signal,
            cancelled_by=job.cancelled_by,
        )
    return explanation.for_job(job.job_id)


def known_reasons() -> List[str]:
    return sorted(REASONS)


def known_states() -> List[str]:
    return sorted(STATES)


def known_exit_codes() -> List[int]:
    return sorted(EXIT_CODES)


# ---------------------------------------------------------------------------
# The catalog: everything above as data, for tools
# ---------------------------------------------------------------------------

#: squeue's short code of each long state (``PENDING`` → ``PD``).
SHORT_STATES: Dict[str, str] = {long: short for short, long in STATE_CODES.items()}


def catalog() -> Dict[str, Any]:
    """Every explanation euler-slurm knows, as JSON-ready data.

    For tools that show SLURM's codes in plain words without running
    euler-slurm for each one (euler-view vendors it, and reads the current one
    from the ``slurm_capability_report``)::

        {
          "pending_reasons": [{code, kind, label, meaning, fix, severity,
                               blocked, category}],          # sorted by code
          "states": [{code, kind, short, label, meaning, fix, severity,
                      category, terminal, pending}],        # sorted by code
          "exit_codes": [{code, kind, label, meaning, fix, severity,
                          category}],   # "0" … "255" by number, then "0:125"
          "signals": [{number, name}],  # exit code 128 + number
          "categories": {"pending_reason": {code: words}, "state": {…},
                         "exit_code": {…}},
        }

    Texts keep the ``<job_id>`` placeholder (fill it in for a job). The entries
    are what :func:`explain` returns for the code; ``0:125`` is SLURM's
    out-of-memory marker, as :func:`explain_exit` ``(0, 125)`` explains it.
    """
    pending_reasons = []
    for code in known_reasons():
        explanation = explain_reason(code)
        pending_reasons.append(
            {
                "code": explanation.code,
                "kind": explanation.kind,
                "label": explanation.label,
                "meaning": explanation.meaning,
                "fix": explanation.fix,
                "severity": explanation.severity,
                "blocked": explanation.blocked,
                "category": explanation.category,
            }
        )
    states = []
    for code in known_states():
        explanation = explain_state(code)
        states.append(
            {
                "code": explanation.code,
                "kind": explanation.kind,
                "short": SHORT_STATES.get(code),
                "label": explanation.label,
                "meaning": explanation.meaning,
                "fix": explanation.fix,
                "severity": explanation.severity,
                "category": explanation.category,
                "terminal": code in TERMINAL_STATES,
                "pending": code in PENDING_STATES,
            }
        )
    exit_explanations = [explain_exit(code) for code in known_exit_codes()]
    exit_explanations.append(explain_exit(0, SLURM_OOM_SIGNAL))
    exit_codes = [
        {
            "code": explanation.code,
            "kind": explanation.kind,
            "label": explanation.label,
            "meaning": explanation.meaning,
            "fix": explanation.fix,
            "severity": explanation.severity,
            "category": explanation.category,
        }
        for explanation in exit_explanations
    ]
    return {
        "pending_reasons": pending_reasons,
        "states": states,
        "exit_codes": exit_codes,
        "signals": [
            {"number": number, "name": name}
            for number, name in sorted(SIGNAL_NAMES.items())
        ],
        "categories": {
            "pending_reason": dict(REASON_CATEGORIES),
            "state": dict(STATE_CATEGORIES),
            "exit_code": dict(EXIT_CATEGORIES),
        },
    }


__all__ = [
    "EXIT_CATEGORIES",
    "EXIT_CODES",
    "Explanation",
    "REASONS",
    "REASON_CATEGORIES",
    "SHORT_STATES",
    "SIGNAL_NAMES",
    "STATES",
    "STATE_CATEGORIES",
    "UNKNOWN_CATEGORY",
    "catalog",
    "explain",
    "explain_exit",
    "explain_job",
    "explain_reason",
    "explain_state",
    "known_exit_codes",
    "known_reasons",
    "known_states",
    "split_reason",
]
