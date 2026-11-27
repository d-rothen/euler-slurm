# euler-slurm

SLURM jobs as typed, versioned JSON, and SLURM explained in plain words.

`squeue`, `sacct` and `scontrol` print terse codes (`PD`, `QOSMaxGRESPerUser`,
`ExitCode=0:9`) in formats that change between SLURM versions. euler-slurm reads
them on the cluster (JSON where SLURM offers it, text otherwise) and gives you:

- **one typed job record** (`Job`) whatever the SLURM version;
- **plain-language explanations with a fix** for every pending reason, job
  state and exit code it knows, and an honest "I don't know this one" for the
  rest;
- **a diagnosis** of a failed job from its record *and* the end of its logs
  (CUDA out of memory, NCCL timeouts, missing packages, full disks, time limits…),
  led by one verdict: summary, cause, fix;
- **versioned JSON envelopes** (`--json`) for tools such as euler-view.

Part of the Euler suite. Zero runtime dependencies, Python 3.9+.

```sh
pip install euler-slurm
```

## Commands

```console
$ euler-slurm jobs
JOB ID            NAME                    STATE          TIME/LIMIT       GPUS          WHERE / WHY
91000003          ev-rtx-1120-1430        RUNNING        1:19:50/4:00:00  1x rtx_4090   eu-g6-023
91000009          metric_train-1120-1243  PENDING        0:00/2-00:00:00  1 GPU         Waiting for its turn (Priority)
91000020_[2-4%2]  sweep                   PENDING        0:00/4:00:00     1 GPU         Array runs its maximum at once (JobArrayTaskLimit)
91000014          multinode               PENDING        0:00/1-00:00:00  8x rtx_4090   Nodes it needs are unavailable (ReqNodeNotAvail, UnavailableNodes:eu-g1-[001-004])
91000015          held-eval               PENDING        0:00/4:00:00     1 GPU         Held by you (JobHeldUser)
90999002          big-data                OUT_OF_MEMORY  10:00/4:00:00    1x rtx_4090   Ran out of memory
90999001          train                   FAILED         42:00/4:00:00    1x rtx_4090   Failed (exit code 1)

$ euler-slurm explain QOSMaxGRESPerUser
QOSMaxGRESPerUser: At your GPU limit (pending reason)
  Meaning:    Your running jobs already use as many GPUs as your QOS allows per user (or
              per account). It starts by itself when some of your GPUs are released.
  What to do: Wait, or cancel or shrink other GPU jobs of yours. `sacctmgr show qos
              format=Name,MaxTRESPerUser%40,MaxTRESPerAccount%40` shows the limit.

$ euler-slurm diagnose 90999001
Job 90999001 (train) - FAILED, exit code 1

  Summary:    Failed (exit code 1)
  Cause:      Python could not import `euler_loading`: the job's environment is missing
              it, or has a version that does not fit.
  Fix:        Run `euler-suite doctor` in the job's environment (the same `module load`
              / `conda activate` / `source .venv/bin/activate` lines the job script
              runs) to see which packages are missing or disagree, then install them
              there.
  Log line:   ModuleNotFoundError: No module named 'euler_loading'
  In:         /cluster/home/alice/proj/train-90999001.err

Also found:

  [error]     Failed (exit code 1): Its batch script exited with a non-zero exit code.
              Exit code 1: The program reported an error and exited with 1. For Python
              this usually means an exception that nothing caught.
  Fix:        The last traceback near the end of the error log names it (`euler-slurm
              diagnose 90999001`).
```

| Command | What it does | `--json` envelope |
|---|---|---|
| `euler-slurm jobs [--user U] [--since-hours N] [--active]` | your active jobs (squeue) and the ones that ended in the last N hours (sacct) | `slurm_jobs` 1.0 |
| `euler-slurm job <id>` | one job in detail: state or pending reason explained, resources, times, log paths | `slurm_job` 1.0 |
| `euler-slurm diagnose <id> [--log PATH]... [--lines N]` | why it failed or waits: the verdict (summary, cause, fix), then every finding from the record and the end of its stdout/stderr and of each `--log` file. When SLURM no longer knows the job, `--log` files alone are enough | `slurm_diagnosis` 1.1 |
| `euler-slurm explain <REASON\|STATE\|EXIT_CODE>` | `Priority`, `PD`, `TIMEOUT`, `137`, `0:9`, `launch failed requeued held` … | `slurm_explanations` 1.0 (`explanations`) |
| `euler-slurm explain --all` | every code it can explain, by kind (`--json`: the whole catalog) | `slurm_explanations` 1.0 (`catalog`) |
| `euler-slurm capabilities` | what SLURM offers here: version, `--json` support, partitions with time limits and GPUs | `slurm_capability_report` 1.1 (always JSON) |

Job ids are `123`, `123_4` (an array task), `123_[1-3]` (array tasks still
pending) or `123+1` (a heterogeneous job component). Exit status: 0 success,
1 the command failed (unknown job, SLURM error), 2 wrong usage, 3 SLURM is not
installed here. Global options: `--no-slurm-json` (read SLURM's text output
only) and `--timeout SECONDS`.

## What does my pending reason mean?

`squeue` shows why a job has not started in its `NODELIST(REASON)` column. The
common ones, in short (`euler-slurm explain <reason>` gives the full text and
the fix for your job):

| Reason | What it means | What to do |
|---|---|---|
| `Priority` | Other jobs are ahead of yours in the queue. Normal. | Wait. `squeue --start -j <id>` shows SLURM's estimate. Smaller jobs (fewer GPUs, shorter `--time`) start sooner. |
| `Resources` | Yours is next, but the GPUs/CPUs/memory it asks for are busy. Normal. | Wait, or ask for less / another GPU type (`sinfo -o '%P %G %D %T'`). |
| `Dependency` | It waits for the job named in `--dependency`. | Nothing to do. |
| `DependencyNeverSatisfied` | The job it waited for failed, so this one **never** starts. | `scancel <id>`, fix the first job, resubmit. |
| `BeginTime` | It was submitted with `--begin` in the future. | Wait, or `scontrol update JobId=<id> StartTime=now`. |
| `JobHeldUser` | It is on hold (`scontrol hold` / `sbatch --hold`). | `scontrol release <id>`. |
| `JobHeldAdmin` | An administrator put it on hold. | Ask cluster support. |
| `QOSMaxJobsPerUserLimit` | You already run as many jobs as your QOS allows. | Wait for one to end, or cancel one. |
| `AssocMaxJobsLimit` | Same, a limit of your account (association). | Wait, or cancel a job. |
| `QOSMaxGRESPerUser` | Your jobs already use your GPU limit. | Wait, or shrink/cancel other GPU jobs. |
| `AssocGrpGRES` | Your **group** uses all its GPUs (colleagues' jobs count too). | Wait; fewer GPUs per job start sooner. |
| `ReqNodeNotAvail` | A node it needs is down, drained or reserved (often maintenance). | Drop `--nodelist`; before a maintenance, a shorter `--time` lets it start now. |
| `PartitionTimeLimit` | `--time` is longer than the partition allows: it **never** starts. | `scontrol update JobId=<id> TimeLimit=HH:MM:SS`, or another partition. |
| `PartitionNodeLimit` | More (or fewer) nodes than the partition allows. | Resubmit with a fitting `--nodes`. |
| `InvalidAccount` | Submitted under an account you may not use: **never** starts. | Resubmit with `--account=…` (`sacctmgr show assoc user=$USER`). |
| `InvalidQOS` | Submitted with a QOS you may not use: **never** starts. | Resubmit with `--qos=…`. |
| `launch failed requeued held` | SLURM could not start it on the node and held it. | Check the work dir and output paths exist, then `scontrol release <id>`. |
| `JobArrayTaskLimit` | The array already runs its maximum number of tasks at once. | Nothing to do. |

euler-slurm knows 86 reason codes (`euler-slurm explain --all` lists them). Codes it does not know are reported as
unknown, with a pointer to `man squeue` (JOB REASON CODES). It does not guess
from the code's name.

### And when a job has ended?

| State / exit | Plain words | Usual fix |
|---|---|---|
| `TIMEOUT` | It reached `--time` and was stopped. | More `--time`, or checkpoint and resume (`--signal=B:USR1@300`). |
| `OUT_OF_MEMORY`, exit `0:125` | It used more RAM than `--mem`. (GPU memory shows as `CUDA out of memory` in the log.) | More `--mem`, fewer DataLoader workers; `sacct -j <id> -o ReqMem,MaxRSS`. |
| `NODE_FAIL` | A node broke. Not your code. | Resubmit, `--exclude=<node>`. |
| `PREEMPTED` | A higher-priority job took the resources. | Resubmit; `--requeue` and frequent checkpoints. |
| `CANCELLED` | `scancel` (by you, an admin or a tool). | `sacct -j <id> -o State%30` shows who. |
| `FAILED`, exit 1 | Your program raised an error. | The last traceback in the log (`euler-slurm diagnose <id>`). |
| exit 137 (`0:9`) | Killed with SIGKILL: usually out of memory, or stopped after a cancel/time limit. | `sacct -j <id> -o State,ReqMem,MaxRSS`. |
| exit 139 (`0:11`) | Segmentation fault in native code: often mismatched CUDA/PyTorch builds. | Rebuild extensions; `euler-suite doctor`. |
| exit 143 (`0:15`) | SIGTERM: cancelled, time limit or preemption. | `sacct -j <id> -o State` says which. |

## Python API

```python
from euler_slurm import list_jobs, get_job, diagnose, explain, capability_report

listing = list_jobs()  # your jobs; list_jobs(job_ids=["123_4"])
for job in listing:  # Job dataclasses
    print(job.job_id, job.state, job.state_reason, job.gpus, job.time_used_s)
print(listing.sources, listing.problems)  # how it was read, what is missing

job = get_job("123")  # scontrol while SLURM remembers it, then sacct
print(explain(job.state_reason or job.state).fix)

report = diagnose("123", log_tail_lines=200, logs=["/scratch/alice/run.log"])
print(report.verdict.summary, report.verdict.cause, report.verdict.fix)
for finding in report.findings:  # {code, severity, subject, message, hint}
    print(finding.severity, finding.message, finding.hint)

from euler_slurm import catalog

catalog()["pending_reasons"][0]  # {code, kind, label, meaning, fix, …, category}
```

Everything goes through a `Runner` (`run(args, timeout) -> (rc, stdout, stderr)`),
so tests and remote tools can pass their own: `list_jobs(runner=my_runner)`.
The default runs the commands on this machine without a shell; a missing
command raises `SlurmUnavailable`, never a crash.

`Job` fields: `job_id`, `raw_job_id`, `array_job_id`, `array_task_id`,
`array_task_range`, `name`, `user`, `state` (normalized long name),
`state_flags`, `state_reason`, `cancelled_by`, `partition`, `nodes`,
`node_count`, `cpus`, `gpus` (`{count, type}` for the whole job, from TRES or
GRES), `time_used_s`, `time_limit_s`, `time_limit_unlimited`, `submit_time`,
`start_time`, `end_time` (ISO 8601 with offset; estimates while the job has not
ended), `exit_code`, `signal`, `shell_exit_code`, `work_dir`, `stdout_path`,
`stderr_path`, `source`.

## How it reads SLURM

| Facts | Command | JSON (Slurm ≥ 21.08) | Fallback |
|---|---|---|---|
| active jobs | `squeue` | `squeue --json` | `squeue --noheader --format '%i\|%A\|…\|%j'`, plus `--Format JobID,tres-alloc` for `--gpus` jobs |
| ended jobs | `sacct --allocations` | `sacct --json` | `sacct --noheader --parsable2 --format=JobID,…,JobName` |
| one job | `scontrol show job` | `scontrol --json show job` | `scontrol --oneliner show job` |
| partitions | `sinfo` | – | `sinfo --noheader --format '%P\|%a\|%l\|%D\|%c\|%m\|%G'` |

- JSON is tried first; when it is not available the text form is read, and
  both produce the same `Job` (the tests check that they agree). Which one
  answered is in `sources` / `Job.source`.
- User and job-id filters are applied again to JSON, because Slurm before
  23.02 ignored them with `--json`.
- Commands run with `SLURM_TIME_FORMAT=standard` and without `SQUEUE_*`,
  `SACCT_*`, `SINFO_*`, `SCONTROL_*` variables, so personal settings cannot
  change the output format.
- Handled forms: short and long states, `CANCELLED by <uid>`, `+` suffixes,
  JSON state flag lists (`["COMPLETED", "COMPLETING"]` is shown as
  `COMPLETING`, like squeue does), array ids `123_4` and `123_[1-3%2]`,
  `UNLIMITED`/`infinite`, `1-02:03:04` durations, both JSON number forms
  (plain and `{set, infinite, number}`), `ExitCode=0:125` (SLURM's out-of-memory
  marker), names containing the delimiter.
- No accounting database (`sacct` fails)? Active jobs are still listed and the
  listing says that finished jobs are missing (`problems`).

## For tools: envelopes and the capability report

Every JSON document has `contract: {kind, version}` and `package: {name,
version}`. Readers should check the kind and the major version and ignore
fields they do not know. A failed command prints the same kind with
`error: {code, message}` (`slurm_unavailable`, `job_not_found`,
`slurm_command_failed`, `invalid_argument`).

| Envelope | Version | Printed by |
|---|---|---|
| `slurm_jobs` | 1.0 | `euler-slurm jobs --json` |
| `slurm_job` | 1.0 | `euler-slurm job <id> --json` |
| `slurm_diagnosis` | 1.1 | `euler-slurm diagnose <id> --json`: `job` (`null` when only logs were read), `summary`, `verdict`, `findings`, `logs` |
| `slurm_explanations` | 1.0 | `euler-slurm explain --all --json` (`catalog`), `euler-slurm explain <code> --json` (`explanations`) |
| `slurm_capability_report` | 1.1 | `euler-slurm capabilities`, the `euler.capabilities` entry point |

`euler-slurm` registers `capability_report` in the suite's
`euler.capabilities` entry-point group, so `euler-suite report`
(euler-dataset-contract) includes a `slurm_capability_report` 1.1: whether SLURM
is available here (and why not), its version, whether `--json` is supported,
the partitions with their time limits, node counts and GPUs (`{type,
per_node, nodes, total}`), the commands and envelopes this package offers, the
codes it can explain and the whole explanation `catalog`. The report never
raises; on a laptop it says `available: false` and still carries the catalog.

**The catalog** (`catalog()`, `explain --all --json`, and `catalog` in the
capability report) is every explanation as data, so a tool can show SLURM's
codes in plain words without asking euler-slurm each time: `pending_reasons`
(`{code, kind, label, meaning, fix, severity, blocked, category}`, sorted by
code), `states` (also `short`, e.g. `PD`, and `terminal`/`pending`),
`exit_codes` (`"0"` … `"255"`, then SLURM's out-of-memory marker `"0:125"`),
`signals` (`{number, name}`: exit code 128 + number) and `categories` (each
kind's closed set, code → a few words). Categories group codes by what they
mean for the user: pending reasons are `queue`, `dependency`, `schedule`,
`hold`, `limit` (clears by itself), `request` (never starts as it is), `budget`,
`nodes`, `startup`, `requeued` or `other`; states `waiting`, `held`, `running`,
`paused`, `succeeded`, `failed`, `interrupted` or `cancelled`; exit codes
`success`, `error`, `environment`, `signal` or `memory`. A code euler-slurm
does not know has category `unknown`. Texts keep the `<job_id>` placeholder.

**The verdict** of `slurm_diagnosis` is the one thing to tell a newcomer, chosen
by euler-slurm, so tools show it instead of ranking findings themselves:
`{summary, cause, fix, code, severity, evidence}`. `summary` is the job in a few
words ("Failed (exit code 1)", "Waiting for its turn", "No SLURM record");
`cause` and `fix` come from the most specific finding: a cause found in the
logs (CUDA out of memory beats a bare Python traceback), else the state or
pending reason. When SLURM itself stopped the job (time limit, preemption,
node failure, cancellation, out of memory), a traceback or `Killed` line is
taken as its consequence and the state stays the cause. `evidence` is the log
line it rests on (`{path, line}`) or `null`. Tools that know more log files
than SLURM pass them with `--log PATH` (repeatable); when SLURM has forgotten
the job, `diagnose --log` still answers from the logs, with `job: null` and a
first finding `job.not_found`.

## Development

```sh
uv venv && uv pip install -e '.[dev]'
pytest
ruff check .
```

The tests run against recorded outputs in `tests/fixtures/` (Slurm 23.11 and
21.08 JSON, `--format`/`--parsable2` text, `scontrol -o`, `sinfo`) through a fake
runner; no SLURM is needed.

## License

MIT
