# Changelog

## 0.2.0

Explanations as data for tools, and a verdict on every diagnosis. All JSON
changes are additive (minor versions).

- **Categories.** Every explanation has a `category` from a closed set per
  kind, exported as `REASON_CATEGORIES`, `STATE_CATEGORIES` and
  `EXIT_CATEGORIES` (code → a few plain words). Pending reasons: `queue`,
  `dependency`, `schedule`, `hold`, `limit` (clears by itself), `request`
  (never starts as it is), `budget`, `nodes`, `startup`, `requeued`, `other`.
  States: `waiting`, `held`, `running`, `paused`, `succeeded`, `failed`,
  `interrupted`, `cancelled`. Exit codes: `success`, `error`, `environment`,
  `signal`, `memory` (SLURM's `0:125`). Unknown codes are `unknown`, other
  128 + n exits `signal`, other exit codes `error`. `Explanation.to_dict()`
  carries `category`.
- **The catalog.** `catalog()` (exported from the package) returns every
  explanation as JSON-ready data: `pending_reasons` (sorted by code, with
  `blocked`), `states` (with the squeue short code, `terminal`, `pending`),
  `exit_codes` (`"0"` … `"255"`, then `"0:125"`), `signals` and `categories`.
  Texts keep the `<job_id>` placeholder.
- `slurm_capability_report` **1.1** carries the `catalog`, also when SLURM is
  not available here.
- New envelope `slurm_explanations` **1.0**: `euler-slurm explain --all --json`
  prints the catalog, `euler-slurm explain CODE... --json` the explanation of
  the code (`explanations: [...]`). `euler-slurm explain --all` lists every code
  with its label, by kind. The capability report's `commands` name the
  envelope.
- `euler-slurm diagnose JOB_ID --log PATH` (repeatable; Python:
  `diagnose(..., logs=[...])`, `diagnose_job(job, lines, logs)`): further log
  files, read and scanned like the job's stdout/stderr (stream `log`); a file
  named twice is read once.
- **Logs-only diagnosis.** When SLURM no longer knows the job (aged out of
  `scontrol`, no accounting record) and `--log` files are given, `diagnose`
  reads only them (`diagnose_logs(job_id, logs, lines)`) instead of failing:
  `job` is `null` and the first finding is `job.not_found`. Without `--log` it
  stays the `job_not_found` error.
- **The verdict.** `slurm_diagnosis` **1.1** adds `verdict: {summary, cause,
  fix, code, severity, evidence}` (`Diagnosis.verdict`, a `Verdict`), the one
  thing to tell a newcomer: the job in a few words (its explanation's label, or
  "No SLURM record"), and the cause and fix of the most specific finding: the
  first log finding of severity error, then warning, in the order of the log
  rules (a bare Python exception after the specific ones), else the state or
  pending reason. When SLURM itself ended the job (TIMEOUT, PREEMPTED,
  NODE_FAIL, BOOT_FAIL, DEADLINE, CANCELLED, OUT_OF_MEMORY), a Python exception
  or `Killed` line does not count: the state is the cause. `euler-slurm
  diagnose` prints the verdict first (Summary, Cause, Fix), then the other
  findings.
- Log rules now also yield across files (a `Killed` line in stdout gives way
  to the oom-kill in stderr), and named logs are not checked for SLURM's `%`
  path patterns.

## 0.1.0

First release: SLURM's `squeue`/`sacct`/`scontrol`/`sinfo` as one typed,
versioned JSON CLI, with plain-language explanations (roadmap item SL1 of the
Euler suite's north star).

- `euler_slurm.runner`: one injectable `Runner` protocol
  (`run(args, timeout) -> (rc, stdout, stderr)`) and `SubprocessRunner` (no
  shell). A missing command is a typed `SlurmUnavailable`; a slow one
  `SlurmTimeout`. Commands run with `SLURM_TIME_FORMAT=standard` and without
  `SQUEUE_*`/`SACCT_*`/`SINFO_*`/`SCONTROL_*` variables, so personal settings
  cannot change the output format.
- `euler_slurm.jobs`: `list_jobs(user=None, job_ids=None)` (active jobs from
  squeue, recently ended ones from sacct, with `sources` and `problems`) and
  `get_job(job_id)` (scontrol while SLURM remembers the job, then sacct),
  returning `Job` dataclasses. `--json` is tried first (Slurm 21.08+; data_parser
  v0.0.37 to v0.0.41 number forms), with a fallback to `--format` / `--parsable2` /
  `--oneliner` text. Both paths give the same record, and the tests check that.
  Handles short and long states, JSON state flags, `CANCELLED by <uid>`, `+`
  suffixes, array ids (`123_4`, `123_[1-3%2]`), heterogeneous ids, `UNLIMITED`,
  `D-HH:MM:SS`, `ExitCode=a:b` and the JSON exit forms, GPUs from TRES/GRES (whole
  job), output path patterns (`%A_%a`, `%x-%j`), and names containing the
  delimiter. Filters are applied again to JSON (Slurm before 23.02 ignored them).
- `euler_slurm.explain`: plain-language meaning and fix for 86 pending
  reasons, every job state and the common exit codes (incl. 137/139/143 and
  SLURM's `0:125` out-of-memory marker). Unknown codes get an honest generic
  explanation (`known: false`), never a guess.
- `euler_slurm.diagnose`: `diagnose(job_id, log_tail_lines=200)` combines the
  job record with the end of its stdout/stderr into findings
  `{code, severity, subject, message, hint, evidence}` (the `euler-suite`
  finding shape). Covers CUDA out of memory, host OOM, NCCL/distributed
  timeouts, `ModuleNotFoundError`/`ImportError` (hint: `euler-suite doctor` in
  the job's environment), disk quota / no space, time limit, preemption, node
  failure, cancellation, bare `Killed`, the last Python exception, and jobs near
  their time limit. Reads tqdm `\r` lines and huge logs safely.
- `euler_slurm.report`: envelopes `slurm_jobs`, `slurm_job`, `slurm_diagnosis`
  and `slurm_capability_report` (all 1.0) with `contract: {kind, version}` and
  `package`; errors as the same kind with `error: {code, message}`.
  `capability_report()` (never raises) is registered as the `euler-slurm`
  entry of `euler.capabilities`: availability and reason, SLURM version,
  `--json` support, partitions with time limits and GPUs from `sinfo`, the
  commands, envelopes and explained codes.
- CLI `euler-slurm jobs|job|diagnose|explain|capabilities` (`--json` on the first
  three), also `python -m euler_slurm`.
