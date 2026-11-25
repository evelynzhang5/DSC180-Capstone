# Running on DSMLP

- Login: `ssh -i ~/.ssh/id_ed25519_dsmlp <user>@dsmlp-login.ucsd.edu` (Duo push).
- Workspace path on the login node: `/dsmlp/workspaces-fs05/DSC180A_FA26_001/home/<user>/`. Inside a pod the same
  folder is `$HOME`, so pod scripts must use `$HOME/...` paths.
- Sync code from a laptop: `rsync -az -e "ssh -i ~/.ssh/id_ed25519_dsmlp" <folder> <user>@dsmlp-login.ucsd.edu:<workspace>/DSC180-Capstone/`.

## Launch a background GPU pod

```bash
source /opt/launch-sh/lib/kubevars.sh
launch-scipy-ml.sh -W DSC180A_FA26_001 -g 1 -c 8 -m 32 -B -N <pod-name> -- bash -c '<command>'
```

- Pass the command base64-encoded (`echo $B64 | base64 -d | bash`) to avoid quoting problems.
- Inside it: `cd $HOME/DSC180-Capstone`, `exec > >(tee $HOME/runs/<name>-$(date +%Y%m%d-%H%M%S).txt) 2>&1`, run with
  `SMOKE=1` first, check the smoke outputs, then the full run, then `echo "##### ALL DONE"`.
- Logs: `$HOME/runs/`. Pod status: `kubectl get pods`. Clean up: `kubectl delete pod <pod-name>` (source kubevars.sh
  first). GPUs vary (GTX 1080 Ti or A30), and queue waits can reach a few hours.

## Conventions we used

- Every experiment, including smoke tests, runs on a pod. Locally: syntax checks, CPU dry runs, analysis of pulled
  outputs.
- The design and decision rule are frozen and committed before the run. Outputs stay in `outputs/` (not in git);
  small result files are copied into the repo.
