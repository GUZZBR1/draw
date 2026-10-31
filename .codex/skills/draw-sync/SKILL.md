---
name: draw-sync
description: Synchronize the GUZZBR1/draw fork from its santleme/draw parent when requested or when a scheduled check detects missing upstream commits.
---

# Synchronize the draw fork

Use this skill whenever a user requests synchronization or a scheduled check monitors the draw fork.

## Repositories

- Upstream source: `santleme/draw`
- Fork destination: `GUZZBR1/draw`
- Default branch: `main`

## Procedure

1. Confirm with `gh repo view GUZZBR1/draw --json isFork,parent,nameWithOwner` that the destination is still a fork of `santleme/draw`.
2. Compare the upstream and fork branches with:

   ```sh
   gh api repos/GUZZBR1/draw/compare/santleme:main...GUZZBR1:main --jq '{status, ahead_by, behind_by}'
   ```

3. If `behind_by` is `0`, the fork already contains upstream `main`; do not sync even if the branches have different SHAs or the fork is ahead.
4. If `behind_by` is greater than `0`, run:

   ```sh
   gh repo sync GUZZBR1/draw --source santleme/draw --branch main
   ```

5. Compare again and report whether the fork is now current.

Never pass `--force`, reset the fork, push local commits, or merge through the local working tree. Leave uncommitted work untouched. If authentication fails, the fork relationship is uncertain, the comparison fails, or GitHub cannot fast-forward the branch, stop and report the blocker without attempting a destructive repair.

For a scheduled run, follow the time window and expiration specified in its task. Outside the window, perform no repository checks. At or after expiration, disable that exact scheduled task and stop.
