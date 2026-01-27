# Land the Plane

Wrap up the current session by:

1. **Commit CLAUDE.md** if it has changes (staged or unstaged)
2. **Git push** to remote
3. **Show beads status** with `bd ready`

## Steps

1. Check if CLAUDE.md has any changes:
   ```bash
   git diff --name-only CLAUDE.md
   git diff --staged --name-only CLAUDE.md
   ```

2. If CLAUDE.md has changes, stage and commit it:
   ```bash
   git add CLAUDE.md
   git commit -m "docs: Update CLAUDE.md"
   ```

3. Push to remote:
   ```bash
   git push
   ```

4. Show beads status:
   ```bash
   bd ready
   bd stats
   ```

5. Report what was done to the user.
