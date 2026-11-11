# Publish this prepared repository

The local folder is ready for a new public GitHub repository. Automatic publishing was attempted, but the stored GitHub credential returned HTTP 401. No remote repository was created during preparation.

1. In your GitHub account `djene-mengistu`, create an **empty** public repository named `UAV-Crack-Segmentation`. Do not initialize it with a README, license or `.gitignore`; they are already included here.
2. Open PowerShell in this folder and run:

```powershell
git remote -v
git push -u origin main
```

The prepared `origin` points to `https://github.com/djene-mengistu/UAV-Crack-Segmentation.git`. If your Git authentication needs refreshing, sign in through your normal Git Credential Manager flow, then repeat the push. Keep credentials out of source files.

Both checkpoints are approximately 22 MB and are tracked as ordinary Git files, below GitHub's 100 MB per-file limit. Git LFS and separate release uploads are not required for this initial package.

After the push, open the repository URL and verify that README and both checkpoint files are visible. Then add this link to the submission description:

https://github.com/djene-mengistu/UAV-Crack-Segmentation

Only after that step is it a live code link. `SUBMISSION_DESCRIPTION.md` contains the copy-ready text.
