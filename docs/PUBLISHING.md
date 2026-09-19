# Publishing to GitHub

## 1. Complete the release metadata

Before publishing, complete `PUBLIC_RELEASE_CHECKLIST.md`. In particular, add a
license, final citation metadata, the OSM redistribution decision, and the
correct GitHub account name.

## 2. Create the remote repository

Create an empty GitHub repository named `crowdsys`. Do not add
a README, `.gitignore`, or license from the GitHub interface because those files
are managed locally.

## 3. Review and commit locally

```bash
git status
git add .
git commit -m "Initial public release"
```

## 4. Connect and push

HTTPS:

```bash
git remote add origin https://github.com/km0204/crowdsys.git
git push -u origin main
```

SSH:

```bash
git remote add origin git@github.com:km0204/crowdsys.git
git push -u origin main
```

## 5. Verify the public repository

- Confirm that the documented smoke runs and all tests pass.
- Confirm that the GitHub Actions test matrix passes.
- Check the repository for local paths, credentials, restricted data, and
  unpublished personal information.
- Create a version tag after verification:

```bash
git tag -a v0.2.0 -m "Intervention portfolio release"
git push origin v0.2.0
```
