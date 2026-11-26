# 07 · Git and GitHub

## 7.1 Adding your SSH key to GitHub

Print your public key (create one with `ssh-keygen -t ed25519` if the file does not exist) and register it on GitHub. The private key must be readable only by you: `chmod 600 ~/.ssh/id_ed25519`.

```bash
cat ~/.ssh/id_ed25519.pub
#copy the output
```
(If you have org access) GitHub → Settings → SSH and GPG keys → New SSH key → paste the copied key → Add SSH key

## 7.2 Adding the GitHub remote
To ensure that your SSH keys are properly loaded into the SSH agent, you can use the following command:

```bash
#git remote set-url github https://github.com/<Organization>/<repository>.git
git remote add github git@github.com:IIS-ACE-lab/TraceFlow.git
git remote set-url github https://github.com/IIS-ACE-lab/TraceFlow.git

# if your self hosted git  repo is ahead  of local relo
git push github dev:dev
```
## 7.3 Keeping GitHub in sync with the self-hosted remote

To ensure that your GitHub repository receives the same commits as your self-hosted repository, you can push your local changes to GitHub using the following command:

```bash

#Confirm your remotes
git remote -v
#Make sure your local dev includes everything from self-hosted origin/dev
git fetch origin
git fetch github
git log --oneline --decorate --graph --max-count=20 dev
#If origin/dev is ahead of your local, sync your local to match it.
#WARNING: reset --hard discards uncommitted changes and local commits; commit or stash first.
git reset --hard origin/dev
#Push your local branch to GitHub
git push github <branch-name>:<branch-name>
