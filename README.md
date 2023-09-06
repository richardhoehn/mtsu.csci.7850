# CSCI 7850 - Deep Learning
CSCI 7850 - Deep Learning class at MTSU in teh Fall of 2023.

This repo will serve for the labs being worked on in this class.

## Login to bio-sim
I used the following command (`ssh rhoehn@login.hpc.svc.cluster.local`) to get acces to biosim. You will need to use your MTSU passord to gain access.

## Runnig Python & Scrips
I had to add teh shebang for python and shell scripts like this:
1. Python => `#!/usr/bin/env python3` at the top of the file.
2. Shell Script => `#!/bin/sh` at the top of the file.

## GII Push and Commit
I use the following commands to push to GIT:
```
git add .
git commit -a -m "Commit Message"
git push
```
You can also add the following to your `~/.bashrc` file like this:
```
gitpush() {
    git add .
    git commit -m "$*"
    git push
}
alias gp=gitpush
```

## Setup GIT on Linux Servers
I setup git on the MTSU servers by using a Persoanl Access token that is open to only the `public` repos I have for class. This can be done by creating the following:
1. Create a file in `~/.git-credentials`.
2. Add a single line in this file like this: `https://richardhoehn:{personal_access_token}@github.com`.
3. Run teh following command: `git config --global credential.helper store`.
