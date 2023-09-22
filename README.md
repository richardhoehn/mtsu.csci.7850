# CSCI 7850 - Deep Learning
CSCI 7850 - Deep Learning class at MTSU in teh Fall of 2023.

This repo will serve for the labs being worked on in this class.

## Jupyter Lab
Using the following coomand works for me:
`docker run -it --rm -p 8888:8888 --user root -e JUPYTER_ENABLE_LAB=yes -e GRANT_SUDO=yes -v "$(pwd)":/home/jovyan/work jlphillips/csci:2023-Fall`

## Login to bio-sim
I used the following command (`ssh rhoehn@login.hpc.svc.cluster.local`) to get acces to biosim. You will need to use your MTSU passord to gain access.
This works: `for((x=0;x<10;x++)); do srun -c 4 apptainer exec /home/shared/sif/csci-2023-Fall.sif ./OL1.sh; done;`

## Hamiton

Run this on Hamilton
`srun -G 1 -p research apptainer run --nv --env NB_UID=${UID} --writable-tmpfs -H jlab:${HOME} --env NOTEBOOK_ARGS="--NotebookApp.base_url=/biosim/user/${USER}/proxy/absolute/9000/ --NotebookApp.custom_display_url=https://jupyterhub.cs.mtsu.edu" /home/shared/sif/csci-2023-Fall.sif`

`srun -G 1 -p research apptainer run --nv --env NB_UID=${UID} --writable-tmpfs -H mtsu.csci.7850:${HOME} --env NOTEBOOK_ARGS="--NotebookApp.base_url=/biosim/user/${USER}/proxy/absolute/9000/ --NotebookApp.custom_display_url=https://jupyterhub.cs.mtsu.edu" /home/shared/sif/csci-2023-Fall.sif`

Then...

`ssh -L 9000:cX:8888 rhoehn@hamilton.cs.mtsu.edu`

`cat slurm* | grep "Validation accuracy:" |  cut -c 22-`


## Create And Link SSH Keys - No Password Needed
1. SSH into device that you want to add the public key to.
2. Create the folders: `mkdir -p ~/.ssh` and `touch ~/.ssh/authorized_keys`
3. Run the following: `cat ~/.ssh/id_rsa.pub  | ssh rhoehn@login.hpc.svc.cluster.local bash -c "cat >> .ssh/authorized_keys"`
4. Should work now.

If the SSH keys get messed up us the following command to clear them. Do this on the local system.

```
ssh-keygen -f "/home/jovyan/.ssh/known_hosts" -R "login.hpc.svc.cluster.local"
```



## Runnig Python & Scrips
I had to add teh shebang for python and shell scripts like this:
1. Python => `#!/usr/bin/env python3` at the top of the file.
2. Shell Script => `#!/bin/sh` at the top of the file.
3. You can also run multiple times the same sheel script like this: `for((x=0;x<100;x++)); do ./OL1.sh; done;`

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
3. Run the following command: `git config --global credential.helper store`.
