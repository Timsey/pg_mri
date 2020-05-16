#!/usr/bin/env bash

ssh taco -R 10000:146.50.28.109:22
cd timu/
sshfs -p 10000 -o idmap=user,nonempty,allow_other,default_permissions timsey@127.0.0.1:/home/timsey/ ./mnt

cd mnt/tmp/pycharm_project_042/

bash taco_runfiles/taco_train_RL_self.sh



sudo sshfs -o allow_other,default_permissions timsey@146.50.28.109:/home/timsey/ ./mnt/

sshfs -o allow_other,default_permissions timsey@146.50.28.109:/home/timsey/ ./mnt/