# The ForagerSim project
TODO

## Running lots of experiments (TODO):
Basic flow:

ssh user@server
tmux new -s ratsim          # create session named "ratsim"
# inside tmux, start everything:
./scripts/start_ratsim_headless.sh /path/to/build
cd ~/git/meta_ratsim/ratsim_experiments
source ~/ratvenv/dreamer_venv/bin/activate
python train_dreamerv3.py def=...

# detach: Ctrl-b then d
# now you can close the ssh session / shut down laptop

Reconnect later:
ssh user@server
tmux attach -t ratsim       # or: tmux a
tmux ls                     # list sessions
