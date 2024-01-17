run_script_path="$(pwd)/run.py"
exp_path="$(pwd)/experiments"


alias tboard='tensorboard --logdir $exp_path --port 6060'


# Set environment variables with the experiment/run name for easier access
function setexp() {
    export CURRENT_EXP="$1"
}
function setrun() {
    export CURRENT_RUN="$1"
}

function setup() {
    echo "------ Experiment Environment Setup ------"
    echo "  Current experiment ---> $CURRENT_EXP"
    echo "  Current run        ---> $CURRENT_RUN"
}