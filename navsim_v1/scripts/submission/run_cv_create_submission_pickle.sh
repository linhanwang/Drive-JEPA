TEAM_NAME="ke"
AUTHORS="ke_guo"
EMAIL="u3006612@connect.hku.hk"
INSTITUTION="cainiao"
COUNTRY="China"

TRAIN_TEST_SPLIT=navtest

python navsim/planning/script/run_create_submission_pickle.py \
train_test_split=$TRAIN_TEST_SPLIT \
agent=constant_velocity_agent \
experiment_name=submission_my_agent \
team_name=$TEAM_NAME \
authors=$AUTHORS \
email=$EMAIL \
institution=$INSTITUTION \
country=$COUNTRY \
