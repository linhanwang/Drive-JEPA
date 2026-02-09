TRAIN_TEST_SPLIT=navtest
CACHE_PATH=$NAVSIM_EXP_ROOT/metric_cache

python navsim/planning/script/run_metric_caching.py \
train_test_split=$TRAIN_TEST_SPLIT \
cache.cache_path=$CACHE_PATH