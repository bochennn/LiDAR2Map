CODE_ROOT=$(realpath $(dirname $0)/../..)

cd $CODE_ROOT
PYTHONPATH=$CODE_ROOT python3 tools/create_data.py ${@:1}