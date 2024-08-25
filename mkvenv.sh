export PKG_CONFIG_PATH="/opt/homebrew/opt/mysql-client/lib/pkgconfig"
python3 -m venv ./venv
source ./venv/bin/activate
pip install --upgrade pip
pip3 install -r requirements.txt