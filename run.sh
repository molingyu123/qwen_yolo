#!/bin/bash
set -e
cd "$(dirname "$0")"

if [ ! -d "venv" ]; then
    echo "创建虚拟环境..."
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip -q
    pip install -r requirements.txt -q
else
    source venv/bin/activate
fi

if [ ! -f ".env" ]; then
    cp .env.example .env
    KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
    sed -i "s|API_KEY=change-me-in-production|API_KEY=${KEY}|" .env
    echo "已生成 .env，API Key: ${KEY}"
fi

echo "启动服务..."
exec python -m app.main