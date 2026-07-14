@echo off
REM Запуск демо адресного поиска (ФИАС + карта Яндекса).
REM Двойной клик по этому файлу — или запусти из терминала.
cd /d "%~dp0"
echo ================================================================
echo   Address search demo (FIAS + Yandex map)
echo.
echo   Локально:  http://127.0.0.1:8000
echo   По сети — этой машины IPv4 адреса (порт 8000):
python -c "import socket; [print('     http://%%s:8000'%%ip) for ip in {i[4][0] for i in socket.getaddrinfo(socket.gethostname(),None) if i[4][0].count('.')==3 and not i[4][0].startswith('127.')}]" 2>nul
echo.
echo   Остановить: Ctrl+C
echo ================================================================
python -m uvicorn service.app:app --host 0.0.0.0 --port 8000
pause
