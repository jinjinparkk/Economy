@echo off
cd /d C:\Users\ParkEunJin\stock_daily_blog
py -3 -m src.main --pre-market >> output\pre_market.log 2>&1
