@echo off
:: ============================================================
:: 프리마켓 브리핑 스케줄러 등록 (관리자 권한 필요)
:: - 매일 05:50 AM 실행
:: - 절전 모드에서 깨워서 실행
:: ============================================================

:: 관리자 권한 체크
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] 관리자 권한으로 실행해주세요.
    echo         이 파일을 우클릭 → "관리자 권한으로 실행"
    pause
    exit /b 1
)

set TASK_NAME=StockDailyBlog_PreMarket
set BAT_PATH=C:\Users\ParkEunJin\stock_daily_blog\scripts\pre_market.bat
set WORK_DIR=C:\Users\ParkEunJin\stock_daily_blog

:: 기존 태스크 삭제 (있으면)
schtasks /delete /tn "%TASK_NAME%" /f >nul 2>&1

:: 새 태스크 등록
schtasks /create ^
    /tn "%TASK_NAME%" ^
    /tr "\"%BAT_PATH%\"" ^
    /sc daily ^
    /st 05:50 ^
    /rl HIGHEST ^
    /f

if %errorlevel% neq 0 (
    echo [ERROR] 스케줄 등록 실패
    pause
    exit /b 1
)

:: 절전 모드 해제 옵션 활성화 (XML로 업데이트)
set XML_PATH=%TEMP%\stock_premarket_task.xml
schtasks /query /tn "%TASK_NAME%" /xml > "%XML_PATH%"

:: PowerShell로 WakeToRun 설정 추가
powershell -Command ^
    "$xml = [xml](Get-Content '%XML_PATH%'); ^
     $ns = New-Object Xml.XmlNamespaceManager($xml.NameTable); ^
     $ns.AddNamespace('t','http://schemas.microsoft.com/windows/2004/02/mit/task'); ^
     $settings = $xml.SelectSingleNode('//t:Settings', $ns); ^
     $wake = $settings.SelectSingleNode('t:WakeToRun', $ns); ^
     if ($wake) { $wake.InnerText = 'true' } ^
     else { $el = $xml.CreateElement('WakeToRun','http://schemas.microsoft.com/windows/2004/02/mit/task'); ^
            $el.InnerText = 'true'; ^
            $settings.AppendChild($el) | Out-Null }; ^
     $idle = $settings.SelectSingleNode('t:IdleSettings/t:StopOnIdleEnd', $ns); ^
     if ($idle) { $idle.InnerText = 'false' }; ^
     $disallow = $settings.SelectSingleNode('t:DisallowStartIfOnBatteries', $ns); ^
     if ($disallow) { $disallow.InnerText = 'false' }; ^
     $stopBatt = $settings.SelectSingleNode('t:StopIfGoingOnBatteries', $ns); ^
     if ($stopBatt) { $stopBatt.InnerText = 'false' }; ^
     $xml.Save('%XML_PATH%')"

schtasks /delete /tn "%TASK_NAME%" /f >nul 2>&1
schtasks /create /tn "%TASK_NAME%" /xml "%XML_PATH%" /f

del "%XML_PATH%" >nul 2>&1

if %errorlevel% equ 0 (
    echo.
    echo ============================================================
    echo [OK] 스케줄 등록 완료
    echo   태스크명:  %TASK_NAME%
    echo   실행시간:  매일 05:50 AM
    echo   절전해제:  YES (잠들어 있어도 깨워서 실행)
    echo   배터리:    YES (배터리 모드에서도 실행)
    echo ============================================================
    echo.
    echo 확인: schtasks /query /tn "%TASK_NAME%" /fo LIST /v
) else (
    echo [ERROR] WakeToRun 설정 실패
)

pause
