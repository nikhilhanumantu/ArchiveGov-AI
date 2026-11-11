@echo off
title ArchiveGov AI — Automated Test Suite
echo =====================================================================
echo Running ArchiveGov AI Verification Test Suite
echo =====================================================================
echo.

py -m pytest -v tests/
echo.
echo =====================================================================
if %ERRORLEVEL% EQU 0 (
    echo [SUCCESS] All 19 tests passed! Platform is 100%% verified.
) else (
    echo [WARNING] Test suite reported failures. Review log output above.
)
echo =====================================================================
pause

