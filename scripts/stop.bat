@echo off
REM Usage: scripts\stop.bat worker | scheduler
if "%1"=="worker" (
  docker compose -f docker-compose.worker.yml down
) else if "%1"=="scheduler" (
  docker compose -f docker-compose.scheduler.yml down
) else (
  echo Usage: scripts\stop.bat worker ^| scheduler
)
